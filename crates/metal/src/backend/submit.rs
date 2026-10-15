//! One-shot Metal queue submission and host completion evidence.
//!
//! Core supplies the complete order and owns graph policy. This module accepts
//! only the implemented single graphics lane, Commands, and one optional
//! Present step, committing each native command buffer in Core's order. It does
//! not build a CPU command list or schedule/reorder GPU work. Completion
//! handlers observe Metal's terminal command-buffer result; cancellation only
//! drops the observer.

use super::{Metal, MetalDevice};
use fluxel_rhi_core::api::{
    self, CompletionBackend, CompletionState, DeferredBackend, Error, LaneId, Result, StepKind,
    SubmissionPlan, SubmitOutcome,
};
#[cfg(target_vendor = "apple")]
use fluxel_rhi_core::api::{CompletionCoverage, CompletionPoint, StepId, StepPayload};
use std::{
    future::Future,
    pin::Pin,
    sync::{Arc, Mutex},
    task::{Context, Poll, Waker},
};

#[cfg(target_vendor = "apple")]
type NativeCommandBuffer =
    objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandBuffer>>;
#[cfg(target_vendor = "apple")]
type CheckpointMarker = (
    StepId,
    NativeCommandBuffer,
    Completion,
    Vec<(LaneId, StepId)>,
);

#[derive(Clone)]
pub struct Completion(Arc<CompletionInner>);

struct CompletionInner {
    result: Mutex<Option<Result<()>>>,
    wakers: Mutex<Vec<Waker>>,
}

impl Completion {
    fn new() -> Self {
        Self(Arc::new(CompletionInner {
            result: Mutex::new(None),
            wakers: Mutex::new(Vec::new()),
        }))
    }

    fn result(&self) -> Option<Result<()>> {
        self.0
            .result
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    fn publish(&self, result: Result<()>) {
        let mut slot = self
            .0
            .result
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if slot.is_some() {
            return;
        }
        *slot = Some(result);
        drop(slot);

        let wakers = std::mem::take(
            &mut *self
                .0
                .wakers
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()),
        );
        for waker in wakers {
            // A user-provided waker must not unwind through Metal's Objective-C
            // callback boundary.
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| waker.wake()));
        }
    }

    #[cfg(target_vendor = "apple")]
    fn install(
        command: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
    ) -> Self {
        use objc2_metal::{MTLCommandBuffer as _, MTLCommandBufferStatus};

        let completion = Self::new();
        let callback_completion = completion.clone();
        let callback = block2::RcBlock::new(
            move |pointer: std::ptr::NonNull<
                objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
            >| {
                // SAFETY: Metal invokes a completion handler with its live
                // command buffer, and the NonNull argument is valid for this
                // synchronous callback invocation.
                let native = unsafe { pointer.as_ref() };
                let result = if native.status() == MTLCommandBufferStatus::Completed {
                    Ok(())
                } else if native.error().is_some_and(|error| {
                    error.code() == objc2_metal::MTLCommandBufferError::DeviceRemoved.0 as isize
                }) {
                    Err(Error::DeviceLost)
                } else {
                    Err(Error::Backend(
                        "Metal command buffer execution failed".into(),
                    ))
                };
                callback_completion.publish(result);
            },
        );
        // SAFETY: `callback` is an owning block and remains alive through this
        // call; Metal's addCompletedHandler contract copies the block and calls
        // it once after the command buffer reaches a terminal execution state.
        unsafe { command.addCompletedHandler(block2::RcBlock::as_ptr(&callback).cast()) };
        completion
    }
}

pub struct CompletionFuture(Completion);

pub struct MetalRecordedWork {
    #[cfg(target_vendor = "apple")]
    command: objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandBuffer>>,
    #[cfg(target_vendor = "apple")]
    indirect_arguments:
        Vec<objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>>,
}

impl MetalRecordedWork {
    #[cfg(target_vendor = "apple")]
    fn retain_indirect_arguments_until_completed(&self) {
        use objc2_metal::MTLCommandBuffer as _;
        if self.indirect_arguments.is_empty() {
            return;
        }
        let retained = self.indirect_arguments.clone();
        let callback = block2::RcBlock::new(
            move |_: std::ptr::NonNull<
                objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
            >| {
                let _ = retained.len();
            },
        );
        // SAFETY: the owning block captures strong references to every indirect
        // argument buffer and is copied by Metal. It is installed before commit
        // and released only after the command buffer reaches terminal status.
        unsafe {
            self.command
                .addCompletedHandler(block2::RcBlock::as_ptr(&callback).cast())
        };
    }
}

impl Future for CompletionFuture {
    type Output = Result<()>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let completion = &self.get_mut().0;
        let mut wakers = completion
            .0
            .wakers
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(result) = completion.result() {
            return Poll::Ready(result);
        }
        if !wakers.iter().any(|waker| waker.will_wake(cx.waker())) {
            wakers.push(cx.waker().clone());
        }
        Poll::Pending
    }
}

impl CompletionBackend for Metal {
    type WaitFuture<'a> = CompletionFuture;

    fn completion_status(_: &MetalDevice, point: &Completion) -> CompletionState {
        match point.result() {
            Some(Ok(())) => CompletionState::Complete,
            Some(Err(error)) => CompletionState::Failed(error),
            None => CompletionState::Pending,
        }
    }

    fn wait_completion<'a>(_: &'a MetalDevice, point: &'a Completion) -> Self::WaitFuture<'a> {
        CompletionFuture(point.clone())
    }

    fn wait_idle(device: &MetalDevice) -> Self::WaitFuture<'_> {
        #[cfg(target_vendor = "apple")]
        {
            use objc2_metal::MTLCommandQueue as _;
            let completion = (|| {
                let command = device
                    ._state
                    ._queue
                    .commandBuffer()
                    .ok_or(Error::OutOfMemory)?;
                let completion = Completion::install(&command);
                {
                    let _gate = device
                        ._state
                        .queue_gate
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    // The marker is enqueued under the same issue-order gate as
                    // normal submission, so it covers the preceding queue prefix.
                    use objc2_metal::MTLCommandBuffer as _;
                    command.commit();
                }
                Ok(completion)
            })();
            CompletionFuture(completion.unwrap_or_else(|error| {
                let failed = Completion::new();
                failed.publish(Err(error));
                failed
            }))
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = device;
            let failed = Completion::new();
            failed.publish(Err(Error::Unsupported("Metal requires an Apple target")));
            CompletionFuture(failed)
        }
    }
}

impl DeferredBackend for Metal {
    type NativeRecordedWork = MetalRecordedWork;

    fn finish_encoder(
        encoder: super::MetalCommandEncoder,
        lane: LaneId,
    ) -> Result<Self::NativeRecordedWork> {
        if lane.ordinal != 0 {
            return Err(Error::InvalidState);
        }
        #[cfg(target_vendor = "apple")]
        {
            if encoder.render.is_some() || encoder.compute.is_some() {
                return Err(Error::InvalidState);
            }
            // The queue's commandBuffer() factory retains native objects used
            // in encoded commands. The backend does not duplicate every bind
            // in a Rust keep-alive vector; Core still owns public resource
            // lifetime decisions through actual GPU completion.
            Ok(MetalRecordedWork {
                command: encoder.command,
                indirect_arguments: encoder.indirect_arguments,
            })
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = encoder;
            Err(Error::Unsupported("Metal requires an Apple target"))
        }
    }

    // SAFETY: Core's submit contract requires the caller to keep every resource
    // and synchronization object reachable from `packet` alive until the
    // returned completion frontier is truly complete. This implementation
    // preserves the packet's Step order and reports only command buffers that
    // were committed to Metal; partial acceptance carries a completion marker.
    unsafe fn submit(device: &MetalDevice, packet: api::SubmitPacket<Self>) -> SubmitOutcome<Self> {
        // Validate every unsupported shape before consuming RecordedWork. This
        // preserves Core's clean-rejection ownership contract.
        if !supports_command_plan(&packet.plan) {
            return SubmitOutcome::Rejected {
                error: Error::Unsupported("Metal supports Commands on one graphics lane"),
                work: api::RetryableSubmit::Fresh(packet),
            };
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = device;
            SubmitOutcome::Rejected {
                error: Error::Unsupported("Metal requires an Apple target"),
                work: api::RetryableSubmit::Fresh(packet),
            }
        }
        #[cfg(target_vendor = "apple")]
        {
            use objc2_metal::{MTLCommandBuffer as _, MTLCommandQueue as _};
            let present_steps = packet
                .steps
                .iter()
                .filter_map(|step| {
                    matches!(step.payload, StepPayload::Present(_)).then_some(step.step)
                })
                .collect::<Vec<_>>();
            let mut present_commands = Vec::with_capacity(present_steps.len());
            for bound in &packet.steps {
                if let StepPayload::Present(binding) = &bound.payload {
                    let Some(command) = device._state._queue.commandBuffer() else {
                        return SubmitOutcome::Rejected {
                            error: Error::OutOfMemory,
                            work: api::RetryableSubmit::Fresh(packet),
                        };
                    };
                    // SAFETY: Core validated this current exclusive lease for
                    // the plan's device; the drawable stays owned by binding.
                    let image = unsafe { binding.image().native_ref() };
                    let drawable =
                        objc2::runtime::ProtocolObject::<dyn objc2_metal::MTLDrawable>::from_ref(
                            &*image.drawable,
                        );
                    command.presentDrawable(drawable);
                    present_commands.push((bound.step, command));
                }
            }
            let api::SubmitPacket {
                plan,
                mut steps,
                completion: request,
            } = packet;
            let requested_steps = match &request {
                api::CompletionRequest::Steps(steps) => steps.to_vec(),
                _ => Vec::new(),
            };

            // Allocate every checkpoint marker and install every callback
            // before consuming the packet or committing native work. A setup
            // failure therefore remains a clean rejection with the fresh plan.
            let mut markers: Vec<CheckpointMarker> = Vec::new();
            let marker_targets = match &request {
                api::CompletionRequest::Steps(targets) => {
                    let mut unique = Vec::new();
                    for target in targets.iter().copied() {
                        if !unique.contains(&target) {
                            unique.push(target);
                        }
                    }
                    unique
                }
                api::CompletionRequest::End if steps.is_empty() => {
                    vec![StepId {
                        plan: plan.id(),
                        index: 0,
                    }]
                }
                api::CompletionRequest::End => vec![plan.step(steps.len() - 1)],
                api::CompletionRequest::None => Vec::new(),
            };
            for target in marker_targets {
                let Some(command) = device._state._queue.commandBuffer() else {
                    return SubmitOutcome::Rejected {
                        error: Error::OutOfMemory,
                        work: api::RetryableSubmit::Fresh(api::SubmitPacket {
                            plan,
                            steps,
                            completion: request,
                        }),
                    };
                };
                let completion = Completion::install(&command);
                markers.push((target, command, completion, Vec::new()));
            }

            // All setup that can return an error is complete. Limit this path
            // to one Present step so lease handoff cannot partially succeed.
            for bound in &mut steps {
                if let StepPayload::Present(binding) = &mut bound.payload {
                    // SAFETY: this is the single native present handoff after
                    // command buffers and completion handlers are prepared.
                    if let Err(error) = unsafe { binding.try_handoff() } {
                        return SubmitOutcome::Rejected {
                            error,
                            work: api::RetryableSubmit::Fresh(api::SubmitPacket {
                                plan,
                                steps,
                                completion: request,
                            }),
                        };
                    }
                }
            }

            let mut native_steps = Vec::with_capacity(steps.len());
            for bound in steps {
                let work = match bound.payload {
                    StepPayload::Commands(work) => work,
                    StepPayload::Present(_) => Vec::new(),
                    StepPayload::SparseBind(_) => unreachable!("prevalidated Metal plan"),
                };
                let native = work
                    .into_iter()
                    .map(|record| {
                        // SAFETY: Core validated device/lane provenance and this
                        // one-shot packet consumes each native recording once.
                        unsafe { record.into_native() }
                    })
                    .collect::<Vec<_>>();
                native_steps.push((bound.step, native));
            }

            let mut accepted = Vec::with_capacity(native_steps.len());
            let mut frontiers: Vec<(LaneId, StepId)> = Vec::new();
            let _gate = device
                ._state
                .queue_gate
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());

            for (step, commands) in &native_steps {
                let lane = plan.steps()[step.index as usize].lane;
                for work in commands {
                    work.retain_indirect_arguments_until_completed();
                    work.command.commit();
                }
                for (present_step, command) in &present_commands {
                    if present_step == step {
                        command.commit();
                    }
                }
                accepted.push(*step);
                record_frontier(&mut frontiers, lane, *step);

                for (target, marker, _, coverage) in markers
                    .iter_mut()
                    .filter(|(target, _, _, _)| target == step)
                {
                    *coverage = frontiers.clone();
                    marker.commit();
                    let _ = target;
                }
            }
            if native_steps.is_empty() {
                for (_, marker, _, _) in &markers {
                    marker.commit();
                }
            }

            // A marker after the final plan step is naturally placed after all
            // command buffers. For requested checkpoints each marker was
            // committed immediately after its matching Commands step.
            let mut checkpoints = Vec::new();
            for requested in requested_steps {
                if let Some((_, _, completion, coverage)) = markers
                    .iter()
                    .find(|(target, _, _, _)| *target == requested)
                {
                    checkpoints.push(api::Checkpoint {
                        // SAFETY: this callback belongs to the queue marker
                        // issued at the requested plan boundary.
                        point: unsafe {
                            CompletionPoint::from_native(plan.id().device, completion.clone())
                        },
                        covers: CompletionCoverage {
                            lane_frontiers: coverage.clone(),
                        },
                    });
                }
            }
            if matches!(request, api::CompletionRequest::End) {
                if let Some((_, _, completion, coverage)) = markers.last() {
                    checkpoints.push(api::Checkpoint {
                        // SAFETY: End marker follows the full accepted queue prefix.
                        point: unsafe {
                            CompletionPoint::from_native(plan.id().device, completion.clone())
                        },
                        covers: CompletionCoverage {
                            lane_frontiers: coverage.clone(),
                        },
                    });
                }
            }
            SubmitOutcome::Accepted(api::SubmissionReceipt {
                device: plan.id().device,
                index: None,
                accepted,
                checkpoints,
                presents: present_steps
                    .into_iter()
                    .map(|step| api::PresentReceipt {
                        step: Some(step),
                        accepted: true,
                    })
                    .collect(),
            })
        }
    }

    // SAFETY: resuming requires proof that the accepted prefix completed and
    // that referenced resources remain alive. Metal does not implement tails,
    // so it rejects the tail unchanged and performs no native operation.
    unsafe fn resume(_: &MetalDevice, tail: api::DeferredTail<Self>) -> SubmitOutcome<Self> {
        SubmitOutcome::Rejected {
            error: Error::Unsupported("Metal submission tails are not implemented"),
            work: api::RetryableSubmit::Tail(tail),
        }
    }
}

fn supports_command_plan(plan: &SubmissionPlan<Metal>) -> bool {
    let mut present_count = 0;
    plan.steps().iter().enumerate().all(|(index, desc)| {
        if desc.kind == StepKind::Present {
            present_count += 1;
        }
        matches!(desc.kind, StepKind::Commands | StepKind::Present)
            && present_count <= 1
            && desc.lane.device == plan.id().device
            && desc.lane.ordinal == 0
            && desc.waits.iter().all(|wait| {
                wait.previous.plan == plan.id()
                    && (wait.previous.index as usize) < index
                    && plan.steps()[wait.previous.index as usize].lane == desc.lane
            })
    })
}

/// Keep the latest accepted step for each lane. A completion marker snapshots
/// this list at its issue boundary, so later steps cannot widen earlier proof.
fn record_frontier(frontiers: &mut Vec<(LaneId, api::StepId)>, lane: LaneId, step: api::StepId) {
    frontiers.retain(|(existing, _)| *existing != lane);
    frontiers.push((lane, step));
}

#[cfg(test)]
mod tests;
