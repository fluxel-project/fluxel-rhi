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
    self, DeferredBackend, Error, LaneId, Result, StepKind, SubmissionPlan, SubmitOutcome,
};
#[cfg(target_vendor = "apple")]
use fluxel_rhi_core::api::{CompletionCoverage, CompletionPoint, StepId, StepPayload};
use std::sync::Arc;

#[cfg(target_vendor = "apple")]
type NativeCommandBuffer =
    objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandBuffer>>;

mod completion;
pub use completion::Completion;
#[cfg(test)]
use completion::CompletionFuture;
use completion::CompletionJoin;
pub(super) use completion::join_into;
#[cfg(target_vendor = "apple")]
pub(super) use completion::{SubmissionTracker, track_command};

#[cfg(target_vendor = "apple")]
type CheckpointMarker = (
    StepId,
    NativeCommandBuffer,
    Completion,
    Vec<(LaneId, StepId)>,
);

pub struct MetalRecordedWork {
    #[cfg(target_vendor = "apple")]
    upload_slices: Vec<super::upload::UploadSlice>,
    #[cfg(target_vendor = "apple")]
    pub(super) command:
        objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandBuffer>>,
    #[cfg(target_vendor = "apple")]
    prepared_draws: Vec<super::encoder::indirect_count::PreparedDraws>,
    #[cfg(target_vendor = "apple")]
    pub(super) lane: u32,
    #[cfg(target_vendor = "apple")]
    external_buffers: Vec<super::MetalBuffer>,
    #[cfg(target_vendor = "apple")]
    argument_arrays: Vec<Arc<super::argument::ArgumentArray>>,
    #[cfg(target_vendor = "apple")]
    external_textures: Vec<super::MetalTexture>,
    #[cfg(target_vendor = "apple")]
    pub(super) query_ops: Vec<super::query::QueryOp>,
    #[cfg(target_vendor = "apple")]
    accelerations: Vec<super::ray::MetalAcceleration>,
    #[cfg(target_vendor = "apple")]
    pub(super) ray_builds: Vec<Arc<super::ray::BuildReservation>>,
    #[cfg(target_vendor = "apple")]
    indirect_arguments:
        Vec<objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>>,
}

impl MetalRecordedWork {
    #[cfg(target_vendor = "apple")]
    pub(super) fn submit_ray_builds(&self) {
        if self.ray_builds.is_empty() {
            return;
        }
        for build in &self.ray_builds {
            build.submit(&self.command);
        }
    }

    #[cfg(target_vendor = "apple")]
    pub(super) fn guard_ray_completion(&self, covered: &mut Vec<Completion>) {
        if self.ray_builds.is_empty() {
            return;
        }
        // Native terminal handlers need not provide ordering between callbacks.
        // The public checkpoint must also cover publication of the accepted
        // BLAS snapshot before a caller can bind/rebuild the completed TLAS.
        covered.extend(
            self.ray_builds
                .iter()
                .map(|build| build.publication_completion()),
        );
        let output = Completion::new();
        Arc::new(CompletionJoin {
            children: std::mem::take(covered),
            output: output.clone(),
            ray_builds: self
                .ray_builds
                .iter()
                .map(|build| build.completion())
                .collect(),
        })
        .update();
        covered.push(output);
    }
    #[cfg(target_vendor = "apple")]
    pub(super) fn retain_indirect_arguments_until_completed(&self) {
        use objc2_metal::MTLCommandBuffer as _;
        if self.indirect_arguments.is_empty()
            && self.accelerations.is_empty()
            && self.prepared_draws.is_empty()
            && self.argument_arrays.is_empty()
            && self.external_buffers.is_empty()
            && self.external_textures.is_empty()
            && self.upload_slices.is_empty()
        {
            return;
        }
        let retained = self.indirect_arguments.clone();
        let accelerations = self.accelerations.clone();
        let prepared = self.prepared_draws.clone();
        let argument_arrays = self.argument_arrays.clone();
        let external_buffers = self.external_buffers.clone();
        let external_textures = self.external_textures.clone();
        let upload_slices = self.upload_slices.clone();
        let callback = block2::RcBlock::new(
            move |_: std::ptr::NonNull<
                objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
            >| {
                let _ = retained.len();
                let _ = accelerations.len();
                let _ = (
                    prepared.len(),
                    argument_arrays.len(),
                    external_buffers.len(),
                    external_textures.len(),
                    upload_slices.len(),
                );
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

impl DeferredBackend for Metal {
    type NativeRecordedWork = MetalRecordedWork;

    fn finish_encoder(
        encoder: super::MetalCommandEncoder,
        lane: LaneId,
    ) -> Result<Self::NativeRecordedWork> {
        if lane.ordinal > 1 {
            return Err(Error::InvalidState);
        }
        #[cfg(target_vendor = "apple")]
        {
            if encoder.render.is_some() || encoder.compute.is_some() {
                return Err(Error::InvalidState);
            }
            if encoder.lane != lane.ordinal {
                return Err(Error::WrongLane);
            }
            // The queue's commandBuffer() factory retains native objects used
            // in encoded commands. The backend does not duplicate every bind
            // in a Rust keep-alive vector; Core still owns public resource
            // lifetime decisions through actual GPU completion.
            Ok(MetalRecordedWork {
                upload_slices: encoder.upload_slices,
                command: encoder.command,
                prepared_draws: encoder.prepared_draws,
                lane: encoder.lane,
                query_ops: encoder.query_ops,
                external_buffers: encoder.external_buffers,
                argument_arrays: encoder.argument_arrays,
                external_textures: encoder.external_textures,
                accelerations: encoder.accelerations,
                ray_builds: encoder.ray_builds,
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
        if packet
            .steps
            .iter()
            .filter(|step| matches!(step.payload, api::StepPayload::Present(_)))
            .count()
            > 1
            || packet.steps.iter().any(|step| {
                matches!(step.payload, api::StepPayload::SparseBind(_))
                    || packet.plan.steps()[step.step.index as usize].lane.ordinal != 0
            })
        {
            // SAFETY: the caller's packet and lifetime obligations are unchanged;
            // the multi-queue lowering preflights every mapping before handoff.
            let outcome = unsafe {
                <Metal as api::MultiQueueBackend>::submit_multi(
                    device,
                    api::MultiQueuePacket {
                        work: packet,
                        signals: Vec::new(),
                        waits: Vec::new(),
                    },
                )
            };
            return match outcome {
                api::MultiQueueSubmitOutcome::Accepted(receipt) => {
                    SubmitOutcome::Accepted(receipt.work)
                }
                api::MultiQueueSubmitOutcome::Rejected { error, packet } => {
                    SubmitOutcome::Rejected {
                        error,
                        work: api::RetryableSubmit::Fresh(packet.work),
                    }
                }
                api::MultiQueueSubmitOutcome::Indeterminate { error, uncertain } => {
                    SubmitOutcome::Indeterminate { error, uncertain }
                }
                api::MultiQueueSubmitOutcome::PartiallyAccepted {
                    error,
                    accepted,
                    unaccepted,
                } => {
                    let next_step = unaccepted
                        .pending
                        .first()
                        .map_or(unaccepted.plan.steps().len() as u32, |step| step.step.index);
                    let accepted_frontier = accepted
                        .work
                        .checkpoints
                        .iter()
                        .map(|checkpoint| api::Checkpoint {
                            point: checkpoint.point.clone(),
                            covers: api::CompletionCoverage {
                                lane_frontiers: checkpoint.covers.lane_frontiers.clone(),
                            },
                        })
                        .collect();
                    SubmitOutcome::PartiallyAccepted {
                        error,
                        accepted: accepted.work,
                        tail: api::DeferredTail {
                            plan: unaccepted.plan,
                            next_step,
                            pending: unaccepted.pending,
                            accepted_frontier,
                            completion: unaccepted.completion,
                        },
                    }
                }
            };
        }
        // Validate every unsupported shape before consuming RecordedWork. This
        // preserves Core's clean-rejection ownership contract.
        if !supports_command_plan(
            &packet.plan,
            packet
                .steps
                .first()
                .map_or(packet.plan.steps().len(), |s| s.step.index as usize),
        ) {
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
            let _gate = device
                ._state
                .queue_gate
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            let ops = packet
                .steps
                .iter()
                .flat_map(|step| match &step.payload {
                    StepPayload::Commands(records) => records.as_slice(),
                    _ => &[],
                })
                .flat_map(|record| {
                    // SAFETY: Core validates the backend provenance of each recording.
                    let work = unsafe { record.native_ref() };
                    work.query_ops.iter().map(move |op| (work.lane, op))
                });
            if let Err(error) = super::query::validate_ops(ops) {
                return SubmitOutcome::Rejected {
                    error,
                    work: api::RetryableSubmit::Fresh(packet),
                };
            }
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
                api::CompletionRequest::End => vec![steps.last().expect("nonempty suffix").step],
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
                let completion = Completion::new();
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
            let mut prefix = Vec::new();
            if let Some(previous) = device
                ._state
                .lane_frontier
                .lock()
                .unwrap_or_else(|p| p.into_inner())[0]
                .clone()
            {
                prefix.push(previous);
            }
            for (step, commands) in &native_steps {
                let lane = plan.steps()[step.index as usize].lane;
                for work in commands {
                    let query_commit =
                        Arc::new(super::query::commit_ops(&work.query_ops, work.lane));
                    work.retain_indirect_arguments_until_completed();
                    work.submit_ray_builds();
                    prefix.push(track_command(
                        device,
                        &work.command,
                        Some((query_commit, work.lane)),
                    ));
                    work.command.commit();
                    work.guard_ray_completion(&mut prefix);
                }
                for (present_step, command) in &present_commands {
                    if present_step == step {
                        prefix.push(track_command(device, command, None));
                        command.commit();
                    }
                }
                accepted.push(*step);
                record_frontier(&mut frontiers, lane, *step);
                let lane_completion = Completion::new();
                join_into(&lane_completion, prefix.clone());
                device
                    ._state
                    .lane_frontier
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())[0] = Some(lane_completion);

                for (target, marker, completion, coverage) in markers
                    .iter_mut()
                    .filter(|(target, _, _, _)| target == step)
                {
                    *coverage = frontiers.clone();
                    let marker_completion = track_command(device, marker, None);
                    let mut covered = prefix.clone();
                    covered.push(marker_completion);
                    join_into(completion, covered);
                    marker.commit();
                    let _ = target;
                }
            }
            if native_steps.is_empty() {
                for (_, marker, completion, _) in &markers {
                    let tracked = track_command(device, marker, None);
                    join_into(completion, vec![tracked]);
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

    unsafe fn resume(device: &MetalDevice, tail: api::DeferredTail<Self>) -> SubmitOutcome<Self> {
        let api::DeferredTail {
            plan,
            next_step,
            pending,
            accepted_frontier,
            completion,
        } = tail;
        let suffix_completion = match &completion {
            api::CompletionRequest::None => api::CompletionRequest::None,
            api::CompletionRequest::End => api::CompletionRequest::End,
            api::CompletionRequest::Steps(steps) => api::CompletionRequest::Steps(
                steps
                    .iter()
                    .copied()
                    .filter(|step| step.index >= next_step)
                    .collect::<Vec<_>>()
                    .into(),
            ),
        };
        // SAFETY: Core validated the original plan and suffix; the caller has
        // observed the accepted frontier complete, so earlier dependencies are satisfied.
        let result = unsafe {
            Self::submit(
                device,
                api::SubmitPacket {
                    plan,
                    steps: pending,
                    completion: suffix_completion,
                },
            )
        };
        match result {
            SubmitOutcome::Rejected {
                error,
                work: api::RetryableSubmit::Fresh(packet),
            } => SubmitOutcome::Rejected {
                error,
                work: api::RetryableSubmit::Tail(api::DeferredTail {
                    plan: packet.plan,
                    next_step,
                    pending: packet.steps,
                    accepted_frontier,
                    completion,
                }),
            },
            SubmitOutcome::PartiallyAccepted {
                error,
                mut accepted,
                mut tail,
            } => {
                let prior: Vec<_> = accepted_frontier
                    .iter()
                    .map(|checkpoint| api::Checkpoint {
                        point: checkpoint.point.clone(),
                        covers: api::CompletionCoverage {
                            lane_frontiers: checkpoint.covers.lane_frontiers.clone(),
                        },
                    })
                    .collect();
                accepted.checkpoints.splice(0..0, prior);
                tail.accepted_frontier.splice(0..0, accepted_frontier);
                tail.completion = completion;
                SubmitOutcome::PartiallyAccepted {
                    error,
                    accepted,
                    tail,
                }
            }
            SubmitOutcome::Accepted(mut receipt) => {
                let mut previous_frontiers = Vec::new();
                for checkpoint in &accepted_frontier {
                    for &(lane, step) in &checkpoint.covers.lane_frontiers {
                        if !previous_frontiers
                            .iter()
                            .any(|(existing, _)| *existing == lane)
                        {
                            previous_frontiers.push((lane, step));
                        } else if let Some((_, previous)) = previous_frontiers
                            .iter_mut()
                            .find(|(existing, _)| *existing == lane)
                        {
                            if step.index > previous.index {
                                *previous = step;
                            }
                        }
                    }
                }
                for checkpoint in &mut receipt.checkpoints {
                    for &(lane, step) in &previous_frontiers {
                        if !checkpoint
                            .covers
                            .lane_frontiers
                            .iter()
                            .any(|(existing, _)| *existing == lane)
                        {
                            checkpoint.covers.lane_frontiers.push((lane, step));
                        }
                    }
                }
                if matches!(completion, api::CompletionRequest::Steps(_)) {
                    receipt.checkpoints.splice(0..0, accepted_frontier);
                }
                SubmitOutcome::Accepted(receipt)
            }
            other => other,
        }
    }
}

fn supports_command_plan(plan: &SubmissionPlan<Metal>, start: usize) -> bool {
    let mut present_count = 0;
    plan.steps()
        .iter()
        .enumerate()
        .skip(start)
        .all(|(index, desc)| {
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
                        && ((wait.previous.index as usize) < start
                            || plan.steps()[wait.previous.index as usize].lane == desc.lane)
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
