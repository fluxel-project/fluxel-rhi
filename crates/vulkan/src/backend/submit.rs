//! Native Vulkan submission and fence-owned command lifetimes.
//!
//! `submit_gate` keeps one Core plan contiguous while `queue_gate` covers each
//! externally synchronized queue call. Core owns work ordering; this module
//! only lowers its plan and retains issued native work until fence completion.

use super::{DeviceState, Vulkan, VulkanDevice, VulkanEncoder, vk_error};
use ash::vk;
use fluxel_rhi_core::api::{
    self, DeferredBackend, Error, Result, RetryableSubmit, StepPayload, SubmitOutcome,
};
use std::sync::Arc;

pub struct RecordedWork {
    pub(super) encoder: VulkanEncoder,
}
impl RecordedWork {
    fn command(&self) -> vk::CommandBuffer {
        self.encoder.command
    }
}

mod completion;
pub(super) mod multi_queue;
pub(super) use completion::Completion;
use completion::{fence_waiter, retain_until_fence};

struct PendingPresentSignal {
    device: Arc<DeviceState>,
    semaphore: Option<vk::Semaphore>,
}
impl Drop for PendingPresentSignal {
    fn drop(&mut self) {
        if let Some(semaphore) = self.semaphore.take() {
            // SAFETY: this guard owns a semaphore that has not been handed to
            // a surface lease, so no queued present operation can use it.
            unsafe { self.device.device.destroy_semaphore(semaphore, None) };
        }
    }
}
impl DeferredBackend for Vulkan {
    type NativeRecordedWork = RecordedWork;

    fn finish_encoder(encoder: VulkanEncoder, lane: api::LaneId) -> Result<RecordedWork> {
        if encoder.lane != lane {
            return Err(Error::WrongLane);
        }
        if encoder.raster
            || encoder.active_occlusion.is_some()
            || encoder.active_statistics.is_some()
        {
            return Err(Error::InvalidState);
        }
        // SAFETY: VulkanEncoder exclusively owns its begun command buffer and
        // Core has established that no raster pass remains active.
        unsafe { encoder.device.device.end_command_buffer(encoder.command) }.map_err(vk_error)?;
        Ok(RecordedWork { encoder })
    }

    unsafe fn submit(
        device: &VulkanDevice,
        packet: api::SubmitPacket<Self>,
    ) -> SubmitOutcome<Self> {
        if (device.0.sparse_image && device.0.timeline_semaphore)
            || packet
                .plan
                .steps()
                .iter()
                .any(|step| step.lane.ordinal != 0 || step.kind == api::StepKind::SparseBind)
        {
            use api::MultiQueueBackend;
            // SAFETY: Core has validated the packet and its compiled dependencies.
            return match unsafe {
                Vulkan::submit_multi(
                    device,
                    api::MultiQueuePacket {
                        work: packet,
                        signals: Vec::new(),
                        waits: Vec::new(),
                    },
                )
            } {
                api::MultiQueueSubmitOutcome::Accepted(receipt) => {
                    SubmitOutcome::Accepted(receipt.work)
                }
                api::MultiQueueSubmitOutcome::Rejected { error, packet } => {
                    SubmitOutcome::Rejected {
                        error,
                        work: RetryableSubmit::Fresh(packet.work),
                    }
                }
                api::MultiQueueSubmitOutcome::Indeterminate { error, uncertain } => {
                    SubmitOutcome::Indeterminate { error, uncertain }
                }
                api::MultiQueueSubmitOutcome::PartiallyAccepted { .. } => {
                    unreachable!("Vulkan multi-queue has no partial result")
                }
            };
        }
        // SAFETY: Core transfers each recorded work item exactly once after
        // validating the packet's backend, device and lane identities.
        // A Vulkan queue is already an ordered executor.  The Core plan has
        // chosen the order and synchronization; this backend only lowers its
        // one graphics-lane prefix and never reconstructs a dependency graph.
        let has_commands = packet
            .steps
            .iter()
            .any(|step| matches!(&step.payload, StepPayload::Commands(work) if !work.is_empty()));
        let plan_shape = has_commands
            && packet
                .steps
                .iter()
                .enumerate()
                .all(|(index, bound)| match &bound.payload {
                    StepPayload::Commands(_) => {
                        packet.plan.steps()[index].lane == packet.plan.steps()[0].lane
                    }
                    StepPayload::Present(_) => {
                        index + 1 == packet.steps.len()
                            && packet.plan.steps()[index].lane == packet.plan.steps()[0].lane
                    }
                    StepPayload::SparseBind(_) => false,
                })
            && packet
                .steps
                .iter()
                .filter(|step| matches!(&step.payload, StepPayload::Present(_)))
                .count()
                <= 1;
        if !plan_shape {
            return SubmitOutcome::Rejected {
                error: Error::Unsupported("Vulkan single graphics-queue submission plan"),
                work: RetryableSubmit::Fresh(packet),
            };
        }
        let _submit = device
            .0
            .submit_gate
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        if let Err(error) = super::query::validate_ops(packet.steps.iter().flat_map(|step| {
            match &step.payload {
                StepPayload::Commands(work) => work.as_slice(),
                _ => &[],
            }
            .iter()
            .flat_map(|record| {
                // SAFETY: Core validated the packet's backend and device identity.
                unsafe { record.native_ref() }
                    .encoder
                    .query_ops
                    .iter()
                    .map(|op| (0, op))
            })
        })) {
            return SubmitOutcome::Rejected {
                error,
                work: RetryableSubmit::Fresh(packet),
            };
        }
        let requested = match &packet.completion {
            api::CompletionRequest::Steps(steps) => steps.as_ref(),
            _ => &[],
        };
        // Allocate every native synchronization object before issuing any
        // queue operation.  Allocation failure remains a clean retryable
        // rejection, rather than creating a half-lowered packet.
        let mut step_fences = (0..packet.steps.len())
            .map(|_| None)
            .collect::<Vec<Option<Completion>>>();
        for step in requested {
            if step_fences[step.index as usize].is_some() {
                continue;
            }
            // SAFETY: DeviceState owns the logical device and Completion takes
            // ownership of the unpublished fence before it can be submitted.
            let fence = match unsafe {
                device
                    .0
                    .device
                    .create_fence(&vk::FenceCreateInfo::default(), None)
            } {
                Ok(fence) => fence,
                Err(error) => {
                    return SubmitOutcome::Rejected {
                        error: vk_error(error),
                        work: RetryableSubmit::Fresh(packet),
                    };
                }
            };
            step_fences[step.index as usize] = Some(Completion::new(device.0.clone(), fence));
        }
        // SAFETY: as above, the terminal Completion owns this unpublished fence.
        let terminal_fence = match unsafe {
            device
                .0
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)
        } {
            Ok(fence) => Completion::new(device.0.clone(), fence),
            Err(error) => {
                return SubmitOutcome::Rejected {
                    error: vk_error(error),
                    work: RetryableSubmit::Fresh(packet),
                };
            }
        };
        let present_signal = if packet
            .steps
            .iter()
            .any(|step| matches!(&step.payload, StepPayload::Present(_)))
        {
            // SAFETY: PendingPresentSignal owns this semaphore until the image
            // lease accepts the explicit handoff below.
            match unsafe {
                device
                    .0
                    .device
                    .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
            } {
                Ok(semaphore) => Some(semaphore),
                Err(error) => {
                    return SubmitOutcome::Rejected {
                        error: vk_error(error),
                        work: RetryableSubmit::Fresh(packet),
                    };
                }
            }
        } else {
            None
        };
        let mut present_signal = present_signal.map(|semaphore| PendingPresentSignal {
            device: device.0.clone(),
            semaphore: Some(semaphore),
        });
        if let Err(error) = fence_waiter() {
            return SubmitOutcome::Rejected {
                error,
                work: RetryableSubmit::Fresh(packet),
            };
        }
        let api::SubmitPacket {
            plan,
            steps,
            completion: request,
        } = packet;
        let mut issued = Vec::new();
        let mut accepted = Vec::new();
        let mut presents = Vec::new();
        let mut checkpoints = Vec::new();
        let mut lane_frontiers = Vec::new();
        // Hold the plan-level gate from the first queue issue through its
        // terminal marker.  Each native queue call still takes `queue_gate`,
        // so Surface present and wait-idle retain their narrow external
        // synchronization boundary without allowing another plan to splice
        // commands between this plan's explicit transitions.
        let indeterminate = |error, issued: Vec<RecordedWork>| {
            for record in &issued {
                super::query::invalidate_ops(&record.encoder.query_ops, 0);
            }
            // vkQueueSubmit may have crossed the driver boundary even on an
            // error.  Keep command pools and their uploads alive unless an
            // externally synchronized queue-idle probe proves retirement.
            let idle = {
                let _queue = device
                    .0
                    .queue_gate
                    .lock()
                    .unwrap_or_else(|p| p.into_inner());
                // SAFETY: queue_gate serializes this idle probe with every
                // queue submit and present call, as Vulkan external sync needs.
                unsafe { device.0.device.queue_wait_idle(device.0.graphics_queue) }
            };
            if idle.is_err() {
                std::mem::forget(issued);
            }
            SubmitOutcome::Indeterminate {
                error,
                // SAFETY: this plan belongs to the device; unit evidence only
                // records that a prior native queue call may have been accepted.
                uncertain: unsafe { api::NativeUncertainty::from_native(plan.id().device, ()) },
            }
        };

        for bound in steps {
            let id = bound.step;
            match bound.payload {
                StepPayload::Commands(records) => {
                    let start = issued.len();
                    issued.extend(records.into_iter().map(|record| {
                        // SAFETY: the packet owns each recorded work item
                        // and Vulkan submission consumes it exactly once.
                        unsafe { record.into_native() }
                    }));
                    let commands = issued[start..]
                        .iter()
                        .map(RecordedWork::command)
                        .collect::<Vec<_>>();
                    if !commands.is_empty() {
                        let info = vk::SubmitInfo::default().command_buffers(&commands);
                        let result = {
                            let _queue = device
                                .0
                                .queue_gate
                                .lock()
                                .unwrap_or_else(|p| p.into_inner());
                            // SAFETY: queue_gate serializes submission and `issued`
                            // retains every command buffer until its fence completes.
                            unsafe {
                                device.0.device.queue_submit(
                                    device.0.graphics_queue,
                                    &[info],
                                    vk::Fence::null(),
                                )
                            }
                        };
                        if let Err(error) = result {
                            return indeterminate(vk_error(error), issued);
                        }
                        for record in &issued[start..] {
                            super::query::commit_ops(&record.encoder.query_ops, 0);
                        }
                    }
                    accepted.push(id);
                }
                StepPayload::Present(mut binding) => {
                    let pending = present_signal
                        .as_mut()
                        .expect("validated one Vulkan present");
                    let signal = pending.semaphore.expect("present signal is still owned");
                    if let Err(error) =
                        // SAFETY: Core validated the binding's device; borrow the
                        // exclusive image lease only for semaphore handoff.
                        unsafe { binding.image().native_ref() }.set_present_wait(signal)
                    {
                        return indeterminate(error, issued);
                    }
                    // The surface image now owns the semaphore even if queue
                    // submission or handoff later fails.
                    pending.semaphore = None;
                    let signals = [signal];
                    let info = vk::SubmitInfo::default().signal_semaphores(&signals);
                    let result = {
                        let _queue = device
                            .0
                            .queue_gate
                            .lock()
                            .unwrap_or_else(|p| p.into_inner());
                        // SAFETY: queue_gate serializes this signal-only submit;
                        // PendingPresentSignal retains the semaphore until handoff.
                        unsafe {
                            device.0.device.queue_submit(
                                device.0.graphics_queue,
                                &[info],
                                vk::Fence::null(),
                            )
                        }
                    };
                    if let Err(error) = result {
                        return indeterminate(vk_error(error), issued);
                    }
                    // SAFETY: the handoff is irrevocable only after queue signal
                    // submission succeeded, which is true on this path.
                    if let Err(error) = unsafe { binding.try_handoff() } {
                        return indeterminate(error, issued);
                    }
                    accepted.push(id);
                    presents.push(api::PresentReceipt {
                        step: Some(id),
                        accepted: true,
                    });
                }
                StepPayload::SparseBind(_) => unreachable!("submission shape was validated"),
            }

            if matches!(
                plan.steps()[id.index as usize].kind,
                api::StepKind::Commands
            ) {
                let lane = plan.steps()[id.index as usize].lane;
                lane_frontiers.retain(|(existing, _)| *existing != lane);
                lane_frontiers.push((lane, id));
            }
            if let Some(completion) = step_fences[id.index as usize].take() {
                let result = {
                    let _queue = device
                        .0
                        .queue_gate
                        .lock()
                        .unwrap_or_else(|p| p.into_inner());
                    // SAFETY: queue_gate serializes the marker and Completion
                    // owns its fence until the observer records completion.
                    unsafe {
                        device.0.device.queue_submit(
                            device.0.graphics_queue,
                            &[],
                            completion.fence(),
                        )
                    }
                };
                if let Err(error) = result {
                    std::mem::forget(completion);
                    return indeterminate(vk_error(error), issued);
                }
                checkpoints.push(api::Checkpoint {
                    // SAFETY: this marker fence belongs to the plan's device and
                    // follows the recorded lane frontier in native queue order.
                    point: unsafe {
                        api::CompletionPoint::from_native(plan.id().device, completion.clone())
                    },
                    covers: api::CompletionCoverage {
                        lane_frontiers: lane_frontiers.clone(),
                    },
                });
                retain_until_fence(completion, Vec::new());
            }
        }

        // A terminal fence is needed even for CompletionRequest::None: it is
        // the native proof that command pools, uploads and descriptors may be
        // released.  Its marker follows every accepted queue operation.
        let terminal = {
            let _queue = device
                .0
                .queue_gate
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            // SAFETY: queue_gate serializes this terminal marker; `issued` is
            // retained until its fence completes.
            unsafe {
                device
                    .0
                    .device
                    .queue_submit(device.0.graphics_queue, &[], terminal_fence.fence())
            }
        };
        if let Err(error) = terminal {
            std::mem::forget(terminal_fence);
            return indeterminate(vk_error(error), issued);
        }
        retain_until_fence(terminal_fence.clone(), issued);
        if matches!(request, api::CompletionRequest::End) {
            checkpoints.push(api::Checkpoint {
                // SAFETY: terminal_fence belongs to this plan and follows every
                // accepted native queue operation.
                point: unsafe {
                    api::CompletionPoint::from_native(plan.id().device, terminal_fence)
                },
                covers: api::CompletionCoverage { lane_frontiers },
            });
        }
        SubmitOutcome::Accepted(api::SubmissionReceipt {
            device: plan.id().device,
            index: None,
            accepted,
            checkpoints,
            presents,
        })
    }

    // SAFETY: Core supplies a tail associated with this backend/device; it is
    // returned unchanged because partial Vulkan resume is unsupported.
    unsafe fn resume(_: &VulkanDevice, tail: api::DeferredTail<Self>) -> SubmitOutcome<Self> {
        SubmitOutcome::Rejected {
            error: Error::Unsupported("Vulkan partial submission resume"),
            work: RetryableSubmit::Tail(tail),
        }
    }
}
