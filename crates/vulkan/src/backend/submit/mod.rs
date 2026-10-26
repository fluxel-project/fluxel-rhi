//! Native Vulkan submission and fence-owned command lifetimes.
//!
//! `submit_gate` keeps one Core plan contiguous while `queue_gate` covers each
//! externally synchronized queue call. Core owns work ordering; this module
//! only lowers its plan and retains issued native work until fence completion.

use super::{
    DeviceState, Vulkan, VulkanDevice, VulkanEncoder, encoder::ExternalBoundary, vk_error,
};
use ash::vk;
use fluxel_rhi_core::api::{self, Error, Result, RetryableSubmit, StepPayload, SubmitOutcome};
use fluxel_rhi_core::backend::DeferredBackend;
use std::sync::Arc;

pub struct RecordedWork {
    pub(crate) encoder: VulkanEncoder,
}
impl RecordedWork {
    fn command(&self) -> vk::CommandBuffer {
        self.encoder.command
    }
}

/// Submits one exact native fragment. Timeline values are supplied for the
/// Core plan semaphores and zero for binary external semaphores.
fn submit_fragment(
    device: &VulkanDevice,
    queue: vk::Queue,
    commands: &[vk::CommandBuffer],
    waits: &[(vk::Semaphore, u64)],
    signals: &[(vk::Semaphore, u64)],
    timeline: bool,
) -> Result<()> {
    let wait_semaphores: Vec<_> = waits.iter().map(|(semaphore, _)| *semaphore).collect();
    let wait_values: Vec<_> = waits.iter().map(|(_, value)| *value).collect();
    let signal_semaphores: Vec<_> = signals.iter().map(|(semaphore, _)| *semaphore).collect();
    let signal_values: Vec<_> = signals.iter().map(|(_, value)| *value).collect();
    let stages = vec![vk::PipelineStageFlags::ALL_COMMANDS; wait_semaphores.len()];
    let mut timeline_info = vk::TimelineSemaphoreSubmitInfo::default()
        .wait_semaphore_values(&wait_values)
        .signal_semaphore_values(&signal_values);
    let mut info = vk::SubmitInfo::default()
        .wait_semaphores(&wait_semaphores)
        .wait_dst_stage_mask(&stages)
        .command_buffers(commands)
        .signal_semaphores(&signal_semaphores);
    if timeline {
        info = info.push_next(&mut timeline_info);
    }
    let _queue = device
        .0
        .queue_gate
        .lock()
        .unwrap_or_else(|p| p.into_inner());
    // SAFETY: queue access is serialized; command buffers and every semaphore
    // owner remain retained by RecordedWork through completion/uncertainty.
    unsafe {
        device
            .0
            .device
            .queue_submit(queue, &[info], vk::Fence::null())
    }
    .map_err(vk_error)
}

// Each imported binary semaphore brackets its exact Core record boundary.
// Waits are attached to the following command segment and signals to the
// preceding one, so Vulkan's submit scopes match Core source order directly.
// Timeline dependencies wrap the first and final native submissions.
fn submit_external_records(
    device: &VulkanDevice,
    queue: vk::Queue,
    records: &[RecordedWork],
    waits: &[(vk::Semaphore, u64)],
    signals: &[(vk::Semaphore, u64)],
    timeline: bool,
) -> Result<()> {
    let mut first = true;
    let mut pending_wait: Option<vk::Semaphore> = None;
    for (index, record) in records.iter().enumerate() {
        for (segment_index, segment) in record.encoder.external_segments.iter().enumerate() {
            let command = [segment.command];
            let mut segment_waits = if first { waits.to_vec() } else { Vec::new() };
            if let Some(wait) = pending_wait.take() {
                segment_waits.push((wait, 0));
            }
            let segment_signals = match &segment.boundary {
                ExternalBoundary::Wait(_) => Vec::new(),
                ExternalBoundary::Signal(semaphore) => vec![(semaphore.semaphore, 0)],
            };
            submit_fragment(
                device,
                queue,
                &command,
                &segment_waits,
                &segment_signals,
                timeline,
            )?;
            super::ray::retention::publish_segment(record, segment_index);
            first = false;
            if let ExternalBoundary::Wait(semaphore) = &segment.boundary {
                pending_wait = Some(semaphore.semaphore);
            }
        }
        let command = [record.command()];
        let final_signals = if index + 1 == records.len() {
            signals
        } else {
            &[]
        };
        let mut final_waits = if first { waits.to_vec() } else { Vec::new() };
        if let Some(wait) = pending_wait.take() {
            final_waits.push((wait, 0));
        }
        submit_fragment(
            device,
            queue,
            &command,
            &final_waits,
            final_signals,
            timeline,
        )?;
        super::ray::retention::publish_segment(record, record.encoder.external_segments.len());
        first = false;
    }
    Ok(())
}

mod completion;
pub(super) mod multi_queue;
pub(super) mod queue_timeline;
pub(super) use completion::Completion;
use completion::{fence_waiter, retain_until_fence};
pub(super) use queue_timeline::Timeline;

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
        // Route by this packet's actual work, never by optional device
        // capabilities. A sparse/timeline-capable device still has a cheap
        // single-queue path for ordinary graphics-only submissions.
        if packet.steps.iter().any(|bound| {
            let desc = &packet.plan.steps()[bound.step.index as usize];
            desc.lane.ordinal != 0 || matches!(bound.payload, StepPayload::SparseBind(_))
        }) {
            use fluxel_rhi_core::backend::MultiQueueBackend;
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
        let plan_shape = packet.steps.iter().all(|bound| match &bound.payload {
            StepPayload::Commands(_) => {
                packet.plan.steps()[bound.step.index as usize].lane.ordinal == 0
            }
            StepPayload::Present(_) => {
                packet.plan.steps()[bound.step.index as usize].lane.ordinal == 0
            }
            StepPayload::SparseBind(_) => false,
        });
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
        let mut step_fences = (0..packet.plan.steps().len())
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
        let first = packet
            .steps
            .first()
            .map_or(packet.plan.steps().len(), |step| step.step.index as usize);
        let mut present_signals = (0..first).map(|_| None).collect::<Vec<_>>();
        for step in &packet.steps {
            if matches!(step.payload, StepPayload::Present(_)) {
                // SAFETY: each fresh semaphore remains owned until image handoff.
                let semaphore = match unsafe {
                    device
                        .0
                        .device
                        .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                } {
                    Ok(semaphore) => semaphore,
                    Err(error) => {
                        return SubmitOutcome::Rejected {
                            error: vk_error(error),
                            work: RetryableSubmit::Fresh(packet),
                        };
                    }
                };
                present_signals.push(Some(PendingPresentSignal {
                    device: device.0.clone(),
                    semaphore: Some(semaphore),
                }));
            } else {
                present_signals.push(None);
            }
        }
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
                    super::ray::retention::pin_batch(&mut issued[start..]);
                    let commands = issued[start..]
                        .iter()
                        .map(RecordedWork::command)
                        .collect::<Vec<_>>();
                    if !commands.is_empty() {
                        let info = vk::SubmitInfo::default().command_buffers(&commands);
                        let has_external = issued[start..]
                            .iter()
                            .any(|record| !record.encoder.external_segments.is_empty());
                        let result = if has_external {
                            submit_external_records(
                                device,
                                device.0.graphics_queue,
                                &issued[start..],
                                &[],
                                &[],
                                false,
                            )
                        } else {
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
                            .map_err(vk_error)
                        };
                        if let Err(error) = result {
                            return indeterminate(error, issued);
                        }
                        for record in &issued[start..] {
                            if !has_external {
                                super::ray::retention::publish_segment(
                                    record,
                                    record.encoder.external_segments.len(),
                                );
                            }
                            super::query::commit_ops(&record.encoder.query_ops, 0);
                        }
                    }
                    accepted.push(id);
                }
                StepPayload::Present(mut binding) => {
                    let pending = present_signals[id.index as usize]
                        .as_mut()
                        .expect("preallocated Vulkan present");
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
                        // The queue may have observed this image's semaphore.
                        // Keep the lease and its native owners alive when the
                        // submission outcome cannot be established.
                        std::mem::forget(binding);
                        return indeterminate(vk_error(error), issued);
                    }
                    // SAFETY: the handoff is irrevocable only after queue signal
                    // submission succeeded, which is true on this path.
                    if let Err(error) = unsafe { binding.try_handoff() } {
                        std::mem::forget(binding);
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

            // A present step advances the queue frontier once its handoff is
            // accepted. Completion covers that handoff, not display scanout.
            let lane = plan.steps()[id.index as usize].lane;
            lane_frontiers.retain(|(existing, _)| *existing != lane);
            lane_frontiers.push((lane, id));
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

    unsafe fn resume(device: &VulkanDevice, tail: api::DeferredTail<Self>) -> SubmitOutcome<Self> {
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
                work: RetryableSubmit::Fresh(packet),
            } => SubmitOutcome::Rejected {
                error,
                work: RetryableSubmit::Tail(api::DeferredTail {
                    plan: packet.plan,
                    next_step,
                    pending: packet.steps,
                    accepted_frontier,
                    completion,
                }),
            },
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

#[cfg(all(test, feature = "hardware-tests"))]
#[path = "../../tests/backend/submit/hardware_tests.rs"]
mod hardware_tests;
