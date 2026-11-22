//! D3D12 queue submission, fence facts, and fence-retired work ownership.
//!
//! A Core `SubmissionPlan` fixes command order and dependencies. This module
//! issues steps on the direct and compute queues, with native fence signals and
//! waits for cross-queue dependencies. The submission gate serializes packet
//! validation and handoff; each queue lock serializes its native operations.
//! Each queue retains its own submitted owners behind its own fence: a direct
//! fence is never completion evidence for compute work. A fence value of
//! `u64::MAX` denotes device removal or quarantined uncertainty.

use super::*;

mod failure;
mod owners;
mod preflight;
mod tail;
use failure::checkpoint;
pub use failure::{Dx12SubmissionFailure, Dx12SubmissionOperation};
use owners::{
    SignaledOwners, register_retirements, retain_failed_handoff, retain_sparse, retain_submitted,
};
impl fluxel_rhi_core::backend::CompletionBackend for Dx12 {
    type WaitFuture<'a>
        = super::fence_wait::Dx12FenceFuture
    where
        Self: 'a;
    fn completion_status(device: &Dx12Device, point: &Dx12Completion) -> CompletionState {
        // GetCompletedValue is native queue evidence; no host-side counter is
        // treated as completion.
        // SAFETY: `point` retains this fence COM interface; querying a fence
        // has no ownership or queue-order side effect.
        match super::fence_wait::fence_progress(
            unsafe { point.fence.GetCompletedValue() },
            point.value,
        ) {
            Err(error) => CompletionState::Failed(error),
            Ok(true) => {
                retire(device);
                CompletionState::Complete
            }
            Ok(false) => CompletionState::Pending,
        }
    }
    fn wait_completion<'a>(
        device: &'a Dx12Device,
        point: &'a Dx12Completion,
    ) -> Self::WaitFuture<'a> {
        super::fence_wait::Dx12FenceFuture::new(device.clone(), point.fence.clone(), point.value)
    }
    fn wait_completion_timeout<'a>(
        device: &'a Dx12Device,
        point: &'a Dx12Completion,
        timeout: std::time::Duration,
    ) -> impl std::future::Future<Output = Result<(), Error>> + 'a {
        super::fence_wait::Dx12FenceFuture::with_timeout(
            device.clone(),
            point.fence.clone(),
            point.value,
            timeout,
        )
    }
    fn wait_idle(device: &Dx12Device) -> Self::WaitFuture<'_> {
        // A direct-queue fence after a GPU wait on the compute frontier
        // proves both queues idle without blocking the calling CPU thread.
        let _submission = device
            .0
            .query_submission_gate
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        use super::super::queue;
        let result = device
            .require_active()
            .and_then(|()| queue::signal(device, 1))
            .and_then(|value| {
                queue::wait(device, 0, &device.0.multi_queue.fence, value)
            })
            .and_then(|()| queue::signal(device, 0));
        let value = match result {
            Ok(value) => value,
            Err(error) => {
                return super::fence_wait::Dx12FenceFuture::immediate(
                    device.clone(),
                    device.0.fence.clone(),
                    0,
                    Err(error),
                );
            }
        };
        super::fence_wait::Dx12FenceFuture::new(device.clone(), device.0.fence.clone(), value)
    }
}

pub(in crate::backend) fn retire(device: &Dx12Device) {
    // SAFETY: the device state owns this fence for every in-flight record;
    // querying it is side-effect free and device removal is handled below.
    let completed = unsafe { device.0.fence.GetCompletedValue() };
    device
        .0
        .abandoned_fence_events
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .retain(|event| {
            // SAFETY: each abandoned event retains its own native fence.
            let event_completed = unsafe { event.fence.GetCompletedValue() };
            event.retain_after(event_completed)
        });
    // SAFETY: this independent fence belongs to the device's compute queue.
    let compute_completed = unsafe { device.0.multi_queue.fence.GetCompletedValue() };
    if compute_completed != u64::MAX {
        device
            .0
            .multi_queue
            .in_flight
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .retain(|(value, _)| retain_after(compute_completed, *value));
    }
    if completed == u64::MAX {
        return;
    }
    device
        .0
        .in_flight
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .retain(|(value, _)| retain_after(completed, *value));
    device
        .0
        .sparse_in_flight
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .retain(|(value, _)| retain_after(completed, *value));
}

fn retain_after(completed: u64, value: u64) -> bool {
    completed == u64::MAX || value == u64::MAX || value > completed
}

fn seal_pending(
    issued: &[Dx12RecordedWork],
    pending: &mut Vec<usize>,
    lane: u32,
    fence: &ID3D12Fence,
    value: u64,
) {
    for index in pending.drain(..) {
        let encoder = &issued[index].encoder;
        super::query::seal_ops(&encoder.query_ops, lane, fence, value);
        for receipt in &encoder.readback_receipts {
            receipt.seal(fence, value);
        }
    }
}

/// Executes one encoder's native lists in their recorded order. Ordinary work
/// remains batched by the caller; this slower path is used only when an
/// explicit external queue boundary split the list.
fn execute_segmented(queue: &ID3D12CommandQueue, work: &Dx12RecordedWork) -> Result<(), Error> {
    use super::external_segment::QueueBoundary;
    debug_assert_eq!(
        work.encoder.closed_segments.len(),
        work.encoder.queue_boundaries.len()
    );
    for (index, (segment, boundary)) in work
        .encoder
        .closed_segments
        .iter()
        .zip(&work.encoder.queue_boundaries)
        .enumerate()
    {
        let list = Some(
            segment
                .list
                .cast::<ID3D12CommandList>()
                .expect("graphics command list interface"),
        );
        // SAFETY: the closed list belongs to this selected lane and the queue
        // lock held by the caller serializes all native queue operations.
        unsafe { queue.ExecuteCommandLists(&[list]) };
        super::super::ray::publish_segment(work, index);
        crate::diagnostics::record(crate::diagnostics::Operation::ExecuteCalls, 1);
        crate::diagnostics::record(crate::diagnostics::Operation::SubmittedLists, 1);
        match boundary {
            QueueBoundary::Wait { fence, value } => {
                // SAFETY: the external fence is retained by the recorded work
                // and this wait appears exactly at its explicit Core boundary.
                unsafe { queue.Wait(fence, *value) }.map_err(native_error)?;
            }
            QueueBoundary::Signal { fence, value } => {
                // SAFETY: this shared fence is retained by the recorded work
                // and receives the exact value created for this release.
                unsafe { queue.Signal(fence, *value) }.map_err(native_error)?;
                crate::diagnostics::record(crate::diagnostics::Operation::QueueSignal, 1);
            }
        }
    }
    let final_list = Some(
        work.encoder
            .list
            .cast::<ID3D12CommandList>()
            .expect("graphics command list interface"),
    );
    // SAFETY: as above, the final list is closed and the caller owns queue
    // serialization. It follows every explicit boundary in the source order.
    unsafe { queue.ExecuteCommandLists(&[final_list]) };
    super::super::ray::publish_segment(work, work.encoder.closed_segments.len());
    crate::diagnostics::record(crate::diagnostics::Operation::ExecuteCalls, 1);
    crate::diagnostics::record(crate::diagnostics::Operation::SubmittedLists, 1);
    Ok(())
}

/// A GPU signal retains the exact producer fence and value for one token.
pub struct Dx12GpuSignal {
    fence: ID3D12Fence,
    value: u64,
}
impl fluxel_rhi_core::backend::MultiQueueBackend for Dx12 {
    type NativeGpuSignal = Dx12GpuSignal;
    unsafe fn submit_multi(
        device: &Dx12Device,
        packet: api::MultiQueuePacket<Self>,
    ) -> api::MultiQueueSubmitOutcome<Self> {
        // SAFETY: Core has validated packet identity and the caller's execution relation.
        unsafe { submit_packet(device, packet, 0) }
    }
}

unsafe fn submit_packet(
    device: &Dx12Device,
    packet: api::MultiQueuePacket<Dx12>,
    first_step: usize,
) -> api::MultiQueueSubmitOutcome<Dx12> {
    use api::MultiQueueSubmitOutcome as Outcome;
    let prepared = match preflight::packet(device, &packet, first_step) {
        Ok(prepared) => prepared,
        Err(error) => return Outcome::Rejected { error, packet },
    };
    let _submission = crate::diagnostics::lock_submission_gate(&device.0.query_submission_gate);
    // Device removal can occur after the lock-free packet preflight. Recheck
    // it while the query/native handoff gate is held, alongside mutable query
    // admission, so no accepted query state races a removed device.
    let validate_under_gate = || -> Result<(), Error> {
        device.require_active()?;
        super::super::ray::validate_acceleration_recordings(packet.work.steps.iter().flat_map(
            |step| {
                let work = match &step.payload {
                    StepPayload::Commands(work) => work.as_slice(),
                    _ => &[],
                };
                work.iter().map(|record| {
                    // SAFETY: packet preflight and Core checked recorded-work ownership.
                    &unsafe { record.native_ref() }.encoder
                })
            },
        ))?;
        super::query::validate_ops(
            packet.work.steps.iter().flat_map(|step| {
                let index = step.step.index as usize;
                let lane = packet.work.plan.steps()[index].lane.ordinal;
                match &step.payload {
                    StepPayload::Commands(work) => work.as_slice(),
                    _ => &[],
                }
                .iter()
                .flat_map(move |record| {
                    // SAFETY: packet identity was checked by Core and preflight;
                    // this only borrows immutable recorded query operations.
                    unsafe { record.native_ref() }
                        .encoder
                        .query_ops
                        .iter()
                        .map(move |op| (lane, index, op))
                })
            }),
            &prepared.coverage,
            &prepared.external_waits,
        )
    };
    if let Err(error) = validate_under_gate() {
        return Outcome::Rejected { error, packet };
    }
    retire(device);
    let api::MultiQueuePacket {
        work: SubmitPacket {
            plan,
            steps,
            completion,
        },
        signals,
        waits,
    } = packet;
    let requested = match &completion {
        api::CompletionRequest::Steps(s) => s.as_ref(),
        _ => &[],
    };
    // Only cross-lane dependencies and externally requested tokens require
    // a signal at the exact producer step. Same-lane ordering is implicit.
    let mut needs_signal = vec![false; plan.steps().len()];
    for desc in plan.steps() {
        for wait in &desc.waits {
            let producer = wait.previous.index as usize;
            if plan.steps()[producer].lane != desc.lane {
                needs_signal[producer] = true;
            }
        }
    }
    for request in &signals {
        needs_signal[request.producer.index as usize] = true;
    }
    let mut waits: Vec<_> = waits.into_iter().map(Some).collect();
    let mut accepted = Vec::new();
    let mut presents = Vec::new();
    let mut frontiers = [None; 2];
    let mut checkpoints = Vec::new();
    let mut issued = Vec::new();
    let mut sparse_issued = Vec::new();
    let mut values = vec![None::<(u32, u64)>; first_step];
    let mut pending_receipts = [Vec::<usize>::new(), Vec::<usize>::new()];
    // A lane is dirty from its most recent native handoff until a fence is
    // queued after it.  Terminal ownership and checkpoints use that same
    // per-lane fence rather than joining compute into direct.
    let mut lane_dirty = [false; 2];
    let mut lane_values = [None; 2];
    let mut signaled = SignaledOwners::default();
    let mut tokens = Vec::new();
    macro_rules! handoff {
        ($kind:ident, $step:expr, $lane:expr, $operation:expr) => {
            match $operation {
                Ok(value) => value,
                Err(error) => {
                    retain_failed_handoff(
                        device,
                        lane_values,
                        &signaled,
                        issued,
                        sparse_issued,
                        &error,
                    );
                    return Outcome::Indeterminate {
                        error,
                        uncertain: failure::uncertain(
                            device,
                            plan.id(),
                            Dx12SubmissionOperation::$kind,
                            $step,
                            $lane,
                            accepted,
                            &signaled,
                        ),
                    };
                }
            }
        };
    }
    let mut steps = steps.into_iter();
    let mut stopped = None;
    while let Some(step) = steps.next() {
        let id = step.step;
        let desc = &plan.steps()[id.index as usize];
        let lane = desc.lane;
        // The lease can become stale after Core validates the packet. Check
        // before any native wait or token consumption for this step, so the
        // full step and remaining tail can still be returned to the caller.
        let admission = if matches!(&step.payload, StepPayload::Present(binding) if !binding.image().is_current())
        {
            Err(Error::SurfaceOutdated)
        } else if !lane_dirty[lane.ordinal as usize] {
            // Admit a new lane prefix only if it can receive a native fence
            // value. An already-dirty lane has reserved no new values since
            // this check, under the same submission gate.
            super::super::queue::require_signal_capacity(device, lane.ordinal)
        } else {
            Ok(())
        };
        if let Err(error) = admission {
            stopped = Some((
                error,
                std::iter::once(step).chain(steps).collect::<Vec<_>>(),
            ));
            break;
        }
        for dependency in &desc.waits {
            if (dependency.previous.index as usize) < first_step {
                // Resume requires actual completion on each accepted lane.
                continue;
            }
            let producer_lane = plan.steps()[dependency.previous.index as usize]
                .lane
                .ordinal;
            if producer_lane != lane.ordinal {
                let (_, value) = values[dependency.previous.index as usize]
                    .expect("cross-lane producer signaled");
                handoff!(
                    QueueWait,
                    Some(id),
                    lane,
                    super::super::queue::wait(
                        device,
                        lane.ordinal,
                        super::super::queue::fence(device, producer_lane).expect("validated lane"),
                        value
                    )
                );
            }
        }
        for entry in &mut waits {
            if entry
                .as_ref()
                .is_some_and(|wait| wait.signal.consumer() == id)
            {
                let wait = entry.take().expect("present wait");
                // SAFETY: each token is consumed exactly once at its designated step.
                let signal = unsafe { wait.signal.into_native() };
                handoff!(
                    QueueWait,
                    Some(id),
                    lane,
                    super::super::queue::wait(device, lane.ordinal, &signal.fence, signal.value)
                );
            }
        }
        let first = issued.len();
        match step.payload {
            StepPayload::Commands(work) => {
                for record in work {
                    // SAFETY: preflight checked closed work and its lane, and each work is consumed once.
                    issued.push(unsafe { record.into_native() });
                }
                let (queue, lock) = if lane.ordinal == 0 {
                    (&device.0.queue, &device.0.queue_lock)
                } else {
                    (&device.0.multi_queue.queue, &device.0.multi_queue.lock)
                };
                let _queue = lock.lock().unwrap_or_else(|p| p.into_inner());
                let segmented = issued[first..]
                    .iter()
                    .any(|native| !native.encoder.closed_segments.is_empty());
                // Pin submitted TLAS dependencies before native handoff. This
                // deliberately covers every retained bind group, including
                // inline ray queries in raster and compute shaders.
                super::super::ray::pin_batch(&mut issued[first..]);
                if segmented {
                    for native in &issued[first..] {
                        handoff!(
                            CommandListHandoff,
                            Some(id),
                            lane,
                            execute_segmented(queue, native)
                        );
                    }
                } else {
                    let lists: Vec<_> = issued[first..]
                        .iter()
                        .map(|native| {
                            Some(
                                native
                                    .encoder
                                    .list
                                    .cast::<ID3D12CommandList>()
                                    .expect("graphics command list interface"),
                            )
                        })
                        .collect();
                    if !lists.is_empty() {
                        // SAFETY: preflight checked every closed list and its lane;
                        // execution order and hazards are supplied by the caller.
                        crate::diagnostics::record(crate::diagnostics::Operation::ExecuteCalls, 1);
                        crate::diagnostics::record(
                            crate::diagnostics::Operation::SubmittedLists,
                            lists.len() as u64,
                        );
                        unsafe {
                            queue.ExecuteCommandLists(&lists);
                        }
                        for native in &issued[first..] {
                            super::super::ray::publish_segment(native, 0);
                        }
                    }
                }
                if first != issued.len() {
                    for index in first..issued.len() {
                        super::query::commit_ops(&issued[index].encoder.query_ops, lane.ordinal);
                        let encoder = &issued[index].encoder;
                        if !encoder.query_ops.is_empty() || !encoder.readback_receipts.is_empty() {
                            pending_receipts[lane.ordinal as usize].push(index);
                        }
                    }
                }
            }
            StepPayload::SparseBind(binds) => {
                let _queue = device
                    .0
                    .queue_lock
                    .lock()
                    .unwrap_or_else(|p| p.into_inner());
                for bind in &binds {
                    sparse_issued.push(super::super::sparse::issue_bind(device, bind));
                }
            }
            StepPayload::Present(mut binding) => {
                // SAFETY: exclusive current image lease is handed to DXGI once.
                handoff!(Present, Some(id), lane, unsafe { binding.try_handoff() });
                presents.push(PresentReceipt {
                    step: Some(id),
                    accepted: true,
                });
            }
        }
        accepted.push(id);
        frontiers[lane.ordinal as usize] = Some((lane, id));
        lane_dirty[lane.ordinal as usize] = true;
        let value = if needs_signal[id.index as usize] {
            let value = handoff!(
                QueueSignal,
                Some(id),
                lane,
                super::super::queue::signal(device, lane.ordinal)
            );
            let fence = super::super::queue::fence(device, lane.ordinal).expect("validated lane");
            seal_pending(
                &issued,
                &mut pending_receipts[lane.ordinal as usize],
                lane.ordinal,
                fence,
                value,
            );
            lane_values[lane.ordinal as usize] = Some(value);
            signaled.record(lane, id, value, issued.len(), sparse_issued.len());
            lane_dirty[lane.ordinal as usize] = false;
            Some(value)
        } else {
            None
        };
        values.push(value.map(|v| (lane.ordinal, v)));
        for request in signals.iter().filter(|s| s.producer == id) {
            let native = Dx12GpuSignal {
                fence: super::super::queue::fence(device, lane.ordinal)
                    .expect("validated lane")
                    .clone(),
                value: value.expect("explicit signal producer is marked"),
            };
            // SAFETY: this signal was just enqueued after the requested producer step.
            tokens.push(unsafe {
                api::GpuSignalPoint::from_native(
                    plan.id().device,
                    id,
                    lane,
                    request.consumer,
                    request.consumer_lane,
                    native,
                )
            });
        }
        if requested.contains(&id) {
            // A requested checkpoint covers this step's lane prefix only.
            // Do not add a direct<-compute wait here: cross-lane visibility is
            // already represented by the plan dependency processed above.
            let fence_value = if lane_dirty[lane.ordinal as usize] {
                handoff!(
                    QueueSignal,
                    Some(id),
                    lane,
                    super::super::queue::signal(device, lane.ordinal)
                )
            } else {
                lane_values[lane.ordinal as usize]
                    .expect("accepted lane has a previously signaled frontier")
            };
            seal_pending(
                &issued,
                &mut pending_receipts[lane.ordinal as usize],
                lane.ordinal,
                super::super::queue::fence(device, lane.ordinal).expect("validated lane"),
                fence_value,
            );
            lane_values[lane.ordinal as usize] = Some(fence_value);
            signaled.record(lane, id, fence_value, issued.len(), sparse_issued.len());
            lane_dirty[lane.ordinal as usize] = false;
            checkpoints.push(checkpoint(device, plan.id().device, lane, fence_value, id));
        }
    }
    if !accepted.is_empty() {
        for ordinal in 0..2 {
            if lane_dirty[ordinal] {
                let (lane, step) = frontiers[ordinal].expect("dirty lane has an issued step");
                let value = handoff!(
                    QueueSignal,
                    None,
                    lane,
                    super::super::queue::signal(device, lane.ordinal)
                );
                seal_pending(
                    &issued,
                    &mut pending_receipts[ordinal],
                    ordinal as u32,
                    super::super::queue::fence(device, ordinal as u32).expect("validated lane"),
                    value,
                );
                lane_values[ordinal] = Some(value);
                signaled.record(lane, step, value, issued.len(), sparse_issued.len());
                lane_dirty[ordinal] = false;
            }
        }
        let retained_lanes = retain_submitted(device, lane_values, &mut issued);
        // Sparse mappings are direct-queue operations. Validation above
        // rejects compute-lane sparse steps, so their owner follows only the
        // direct timeline even when compute work is also present.
        let retained_sparse = !sparse_issued.is_empty();
        if retained_sparse {
            retain_sparse(
                device,
                lane_values[0].expect("accepted direct sparse work has a fence"),
                sparse_issued,
            );
        }
        // Every terminal fence gets one autonomous OS completion observation.
        // Register only after every owner for that lane has entered its
        // in-flight store, otherwise a fast completion could retire an earlier
        // subset and strand later ownership without another API call.
        if let Err((ordinal, error)) =
            register_retirements(device, lane_values, retained_lanes, retained_sparse)
        {
            // Handoff has happened and in-flight stores retain each owner.
            return Outcome::Indeterminate {
                error,
                uncertain: failure::uncertain(
                    device,
                    plan.id(),
                    Dx12SubmissionOperation::RetirementRegistration,
                    None,
                    frontiers[ordinal as usize]
                        .expect("retirement lane has issued work")
                        .0,
                    accepted,
                    &signaled,
                ),
            };
        }
        if stopped.is_some() || matches!(completion, api::CompletionRequest::End) {
            for (ordinal, frontier) in frontiers.into_iter().enumerate() {
                if let Some((lane, step)) = frontier {
                    if checkpoints
                        .iter()
                        .any(|p| p.covers.lane_frontiers.as_slice() == [(lane, step)])
                    {
                        continue;
                    }
                    checkpoints.push(checkpoint(
                        device,
                        plan.id().device,
                        lane,
                        lane_values[ordinal].expect("accepted lane has a terminal fence"),
                        step,
                    ));
                }
            }
        }
        retire(device);
    }
    let receipt = api::MultiQueueReceipt {
        work: SubmissionReceipt {
            device: plan.id().device,
            index: None,
            accepted,
            checkpoints,
            presents,
        },
        signals: tokens,
    };
    if let Some((error, pending)) = stopped {
        return tail::stopped(
            error,
            receipt,
            api::MultiQueuePacket {
                work: SubmitPacket {
                    plan,
                    steps: pending,
                    completion,
                },
                signals,
                waits: waits.into_iter().flatten().collect(),
            },
        );
    }
    Outcome::Accepted(receipt)
}

#[cfg(test)]
#[path = "../../../tests/backend/encoder/submit.rs"]
mod tests;
