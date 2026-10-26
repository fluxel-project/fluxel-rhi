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

mod preflight;
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
    fn wait_idle(device: &Dx12Device) -> Self::WaitFuture<'_> {
        // A direct-queue fence after a GPU wait on the compute frontier
        // proves both queues idle without blocking the calling CPU thread.
        let _submission = device
            .0
            .query_submission_gate
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let result = super::super::queue::signal(device, 1)
            .and_then(|value| {
                super::super::queue::wait(device, 0, &device.0.multi_queue.fence, value)
            })
            .and_then(|()| signal_queue(device));
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

pub(super) fn retire(device: &Dx12Device) {
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

// The queue gate serializes each fence allocation with its native Signal.
// u64::MAX is reserved for device removal and never used as ordinary
// completion evidence.
fn signal_queue(device: &Dx12Device) -> Result<u64, Error> {
    let _queue = device
        .0
        .queue_lock
        .lock()
        .unwrap_or_else(|p| p.into_inner());
    let value = device
        .0
        .next_fence
        .fetch_update(
            std::sync::atomic::Ordering::Relaxed,
            std::sync::atomic::Ordering::Relaxed,
            |next| next.checked_add(1),
        )
        .map_err(|_| Error::InvalidRange)?;
    // SAFETY: `queue_lock` holds the queue's external synchronization boundary,
    // and the device owns both queue and fence for their whole shared lifetime.
    crate::diagnostics::record(crate::diagnostics::Operation::QueueSignal, 1);
    unsafe { device.0.queue.Signal(&device.0.fence, value) }.map_err(native_error)?;
    Ok(value)
}

fn checkpoint(
    device: &Dx12Device,
    device_id: api::DeviceId,
    lane: LaneId,
    value: u64,
    frontier: api::StepId,
) -> api::Checkpoint<Dx12> {
    api::Checkpoint {
        // SAFETY: this completion point carries only a cloned device-owned
        // fence and the monotonically allocated value serialized above.
        point: unsafe {
            CompletionPoint::from_native(
                device_id,
                Dx12Completion {
                    fence: super::super::queue::fence(device, lane.ordinal)
                        .expect("validated lane")
                        .clone(),
                    value,
                },
            )
        },
        covers: CompletionCoverage {
            // A queue fence proves only its own ordered prefix. Dependencies
            // that make another lane observable are represented by the plan's
            // explicit native queue waits, never by an implicit terminal join.
            lane_frontiers: vec![(lane, frontier)],
        },
    }
}

fn seal_pending(
    issued: &[Dx12RecordedWork],
    pending: &mut Vec<usize>,
    lane: u32,
    fence: &ID3D12Fence,
    value: u64,
) {
    for index in pending.drain(..) {
        super::query::seal_ops(&issued[index].encoder.query_ops, lane, fence, value);
    }
}

fn retain_indeterminate(device: &Dx12Device, issued: Vec<Dx12RecordedWork>) {
    if !issued.is_empty() {
        for record in &issued {
            super::query::invalidate_ops(&record.encoder.query_ops);
        }
        let mut direct = Vec::new();
        let mut compute = Vec::new();
        for record in issued {
            match record.lane.ordinal {
                0 => direct.push(record),
                1 => compute.push(record),
                _ => unreachable!("submission preflight validates DX12 lanes"),
            }
        }
        if !direct.is_empty() {
            device
                .0
                .in_flight
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .push((u64::MAX, direct));
        }
        if !compute.is_empty() {
            device
                .0
                .multi_queue
                .in_flight
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .push((u64::MAX, compute));
        }
    }
}

fn retain_submitted(
    device: &Dx12Device,
    values: [Option<u64>; 2],
    issued: &mut Vec<Dx12RecordedWork>,
) -> [bool; 2] {
    let mut direct = Vec::new();
    let mut compute = Vec::new();
    for record in issued.drain(..) {
        match record.lane.ordinal {
            0 => direct.push(record),
            1 => compute.push(record),
            _ => unreachable!("submission preflight validates DX12 lanes"),
        }
    }
    let retained = [!direct.is_empty(), !compute.is_empty()];
    if retained[0] {
        let value = values[0].expect("direct work has a terminal direct fence");
        device
            .0
            .in_flight
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .push((value, direct));
    }
    if retained[1] {
        let value = values[1].expect("compute work has a terminal compute fence");
        device
            .0
            .multi_queue
            .in_flight
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .push((value, compute));
    }
    retained
}

fn retain_sparse(
    device: &Dx12Device,
    value: u64,
    retained: Vec<super::super::sparse::SparseRetained>,
) {
    if !retained.is_empty() {
        device
            .0
            .sparse_in_flight
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .push((value, retained));
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

impl DeferredBackend for Dx12 {
    type NativeRecordedWork = Dx12RecordedWork;
    fn finish_encoder(mut encoder: Dx12Encoder, lane: LaneId) -> Result<Dx12RecordedWork, Error> {
        if encoder.active_occlusion.is_some() || encoder.active_statistics.is_some() {
            return Err(Error::InvalidState);
        }
        // SAFETY: `encoder` exclusively owns its still-open list; Core calls
        // this once before any list can cross the queue boundary.
        unsafe { encoder.list.Close() }.map_err(native_error)?;
        encoder.closed = true;
        Ok(Dx12RecordedWork { lane, encoder })
    }
    unsafe fn submit(device: &Dx12Device, packet: SubmitPacket<Self>) -> SubmitOutcome<Self> {
        // SAFETY: ordinary submission has the same queue ordering requirements.
        match unsafe {
            submit_packet(
                device,
                api::MultiQueuePacket {
                    work: packet,
                    signals: vec![],
                    waits: vec![],
                },
            )
        } {
            api::MultiQueueSubmitOutcome::Accepted(r) => SubmitOutcome::Accepted(r.work),
            api::MultiQueueSubmitOutcome::Rejected { error, packet } => SubmitOutcome::Rejected {
                error,
                work: api::RetryableSubmit::Fresh(packet.work),
            },
            api::MultiQueueSubmitOutcome::Indeterminate { error, uncertain } => {
                SubmitOutcome::Indeterminate { error, uncertain }
            }
            api::MultiQueueSubmitOutcome::PartiallyAccepted { .. } => {
                unreachable!("DX12 reports uncertain native handoffs as indeterminate")
            }
        }
    }
    // SAFETY: this route issues no native work and returns the owning tail
    // unchanged, so it cannot advance or discard the accepted frontier.
    unsafe fn resume(_: &Dx12Device, tail: api::DeferredTail<Self>) -> SubmitOutcome<Self> {
        SubmitOutcome::Rejected {
            error: Error::Unsupported("DX12 submit resume"),
            work: api::RetryableSubmit::Tail(tail),
        }
    }
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
        unsafe { submit_packet(device, packet) }
    }
}

unsafe fn submit_packet(
    device: &Dx12Device,
    packet: api::MultiQueuePacket<Dx12>,
) -> api::MultiQueueSubmitOutcome<Dx12> {
    use api::MultiQueueSubmitOutcome as Outcome;
    let prepared = match preflight::packet(device, &packet) {
        Ok(prepared) => prepared,
        Err(error) => return Outcome::Rejected { error, packet },
    };
    let _submission = crate::diagnostics::lock_submission_gate(&device.0.query_submission_gate);
    // Device removal can occur after the lock-free packet preflight. Recheck
    // it while the query/native handoff gate is held, alongside mutable query
    // admission, so no accepted query state races a removed device.
    let validate_under_gate = || -> Result<(), Error> {
        if unsafe { device.0.fence.GetCompletedValue() } == u64::MAX
            || unsafe { device.0.multi_queue.fence.GetCompletedValue() } == u64::MAX
        {
            return Err(Error::DeviceLost);
        }
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
    let mut values = Vec::<Option<(u32, u64)>>::new();
    let mut pending_queries = [Vec::<usize>::new(), Vec::<usize>::new()];
    // A lane is dirty from its most recent native handoff until a fence is
    // queued after it.  Terminal ownership and checkpoints use that same
    // per-lane fence rather than joining compute into direct.
    let mut lane_dirty = [false; 2];
    let mut lane_values = [None; 2];
    let mut tokens = Vec::new();
    macro_rules! handoff {
        ($operation:expr) => {
            match $operation {
                Ok(value) => value,
                Err(error) => {
                    retain_indeterminate(device, issued);
                    retain_sparse(device, u64::MAX, sparse_issued);
                    return Outcome::Indeterminate {
                        error,
                        uncertain: unsafe {
                            api::NativeUncertainty::from_native(plan.id().device, ())
                        },
                    };
                }
            }
        };
    }
    for step in steps {
        let id = step.step;
        let desc = &plan.steps()[id.index as usize];
        let lane = desc.lane;
        for dependency in &desc.waits {
            let producer_lane = plan.steps()[dependency.previous.index as usize]
                .lane
                .ordinal;
            if producer_lane != lane.ordinal {
                let (_, value) = values[dependency.previous.index as usize]
                    .expect("cross-lane producer signaled");
                handoff!(super::super::queue::wait(
                    device,
                    lane.ordinal,
                    super::super::queue::fence(device, producer_lane).expect("validated lane"),
                    value
                ));
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
                handoff!(super::super::queue::wait(
                    device,
                    lane.ordinal,
                    &signal.fence,
                    signal.value
                ));
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
                        handoff!(execute_segmented(queue, native));
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
                        if !issued[index].encoder.query_ops.is_empty() {
                            pending_queries[lane.ordinal as usize].push(index);
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
                handoff!(unsafe { binding.try_handoff() });
                presents.push(PresentReceipt {
                    step: Some(id),
                    accepted: true,
                });
            }
        }
        lane_dirty[lane.ordinal as usize] = true;
        let value = if needs_signal[id.index as usize] {
            let value = handoff!(super::super::queue::signal(device, lane.ordinal));
            let fence = super::super::queue::fence(device, lane.ordinal).expect("validated lane");
            seal_pending(
                &issued,
                &mut pending_queries[lane.ordinal as usize],
                lane.ordinal,
                fence,
                value,
            );
            lane_values[lane.ordinal as usize] = Some(value);
            lane_dirty[lane.ordinal as usize] = false;
            Some(value)
        } else {
            None
        };
        values.push(value.map(|v| (lane.ordinal, v)));
        accepted.push(id);
        frontiers[lane.ordinal as usize] = Some((lane, id));
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
                handoff!(super::super::queue::signal(device, lane.ordinal))
            } else {
                lane_values[lane.ordinal as usize]
                    .expect("accepted lane has a previously signaled frontier")
            };
            seal_pending(
                &issued,
                &mut pending_queries[lane.ordinal as usize],
                lane.ordinal,
                super::super::queue::fence(device, lane.ordinal).expect("validated lane"),
                fence_value,
            );
            lane_values[lane.ordinal as usize] = Some(fence_value);
            lane_dirty[lane.ordinal as usize] = false;
            checkpoints.push(checkpoint(device, plan.id().device, lane, fence_value, id));
        }
    }
    if !accepted.is_empty() {
        for ordinal in 0..2 {
            if lane_dirty[ordinal] {
                let value = handoff!(super::super::queue::signal(device, ordinal as u32));
                seal_pending(
                    &issued,
                    &mut pending_queries[ordinal],
                    ordinal as u32,
                    super::super::queue::fence(device, ordinal as u32).expect("validated lane"),
                    value,
                );
                lane_values[ordinal] = Some(value);
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
        for ordinal in 0..2 {
            if retained_lanes[ordinal] || (ordinal == 0 && retained_sparse) {
                if let Err(error) = retirement::register(
                    device,
                    ordinal as u32,
                    lane_values[ordinal].expect("retained lane has a terminal fence"),
                ) {
                    // Handoff and query commit have already occurred. The
                    // in-flight stores retain every owner, so report the
                    // truthful uncertain boundary rather than Accepted.
                    return Outcome::Indeterminate {
                        error,
                        uncertain: unsafe {
                            api::NativeUncertainty::from_native(plan.id().device, ())
                        },
                    };
                }
            }
        }
        if matches!(completion, api::CompletionRequest::End) {
            for (ordinal, frontier) in frontiers.into_iter().enumerate() {
                if let Some((lane, step)) = frontier {
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
    Outcome::Accepted(api::MultiQueueReceipt {
        work: SubmissionReceipt {
            device: plan.id().device,
            index: None,
            accepted,
            checkpoints,
            presents,
        },
        signals: tokens,
    })
}

#[cfg(test)]
#[path = "../../../tests/backend/encoder/submit.rs"]
mod tests;
