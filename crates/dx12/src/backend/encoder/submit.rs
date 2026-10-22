//! D3D12 queue submission, fence facts, and fence-retired work ownership.
//!
//! A Core `SubmissionPlan` fixes command order and dependencies. This module
//! issues steps on the direct and compute queues, with native fence signals and
//! waits for cross-queue dependencies. The submission gate serializes packet
//! validation and handoff; each queue lock serializes its native operations.
//! An aggregate direct-queue fence waits for the compute frontier before it
//! provides completion evidence for retained work from both queues. A fence
//! value of `u64::MAX` denotes device removal or quarantined uncertainty.

use super::*;
impl api::CompletionBackend for Dx12 {
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
            event_completed != u64::MAX && event.value > event_completed
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
    value: u64,
    lane_frontiers: &[(LaneId, api::StepId)],
) -> api::Checkpoint<Dx12> {
    api::Checkpoint {
        // SAFETY: this completion point carries only a cloned device-owned
        // fence and the monotonically allocated value serialized above.
        point: unsafe {
            CompletionPoint::from_native(
                device_id,
                Dx12Completion {
                    fence: device.0.fence.clone(),
                    value,
                },
            )
        },
        covers: CompletionCoverage {
            lane_frontiers: lane_frontiers.to_vec(),
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
        device
            .0
            .in_flight
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .push((u64::MAX, issued));
    }
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
impl api::MultiQueueBackend for Dx12 {
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
    let _submission = device
        .0
        .query_submission_gate
        .lock()
        .unwrap_or_else(|p| p.into_inner());
    let validate = || -> Result<(), Error> {
        // SAFETY: querying both device-owned fences has no queue side effect.
        if unsafe { device.0.fence.GetCompletedValue() } == u64::MAX
            || unsafe { device.0.multi_queue.fence.GetCompletedValue() } == u64::MAX
        {
            return Err(Error::DeviceLost);
        }
        for (index, step) in packet.work.steps.iter().enumerate() {
            let desc = &packet.work.plan.steps()[index];
            if desc.lane.device != packet.work.plan.id().device || desc.lane.ordinal > 1 {
                return Err(Error::WrongLane);
            }
            for wait in &desc.waits {
                if wait.previous.plan != packet.work.plan.id()
                    || wait.previous.index as usize >= index
                {
                    return Err(Error::InvalidState);
                }
            }
            match &step.payload {
                StepPayload::Commands(work) => {
                    for record in work {
                        // SAFETY: Core checked owning backend and device before calling this method.
                        let native = unsafe { record.native_ref() };
                        if !native.encoder.closed || native.lane != desc.lane {
                            return Err(Error::InvalidState);
                        }
                    }
                }
                StepPayload::SparseBind(binds) => {
                    if desc.lane.ordinal != 0 {
                        return Err(Error::Unsupported(
                            "DX12 sparse mappings require direct lane",
                        ));
                    }
                    if !device.0.sparse_supported || binds.is_empty() {
                        return Err(Error::InvalidRange);
                    }
                    binds
                        .iter()
                        .try_for_each(super::super::sparse::validate_bind)?;
                }
                StepPayload::Present(_) if desc.lane.ordinal != 0 || index == 0 => {
                    return Err(Error::InvalidState);
                }
                _ => {}
            }
        }
        // Track only ordering already present in the submitted plan. Same-lane
        // order is implicit; explicit waits import the producer's frontier.
        let mut coverage = Vec::<[Option<usize>; 2]>::new();
        let mut external_waits = Vec::<Vec<(ID3D12Fence, u64)>>::new();
        let mut lanes: [Option<usize>; 2] = [None; 2];
        for (index, desc) in packet.work.plan.steps().iter().enumerate() {
            let lane = desc.lane.ordinal as usize;
            let mut frontier = lanes[lane]
                .map(|previous| coverage[previous])
                .unwrap_or([None; 2]);
            let mut native_frontier = lanes[lane]
                .map(|previous| external_waits[previous].clone())
                .unwrap_or_default();
            for wait in &desc.waits {
                for (slot, imported) in frontier
                    .iter_mut()
                    .zip(coverage[wait.previous.index as usize])
                {
                    *slot = (*slot).max(imported);
                }
                for (fence, value) in &external_waits[wait.previous.index as usize] {
                    if let Some((_, current)) = native_frontier
                        .iter_mut()
                        .find(|(known, _)| known.as_raw() == fence.as_raw())
                    {
                        *current = (*current).max(*value);
                    } else {
                        native_frontier.push((fence.clone(), *value));
                    }
                }
            }
            for wait in packet
                .waits
                .iter()
                .filter(|wait| wait.signal.consumer().index as usize == index)
            {
                // SAFETY: Core validated token ownership; preflight only borrows its immutable native point.
                let signal = unsafe { wait.signal.native_ref() };
                if let Some((_, value)) = native_frontier
                    .iter_mut()
                    .find(|(known, _)| known.as_raw() == signal.fence.as_raw())
                {
                    *value = (*value).max(signal.value);
                } else {
                    native_frontier.push((signal.fence.clone(), signal.value));
                }
            }
            frontier[lane] = Some(index);
            coverage.push(frontier);
            external_waits.push(native_frontier);
            lanes[lane] = Some(index);
        }
        super::query::validate_ops(
            packet.work.steps.iter().flat_map(|step| {
                let index = step.step.index as usize;
                let lane = packet.work.plan.steps()[step.step.index as usize]
                    .lane
                    .ordinal;
                match &step.payload {
                    StepPayload::Commands(work) => work.as_slice(),
                    _ => &[],
                }
                .iter()
                .flat_map(move |record| {
                    // SAFETY: packet identity checked by Core, this borrow is read-only.
                    unsafe { record.native_ref() }
                        .encoder
                        .query_ops
                        .iter()
                        .map(move |op| (lane, index, op))
                })
            }),
            &coverage,
            &external_waits,
        )?;
        Ok(())
    };
    if let Err(error) = validate() {
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
    let mut frontiers = Vec::new();
    let mut checkpoints = Vec::new();
    let mut issued = Vec::new();
    let mut sparse_issued = Vec::new();
    let mut values = Vec::<Option<(u32, u64)>>::new();
    let mut pending_queries = [Vec::<usize>::new(), Vec::<usize>::new()];
    let mut compute_seen = false;
    let mut compute_dirty = false;
    let mut compute_value = None;
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
        if lane.ordinal == 1 {
            compute_seen = true;
            compute_dirty = true;
        }
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
            if lane.ordinal == 1 {
                compute_value = Some(value);
                compute_dirty = false;
            }
            Some(value)
        } else {
            None
        };
        values.push(value.map(|v| (lane.ordinal, v)));
        accepted.push(id);
        frontiers.retain(|(l, _)| *l != lane);
        frontiers.push((lane, id));
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
            if compute_seen {
                if compute_dirty {
                    let value = handoff!(super::super::queue::signal(device, 1));
                    seal_pending(
                        &issued,
                        &mut pending_queries[1],
                        1,
                        &device.0.multi_queue.fence,
                        value,
                    );
                    compute_value = Some(value);
                    compute_dirty = false;
                }
                handoff!(super::super::queue::wait(
                    device,
                    0,
                    &device.0.multi_queue.fence,
                    compute_value.expect("compute frontier signaled")
                ));
            }
            let fence_value = handoff!(signal_queue(device));
            seal_pending(
                &issued,
                &mut pending_queries[0],
                0,
                &device.0.fence,
                fence_value,
            );
            checkpoints.push(checkpoint(
                device,
                plan.id().device,
                fence_value,
                &frontiers,
            ));
        }
    }
    let retention = if accepted.is_empty() {
        None
    } else {
        if compute_seen {
            if compute_dirty {
                let value = handoff!(super::super::queue::signal(device, 1));
                seal_pending(
                    &issued,
                    &mut pending_queries[1],
                    1,
                    &device.0.multi_queue.fence,
                    value,
                );
                compute_value = Some(value);
            }
            handoff!(super::super::queue::wait(
                device,
                0,
                &device.0.multi_queue.fence,
                compute_value.expect("compute frontier signaled")
            ));
        }
        Some(handoff!(signal_queue(device)))
    };
    if let Some(value) = retention {
        seal_pending(&issued, &mut pending_queries[0], 0, &device.0.fence, value);
        device
            .0
            .in_flight
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .push((value, issued));
        retain_sparse(device, value, sparse_issued);
        if matches!(completion, api::CompletionRequest::End) {
            checkpoints.push(checkpoint(device, plan.id().device, value, &frontiers));
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
mod tests {
    use super::retain_after;

    #[test]
    fn device_removed_keeps_all_submitted_native_owners() {
        assert!(retain_after(u64::MAX, 1));
        assert!(retain_after(u64::MAX, u64::MAX));
        assert!(retain_after(10, u64::MAX));
        assert!(retain_after(10, 11));
        assert!(!retain_after(10, 10));
    }
}
