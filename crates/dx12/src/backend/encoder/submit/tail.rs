//! Bridge between DX12's packet execution and Core's owning deferred tail.
//!
//! Resuming requires completed native evidence for each lane in the accepted
//! prefix. Only the unissued suffix is passed back to the executor.

use super::*;
use fluxel_rhi_core::backend::CompletionBackend;

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
        // SAFETY: Core checked packet ownership and the caller supplies synchronization.
        let outcome = unsafe { submit_packet(device, ordinary(packet), 0) };
        ordinary_outcome(outcome, None)
    }

    unsafe fn resume(device: &Dx12Device, tail: api::DeferredTail<Self>) -> SubmitOutcome<Self> {
        if let Err(error) = validate_frontier(device, &tail) {
            return SubmitOutcome::Rejected {
                error,
                work: api::RetryableSubmit::Tail(tail),
            };
        }
        let api::DeferredTail {
            plan,
            next_step,
            pending,
            accepted_frontier,
            completion,
        } = tail;
        let packet = ordinary(SubmitPacket {
            plan,
            steps: pending,
            completion,
        });
        // SAFETY: Core checked the tail's plan and payloads. Every prefix lane
        // is complete, and this call consumes only the unissued suffix.
        let outcome = unsafe { submit_packet(device, packet, next_step as usize) };
        ordinary_outcome(outcome, Some((next_step, accepted_frontier)))
    }
}

fn ordinary(work: SubmitPacket<Dx12>) -> api::MultiQueuePacket<Dx12> {
    api::MultiQueuePacket {
        work,
        signals: Vec::new(),
        waits: Vec::new(),
    }
}

pub(super) fn stopped(
    error: Error,
    receipt: api::MultiQueueReceipt<Dx12>,
    packet: api::MultiQueuePacket<Dx12>,
) -> api::MultiQueueSubmitOutcome<Dx12> {
    use api::MultiQueueSubmitOutcome as Outcome;
    if receipt.work.accepted.is_empty() {
        return Outcome::Rejected { error, packet };
    }
    let next = packet.work.steps[0].step.index;
    Outcome::PartiallyAccepted {
        error,
        accepted: receipt,
        unaccepted: api::MultiQueueTail {
            plan: packet.work.plan,
            pending: packet.work.steps,
            completion: packet.work.completion,
            remaining_waits: packet.waits,
            remaining_signals: packet
                .signals
                .into_iter()
                .filter(|request| request.producer.index >= next)
                .collect(),
        },
    }
}

fn frontiers(plan: &api::SubmissionPlan<Dx12>, end: usize) -> [Option<(LaneId, api::StepId)>; 2] {
    let mut latest = [None; 2];
    for (index, step) in plan.steps()[..end].iter().enumerate() {
        latest[step.lane.ordinal as usize] = Some((step.lane, plan.step(index)));
    }
    latest
}

fn validate_frontier(device: &Dx12Device, tail: &api::DeferredTail<Dx12>) -> Result<(), Error> {
    let next = tail.next_step as usize;
    if next == 0 || tail.pending.is_empty() || next >= tail.plan.steps().len() {
        return Err(Error::InvalidState);
    }
    if tail.plan.steps().iter().any(|step| step.lane.ordinal > 1) {
        return Err(Error::WrongLane);
    }
    let mut expected = frontiers(&tail.plan, next);
    for checkpoint in &tail.accepted_frontier {
        let &[(lane, step)] = checkpoint.covers.lane_frontiers.as_slice() else {
            return Err(Error::InvalidState);
        };
        let slot = expected
            .get_mut(lane.ordinal as usize)
            .ok_or(Error::WrongLane)?;
        if slot.take() != Some((lane, step)) {
            return Err(Error::InvalidState);
        }
        // SAFETY: Core checked the checkpoint belongs to this device.
        let point = unsafe { checkpoint.point.native_ref() };
        if super::super::super::queue::fence(device, lane.ordinal)?.as_raw() != point.fence.as_raw()
            || point.value == 0
            || point.value == u64::MAX
        {
            return Err(Error::InvalidState);
        }
        match Dx12::completion_status(device, point) {
            CompletionState::Complete => (),
            CompletionState::Pending => return Err(Error::InvalidState),
            CompletionState::Failed(error) => return Err(error),
        }
    }
    if expected.iter().any(Option::is_some) {
        return Err(Error::InvalidState);
    }
    Ok(())
}

fn ordinary_outcome(
    outcome: api::MultiQueueSubmitOutcome<Dx12>,
    resumed: Option<(u32, Vec<api::Checkpoint<Dx12>>)>,
) -> SubmitOutcome<Dx12> {
    use api::MultiQueueSubmitOutcome as Outcome;
    match outcome {
        Outcome::Accepted(receipt) => SubmitOutcome::Accepted(receipt.work),
        Outcome::Rejected { error, packet } => {
            let work = match resumed {
                None => api::RetryableSubmit::Fresh(packet.work),
                Some((next_step, accepted_frontier)) => {
                    api::RetryableSubmit::Tail(api::DeferredTail {
                        plan: packet.work.plan,
                        next_step,
                        pending: packet.work.steps,
                        accepted_frontier,
                        completion: packet.work.completion,
                    })
                }
            };
            SubmitOutcome::Rejected { error, work }
        }
        Outcome::Indeterminate { error, uncertain } => {
            SubmitOutcome::Indeterminate { error, uncertain }
        }
        Outcome::PartiallyAccepted {
            error,
            accepted,
            unaccepted,
        } => {
            let next_step = unaccepted.pending[0].step.index;
            let previous = resumed.map_or_else(Vec::new, |(_, frontier)| frontier);
            let accepted_frontier = frontiers(&unaccepted.plan, next_step as usize)
                .into_iter()
                .flatten()
                .map(|frontier| {
                    let checkpoint = accepted
                        .work
                        .checkpoints
                        .iter()
                        .chain(previous.iter())
                        .find(|p| p.covers.lane_frontiers.as_slice() == [frontier])
                        .expect("partial submission seals every accepted lane");
                    api::Checkpoint {
                        point: checkpoint.point.clone(),
                        covers: api::CompletionCoverage {
                            lane_frontiers: vec![frontier],
                        },
                    }
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
    }
}
