//! Cold-path DX12 submission diagnostics and native checkpoint construction.

use super::*;
use crate::backend::queue;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dx12SubmissionOperation {
    QueueWait,
    CommandListHandoff,
    Present,
    QueueSignal,
    RetirementRegistration,
}

/// Facts observed during a failed execution of a deferred plan.
pub struct Dx12SubmissionFailure {
    pub plan: api::PlanId,
    pub operation: Dx12SubmissionOperation,
    /// The individual step affected, when the failed operation was step-local.
    pub step: Option<api::StepId>,
    pub lane: LaneId,
    /// Step payloads that crossed native handoff before the reported failure.
    pub issued_steps: Vec<api::StepId>,
    /// Successful fence signals proved these lane prefixes. The points still
    /// need to be observed before the covered GPU work can be retired.
    pub checkpoints: Vec<api::Checkpoint<Dx12>>,
}

#[cold]
pub(super) fn uncertain(
    device: &Dx12Device,
    plan: api::PlanId,
    operation: Dx12SubmissionOperation,
    step: Option<api::StepId>,
    lane: LaneId,
    issued_steps: Vec<api::StepId>,
    signaled: &SignaledOwners,
) -> api::NativeUncertainty<Dx12> {
    let checkpoints = signaled
        .frontiers
        .iter()
        .flatten()
        .map(|&(lane, step, value)| checkpoint(device, plan.device, lane, value, step))
        .collect();
    // SAFETY: the checkpoints only name signals issued by this device. The
    // executor retained all work whose handoff cannot be determined.
    unsafe {
        api::NativeUncertainty::from_native(
            plan.device,
            Dx12SubmissionFailure {
                plan,
                operation,
                step,
                lane,
                issued_steps,
                checkpoints,
            },
        )
    }
}

pub(super) fn checkpoint(
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
                    fence: queue::fence(device, lane.ordinal)
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
