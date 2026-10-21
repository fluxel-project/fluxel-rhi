//! Single-lane example submission and explicit fail-stop host policy.
//!
//! The raw outcome survives until actual GPU drain. Scenes keep their resources;
//! this module neither retires renderer resources nor retries an accepted prefix.
use super::{block_on, lifecycle::unsettled};
use fluxel_rhi::api::*;
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

static NEXT_PLAN: AtomicU64 = AtomicU64::new(1);

/// An issued frame with its complete native acceptance result still attached.
/// Dropping/cancelling potentially accepted work before drain is fail-stop.
#[must_use]
pub struct FrameSubmission<B: DeferredBackend> {
    device: Device<B>,
    outcome: Option<SubmitOutcome<B>>,
    plan: PlanId,
    lane: LaneId,
}

/// A frame whose potentially issued work has been observed idle.
/// Hosts can inspect every acceptance variant before choosing their policy.
pub struct FrameCompletion<B: DeferredBackend> {
    pub outcome: SubmitOutcome<B>,
    plan: PlanId,
    lane: LaneId,
}
impl<B: DeferredBackend> FrameSubmission<B> {
    /// Native acceptance evidence, before applying a host recovery policy.
    pub fn outcome(&self) -> &SubmitOutcome<B> {
        self.outcome.as_ref().expect("pending submission")
    }
    /// Retains the outcome, native dependencies, and device throughout the wait.
    /// This deliberately serial example harness drains even a failed present;
    /// accepting commands and accepting presentation are separate facts.
    pub async fn complete(mut self) -> FrameCompletion<B> {
        if !matches!(self.outcome(), SubmitOutcome::Rejected { .. }) {
            if let Err(error) = self.device.wait_idle().await {
                unsettled(&format!("frame drain failed: {error}"));
            }
        }
        FrameCompletion {
            outcome: self.outcome.take().expect("pending submission"),
            plan: self.plan,
            lane: self.lane,
        }
    }
}
impl<B: DeferredBackend> Drop for FrameSubmission<B> {
    fn drop(&mut self) {
        if self
            .outcome
            .as_ref()
            .is_some_and(|outcome| !matches!(outcome, SubmitOutcome::Rejected { .. }))
        {
            unsettled("submission observation cancelled before GPU drain");
        }
    }
}
impl<B: DeferredBackend> FrameCompletion<B> {
    /// The examples terminate a failed frame instead of retrying its tail or
    /// publishing speculative resource state. Inspect `outcome` for another
    /// host policy; draining has already made dropping its native payload safe.
    pub fn into_presented(self) -> Result<SubmissionReceipt<B>> {
        match self.outcome {
            SubmitOutcome::Accepted(receipt) => {
                let commands = StepId {
                    plan: self.plan,
                    index: 0,
                };
                let present = StepId {
                    plan: self.plan,
                    index: 1,
                };
                if receipt.device != self.plan.device
                    || receipt.accepted != [commands, present]
                    || receipt.presents.len() != 1
                    || !receipt.presents[0].accepted
                    || receipt.presents[0].step != Some(present)
                    || receipt.checkpoints.len() != 1
                    || receipt.checkpoints[0].point.device() != self.plan.device
                    || receipt.checkpoints[0].covers.lane_frontiers.len() != 1
                {
                    return Err(Error::InvalidState);
                }
                // GPU completion must cover Commands. Presentation acceptance
                // is separate: Vulkan's fence precedes queue-present and its
                // truthful frontier can remain at Commands, while other native
                // queues include Present in the same completion frontier.
                let (lane, frontier) = receipt.checkpoints[0].covers.lane_frontiers[0];
                if lane != self.lane || (frontier != commands && frontier != present) {
                    return Err(Error::InvalidState);
                }
                Ok(receipt)
            }
            SubmitOutcome::Rejected { error, .. }
            | SubmitOutcome::PartiallyAccepted { error, .. }
            | SubmitOutcome::Indeterminate { error, .. } => Err(error),
        }
    }
}

/// Issues a fixed Commands -> Present frame, retaining the structured result.
/// The scene must retain all recording dependencies until `complete` finishes.
pub(super) fn frame_plan<B: Backend>(device: DeviceId, lane: LaneId) -> Result<SubmissionPlan<B>> {
    let ordinal = NEXT_PLAN
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
            value.checked_add(1)
        })
        .map_err(|_| Error::OutOfMemory)?;
    // SAFETY: this harness selects the actual graphics lane and explicitly
    // supplies the command-to-present dependency. Resource barriers stay in
    // the scene, and no graph or cross-lane ordering is inferred here.
    let id = PlanId { device, ordinal };
    Ok(unsafe {
        SubmissionPlan::from_compiled(
            id,
            Arc::from([
                StepDesc {
                    lane,
                    kind: StepKind::Commands,
                    waits: Vec::new(),
                },
                StepDesc {
                    lane,
                    kind: StepKind::Present,
                    waits: vec![StepWait {
                        previous: StepId { plan: id, index: 0 },
                        destination: AccessScope {
                            stages: StageMask::ALL_COMMANDS,
                            accesses: AccessMask::ALL,
                        },
                    }],
                },
            ]),
        )
    })
}

pub fn submit_frame<B: DeferredBackend + ExecutionBackend<Mode = DeferredExecution>>(
    device: &Device<B>,
    lane: LaneId,
    work: RecordedWork<B>,
    image: SurfaceImage<B>,
) -> Result<FrameSubmission<B>> {
    if lane.device != device.id() || image.device() != device.id() {
        return Err(Error::WrongDevice);
    }
    let plan = frame_plan::<B>(device.id(), lane)?;
    let id = plan.id();
    let command = plan.step(0);
    let present = plan.step(1);
    // SAFETY: scene owners remain alive throughout the following explicit
    // drain. Failed/uncertain acceptance cannot escape without that proof;
    // cancellation or a failed drain terminates before caller resources drop.
    let mut execution = unsafe { device.begin_execution(plan, CompletionRequest::End)? };
    if let Err(failure) = execution.recorded_commands(command, vec![work]) {
        return Err(failure.error);
    }
    if let Err(failure) = execution.present(present, PresentBinding::new(image)) {
        return Err(failure.error);
    }
    // SAFETY: ownership and lifetime obligations are identical to direct submit.
    let outcome = match unsafe { execution.finish() } {
        DeferredExecutionOutcome::Submitted(outcome) => outcome,
        DeferredExecutionOutcome::Unsubmitted { error, .. } => return Err(error),
    };
    Ok(FrameSubmission {
        device: device.clone(),
        outcome: Some(outcome),
        plan: id,
        lane,
    })
}

pub fn submit_and_present<B: DeferredBackend + ExecutionBackend<Mode = DeferredExecution>>(
    device: &Device<B>,
    lane: LaneId,
    work: RecordedWork<B>,
    image: SurfaceImage<B>,
) -> Result<()> {
    block_on(submit_and_present_async(device, lane, work, image))
}
pub async fn submit_and_present_async<
    B: DeferredBackend + ExecutionBackend<Mode = DeferredExecution>,
>(
    device: &Device<B>,
    lane: LaneId,
    work: RecordedWork<B>,
    image: SurfaceImage<B>,
) -> Result<()> {
    submit_frame(device, lane, work, image)?
        .complete()
        .await
        .into_presented()
        .map(|_| ())
}
pub fn submit_and_present_point<B: DeferredBackend + ExecutionBackend<Mode = DeferredExecution>>(
    device: &Device<B>,
    lane: LaneId,
    work: RecordedWork<B>,
    image: SurfaceImage<B>,
) -> Result<CompletionPoint<B>> {
    block_on(submit_and_present_point_async(device, lane, work, image))
}
pub async fn submit_and_present_point_async<
    B: DeferredBackend + ExecutionBackend<Mode = DeferredExecution>,
>(
    device: &Device<B>,
    lane: LaneId,
    work: RecordedWork<B>,
    image: SurfaceImage<B>,
) -> Result<CompletionPoint<B>> {
    let receipt = submit_frame(device, lane, work, image)?
        .complete()
        .await
        .into_presented()?;
    if receipt.checkpoints.len() != 1 {
        return Err(Error::InvalidState);
    }
    Ok(receipt
        .checkpoints
        .into_iter()
        .next()
        .expect("one checkpoint")
        .point)
}
