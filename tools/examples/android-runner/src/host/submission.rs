//! Ordered single-lane frame submission with a completion point for readback.
//! The caller retains scene resources and maps tickets only after this point completes.

use super::*;
use std::sync::atomic::{AtomicU64, Ordering};

pub(super) fn submit_frame<B: DeferredBackend>(
    device: &Device<B>,
    lane: LaneId,
    work: RecordedWork<B>,
    image: SurfaceImage<B>,
) -> Result<CompletionPoint<B>> {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    let ordinal = NEXT
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
            value.checked_add(1)
        })
        .map_err(|_| Error::OutOfMemory)?;
    // SAFETY: both steps use one graphics lane in command-then-present order.
    // The bound work and surface lease were created on this device.
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: device.id(),
                ordinal,
            },
            Arc::from([
                StepDesc {
                    lane,
                    kind: StepKind::Commands,
                    waits: vec![],
                },
                StepDesc {
                    lane,
                    kind: StepKind::Present,
                    waits: vec![],
                },
            ]),
        )
    };
    // SAFETY: the packet binds each step once; the caller keeps its scene alive
    // until the returned checkpoint, which covers both native commands and present.
    match unsafe {
        device.queue().submit(SubmitPacket {
            steps: vec![
                BoundStep {
                    step: plan.step(0),
                    payload: StepPayload::Commands(vec![work]),
                },
                BoundStep {
                    step: plan.step(1),
                    payload: StepPayload::Present(PresentBinding::new(image)),
                },
            ],
            plan,
            completion: CompletionRequest::End,
        })
    } {
        SubmitOutcome::Accepted(receipt) => {
            if receipt.accepted.len() != 2
                || receipt.presents.len() != 1
                || !receipt.presents[0].accepted
            {
                utils::block_on(device.wait_idle())?;
                return Err(Error::InvalidState);
            }
            receipt
                .checkpoints
                .into_iter()
                .next()
                .map(|checkpoint| checkpoint.point)
                .ok_or(Error::InvalidState)
        }
        SubmitOutcome::Rejected { error, .. } => Err(error),
        SubmitOutcome::PartiallyAccepted { error, .. }
        | SubmitOutcome::Indeterminate { error, .. } => {
            // A failure may still have executed a prefix; retain scene data until
            // that prefix has retired before reporting the error to the host.
            utils::block_on(device.wait_idle())?;
            Err(error)
        }
    }
}
