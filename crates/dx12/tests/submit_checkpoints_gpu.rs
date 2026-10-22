//! Requested intermediate checkpoints cover each lane frontier.
#![cfg(all(windows, feature = "hardware-tests"))]
use fluxel_rhi_core::api::*;
use fluxel_rhi_dx12::Dx12Instance;
use std::{future::Future, sync::Arc};
fn block<F: Future>(f: F) -> F::Output {
    let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
    let mut f = std::pin::pin!(f);
    loop {
        match f.as_mut().poll(&mut cx) {
            std::task::Poll::Ready(v) => return v,
            std::task::Poll::Pending => std::thread::sleep(std::time::Duration::from_millis(1)),
        }
    }
}
#[test]
fn checkpoints_cover_each_ordered_cross_lane_frontier() {
    let d = block(Dx12Instance::new().unwrap().open_default()).unwrap();
    let direct = d
        .lanes()
        .iter()
        .find(|x| x.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let compute = d
        .lanes()
        .iter()
        .find(|x| x.class == LaneClass::Compute)
        .unwrap()
        .id;
    // SAFETY: the plan describes one-shot commands recorded on these device lanes.
    let id = PlanId {
        device: d.id(),
        ordinal: 99700,
    };
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            id,
            Arc::from([
                StepDesc {
                    lane: direct,
                    kind: StepKind::Commands,
                    waits: vec![],
                },
                StepDesc {
                    lane: compute,
                    kind: StepKind::Commands,
                    waits: vec![],
                },
                StepDesc {
                    lane: direct,
                    kind: StepKind::Commands,
                    waits: vec![],
                },
            ]),
        )
    };
    let steps = [plan.step(0), plan.step(1), plan.step(2)];
    let work = [direct, compute, direct]
        .into_iter()
        .map(|lane| d.encoder(lane).unwrap().finish().unwrap())
        .collect::<Vec<_>>();
    // SAFETY: each recording is handed off once on the lane in the plan.
    let outcome = unsafe {
        d.queue().submit(SubmitPacket {
            plan,
            steps: steps
                .into_iter()
                .zip(work)
                .map(|(step, work)| BoundStep {
                    step,
                    payload: StepPayload::Commands(vec![work]),
                })
                .collect(),
            completion: CompletionRequest::Steps(Arc::from(steps)),
        })
    };
    let receipt = match outcome {
        SubmitOutcome::Accepted(v) => v,
        _ => panic!("submit rejected"),
    };
    assert_eq!(receipt.checkpoints.len(), 3);
    for (i, cp) in receipt.checkpoints.iter().enumerate() {
        block(d.wait_completion(&cp.point)).unwrap();
        assert!(
            cp.covers
                .lane_frontiers
                .contains(&(if i == 1 { compute } else { direct }, steps[i]))
        );
    }
}
