//! Real-driver proof that graphics-only work stays on the single queue path.

use super::multi_queue;
use crate::backend::VulkanInstance;
use fluxel_rhi_core::api::*;
use std::{future::Future, sync::Arc, thread, time::Duration};

fn block<F: Future>(future: F) -> F::Output {
    let mut context = std::task::Context::from_waker(std::task::Waker::noop());
    let mut future = std::pin::pin!(future);
    loop {
        match future.as_mut().poll(&mut context) {
            std::task::Poll::Ready(value) => return value,
            std::task::Poll::Pending => thread::sleep(Duration::from_millis(1)),
        }
    }
}

#[test]
fn graphics_only_packet_does_not_enter_multi_queue() {
    let device = block(VulkanInstance::new().unwrap().open_default()).unwrap();
    let graphics = device
        .lanes()
        .iter()
        .find(|lane| lane.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let encoder = device.encoder(graphics).unwrap();
    let work = encoder.finish().unwrap();
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: device.id(),
                ordinal: 74_101,
            },
            Arc::from([StepDesc {
                lane: graphics,
                kind: StepKind::Commands,
                waits: vec![],
            }]),
        )
    };
    multi_queue::reset_test_submit_count();
    // SAFETY: this one finished command buffer belongs to the plan's only lane.
    let receipt = unsafe {
        device.queue().submit(SubmitPacket {
            plan: plan.clone(),
            steps: vec![BoundStep {
                step: plan.step(0),
                payload: StepPayload::Commands(vec![work]),
            }],
            completion: CompletionRequest::End,
        })
    };
    let receipt = match receipt {
        SubmitOutcome::Accepted(receipt) => receipt,
        _ => panic!("graphics packet was not accepted"),
    };
    assert_eq!(multi_queue::test_submit_count(), 0);
    let point = &receipt.checkpoints[0].point;
    block(device.wait_completion(point)).unwrap();
}
