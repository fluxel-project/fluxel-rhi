//! A completed native prefix followed by an original-plan suffix.
#![cfg(feature = "hardware-tests")]
use fluxel_rhi_core::api::*;
use fluxel_rhi_vulkan::{Vulkan, VulkanInstance};
use std::{future::Future, sync::Arc};

fn block<F: Future>(future: F) -> F::Output {
    let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
    let mut future = std::pin::pin!(future);
    loop {
        match future.as_mut().poll(&mut cx) {
            std::task::Poll::Ready(value) => return value,
            std::task::Poll::Pending => std::thread::sleep(std::time::Duration::from_millis(1)),
        }
    }
}

fn exercise_resume(request_steps: bool) {
    let device = block(VulkanInstance::new().unwrap().open_default()).unwrap();
    let graphics = device
        .lanes()
        .iter()
        .find(|lane| lane.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let producer_lane = device
        .lanes()
        .iter()
        .find(|lane| lane.class == LaneClass::Compute)
        .map_or(graphics, |lane| lane.id);
    let buffer = device
        .create_buffer(&BufferDesc {
            label: None,
            size: 16,
            usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let prefix_buffer = device
        .create_buffer(&BufferDesc {
            label: None,
            size: 16,
            usage: BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let id = PlanId {
        device: device.id(),
        ordinal: 8921,
    };
    let first = StepId { plan: id, index: 0 };
    // SAFETY: the dependency is earlier and the completed prefix is observed before resume.
    let plan = unsafe {
        SubmissionPlan::<Vulkan>::from_compiled(
            id,
            Arc::from([
                StepDesc {
                    lane: producer_lane,
                    kind: StepKind::Commands,
                    waits: vec![],
                },
                StepDesc {
                    lane: graphics,
                    kind: StepKind::Commands,
                    waits: vec![StepWait {
                        previous: first,
                        destination: AccessScope {
                            stages: StageMask::COPY,
                            accesses: AccessMask::COPY_WRITE,
                        },
                    }],
                },
            ]),
        )
    };
    let mut producer = device.encoder(producer_lane).unwrap();
    producer.write_buffer(&prefix_buffer, 0, &[1; 16]).unwrap();
    let producer = producer.finish().unwrap();
    // Submit only the prefix through the backend SPI to construct a truthful
    // partial-acceptance fixture. Public Queue::submit requires a full packet;
    // native prefix execution and its completion are real, not replayed.
    let native = unsafe { device.native_ref() };
    let prefix = unsafe {
        <Vulkan as DeferredBackend>::submit(
            native,
            SubmitPacket {
                plan: plan.clone(),
                steps: vec![BoundStep {
                    step: first,
                    payload: StepPayload::Commands(vec![producer]),
                }],
                completion: CompletionRequest::End,
            },
        )
    };
    let prefix = match prefix {
        SubmitOutcome::Accepted(receipt) => receipt,
        _ => panic!("prefix not accepted"),
    };
    assert_eq!(prefix.accepted, vec![first]);
    for checkpoint in &prefix.checkpoints {
        block(device.wait_completion(&checkpoint.point)).unwrap();
    }
    let mut consumer = device.encoder(graphics).unwrap();
    let expected = [7u8; 16];
    consumer.write_buffer(&buffer, 0, &expected).unwrap();
    let ticket = consumer.readback_buffer(&buffer, 0..16).unwrap();
    let work = consumer.finish().unwrap();
    let second = plan.step(1);
    let completion = if request_steps {
        CompletionRequest::Steps(Arc::from([first, second]))
    } else {
        CompletionRequest::End
    };
    let tail = DeferredTail {
        plan,
        next_step: 1,
        pending: vec![BoundStep {
            step: second,
            payload: StepPayload::Commands(vec![work]),
        }],
        accepted_frontier: prefix.checkpoints,
        completion,
    };
    // SAFETY: the test waited for every accepted-frontier checkpoint above.
    let receipt = match unsafe { device.queue().resume(tail) } {
        SubmitOutcome::Accepted(receipt) => receipt,
        _ => panic!("resume failed"),
    };
    assert_eq!(receipt.accepted, vec![second]);
    if request_steps {
        assert_eq!(receipt.checkpoints.len(), 2);
    }
    let checkpoint = receipt.checkpoints.last().unwrap();
    block(device.wait_completion(&checkpoint.point)).unwrap();
    let bytes = block(unsafe { device.read_after(ticket, &checkpoint.point) })
        .unwrap()
        .bytes;
    assert_eq!(bytes, expected);
    assert!(
        checkpoint
            .covers
            .lane_frontiers
            .iter()
            .any(|&(_, step)| step == second)
    );
    if producer_lane != graphics {
        assert!(
            checkpoint
                .covers
                .lane_frontiers
                .iter()
                .any(|&(_, step)| step == first)
        );
    }
}

#[test]
fn resume_completed_prefix_with_end_checkpoint() {
    exercise_resume(false);
}

#[test]
fn resume_completed_prefix_with_step_checkpoints() {
    exercise_resume(true);
}
