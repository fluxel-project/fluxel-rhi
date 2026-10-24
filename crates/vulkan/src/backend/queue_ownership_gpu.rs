//! Explicitly distinct-family ownership acceptance with repeated round trips.
use super::*;
use fluxel_rhi_core::api::*;
use std::{future::Future, sync::Arc};

fn block<F: Future>(future: F) -> F::Output {
    let mut future = std::pin::pin!(future);
    let mut context = std::task::Context::from_waker(std::task::Waker::noop());
    loop {
        match future.as_mut().poll(&mut context) {
            std::task::Poll::Ready(value) => return value,
            std::task::Poll::Pending => std::thread::sleep(std::time::Duration::from_millis(1)),
        }
    }
}

#[test]
#[ignore = "requires distinct native graphics/compute families; run explicitly with validation"]
fn exclusive_buffer_round_trip_requires_distinct_queue_families() {
    let device = block(VulkanInstance::new().unwrap().open_default()).unwrap();
    // SAFETY: immutable native metadata is borrowed from this live device.
    let native = unsafe { device.native_ref() };
    eprintln!(
        "ownership acceptance: graphics_family={}, compute_family={}",
        native.0.graphics_family, native.0.compute_family
    );
    assert!(
        native.0.compute_queue.is_some(),
        "an independent compute queue is required"
    );
    assert_ne!(
        native.0.graphics_family, native.0.compute_family,
        "same-family execution cannot satisfy this acceptance scenario"
    );
    let graphics = device
        .lanes()
        .iter()
        .find(|l| l.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let compute = device
        .lanes()
        .iter()
        .find(|l| l.class == LaneClass::Compute)
        .unwrap()
        .id;
    let make_buffer = |usage, memory| {
        device
            .create_buffer(&BufferDesc {
                label: None,
                size: 64,
                usage,
                memory,
            })
            .unwrap()
    };
    let buffer = make_buffer(
        BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
        MemoryClass::Device,
    );
    let download = make_buffer(
        BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
        MemoryClass::Readback,
    );
    let read = BufferUses {
        stages: StageMask::COPY,
        accesses: AccessMask::COPY_READ,
    };
    let write = BufferUses {
        stages: StageMask::COPY,
        accesses: AccessMask::COPY_WRITE,
    };
    let barrier = |from, to, ownership| {
        ResourceTransition::Buffer(BufferBarrier {
            target: BufferSlice {
                buffer: &buffer,
                range: 0..64,
            },
            usage: StateTransition { from, to },
            ownership: Some(ownership),
        })
    };
    for frame in 0..3u8 {
        let mut producer = device.encoder(compute).unwrap();
        if frame != 0 {
            producer
                .transition_resources(&[barrier(
                    read,
                    write,
                    OwnershipTransfer::Acquire {
                        from: graphics,
                        to: compute,
                    },
                )])
                .unwrap();
        }
        let expected: Vec<u8> = (0..64)
            .map(|i| frame.wrapping_mul(31).wrapping_add(i))
            .collect();
        producer.write_buffer(&buffer, 0, &expected).unwrap();
        producer
            .transition_resources(&[barrier(
                write,
                read,
                OwnershipTransfer::Release {
                    from: compute,
                    to: graphics,
                },
            )])
            .unwrap();
        let mut consumer = device.encoder(graphics).unwrap();
        consumer
            .transition_resources(&[barrier(
                write,
                read,
                OwnershipTransfer::Acquire {
                    from: compute,
                    to: graphics,
                },
            )])
            .unwrap();
        consumer.copy_buffer(&buffer, 0, &download, 0, 64).unwrap();
        let ticket = consumer.readback_buffer(&download, 0..64).unwrap();
        if frame != 2 {
            consumer
                .transition_resources(&[barrier(
                    read,
                    write,
                    OwnershipTransfer::Release {
                        from: graphics,
                        to: compute,
                    },
                )])
                .unwrap();
        }
        let id = PlanId {
            device: device.id(),
            ordinal: 230100 + u64::from(frame),
        };
        // SAFETY: both device-local steps have an explicit producer wait. The
        // previous End completion covers the return transfer before reuse.
        let plan = unsafe {
            SubmissionPlan::from_compiled(
                id,
                Arc::from([
                    StepDesc {
                        lane: compute,
                        kind: StepKind::Commands,
                        waits: vec![],
                    },
                    StepDesc {
                        lane: graphics,
                        kind: StepKind::Commands,
                        waits: vec![StepWait {
                            previous: StepId { plan: id, index: 0 },
                            destination: AccessScope {
                                stages: StageMask::COPY,
                                accesses: AccessMask::COPY_READ,
                            },
                        }],
                    },
                ]),
            )
        };
        // SAFETY: recordings and resources survive the covering completion.
        let receipt = match unsafe {
            device.queue().submit(SubmitPacket {
                steps: vec![
                    BoundStep {
                        step: plan.step(0),
                        payload: StepPayload::Commands(vec![producer.finish().unwrap()]),
                    },
                    BoundStep {
                        step: plan.step(1),
                        payload: StepPayload::Commands(vec![consumer.finish().unwrap()]),
                    },
                ],
                plan,
                completion: CompletionRequest::End,
            })
        } {
            SubmitOutcome::Accepted(receipt) => receipt,
            _ => panic!("distinct-family ownership packet rejected"),
        };
        let point = &receipt.checkpoints.first().expect("End checkpoint").point;
        block(device.wait_completion(point)).unwrap();
        // SAFETY: End covers the graphics readback and return ownership release.
        let actual = unsafe { block(device.read_after(ticket, point)) }
            .unwrap()
            .bytes;
        assert_eq!(actual, expected, "round-trip frame {frame}");
    }
}
