//! Upload page ownership, reuse and GPU readback.
#![cfg(all(windows, feature = "hardware-tests"))]
use fluxel_rhi_core::api::*;
use fluxel_rhi_dx12::Dx12Instance;
use std::{future::Future, sync::Arc};

fn block<F: Future>(f: F) -> F::Output {
    let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
    let mut f = std::pin::pin!(f);
    loop {
        if let std::task::Poll::Ready(v) = f.as_mut().poll(&mut cx) {
            return v;
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
}

#[test]
fn pages_are_exclusive_until_retired_and_abandoned_pages_are_reused() {
    let d = block(Dx12Instance::new().unwrap().open_default()).unwrap();
    let lane = d
        .lanes()
        .iter()
        .find(|l| l.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let size = 1024 * 1024 + 19;
    let output = d
        .create_buffer(&BufferDesc {
            label: None,
            size: size as u64 + 64,
            usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let before = fluxel_rhi_dx12::diagnostics::snapshot();
    let mut invalid = d.encoder(lane).unwrap();
    assert!(matches!(
        invalid.write_buffer(&output, u64::MAX, &[1, 2]),
        Err(Error::InvalidRange)
    ));
    drop(invalid);
    assert_eq!(
        fluxel_rhi_dx12::diagnostics::snapshot().upload_resources,
        before.upload_resources
    );
    let mut abandoned = d.encoder(lane).unwrap();
    abandoned.write_buffer(&output, 0, &[7; 3]).unwrap();
    drop(abandoned);
    let after_drop = fluxel_rhi_dx12::diagnostics::snapshot();
    assert_eq!(after_drop.upload_resources - before.upload_resources, 1);
    let mut first = d.encoder(lane).unwrap();
    first.write_buffer(&output, 0, &[11; 3]).unwrap();
    first.write_buffer(&output, 3, &[12; 5]).unwrap();
    let after_reuse = fluxel_rhi_dx12::diagnostics::snapshot();
    assert_eq!(after_reuse.upload_resources, after_drop.upload_resources);
    let first = first.finish().unwrap();
    let first_plan = unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: d.id(),
                ordinal: 991_240,
            },
            Arc::from([StepDesc {
                lane,
                kind: StepKind::Commands,
                waits: vec![],
            }]),
        )
    };
    let first_outcome = unsafe {
        d.queue().submit(SubmitPacket {
            steps: vec![BoundStep {
                step: first_plan.step(0),
                payload: StepPayload::Commands(vec![first]),
            }],
            plan: first_plan,
            completion: CompletionRequest::End,
        })
    };
    let _first_point = match first_outcome {
        SubmitOutcome::Accepted(r) => r.checkpoints.into_iter().next().unwrap().point,
        _ => panic!("first submission rejected"),
    };
    let mut second = d.encoder(lane).unwrap();
    second.write_buffer(&output, 8, &[13; 3]).unwrap();
    let after_second = fluxel_rhi_dx12::diagnostics::snapshot();
    assert_eq!(
        after_second.upload_resources - after_reuse.upload_resources,
        1
    );
    let large = vec![42; size];
    second.write_buffer(&output, 32, &large).unwrap();
    let after_large = fluxel_rhi_dx12::diagnostics::snapshot();
    assert_eq!(
        after_large.upload_resources - after_second.upload_resources,
        1
    );
    let ticket = second
        .readback_buffer(&output, 0..(size as u64 + 32))
        .unwrap();
    let second = second.finish().unwrap();
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: d.id(),
                ordinal: 991_241,
            },
            Arc::from([StepDesc {
                lane,
                kind: StepKind::Commands,
                waits: vec![],
            }]),
        )
    };
    let outcome = unsafe {
        d.queue().submit(SubmitPacket {
            steps: vec![BoundStep {
                step: plan.step(0),
                payload: StepPayload::Commands(vec![second]),
            }],
            plan,
            completion: CompletionRequest::End,
        })
    };
    let point = match outcome {
        SubmitOutcome::Accepted(r) => r.checkpoints.into_iter().next().unwrap().point,
        _ => panic!("submission rejected"),
    };
    block(d.wait_completion(&point)).unwrap();
    let bytes = unsafe { block(d.read_after(ticket, &point)).unwrap().bytes };
    assert_eq!(&bytes[..11], &[11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13]);
    assert_eq!(&bytes[32..], large.as_slice());

    // A fresh device avoids counting pages already retained by earlier work.
    let d = block(Dx12Instance::new().unwrap().open_default()).unwrap();
    let lane = d
        .lanes()
        .iter()
        .find(|l| l.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let output = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 4,
            usage: BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let before = fluxel_rhi_dx12::diagnostics::snapshot();
    let mut encoders = Vec::new();
    for _ in 0..500 {
        let mut encoder = d.encoder(lane).unwrap();
        encoder.write_buffer(&output, 0, &[1; 4]).unwrap();
        encoders.push(encoder);
    }
    let allocated = fluxel_rhi_dx12::diagnostics::snapshot();
    assert_eq!(allocated.upload_resources - before.upload_resources, 500);
    assert_eq!(allocated.upload_maps - before.upload_maps, 500);
    assert_eq!(
        allocated.upload_capacity_bytes - before.upload_capacity_bytes,
        500 * 65536
    );
    drop(encoders);
    let mut encoders = Vec::new();
    for _ in 0..129 {
        let mut encoder = d.encoder(lane).unwrap();
        encoder.write_buffer(&output, 0, &[2; 4]).unwrap();
        encoders.push(encoder);
    }
    let reused = fluxel_rhi_dx12::diagnostics::snapshot();
    assert_eq!(reused.upload_resources - allocated.upload_resources, 1);
    assert_eq!(
        reused.upload_capacity_bytes - allocated.upload_capacity_bytes,
        65536
    );
}
