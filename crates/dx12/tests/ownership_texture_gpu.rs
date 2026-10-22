//! Cross-queue GPU signal/wait and host completion evidence.
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
fn ownership_direct_compute_direct_texture() {
    let d = block(Dx12Instance::new().unwrap().open_default()).unwrap();
    let direct = d
        .lanes()
        .iter()
        .find(|l| l.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let compute = d
        .lanes()
        .iter()
        .find(|l| l.class == LaneClass::Compute)
        .unwrap()
        .id;
    let domain = d.multi_queue().unwrap();
    let texture = d
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [4, 4, 1],
            array_layers: 1,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::COPY_SRC | TextureUsage::COPY_DST,
            view_formats: vec![],
        })
        .unwrap();
    let data: Vec<u8> = (0..64).map(|x| x as u8).collect();
    let region = || TextureRegion {
        texture: &texture,
        mip: 0,
        origin: [0, 0, 0],
        extent: [4, 4, 1],
        aspect: TextureAspects::COLOR,
    };
    let empty = TextureUses {
        stages: StageMask::EMPTY,
        accesses: AccessMask::EMPTY,
    };
    let read = TextureUses {
        stages: StageMask::COPY,
        accesses: AccessMask::COPY_READ,
    };
    let handoff = |from, to, ownership| {
        ResourceTransition::Texture(TextureBarrier {
            target: Subresources {
                texture: &texture,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..1,
            },
            usage: StateTransition { from, to },
            layout: Some((ImageLayout::Common, ImageLayout::CopySrc)),
            ownership: Some(ownership),
        })
    };
    let mut producer = d.encoder(direct).unwrap();
    producer
        .write_texture(
            region(),
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: 16,
                rows_per_image: 4,
            },
            &data,
        )
        .unwrap();
    producer
        .transition_resources(&[handoff(
            empty,
            read,
            OwnershipTransfer::Release {
                from: direct,
                to: compute,
            },
        )])
        .unwrap();
    let mut consumer = d.encoder(compute).unwrap();
    consumer
        .transition_resources(&[handoff(
            empty,
            read,
            OwnershipTransfer::Acquire {
                from: direct,
                to: compute,
            },
        )])
        .unwrap();
    let compute_readback = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 1024,
            usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
            memory: MemoryClass::Readback,
        })
        .unwrap();
    consumer
        .texture_to_buffer(
            region(),
            &compute_readback,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: 256,
                rows_per_image: 4,
            },
        )
        .unwrap();
    let compute_ticket = consumer
        .readback_buffer(&compute_readback, 0..1024)
        .unwrap();
    let return_handoff = |ownership| {
        ResourceTransition::Texture(TextureBarrier {
            target: Subresources {
                texture: &texture,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..1,
            },
            usage: StateTransition {
                from: read,
                to: read,
            },
            layout: Some((ImageLayout::CopySrc, ImageLayout::CopySrc)),
            ownership: Some(ownership),
        })
    };
    consumer
        .transition_resources(&[return_handoff(OwnershipTransfer::Release {
            from: compute,
            to: direct,
        })])
        .unwrap();
    let dest = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 1024,
            usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
            memory: MemoryClass::Readback,
        })
        .unwrap();
    let mut final_consumer = d.encoder(direct).unwrap();
    final_consumer
        .transition_resources(&[return_handoff(OwnershipTransfer::Acquire {
            from: compute,
            to: direct,
        })])
        .unwrap();
    final_consumer
        .texture_to_buffer(
            region(),
            &dest,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: 256,
                rows_per_image: 4,
            },
        )
        .unwrap();
    let ticket = final_consumer.readback_buffer(&dest, 0..1024).unwrap();
    let mkplan = |ordinal, lane| unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: d.id(),
                ordinal,
            },
            Arc::from([StepDesc {
                lane,
                kind: StepKind::Commands,
                waits: vec![],
            }]),
        )
    };
    let pp = mkplan(98911, direct);
    let cp = mkplan(98912, compute);
    let fp = mkplan(98913, direct);
    let ps = pp.step(0);
    let cs = cp.step(0);
    let fs = fp.step(0);
    let producer_packet = MultiQueuePacket {
        work: SubmitPacket {
            plan: pp,
            steps: vec![BoundStep {
                step: ps,
                payload: StepPayload::Commands(vec![producer.finish().unwrap()]),
            }],
            completion: CompletionRequest::None,
        },
        signals: vec![GpuSignalRequest {
            producer: ps,
            consumer: cs,
            consumer_lane: compute,
        }],
        waits: vec![],
    };
    // SAFETY: packets use distinct plans and the token is consumed by its designated consumer.
    let signal = match unsafe { domain.submit(producer_packet) } {
        MultiQueueSubmitOutcome::Accepted(r) => r.signals.into_iter().next().unwrap(),
        _ => panic!("producer rejected"),
    };
    let consumer_packet = MultiQueuePacket {
        work: SubmitPacket {
            plan: cp,
            steps: vec![BoundStep {
                step: cs,
                payload: StepPayload::Commands(vec![consumer.finish().unwrap()]),
            }],
            completion: CompletionRequest::None,
        },
        signals: vec![GpuSignalRequest {
            producer: cs,
            consumer: fs,
            consumer_lane: direct,
        }],
        waits: vec![GpuWait {
            signal,
            destination: AccessScope {
                stages: StageMask::COPY,
                accesses: AccessMask::COPY_READ,
            },
        }],
    };
    let signal = match unsafe { domain.submit(consumer_packet) } {
        MultiQueueSubmitOutcome::Accepted(r) => r.signals.into_iter().next().unwrap(),
        _ => panic!("direct consumer rejected"),
    };
    let final_packet = MultiQueuePacket {
        work: SubmitPacket {
            plan: fp,
            steps: vec![BoundStep {
                step: fs,
                payload: StepPayload::Commands(vec![final_consumer.finish().unwrap()]),
            }],
            completion: CompletionRequest::End,
        },
        signals: vec![],
        waits: vec![GpuWait {
            signal,
            destination: AccessScope {
                stages: StageMask::COPY,
                accesses: AccessMask::COPY_READ,
            },
        }],
    };
    let point = match unsafe { domain.submit(final_packet) } {
        MultiQueueSubmitOutcome::Accepted(r) => {
            r.work.checkpoints.into_iter().next().unwrap().point
        }
        _ => panic!("final consumer rejected"),
    };
    block(d.wait_completion(&point)).unwrap();
    let compute_bytes = unsafe { block(d.read_after(compute_ticket, &point)).unwrap().bytes };
    for y in 0..4 {
        assert_eq!(
            &compute_bytes[y * 256..y * 256 + 16],
            &data[y * 16..y * 16 + 16]
        );
    }
    let bytes = unsafe { block(d.read_after(ticket, &point)).unwrap().bytes };
    for y in 0..4 {
        assert_eq!(&bytes[y * 256..y * 256 + 16], &data[y * 16..y * 16 + 16]);
    }
}
