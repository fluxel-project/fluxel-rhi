//! BGRA texture writes preserve bytes across an array mip and GPU readback.
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
fn bgra_upload_array_mip_readback() {
    let d = block(Dx12Instance::new().unwrap().open_default()).unwrap();
    let lane = d
        .lanes()
        .iter()
        .find(|v| v.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let tex = d
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [8, 8, 1],
            array_layers: 2,
            format: TextureFormat::Bgra8Unorm,
            mip_levels: 2,
            samples: 1,
            usage: TextureUsage::COPY_SRC | TextureUsage::COPY_DST,
            view_formats: vec![],
        })
        .unwrap();
    let dst = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 1024,
            usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
            memory: MemoryClass::Readback,
        })
        .unwrap();
    let mut data = vec![0u8; 64];
    for (i, v) in data.iter_mut().enumerate() {
        *v = (i as u8).wrapping_mul(3).wrapping_add(1);
    }
    let region = || TextureRegion {
        texture: &tex,
        mip: 1,
        origin: [0, 0, 1],
        extent: [4, 4, 1],
        aspect: TextureAspects::COLOR,
    };
    let mut e = d.encoder(lane).unwrap();
    e.write_texture(
        region(),
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: 16,
            rows_per_image: 4,
        },
        &data,
    )
    .unwrap();
    e.transition_resources(&[ResourceTransition::Texture(TextureBarrier {
        target: Subresources {
            texture: &tex,
            aspects: TextureAspects::COLOR,
            mips: 1..2,
            layers: 1..2,
        },
        usage: StateTransition {
            from: TextureUses {
                stages: StageMask::EMPTY,
                accesses: AccessMask::EMPTY,
            },
            to: TextureUses {
                stages: StageMask::COPY,
                accesses: AccessMask::COPY_READ,
            },
        },
        layout: Some((ImageLayout::Common, ImageLayout::CopySrc)),
        ownership: None,
    })])
    .unwrap();
    let target = || TextureRegion {
        texture: &tex,
        mip: 1,
        origin: [0, 0, 0],
        extent: [4, 4, 1],
        aspect: TextureAspects::COLOR,
    };
    let target_barrier = |from, to, from_layout, to_layout| {
        ResourceTransition::Texture(TextureBarrier {
            target: Subresources {
                texture: &tex,
                aspects: TextureAspects::COLOR,
                mips: 1..2,
                layers: 0..1,
            },
            usage: StateTransition { from, to },
            layout: Some((from_layout, to_layout)),
            ownership: None,
        })
    };
    let empty = TextureUses {
        stages: StageMask::EMPTY,
        accesses: AccessMask::EMPTY,
    };
    let write = TextureUses {
        stages: StageMask::COPY,
        accesses: AccessMask::COPY_WRITE,
    };
    let read = TextureUses {
        stages: StageMask::COPY,
        accesses: AccessMask::COPY_READ,
    };
    e.transition_resources(&[target_barrier(
        empty,
        write,
        ImageLayout::Common,
        ImageLayout::CopyDst,
    )])
    .unwrap();
    e.copy_texture(region(), target()).unwrap();
    e.transition_resources(&[target_barrier(
        write,
        read,
        ImageLayout::CopyDst,
        ImageLayout::CopySrc,
    )])
    .unwrap();
    e.texture_to_buffer(
        target(),
        &dst,
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: 256,
            rows_per_image: 4,
        },
    )
    .unwrap();
    let ticket = e.readback_buffer(&dst, 0..1024).unwrap();
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: d.id(),
                ordinal: 99801,
            },
            Arc::from([StepDesc {
                lane,
                kind: StepKind::Commands,
                waits: vec![],
            }]),
        )
    };
    let step = plan.step(0);
    let outcome = unsafe {
        d.queue().submit(SubmitPacket {
            plan,
            steps: vec![BoundStep {
                step,
                payload: StepPayload::Commands(vec![e.finish().unwrap()]),
            }],
            completion: CompletionRequest::End,
        })
    };
    let point = match outcome {
        SubmitOutcome::Accepted(r) => r.checkpoints.into_iter().next().unwrap().point,
        _ => panic!("submission rejected"),
    };
    block(d.wait_completion(&point)).unwrap();
    let bytes = unsafe { block(d.read_after(ticket, &point)).unwrap().bytes };
    for y in 0..4 {
        assert_eq!(&bytes[y * 256..y * 256 + 16], &data[y * 16..y * 16 + 16]);
    }
}
