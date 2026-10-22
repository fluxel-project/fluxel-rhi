//! Color formats preserve bytes across array mip upload, copy and readback.
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
fn color_formats_array_mip_copy_readback() {
    let d = block(Dx12Instance::new().unwrap().open_default()).unwrap();
    let lane = d
        .lanes()
        .iter()
        .find(|v| v.class == LaneClass::Graphics)
        .unwrap()
        .id;
    for (format, pixel_bytes) in [
        (TextureFormat::R8Unorm, 1usize),
        (TextureFormat::R8Snorm, 1usize),
        (TextureFormat::R8Uint, 1usize),
        (TextureFormat::R8Sint, 1usize),
        (TextureFormat::Rg8Unorm, 2usize),
        (TextureFormat::Rg8Snorm, 2usize),
        (TextureFormat::Rg8Uint, 2usize),
        (TextureFormat::Rg8Sint, 2usize),
        (TextureFormat::R16Unorm, 2usize),
        (TextureFormat::R16Snorm, 2usize),
        (TextureFormat::R16Uint, 2usize),
        (TextureFormat::R16Sint, 2usize),
        (TextureFormat::R16Float, 2usize),
        (TextureFormat::Rgba8Unorm, 4usize),
        (TextureFormat::Rgba8UnormSrgb, 4usize),
        (TextureFormat::Rgba8Snorm, 4usize),
        (TextureFormat::Rgba8Uint, 4usize),
        (TextureFormat::Rgba8Sint, 4usize),
        (TextureFormat::Bgra8Unorm, 4usize),
        (TextureFormat::Bgra8UnormSrgb, 4usize),
        (TextureFormat::Rg16Unorm, 4usize),
        (TextureFormat::Rg16Snorm, 4usize),
        (TextureFormat::Rg16Uint, 4usize),
        (TextureFormat::Rg16Sint, 4usize),
        (TextureFormat::Rg16Float, 4usize),
        (TextureFormat::R32Uint, 4usize),
        (TextureFormat::R32Sint, 4usize),
        (TextureFormat::R32Float, 4usize),
        (TextureFormat::Rgb10a2Unorm, 4usize),
        (TextureFormat::Rgb10a2Uint, 4usize),
        (TextureFormat::Rg11b10Ufloat, 4usize),
        (TextureFormat::Rgb9e5Ufloat, 4usize),
        (TextureFormat::Rgba16Unorm, 8usize),
        (TextureFormat::Rgba16Snorm, 8usize),
        (TextureFormat::Rgba16Uint, 8usize),
        (TextureFormat::Rgba16Sint, 8usize),
        (TextureFormat::Rgba16Float, 8usize),
        (TextureFormat::Rg32Uint, 8usize),
        (TextureFormat::Rg32Sint, 8usize),
        (TextureFormat::Rg32Float, 8usize),
        (TextureFormat::Rgba32Uint, 16usize),
        (TextureFormat::Rgba32Sint, 16usize),
        (TextureFormat::Rgba32Float, 16usize),
    ] {
        let desc = TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [8, 8, 1],
            array_layers: 2,
            format,
            mip_levels: 2,
            samples: 1,
            usage: TextureUsage::COPY_SRC | TextureUsage::COPY_DST,
            view_formats: vec![],
        };
        if matches!(d.texture_support(&desc), Support::Unsupported(_)) {
            eprintln!("unsupported {format:?}");
            continue;
        }
        let tex = d.create_texture(&desc).unwrap();
        let dst = d
            .create_buffer(&BufferDesc {
                label: None,
                size: 1024,
                usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
                memory: MemoryClass::Readback,
            })
            .unwrap();
        let row = 4 * pixel_bytes;
        let mut data = vec![0u8; row * 4];
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
                bytes_per_row: row as u32,
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
            assert_eq!(
                &bytes[y * 256..y * 256 + row],
                &data[y * row..y * row + row],
                "{format:?}"
            );
        }
    }
}
