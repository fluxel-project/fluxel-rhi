//! BC texture uploads and readback preserve encoded blocks.
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
fn bc_non_aligned_top_level() {
    use TextureFormat::*;
    let formats = [
        (Bc1RgbaUnorm, 8),
        (Bc1RgbaUnormSrgb, 8),
        (Bc2RgbaUnorm, 16),
        (Bc2RgbaUnormSrgb, 16),
        (Bc3RgbaUnorm, 16),
        (Bc3RgbaUnormSrgb, 16),
        (Bc4RUnorm, 8),
        (Bc4RSnorm, 8),
        (Bc5RgUnorm, 16),
        (Bc5RgSnorm, 16),
        (Bc6hRgbUfloat, 16),
        (Bc6hRgbFloat, 16),
        (Bc7RgbaUnorm, 16),
        (Bc7RgbaUnormSrgb, 16),
    ];
    let d = block(Dx12Instance::new().unwrap().open_default()).unwrap();
    let lane = d
        .lanes()
        .iter()
        .find(|v| v.class == LaneClass::Graphics)
        .unwrap()
        .id;
    for (index, (format, block_size)) in formats.into_iter().enumerate() {
        let desc = TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [7, 5, 1],
            array_layers: 2,
            format,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::COPY_SRC | TextureUsage::COPY_DST | TextureUsage::TEXTURE_BINDING,
            view_formats: vec![],
        };
        if matches!(d.texture_support(&desc), Support::Unsupported(_)) {
            eprintln!("unsupported unaligned {format:?}");
            continue;
        }
        let tex = d.create_texture(&desc).unwrap();
        let dst = d
            .create_buffer(&BufferDesc {
                label: None,
                size: 512,
                usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
                memory: MemoryClass::Readback,
            })
            .unwrap();
        let row = 2 * block_size;
        let data: Vec<_> = (0..row * 2)
            .map(|n| (n as u8).wrapping_mul(7).wrapping_add(3))
            .collect();
        let region = || TextureRegion {
            texture: &tex,
            mip: 0,
            origin: [0, 0, 1],
            extent: [7, 5, 1],
            aspect: TextureAspects::COLOR,
        };
        let mut e = d.encoder(lane).unwrap();
        e.write_texture(
            region(),
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: row as u32,
                rows_per_image: 2,
            },
            &data,
        )
        .unwrap();
        e.transition_resources(&[ResourceTransition::Texture(TextureBarrier {
            target: Subresources {
                texture: &tex,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
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
        e.texture_to_buffer(
            region(),
            &dst,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: 256,
                rows_per_image: 2,
            },
        )
        .unwrap();
        let ticket = e.readback_buffer(&dst, 0..512).unwrap();
        let source = d
            .create_buffer(&BufferDesc {
                label: None,
                size: 512,
                usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
                memory: MemoryClass::Device,
            })
            .unwrap();
        let mut staged = vec![0u8; 512];
        for y in 0..2 {
            staged[y * 256..y * 256 + row].copy_from_slice(&data[y * row..y * row + row]);
        }
        e.write_buffer(&source, 0, &staged).unwrap();
        e.transition_resources(&[
            ResourceTransition::Buffer(BufferBarrier {
                target: BufferSlice {
                    buffer: &source,
                    range: 0..512,
                },
                usage: StateTransition {
                    from: BufferUses {
                        stages: StageMask::EMPTY,
                        accesses: AccessMask::EMPTY,
                    },
                    to: BufferUses {
                        stages: StageMask::COPY,
                        accesses: AccessMask::COPY_READ,
                    },
                },
                ownership: None,
            }),
            ResourceTransition::Texture(TextureBarrier {
                target: Subresources {
                    texture: &tex,
                    aspects: TextureAspects::COLOR,
                    mips: 0..1,
                    layers: 1..2,
                },
                usage: StateTransition {
                    from: TextureUses {
                        stages: StageMask::COPY,
                        accesses: AccessMask::COPY_READ,
                    },
                    to: TextureUses {
                        stages: StageMask::COPY,
                        accesses: AccessMask::COPY_WRITE,
                    },
                },
                layout: Some((ImageLayout::CopySrc, ImageLayout::CopyDst)),
                ownership: None,
            }),
        ])
        .unwrap();
        e.buffer_to_texture(
            &source,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: 256,
                rows_per_image: 2,
            },
            region(),
        )
        .unwrap();
        e.transition_resources(&[ResourceTransition::Texture(TextureBarrier {
            target: Subresources {
                texture: &tex,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 1..2,
            },
            usage: StateTransition {
                from: TextureUses {
                    stages: StageMask::COPY,
                    accesses: AccessMask::COPY_WRITE,
                },
                to: TextureUses {
                    stages: StageMask::COPY,
                    accesses: AccessMask::COPY_READ,
                },
            },
            layout: Some((ImageLayout::CopyDst, ImageLayout::CopySrc)),
            ownership: None,
        })])
        .unwrap();
        let dst2 = d
            .create_buffer(&BufferDesc {
                label: None,
                size: 512,
                usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
                memory: MemoryClass::Readback,
            })
            .unwrap();
        e.texture_to_buffer(
            region(),
            &dst2,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: 256,
                rows_per_image: 2,
            },
        )
        .unwrap();
        let ticket2 = e.readback_buffer(&dst2, 0..512).unwrap();
        let copied = d.create_texture(&desc).unwrap();
        e.transition_resources(&[ResourceTransition::Texture(TextureBarrier {
            target: Subresources {
                texture: &copied,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 1..2,
            },
            usage: StateTransition {
                from: TextureUses {
                    stages: StageMask::EMPTY,
                    accesses: AccessMask::EMPTY,
                },
                to: TextureUses {
                    stages: StageMask::COPY,
                    accesses: AccessMask::COPY_WRITE,
                },
            },
            layout: Some((ImageLayout::Common, ImageLayout::CopyDst)),
            ownership: None,
        })])
        .unwrap();
        let copied_region = || TextureRegion {
            texture: &copied,
            mip: 0,
            origin: [0, 0, 1],
            extent: [7, 5, 1],
            aspect: TextureAspects::COLOR,
        };
        e.copy_texture(region(), copied_region()).unwrap();
        e.transition_resources(&[ResourceTransition::Texture(TextureBarrier {
            target: Subresources {
                texture: &copied,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 1..2,
            },
            usage: StateTransition {
                from: TextureUses {
                    stages: StageMask::COPY,
                    accesses: AccessMask::COPY_WRITE,
                },
                to: TextureUses {
                    stages: StageMask::COPY,
                    accesses: AccessMask::COPY_READ,
                },
            },
            layout: Some((ImageLayout::CopyDst, ImageLayout::CopySrc)),
            ownership: None,
        })])
        .unwrap();
        let dst3 = d
            .create_buffer(&BufferDesc {
                label: None,
                size: 512,
                usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
                memory: MemoryClass::Readback,
            })
            .unwrap();
        e.texture_to_buffer(
            copied_region(),
            &dst3,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: 256,
                rows_per_image: 2,
            },
        )
        .unwrap();
        let ticket3 = e.readback_buffer(&dst3, 0..512).unwrap();
        let plan = unsafe {
            SubmissionPlan::from_compiled(
                PlanId {
                    device: d.id(),
                    ordinal: 99950 + index as u64,
                },
                Arc::from([StepDesc {
                    lane,
                    kind: StepKind::Commands,
                    waits: vec![],
                }]),
            )
        };
        let step = plan.step(0);
        let result = unsafe {
            d.queue().submit(SubmitPacket {
                plan,
                steps: vec![BoundStep {
                    step,
                    payload: StepPayload::Commands(vec![e.finish().unwrap()]),
                }],
                completion: CompletionRequest::End,
            })
        };
        let point = match result {
            SubmitOutcome::Accepted(r) => r.checkpoints.into_iter().next().unwrap().point,
            _ => panic!("submit {format:?}"),
        };
        block(d.wait_completion(&point)).unwrap();
        let bytes = unsafe { block(d.read_after(ticket, &point)).unwrap().bytes };
        let bytes2 = unsafe { block(d.read_after(ticket2, &point)).unwrap().bytes };
        let bytes3 = unsafe { block(d.read_after(ticket3, &point)).unwrap().bytes };
        for y in 0..2 {
            assert_eq!(
                &bytes[y * 256..y * 256 + row],
                &data[y * row..y * row + row],
                "{format:?}"
            );
            assert_eq!(
                &bytes2[y * 256..y * 256 + row],
                &data[y * row..y * row + row],
                "buffer copy {format:?}"
            );
            assert_eq!(
                &bytes3[y * 256..y * 256 + row],
                &data[y * row..y * row + row],
                "texture copy {format:?}"
            );
        }
    }
}
