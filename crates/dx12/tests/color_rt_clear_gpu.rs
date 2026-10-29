//! Float render-target clear and readback.
#![cfg(all(windows, feature = "hardware-tests"))]
use fluxel_rhi_core::api::*;
use fluxel_rhi_dx12::Dx12Instance;
use std::sync::Arc;
#[path = "support/mod.rs"]
mod support;
use support::block;
#[test]
fn float_render_target_clear_readback() {
    let d = block(Dx12Instance::new().unwrap().open_default()).unwrap();
    let lane = d
        .lanes()
        .iter()
        .find(|v| v.class == LaneClass::Graphics)
        .unwrap()
        .id;
    for (format, expected) in [
        (TextureFormat::R8Unorm, vec![255]),
        (TextureFormat::R8Snorm, vec![127]),
        (TextureFormat::Rg8Unorm, vec![255, 0]),
        (TextureFormat::Rg8Snorm, vec![127, 0]),
        (TextureFormat::R16Unorm, vec![255, 255]),
        (TextureFormat::R16Snorm, vec![255, 127]),
        (TextureFormat::R16Float, vec![0, 60]),
        (TextureFormat::Rg16Unorm, vec![255, 255, 0, 0]),
        (TextureFormat::Rg16Snorm, vec![255, 127, 0, 0]),
        (TextureFormat::Rg16Float, vec![0, 60, 0, 0]),
        (TextureFormat::R32Float, 1.0f32.to_le_bytes().to_vec()),
        (TextureFormat::Rg32Float, vec![0, 0, 128, 63, 0, 0, 0, 0]),
        (TextureFormat::Rgba8Unorm, vec![255, 0, 0, 255]),
        (TextureFormat::Rgba8Snorm, vec![127, 0, 0, 127]),
        (TextureFormat::Bgra8Unorm, vec![0, 0, 255, 255]),
        (TextureFormat::Bgra8UnormSrgb, vec![0, 0, 255, 255]),
        (
            TextureFormat::Rgba16Unorm,
            vec![255, 255, 0, 0, 0, 0, 255, 255],
        ),
        (
            TextureFormat::Rgba16Snorm,
            vec![255, 127, 0, 0, 0, 0, 255, 127],
        ),
        (TextureFormat::Rgba16Float, vec![0, 60, 0, 0, 0, 0, 0, 60]),
        (
            TextureFormat::Rgba32Float,
            vec![0, 0, 128, 63, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 63],
        ),
        (TextureFormat::Rgb10a2Unorm, vec![255, 3, 0, 192]),
        (TextureFormat::Rgba8UnormSrgb, vec![255, 0, 0, 255]),
    ] {
        let desc = TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [4, 4, 1],
            array_layers: 1,
            format,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
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
        let intermediate = d
            .create_buffer(&BufferDesc {
                label: None,
                size: 1024,
                usage: BufferUsage::COPY_DST | BufferUsage::COPY_SRC,
                memory: MemoryClass::Device,
            })
            .unwrap();
        let mut e = d.encoder(lane).unwrap();
        e.transition_resources(&[ResourceTransition::Buffer(BufferBarrier {
            target: BufferSlice {
                buffer: &intermediate,
                range: 0..1024,
            },
            usage: StateTransition {
                from: BufferUses {
                    stages: StageMask::EMPTY,
                    accesses: AccessMask::EMPTY,
                },
                to: BufferUses {
                    stages: StageMask::COPY,
                    accesses: AccessMask::COPY_WRITE,
                },
            },
            ownership: None,
        })])
        .unwrap();
        let barrier = |from, to, before, after| {
            ResourceTransition::Texture(TextureBarrier {
                target: Subresources {
                    texture: &tex,
                    aspects: TextureAspects::COLOR,
                    mips: 0..1,
                    layers: 0..1,
                },
                usage: StateTransition { from, to },
                layout: Some((before, after)),
                ownership: None,
            })
        };
        let empty = TextureUses {
            stages: StageMask::EMPTY,
            accesses: AccessMask::EMPTY,
        };
        let rt = TextureUses {
            stages: StageMask::COLOR_OUTPUT,
            accesses: AccessMask::COLOR_WRITE,
        };
        let copy = TextureUses {
            stages: StageMask::COPY,
            accesses: AccessMask::COPY_READ,
        };
        e.transition_resources(&[barrier(
            empty,
            rt,
            ImageLayout::Common,
            ImageLayout::ColorAttachment,
        )])
        .unwrap();
        e.clear_texture(
            &tex,
            0..1,
            0..1,
            TextureClearValue::Color(ClearColor::Float([1.0, 0.0, 0.0, 1.0])),
        )
        .unwrap();
        e.transition_resources(&[barrier(
            rt,
            copy,
            ImageLayout::ColorAttachment,
            ImageLayout::CopySrc,
        )])
        .unwrap();
        e.texture_to_buffer(
            TextureRegion {
                texture: &tex,
                mip: 0,
                origin: [0, 0, 0],
                extent: [4, 4, 1],
                aspect: TextureAspects::COLOR,
            },
            &intermediate,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: 256,
                rows_per_image: 4,
            },
        )
        .unwrap();
        e.transition_resources(&[ResourceTransition::Buffer(BufferBarrier {
            target: BufferSlice {
                buffer: &intermediate,
                range: 0..1024,
            },
            usage: StateTransition {
                from: BufferUses {
                    stages: StageMask::COPY,
                    accesses: AccessMask::COPY_WRITE,
                },
                to: BufferUses {
                    stages: StageMask::COPY,
                    accesses: AccessMask::COPY_READ,
                },
            },
            ownership: None,
        })])
        .unwrap();
        e.copy_buffer(&intermediate, 0, &dst, 0, 1024).unwrap();
        let ticket = e.readback_buffer(&dst, 0..1024).unwrap();
        let plan = unsafe {
            SubmissionPlan::from_compiled(
                PlanId {
                    device: d.id(),
                    ordinal: 99802,
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
            for x in 0..4 {
                assert_eq!(
                    &bytes[y * 256 + x * expected.len()..y * 256 + (x + 1) * expected.len()],
                    &expected,
                    "{format:?}"
                );
            }
        }
    }
}
