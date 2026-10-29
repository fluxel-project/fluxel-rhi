//! Write tightly packed CPU depth/stencil rows and read them back.
#![cfg(all(windows, feature = "hardware-tests"))]
use fluxel_rhi_core::api::*;
use fluxel_rhi_dx12::Dx12Instance;
use std::sync::Arc;
#[path = "support/mod.rs"]
mod support;
use support::block;
#[test]
fn depth_stencil_plane_buffer_upload() {
    for (index, (format, aspect)) in [
        (TextureFormat::Depth16Unorm, TextureAspects::DEPTH),
        (TextureFormat::Depth32Float, TextureAspects::DEPTH),
        (TextureFormat::Depth32FloatStencil8, TextureAspects::DEPTH),
        (TextureFormat::Depth24PlusStencil8, TextureAspects::STENCIL),
        (TextureFormat::Depth32FloatStencil8, TextureAspects::STENCIL),
    ]
    .into_iter()
    .enumerate()
    {
        check(format, aspect, index as u64);
    }
}
fn check(format: TextureFormat, aspect: TextureAspects, index: u64) {
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
            array_layers: 1,
            format,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::COPY_DST | TextureUsage::COPY_SRC,
            view_formats: vec![],
        })
        .unwrap();
    let mut e = d.encoder(lane).unwrap();
    let bytes_per_texel = if aspect == TextureAspects::STENCIL {
        1
    } else if format == TextureFormat::Depth16Unorm {
        2
    } else {
        4
    };
    let mut data = vec![0u8; 8 * 8 * bytes_per_texel];
    for y in 0..8 {
        for x in 0..8 {
            let p = (y * 8 + x) * bytes_per_texel;
            if aspect == TextureAspects::STENCIL {
                data[p] = (x + y * 8 + 1) as u8;
            } else if bytes_per_texel == 2 {
                data[p..p + 2].copy_from_slice(&((x + y * 8) as u16 * 997).to_le_bytes());
            } else {
                data[p..p + 4].copy_from_slice(&(((x + y * 8) as f32 / 64.0).to_le_bytes()));
            }
        }
    }
    e.write_texture(
        TextureRegion {
            texture: &tex,
            mip: 0,
            origin: [0, 0, 0],
            extent: [8, 8, 1],
            aspect,
        },
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: (8 * bytes_per_texel) as u32,
            rows_per_image: 8,
        },
        &data,
    )
    .unwrap();
    let output = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 4096,
            usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
            memory: MemoryClass::Readback,
        })
        .unwrap();
    e.transition_resources(&[ResourceTransition::Texture(TextureBarrier {
        target: Subresources {
            texture: &tex,
            aspects: aspect,
            mips: 0..1,
            layers: 0..1,
        },
        usage: StateTransition {
            from: TextureUses {
                stages: StageMask::FRAGMENT,
                accesses: AccessMask::DEPTH_WRITE,
            },
            to: TextureUses {
                stages: StageMask::COPY,
                accesses: AccessMask::COPY_READ,
            },
        },
        layout: Some((ImageLayout::DepthStencilAttachment, ImageLayout::CopySrc)),
        ownership: None,
    })])
    .unwrap();
    e.texture_to_buffer(
        TextureRegion {
            texture: &tex,
            mip: 0,
            origin: [0, 0, 0],
            extent: [8, 8, 1],
            aspect,
        },
        &output,
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: 512,
            rows_per_image: 8,
        },
    )
    .unwrap();
    let sample_ticket = e.readback_buffer(&output, 0..4096).unwrap();
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: d.id(),
                ordinal: 99530 + index as u64,
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
    let sample = unsafe { block(d.read_after(sample_ticket, &point)).unwrap().bytes };
    for y in 0..8 {
        for x in 0..8 {
            let source = (y * 8 + x) * bytes_per_texel;
            let target = y * 512 + x * bytes_per_texel;
            assert_eq!(
                &sample[target..target + bytes_per_texel],
                &data[source..source + bytes_per_texel],
                "{format:?} ({x},{y})"
            );
        }
    }
}
