//! Read the defined depth and stencil plane byte representations.
#![cfg(all(windows, feature = "hardware-tests"))]
use fluxel_rhi_core::api::*;
use fluxel_rhi_dx12::Dx12Instance;
use std::sync::Arc;
#[path = "support/mod.rs"]
mod support;
use support::block;
#[test]
fn depth_stencil_plane_buffer_readback() {
    for (index, format) in [
        TextureFormat::Depth24PlusStencil8,
        TextureFormat::Depth32FloatStencil8,
    ]
    .into_iter()
    .enumerate()
    {
        check(format, TextureAspects::STENCIL, index as u64);
        if format == TextureFormat::Depth32FloatStencil8 {
            check(format, TextureAspects::DEPTH, index as u64 + 2);
        }
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
            usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
            view_formats: vec![],
        })
        .unwrap();
    let mut e = d.encoder(lane).unwrap();
    let attachment_view = d
        .create_view(
            &tex,
            &TextureViewDesc {
                format,
                dimension: TextureViewDimension::D2,
                aspects: TextureAspects::DEPTH | TextureAspects::STENCIL,
                mips: 0..1,
                layers: 0..1,
            },
        )
        .unwrap();
    e.transition_resources(&[ResourceTransition::Texture(TextureBarrier {
        target: Subresources {
            texture: &tex,
            aspects: TextureAspects::DEPTH | TextureAspects::STENCIL,
            mips: 0..1,
            layers: 0..1,
        },
        usage: StateTransition {
            from: TextureUses {
                stages: StageMask::EMPTY,
                accesses: AccessMask::EMPTY,
            },
            to: TextureUses {
                stages: StageMask::FRAGMENT,
                accesses: AccessMask::DEPTH_WRITE,
            },
        },
        layout: Some((ImageLayout::Undefined, ImageLayout::DepthStencilAttachment)),
        ownership: None,
    })])
    .unwrap();
    {
        let pass = e
            .begin_raster(&RasterPassDesc {
                label: None,
                colors: &[],
                depth_stencil: Some(DepthStencilAttachment {
                    view: AttachmentView::Texture(&attachment_view),
                    depth: Some((LoadOp::Clear(0.25), StoreOp::Store)),
                    stencil: Some((LoadOp::Clear(37), StoreOp::Store)),
                    depth_read_only: false,
                    stencil_read_only: false,
                }),
                render_area: Rect {
                    x: 0,
                    y: 0,
                    width: 8,
                    height: 8,
                },
                multiview_mask: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                secondary_inheritance: None,
            })
            .unwrap();
        pass.finish().unwrap();
    }
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
            if aspect == TextureAspects::STENCIL {
                assert_eq!(sample[y * 512 + x], 37, "{format:?} ({x},{y})");
            } else {
                let p = y * 512 + x * 4;
                assert_eq!(
                    f32::from_le_bytes(sample[p..p + 4].try_into().unwrap()),
                    0.25,
                    "{format:?} ({x},{y})"
                );
            }
        }
    }
}
