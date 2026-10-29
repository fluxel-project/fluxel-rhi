//! Active raster-pass rectangle clear GPU readback coverage.
#![cfg(all(windows, feature = "hardware-tests"))]

use fluxel_rhi_core::api::*;
use fluxel_rhi_dx12::Dx12Instance;
use std::sync::Arc;

#[path = "support/mod.rs"]
mod support;
use support::block;

#[test]
fn raster_rect_clear_changes_only_selected_color_rectangle() {
    let device = block(Dx12Instance::new().unwrap().open_default()).unwrap();
    assert!(device.capabilities().rect_attachment_clear);
    let lane = device
        .lanes()
        .iter()
        .find(|lane| lane.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let color_desc = TextureDesc {
        label: None,
        dimension: TextureDimension::D2,
        extent: [8, 8, 1],
        array_layers: 1,
        format: TextureFormat::Rgba8Unorm,
        mip_levels: 1,
        samples: 1,
        usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
        view_formats: vec![],
    };
    let first = device.create_texture(&color_desc).unwrap();
    let second = device.create_texture(&color_desc).unwrap();
    let depth = device
        .create_texture(&TextureDesc {
            format: TextureFormat::Depth32Float,
            ..color_desc.clone()
        })
        .unwrap();
    let color_view = |texture: &Texture<_>| {
        device
            .create_view(
                texture,
                &TextureViewDesc {
                    format: TextureFormat::Rgba8Unorm,
                    dimension: TextureViewDimension::D2,
                    aspects: TextureAspects::COLOR,
                    mips: 0..1,
                    layers: 0..1,
                },
            )
            .unwrap()
    };
    let first_view = color_view(&first);
    let second_view = color_view(&second);
    let depth_view = device
        .create_view(
            &depth,
            &TextureViewDesc {
                format: TextureFormat::Depth32Float,
                dimension: TextureViewDimension::D2,
                aspects: TextureAspects::DEPTH,
                mips: 0..1,
                layers: 0..1,
            },
        )
        .unwrap();
    let output = device
        .create_buffer(&BufferDesc {
            label: None,
            size: 12_288,
            usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
            memory: MemoryClass::Readback,
        })
        .unwrap();
    let empty = TextureUses {
        stages: StageMask::EMPTY,
        accesses: AccessMask::EMPTY,
    };
    let render = TextureUses {
        stages: StageMask::COLOR_OUTPUT,
        accesses: AccessMask::COLOR_WRITE,
    };
    let depth_write = TextureUses {
        stages: StageMask::FRAGMENT,
        accesses: AccessMask::DEPTH_WRITE,
    };
    let copy = TextureUses {
        stages: StageMask::COPY,
        accesses: AccessMask::COPY_READ,
    };
    let mut encoder = device.encoder(lane).unwrap();
    for texture in [&first, &second] {
        encoder
            .transition_resources(&[ResourceTransition::Texture(TextureBarrier {
                target: Subresources {
                    texture,
                    aspects: TextureAspects::COLOR,
                    mips: 0..1,
                    layers: 0..1,
                },
                usage: StateTransition {
                    from: empty,
                    to: render,
                },
                layout: Some((ImageLayout::Common, ImageLayout::ColorAttachment)),
                ownership: None,
            })])
            .unwrap();
    }
    encoder
        .transition_resources(&[ResourceTransition::Texture(TextureBarrier {
            target: Subresources {
                texture: &depth,
                aspects: TextureAspects::DEPTH,
                mips: 0..1,
                layers: 0..1,
            },
            usage: StateTransition {
                from: empty,
                to: depth_write,
            },
            layout: Some((ImageLayout::Undefined, ImageLayout::DepthStencilAttachment)),
            ownership: None,
        })])
        .unwrap();
    let colors = [
        Some(ColorAttachment {
            view: AttachmentView::Texture(&first_view),
            resolve_target: None,
            load: LoadOp::Clear(ClearColor::Float([0.0, 0.0, 1.0, 1.0])),
            store: StoreOp::Store,
        }),
        Some(ColorAttachment {
            view: AttachmentView::Texture(&second_view),
            resolve_target: None,
            load: LoadOp::Clear(ClearColor::Float([0.0, 1.0, 0.0, 1.0])),
            store: StoreOp::Store,
        }),
    ];
    let mut pass = encoder
        .begin_raster(&RasterPassDesc {
            label: None,
            colors: &colors,
            depth_stencil: Some(DepthStencilAttachment {
                view: AttachmentView::Texture(&depth_view),
                depth: Some((LoadOp::Clear(0.25), StoreOp::Store)),
                stencil: None,
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
    pass.clear_rect(
        Rect {
            x: 2,
            y: 3,
            width: 4,
            height: 3,
        },
        &[ColorAttachmentClear {
            attachment: 0,
            value: ClearColor::Float([1.0, 0.0, 0.0, 1.0]),
        }],
        None,
        None,
    )
    .unwrap();
    pass.finish().unwrap();
    for texture in [&first, &second] {
        encoder
            .transition_resources(&[ResourceTransition::Texture(TextureBarrier {
                target: Subresources {
                    texture,
                    aspects: TextureAspects::COLOR,
                    mips: 0..1,
                    layers: 0..1,
                },
                usage: StateTransition {
                    from: render,
                    to: copy,
                },
                layout: Some((ImageLayout::ColorAttachment, ImageLayout::CopySrc)),
                ownership: None,
            })])
            .unwrap();
    }
    encoder
        .transition_resources(&[ResourceTransition::Texture(TextureBarrier {
            target: Subresources {
                texture: &depth,
                aspects: TextureAspects::DEPTH,
                mips: 0..1,
                layers: 0..1,
            },
            usage: StateTransition {
                from: depth_write,
                to: copy,
            },
            layout: Some((ImageLayout::DepthStencilAttachment, ImageLayout::CopySrc)),
            ownership: None,
        })])
        .unwrap();
    for (texture, offset, aspect) in [
        (&first, 0, TextureAspects::COLOR),
        (&second, 4096, TextureAspects::COLOR),
        (&depth, 8192, TextureAspects::DEPTH),
    ] {
        encoder
            .texture_to_buffer(
                TextureRegion {
                    texture,
                    mip: 0,
                    origin: [0, 0, 0],
                    extent: [8, 8, 1],
                    aspect,
                },
                &output,
                TexelCopyBufferLayout {
                    offset,
                    bytes_per_row: 256,
                    rows_per_image: 8,
                },
            )
            .unwrap();
    }
    let ticket = encoder.readback_buffer(&output, 0..12_288).unwrap();
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: device.id(),
                ordinal: 95_117,
            },
            Arc::from([StepDesc {
                lane,
                kind: StepKind::Commands,
                waits: vec![],
            }]),
        )
    };
    let step = plan.step(0);
    let point = match unsafe {
        device.queue().submit(SubmitPacket {
            plan,
            steps: vec![BoundStep {
                step,
                payload: StepPayload::Commands(vec![encoder.finish().unwrap()]),
            }],
            completion: CompletionRequest::End,
        })
    } {
        SubmitOutcome::Accepted(receipt) => receipt.checkpoints.into_iter().next().unwrap().point,
        _ => panic!("submission rejected"),
    };
    block(device.wait_completion(&point)).unwrap();
    let bytes = unsafe { block(device.read_after(ticket, &point)).unwrap().bytes };
    for y in 0..8 {
        for x in 0..8 {
            let pixel = y * 256 + x * 4;
            let expected = if (2..6).contains(&x) && (3..6).contains(&y) {
                [255, 0, 0, 255]
            } else {
                [0, 0, 255, 255]
            };
            assert_eq!(&bytes[pixel..pixel + 4], expected, "first ({x}, {y})");
            assert_eq!(
                &bytes[4096 + pixel..4100 + pixel],
                [0, 255, 0, 255],
                "non-target MRT ({x}, {y})"
            );
            assert_eq!(
                f32::from_ne_bytes(bytes[8192 + pixel..8196 + pixel].try_into().unwrap()),
                0.25,
                "non-target depth ({x}, {y})"
            );
        }
    }
}

#[test]
fn raster_rect_clear_rejects_multiview() {
    let device = block(Dx12Instance::new().unwrap().open_default()).unwrap();
    if !device.capabilities().multiview {
        return;
    }
    let lane = device
        .lanes()
        .iter()
        .find(|lane| lane.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let texture = device
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [4, 4, 1],
            array_layers: 2,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::RENDER_ATTACHMENT,
            view_formats: vec![],
        })
        .unwrap();
    let view = device
        .create_view(
            &texture,
            &TextureViewDesc {
                format: TextureFormat::Rgba8Unorm,
                dimension: TextureViewDimension::D2Array,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..2,
            },
        )
        .unwrap();
    let mut encoder = device.encoder(lane).unwrap();
    let attachments = [Some(ColorAttachment {
        view: AttachmentView::Texture(&view),
        resolve_target: None,
        load: LoadOp::Load,
        store: StoreOp::Store,
    })];
    let mut pass = encoder
        .begin_raster(&RasterPassDesc {
            label: None,
            colors: &attachments,
            depth_stencil: None,
            render_area: Rect {
                x: 0,
                y: 0,
                width: 4,
                height: 4,
            },
            multiview_mask: Some(0b11),
            timestamp_writes: None,
            occlusion_query_set: None,
            secondary_inheritance: None,
        })
        .unwrap();
    assert!(matches!(
        pass.clear_rect(
            Rect {
                x: 0,
                y: 0,
                width: 1,
                height: 1,
            },
            &[ColorAttachmentClear {
                attachment: 0,
                value: ClearColor::Float([0.0; 4]),
            }],
            None,
            None,
        ),
        Err(Error::Unsupported(
            "DX12 multiview raster attachment rect clear"
        ))
    ));
}
