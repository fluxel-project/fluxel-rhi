//! Direct RHI proof for D3D12 dynamic depth bias.
#![cfg(all(windows, feature = "hardware-tests"))]

use fluxel_rhi_core::api::*;
use fluxel_rhi_dx12::Dx12Instance;
use std::sync::Arc;

#[path = "support/mod.rs"]
mod support;
use support::block;

fn triangle_pipeline_desc(
    device: &Device<fluxel_rhi_dx12::Dx12>,
) -> RasterPipelineDesc<fluxel_rhi_dx12::Dx12> {
    let shader = |bytes: &'static [u8], name: &str, stage| {
        block(device.shader(ShaderDesc {
            label: None,
            format: ShaderFormat::Dxil,
            bytes: Arc::from(bytes),
            entries: vec![ShaderEntryPoint {
                name: name.into(),
                stage,
                workgroup_size: None,
            }],
            required_features: vec![],
        }))
        .unwrap()
    };
    let layout = device
        .create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &[],
            push_constants: &[],
        })
        .unwrap();
    let face = StencilFaceState {
        compare: CompareFunction::Always,
        fail: StencilOperation::Keep,
        depth_fail: StencilOperation::Keep,
        pass: StencilOperation::Keep,
    };
    RasterPipelineDesc {
        label: Some("dynamic depth bias test".into()),
        layout,
        vertex: PipelineStage {
            module: shader(
                include_bytes!("../../rhi/examples/assets/01_triangle/triangle_vs.dxil"),
                "vs_main",
                ShaderStage::Vertex,
            ),
            entry: "vs_main".into(),
            constants: vec![],
        },
        fragment: Some(PipelineStage {
            module: shader(
                include_bytes!("../../rhi/examples/assets/01_triangle/triangle_ps.dxil"),
                "ps_main",
                ShaderStage::Fragment,
            ),
            entry: "ps_main".into(),
            constants: vec![],
        }),
        vertex_buffers: vec![VertexBufferLayout {
            stride: 24,
            step_mode: VertexStepMode::Vertex,
            attributes: vec![
                VertexAttribute {
                    location: 0,
                    offset: 0,
                    format: VertexFormat::Float32(3),
                },
                VertexAttribute {
                    location: 1,
                    offset: 12,
                    format: VertexFormat::Float32(3),
                },
            ],
        }],
        color_targets: vec![Some(ColorTargetState {
            format: TextureFormat::Rgba8Unorm,
            blend: None,
            write_mask: 15,
        })],
        primitive: PrimitiveState {
            topology: PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: FrontFace::Ccw,
            cull_mode: CullMode::None,
            unclipped_depth: false,
            polygon_mode: PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: Some(DepthStencilState {
            format: TextureFormat::Depth32Float,
            depth_write_enabled: false,
            depth_compare: CompareFunction::Less,
            stencil_front: face,
            stencil_back: face,
            stencil_read_mask: 0,
            stencil_write_mask: 0,
            depth_bias_constant: 0,
            depth_bias_slope_scale: 0.0,
            depth_bias_clamp: 0.0,
        }),
        multisample: MultisampleState {
            count: 1,
            mask: u64::from(u32::MAX),
            alpha_to_coverage_enabled: false,
        },
        multiview: None,
        cache: None,
    }
}

#[test]
fn dynamic_depth_bias_changes_draw_and_rebind_restores_initial_bias() {
    let device = block(Dx12Instance::new().unwrap().open_default()).unwrap();
    if !device.capabilities().dynamic_depth_bias {
        eprintln!("SKIP: adapter has no D3D12 dynamic depth bias support");
        return;
    }
    let lane = device
        .lanes()
        .iter()
        .find(|lane| lane.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let initial = DynamicDepthBias::default();
    let pipeline = block(
        device.raster_pipeline_with_dynamic_depth_bias(triangle_pipeline_desc(&device), initial),
    )
    .unwrap();
    let color = device
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [16, 16, 1],
            array_layers: 1,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
            view_formats: vec![],
        })
        .unwrap();
    let depth = device
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [16, 16, 1],
            array_layers: 1,
            format: TextureFormat::Depth32Float,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::RENDER_ATTACHMENT,
            view_formats: vec![],
        })
        .unwrap();
    let color_view = device
        .create_view(
            &color,
            &TextureViewDesc {
                format: TextureFormat::Rgba8Unorm,
                dimension: TextureViewDimension::D2,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..1,
            },
        )
        .unwrap();
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
    let vertex = device
        .create_buffer(&BufferDesc {
            label: None,
            size: 72,
            usage: BufferUsage::VERTEX | BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let output = device
        .create_buffer(&BufferDesc {
            label: None,
            size: 4096,
            usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
            memory: MemoryClass::Readback,
        })
        .unwrap();
    let mut encoder = device.encoder(lane).unwrap();
    let vertices: [f32; 18] = [
        -0.8, -0.8, 0.5, 1.0, 0.0, 0.0, 0.8, -0.8, 0.5, 1.0, 0.0, 0.0, 0.0, 0.8, 0.5, 1.0, 0.0, 0.0,
    ];
    encoder
        .write_buffer(
            &vertex,
            0,
            &vertices
                .into_iter()
                .flat_map(f32::to_ne_bytes)
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let empty = TextureUses {
        stages: StageMask::EMPTY,
        accesses: AccessMask::EMPTY,
    };
    let render_color = TextureUses {
        stages: StageMask::COLOR_OUTPUT,
        accesses: AccessMask::COLOR_WRITE,
    };
    let render_depth = TextureUses {
        stages: StageMask::FRAGMENT,
        accesses: AccessMask::DEPTH_WRITE,
    };
    encoder
        .transition_resources(&[
            ResourceTransition::Texture(TextureBarrier {
                target: Subresources {
                    texture: &color,
                    aspects: TextureAspects::COLOR,
                    mips: 0..1,
                    layers: 0..1,
                },
                usage: StateTransition {
                    from: empty,
                    to: render_color,
                },
                layout: Some((ImageLayout::Common, ImageLayout::ColorAttachment)),
                ownership: None,
            }),
            ResourceTransition::Texture(TextureBarrier {
                target: Subresources {
                    texture: &depth,
                    aspects: TextureAspects::DEPTH,
                    mips: 0..1,
                    layers: 0..1,
                },
                usage: StateTransition {
                    from: empty,
                    to: render_depth,
                },
                layout: Some((ImageLayout::Undefined, ImageLayout::DepthStencilAttachment)),
                ownership: None,
            }),
        ])
        .unwrap();
    let colors = [Some(ColorAttachment {
        view: AttachmentView::Texture(&color_view),
        resolve_target: None,
        load: LoadOp::Clear(ClearColor::Float([0.0, 0.0, 0.0, 1.0])),
        store: StoreOp::Store,
    })];
    let mut pass = encoder
        .begin_raster(&RasterPassDesc {
            label: Some("dynamic depth bias"),
            colors: &colors,
            depth_stencil: Some(DepthStencilAttachment {
                view: AttachmentView::Texture(&depth_view),
                depth: Some((LoadOp::Clear(0.5), StoreOp::Store)),
                stencil: None,
                depth_read_only: false,
                stencil_read_only: false,
            }),
            render_area: Rect {
                x: 0,
                y: 0,
                width: 16,
                height: 16,
            },
            multiview_mask: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            secondary_inheritance: None,
        })
        .unwrap();
    pass.viewport(Viewport {
        x: 0.0,
        y: 0.0,
        width: 16.0,
        height: 16.0,
        min_depth: 0.0,
        max_depth: 1.0,
    })
    .unwrap();
    pass.set_pipeline(&pipeline).unwrap();
    pass.set_vertex_buffer(0, &vertex, 0..72).unwrap();
    // Initial zero bias fails LESS against the clear value at x=6.
    pass.scissor(Rect {
        x: 5,
        y: 0,
        width: 2,
        height: 16,
    })
    .unwrap();
    pass.draw(0..3, 0..1).unwrap();
    // Negative integral bias makes the coincident primitive pass at x=8.
    pass.set_depth_bias(DynamicDepthBias {
        constant: -1,
        slope_scale: 0.0,
        clamp: 0.0,
    })
    .unwrap();
    pass.scissor(Rect {
        x: 7,
        y: 0,
        width: 2,
        height: 16,
    })
    .unwrap();
    pass.draw(0..3, 0..1).unwrap();
    // Rebinding must reinstall the pipeline's zero initial bias at x=10.
    pass.set_pipeline(&pipeline).unwrap();
    pass.scissor(Rect {
        x: 9,
        y: 0,
        width: 2,
        height: 16,
    })
    .unwrap();
    pass.draw(0..3, 0..1).unwrap();
    pass.finish().unwrap();
    encoder
        .transition_resources(&[ResourceTransition::Texture(TextureBarrier {
            target: Subresources {
                texture: &color,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..1,
            },
            usage: StateTransition {
                from: render_color,
                to: TextureUses {
                    stages: StageMask::COPY,
                    accesses: AccessMask::COPY_READ,
                },
            },
            layout: Some((ImageLayout::ColorAttachment, ImageLayout::CopySrc)),
            ownership: None,
        })])
        .unwrap();
    encoder
        .texture_to_buffer(
            TextureRegion {
                texture: &color,
                mip: 0,
                origin: [0, 0, 0],
                extent: [16, 16, 1],
                aspect: TextureAspects::COLOR,
            },
            &output,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: 256,
                rows_per_image: 16,
            },
        )
        .unwrap();
    let ticket = encoder.readback_buffer(&output, 0..4096).unwrap();
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: device.id(),
                ordinal: 0xDDB,
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
        device.queue().submit(SubmitPacket {
            plan,
            steps: vec![BoundStep {
                step,
                payload: StepPayload::Commands(vec![encoder.finish().unwrap()]),
            }],
            completion: CompletionRequest::End,
        })
    };
    let point = match outcome {
        SubmitOutcome::Accepted(receipt) => receipt.checkpoints.into_iter().next().unwrap().point,
        _ => panic!("submission rejected"),
    };
    block(device.wait_completion(&point)).unwrap();
    let bytes = unsafe { block(device.read_after(ticket, &point)).unwrap().bytes };
    let pixel = |x: usize| &bytes[8 * 256 + x * 4..8 * 256 + x * 4 + 4];
    assert_eq!(pixel(6), &[0, 0, 0, 255], "zero initial bias must fail");
    assert!(pixel(8)[0] > 100, "negative dynamic bias must pass");
    assert_eq!(
        pixel(10),
        &[0, 0, 0, 255],
        "pipeline rebind must restore zero bias"
    );
}
