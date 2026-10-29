//! Real GPU checks for primitive topology and vertex input.
#![cfg(all(windows, feature = "hardware-tests"))]
use fluxel_rhi_core::api::*;
use fluxel_rhi_dx12::Dx12Instance;
use std::sync::Arc;

#[path = "support/mod.rs"]
mod support;
use support::block as block_on;

#[test]
fn point_list() {
    check_primitive(
        PrimitiveTopology::PointList,
        PolygonMode::Fill,
        false,
        false,
    );
}
#[test]
fn line_list() {
    check_primitive(PrimitiveTopology::LineList, PolygonMode::Fill, false, false);
}
#[test]
fn line_strip() {
    check_primitive(
        PrimitiveTopology::LineStrip,
        PolygonMode::Fill,
        false,
        false,
    );
}
#[test]
fn triangle_strip() {
    check_primitive(
        PrimitiveTopology::TriangleStrip,
        PolygonMode::Fill,
        false,
        false,
    );
}
#[test]
fn polygon_line() {
    check_primitive(
        PrimitiveTopology::TriangleList,
        PolygonMode::Line,
        false,
        false,
    );
}

#[test]
fn triangle_strip_indexed_restart() {
    check_primitive(
        PrimitiveTopology::TriangleStrip,
        PolygonMode::Fill,
        true,
        false,
    );
}

#[test]
fn multiview_triangle_strip_indexed_restart() {
    check_primitive(
        PrimitiveTopology::TriangleStrip,
        PolygonMode::Fill,
        true,
        true,
    );
}

fn check_primitive(
    topology: PrimitiveTopology,
    polygon_mode: PolygonMode,
    restart: bool,
    multiview: bool,
) {
    let d = block_on(Dx12Instance::new().unwrap().open_default()).unwrap();
    let lane = d
        .lanes()
        .iter()
        .find(|l| l.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let shader = |bytes: &'static [u8], name: &str, stage| {
        block_on(d.shader(ShaderDesc {
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
    let vs = shader(
        include_bytes!("../../rhi/examples/assets/01_triangle/triangle_vs.dxil"),
        "vs_main",
        ShaderStage::Vertex,
    );
    let ps = shader(
        include_bytes!("../../rhi/examples/assets/01_triangle/triangle_ps.dxil"),
        "ps_main",
        ShaderStage::Fragment,
    );
    let layout = d
        .create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &[],
            push_constants: &[],
        })
        .unwrap();
    let pipeline = block_on(d.raster_pipeline(RasterPipelineDesc {
        label: None,
        layout,
        vertex: PipelineStage {
            module: vs,
            entry: "vs_main".into(),
            constants: vec![],
        },
        fragment: Some(PipelineStage {
            module: ps,
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
            topology,
            strip_index_format: if restart {
                Some(IndexFormat::Uint16)
            } else {
                None
            },
            front_face: FrontFace::Ccw,
            cull_mode: CullMode::None,
            unclipped_depth: false,
            polygon_mode,
            conservative: false,
        },
        depth_stencil: None,
        multisample: MultisampleState {
            count: 1,
            mask: u64::from(u32::MAX),
            alpha_to_coverage_enabled: false,
        },
        multiview: if multiview { Some(2) } else { None },
        cache: None,
    }))
    .unwrap();
    let color = d
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [16, 16, 1],
            array_layers: 2,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::RENDER_ATTACHMENT.union(TextureUsage::COPY_SRC),
            view_formats: vec![],
        })
        .unwrap();
    let views = |texture: &Texture<_>, format, aspect| {
        (0..2)
            .map(|layer| {
                d.create_view(
                    texture,
                    &TextureViewDesc {
                        format,
                        dimension: TextureViewDimension::D2Array,
                        aspects: aspect,
                        mips: 0..1,
                        layers: layer..layer + 1,
                    },
                )
                .unwrap()
            })
            .collect::<Vec<_>>()
    };
    let colors = views(&color, TextureFormat::Rgba8Unorm, TextureAspects::COLOR);
    let full_view = if multiview {
        Some(
            d.create_view(
                &color,
                &TextureViewDesc {
                    format: TextureFormat::Rgba8Unorm,
                    dimension: TextureViewDimension::D2Array,
                    aspects: TextureAspects::COLOR,
                    mips: 0..1,
                    layers: 0..2,
                },
            )
            .unwrap(),
        )
    } else {
        None
    };
    let vertex = d
        .create_buffer(&BufferDesc {
            label: None,
            size: if restart { 144 } else { 72 },
            usage: BufferUsage::VERTEX.union(BufferUsage::COPY_DST),
            memory: MemoryClass::Device,
        })
        .unwrap();
    let dst = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 8192,
            usage: BufferUsage::COPY_DST.union(BufferUsage::MAP_READ),
            memory: MemoryClass::Readback,
        })
        .unwrap();
    let mut e = d.encoder(lane).unwrap();
    let vertices: Vec<f32> = if restart {
        vec![
            -0.95, -0.8, 0.5, 1.0, 0.0, 0.0, -0.15, -0.8, 0.5, 1.0, 0.0, 0.0, -0.55, 0.8, 0.5, 1.0,
            0.0, 0.0, 0.15, -0.8, 0.5, 1.0, 0.0, 0.0, 0.95, -0.8, 0.5, 1.0, 0.0, 0.0, 0.55, 0.8,
            0.5, 1.0, 0.0, 0.0,
        ]
    } else {
        vec![
            -0.8, -0.8, 0.5, 1.0, 0.0, 0.0, 0.8, -0.8, 0.5, 1.0, 0.0, 0.0, 0.0, 0.8, 0.5, 1.0, 0.0,
            0.0,
        ]
    };
    e.write_buffer(
        &vertex,
        0,
        &vertices
            .into_iter()
            .flat_map(f32::to_ne_bytes)
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let indices = if restart {
        let buffer = d
            .create_buffer(&BufferDesc {
                label: None,
                size: 14,
                usage: BufferUsage::INDEX.union(BufferUsage::COPY_DST),
                memory: MemoryClass::Device,
            })
            .unwrap();
        e.write_buffer(
            &buffer,
            0,
            &[0u16, 1, 2, u16::MAX, 3, 4, 5]
                .into_iter()
                .flat_map(u16::to_ne_bytes)
                .collect::<Vec<_>>(),
        )
        .unwrap();
        Some(buffer)
    } else {
        None
    };
    let empty = TextureUses {
        stages: StageMask::EMPTY,
        accesses: AccessMask::EMPTY,
    };
    let render = TextureUses {
        stages: StageMask::COLOR_OUTPUT,
        accesses: AccessMask::COLOR_WRITE,
    };
    let copy = TextureUses {
        stages: StageMask::COPY,
        accesses: AccessMask::COPY_READ,
    };
    let barrier = |from, to, layouts| {
        ResourceTransition::Texture(TextureBarrier {
            target: Subresources {
                texture: &color,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..2,
            },
            usage: StateTransition { from, to },
            layout: Some(layouts),
            ownership: None,
        })
    };
    e.transition_resources(&[barrier(
        empty,
        render,
        (ImageLayout::Common, ImageLayout::ColorAttachment),
    )])
    .unwrap();
    let attachments = [Some(ColorAttachment {
        view: AttachmentView::Texture(full_view.as_ref().unwrap_or(&colors[0])),
        resolve_target: None,
        load: LoadOp::Clear(ClearColor::Float([0.0, 0.0, 0.0, 1.0])),
        store: StoreOp::Store,
    })];
    let mut pass = e
        .begin_raster(&RasterPassDesc {
            label: None,
            colors: &attachments,
            depth_stencil: None,
            render_area: Rect {
                x: 0,
                y: 0,
                width: 16,
                height: 16,
            },
            multiview_mask: if multiview { Some(3) } else { None },
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
    pass.scissor(Rect {
        x: 0,
        y: 0,
        width: 16,
        height: 16,
    })
    .unwrap();
    pass.set_pipeline(&pipeline).unwrap();
    pass.set_vertex_buffer(0, &vertex, 0..if restart { 144 } else { 72 })
        .unwrap();
    if let Some(indices) = &indices {
        pass.set_index_buffer(indices, 0..14, IndexFormat::Uint16)
            .unwrap();
        pass.draw_indexed(0..7, 0, 0..1).unwrap();
    } else {
        pass.draw(0..3, 0..1).unwrap();
    }
    pass.finish().unwrap();
    e.transition_resources(&[barrier(
        render,
        copy,
        (ImageLayout::ColorAttachment, ImageLayout::CopySrc),
    )])
    .unwrap();
    for layer in 0..2 {
        e.texture_to_buffer(
            TextureRegion {
                texture: &color,
                mip: 0,
                origin: [0, 0, layer],
                extent: [16, 16, 1],
                aspect: TextureAspects::COLOR,
            },
            &dst,
            TexelCopyBufferLayout {
                offset: u64::from(layer) * 4096,
                bytes_per_row: 256,
                rows_per_image: 16,
            },
        )
        .unwrap();
    }
    let ticket = e.readback_buffer(&dst, 0..8192).unwrap();
    let id = PlanId {
        device: d.id(),
        ordinal: 755321,
    };
    // SAFETY: the one-step plan uses this device's graphics lane.
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            id,
            Arc::from([StepDesc {
                lane,
                kind: StepKind::Commands,
                waits: vec![],
            }]),
        )
    };
    let step = plan.step(0);
    // SAFETY: the work is consumed once on its recording lane.
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
    block_on(d.wait_completion(&point)).unwrap();
    // SAFETY: the awaited completion point covers the readback.
    let bytes = unsafe { block_on(d.read_after(ticket, &point)).unwrap().bytes };
    if restart {
        for layer in 0..if multiview { 2 } else { 1 } {
            let base = layer * 4096;
            assert_eq!(
                &bytes[base + 8 * 256 + 8 * 4..base + 8 * 256 + 8 * 4 + 4],
                &[0, 0, 0, 255],
                "strip restart must leave layer {layer} gap clear"
            );
            assert!(
                bytes[base + 8 * 256 + 4 * 4] > 100 && bytes[base + 8 * 256 + 12 * 4] > 100,
                "both strips must draw in layer {layer}"
            );
        }
        return;
    }
    let lit = (0..16)
        .flat_map(|y| (0..16).map(move |x| y * 256 + x * 4))
        .filter(|&i| bytes[i] > 100)
        .count();
    assert!(lit > 0, "expected visible primitive, got {lit} pixels");
    match topology {
        PrimitiveTopology::PointList => assert!(lit <= 3, "point list produced {lit} pixels"),
        PrimitiveTopology::LineList | PrimitiveTopology::LineStrip => {
            assert!(lit < 40, "line produced {lit} pixels")
        }
        _ if matches!(polygon_mode, PolygonMode::Line) => {
            assert!(lit < 70, "wireframe produced {lit} pixels")
        }
        _ => assert!(lit > 40, "filled triangle produced {lit} pixels"),
    }
}
