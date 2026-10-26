//! Real GPU checks for read-only depth and stencil attachments.
#![cfg(all(windows, feature = "hardware-tests"))]
use fluxel_rhi_core::api::*;
use fluxel_rhi_dx12::Dx12Instance;
use std::{future::Future, sync::Arc};

fn block_on<F: Future>(f: F) -> F::Output {
    let w = std::task::Waker::noop();
    let mut c = std::task::Context::from_waker(w);
    let mut f = std::pin::pin!(f);
    loop {
        match f.as_mut().poll(&mut c) {
            std::task::Poll::Ready(v) => return v,
            std::task::Poll::Pending => std::thread::sleep(std::time::Duration::from_millis(1)),
        }
    }
}

#[test]
fn readonly_depth_allows_stencil_writes() {
    check_readonly(true);
}
#[test]
fn readonly_stencil_allows_depth_writes() {
    check_readonly(false);
}
fn check_readonly(readonly_depth: bool) {
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
        include_bytes!("../../../rhi/examples/assets/01_triangle/triangle_vs.dxil"),
        "vs_main",
        ShaderStage::Vertex,
    );
    let ps = shader(
        include_bytes!("../../../rhi/examples/assets/01_triangle/triangle_ps.dxil"),
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
    let face = StencilFaceState {
        compare: if readonly_depth {
            CompareFunction::Equal
        } else {
            CompareFunction::Always
        },
        fail: StencilOperation::Keep,
        depth_fail: StencilOperation::Keep,
        pass: StencilOperation::Keep,
    };
    let pipeline = block_on(d.raster_pipeline(RasterPipelineDesc {
        label: None,
        layout: layout.clone(),
        vertex: PipelineStage {
            module: vs.clone(),
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
            topology: PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: FrontFace::Ccw,
            cull_mode: CullMode::None,
            unclipped_depth: false,
            polygon_mode: PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: Some(DepthStencilState {
            format: TextureFormat::Depth24PlusStencil8,
            depth_write_enabled: false,
            depth_compare: if readonly_depth {
                CompareFunction::Always
            } else {
                CompareFunction::Less
            },
            stencil_front: face,
            stencil_back: face,
            stencil_read_mask: 255,
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
    }))
    .unwrap();
    let make_depth_desc = || RasterPipelineDesc {
        label: None,
        layout: layout.clone(),
        vertex: PipelineStage {
            module: vs.clone(),
            entry: "vs_main".into(),
            constants: vec![],
        },
        fragment: None,
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
        color_targets: vec![],
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
            format: TextureFormat::Depth24PlusStencil8,
            depth_write_enabled: !readonly_depth,
            depth_compare: CompareFunction::Always,
            stencil_front: StencilFaceState {
                compare: CompareFunction::Always,
                pass: if readonly_depth {
                    StencilOperation::Replace
                } else {
                    StencilOperation::Keep
                },
                ..face
            },
            stencil_back: StencilFaceState {
                compare: CompareFunction::Always,
                pass: if readonly_depth {
                    StencilOperation::Replace
                } else {
                    StencilOperation::Keep
                },
                ..face
            },
            stencil_read_mask: 255,
            stencil_write_mask: if readonly_depth { 255 } else { 0 },
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
    };
    let mut conflicting = make_depth_desc();
    let state = conflicting.depth_stencil.as_mut().unwrap();
    if readonly_depth {
        state.depth_write_enabled = true;
    } else {
        state.stencil_write_mask = 255;
        state.stencil_front.pass = StencilOperation::Replace;
        state.stencil_back.pass = StencilOperation::Replace;
    }
    let conflicting_pipeline = block_on(d.raster_pipeline(conflicting)).unwrap();
    let depth_pipeline = block_on(d.raster_pipeline(make_depth_desc())).unwrap();
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
    let depth = d
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [16, 16, 1],
            array_layers: 2,
            format: TextureFormat::Depth24PlusStencil8,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::RENDER_ATTACHMENT,
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
    let depths = views(
        &depth,
        TextureFormat::Depth24PlusStencil8,
        TextureAspects::DEPTH | TextureAspects::STENCIL,
    );
    let vertex = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 72,
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
    let vertices: [f32; 18] = [
        -0.8, -0.8, 0.5, 1.0, 0.0, 0.0, 0.8, -0.8, 0.5, 1.0, 0.0, 0.0, 0.0, 0.8, 0.5, 1.0, 0.0, 0.0,
    ];
    e.write_buffer(
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
    e.transition_resources(&[ResourceTransition::Texture(TextureBarrier {
        target: Subresources {
            texture: &depth,
            aspects: TextureAspects::DEPTH | TextureAspects::STENCIL,
            mips: 0..1,
            layers: 0..2,
        },
        usage: StateTransition {
            from: empty,
            to: TextureUses {
                stages: StageMask::FRAGMENT,
                accesses: AccessMask::DEPTH_WRITE,
            },
        },
        layout: Some((ImageLayout::Undefined, ImageLayout::DepthStencilAttachment)),
        ownership: None,
    })])
    .unwrap();
    // Initialize both planes while writable; later passes mark one plane read-only.
    for layer in 0..2 {
        let attachment = DepthStencilAttachment {
            view: AttachmentView::Texture(&depths[layer]),
            depth: Some((LoadOp::Clear(1.0), StoreOp::Store)),
            stencil: Some((LoadOp::Clear(0), StoreOp::Store)),
            depth_read_only: false,
            stencil_read_only: false,
        };
        e.begin_raster(&RasterPassDesc {
            label: None,
            colors: &[],
            depth_stencil: Some(attachment),
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
        .unwrap()
        .finish()
        .unwrap();
    }
    let plane = if readonly_depth {
        TextureAspects::DEPTH
    } else {
        TextureAspects::STENCIL
    };
    let write_plane = TextureUses {
        stages: StageMask::FRAGMENT,
        accesses: if readonly_depth {
            AccessMask::DEPTH_WRITE
        } else {
            AccessMask::STENCIL_WRITE
        },
    };
    let read_plane = TextureUses {
        stages: StageMask::FRAGMENT,
        accesses: if readonly_depth {
            AccessMask::DEPTH_READ
        } else {
            AccessMask::STENCIL_READ
        },
    };
    let plane_barrier = |from, to| {
        ResourceTransition::Texture(TextureBarrier {
            target: Subresources {
                texture: &depth,
                aspects: plane,
                mips: 0..1,
                layers: 1..2,
            },
            usage: StateTransition { from, to },
            layout: Some((
                ImageLayout::DepthStencilAttachment,
                ImageLayout::DepthStencilAttachment,
            )),
            ownership: None,
        })
    };
    e.transition_resources(&[plane_barrier(write_plane, read_plane)])
        .unwrap();
    readonly_conflicts::check(
        &d,
        lane,
        &depths[1],
        &conflicting_pipeline,
        &vertex,
        readonly_depth,
    );
    let attachment = DepthStencilAttachment {
        view: AttachmentView::Texture(&depths[1]),
        depth: Some((LoadOp::Load, StoreOp::Store)),
        stencil: Some((LoadOp::Load, StoreOp::Store)),
        depth_read_only: readonly_depth,
        stencil_read_only: !readonly_depth,
    };
    let mut pass = e
        .begin_raster(&RasterPassDesc {
            label: None,
            colors: &[],
            depth_stencil: Some(attachment),
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
    pass.set_pipeline(&depth_pipeline).unwrap();
    pass.stencil_reference(7).unwrap();
    pass.set_vertex_buffer(0, &vertex, 0..72).unwrap();
    pass.draw(0..3, 0..1).unwrap();
    pass.finish().unwrap();
    e.transition_resources(&[plane_barrier(read_plane, write_plane)])
        .unwrap();
    for layer in 0..2 {
        let attachments = [Some(ColorAttachment {
            view: AttachmentView::Texture(&colors[layer]),
            resolve_target: None,
            load: LoadOp::Clear(ClearColor::Float([0.0, 0.0, 0.0, 1.0])),
            store: StoreOp::Store,
        })];
        let depth_attachment = DepthStencilAttachment {
            view: AttachmentView::Texture(&depths[layer]),
            depth: Some((LoadOp::Load, StoreOp::Store)),
            stencil: Some((LoadOp::Load, StoreOp::Store)),
            depth_read_only: false,
            stencil_read_only: false,
        };
        let mut pass = e
            .begin_raster(&RasterPassDesc {
                label: None,
                colors: &attachments,
                depth_stencil: Some(depth_attachment),
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
        pass.scissor(Rect {
            x: 0,
            y: 0,
            width: 16,
            height: 16,
        })
        .unwrap();
        pass.set_pipeline(&pipeline).unwrap();
        pass.stencil_reference(7).unwrap();
        pass.set_vertex_buffer(0, &vertex, 0..72).unwrap();
        pass.draw(0..3, 0..1).unwrap();
        pass.finish().unwrap();
    }
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
    let center = 8 * 256 + 8 * 4;
    if readonly_depth {
        assert_eq!(&bytes[center..center + 4], &[0, 0, 0, 255]);
        assert!(
            bytes[4096 + center] > 100,
            "writable stencil must contain reference 7"
        );
    } else {
        assert!(
            bytes[center] > 100,
            "unwritten depth 1 must pass comparison"
        );
        assert_eq!(
            &bytes[4096 + center..4096 + center + 4],
            &[0, 0, 0, 255],
            "writable depth 0.5 must reject equal depth"
        );
    }
}

#[path = "conflicts.rs"]
mod readonly_conflicts;
