//! Focused GPU checks for standalone texture clear.
#![cfg(all(windows, feature = "hardware-tests"))]
use fluxel_rhi_core::api::*;
use fluxel_rhi_vulkan::VulkanInstance;
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
fn rejects_invalid_clear_descriptions() {
    let d = block_on(VulkanInstance::new().unwrap().open_default()).unwrap();
    let lane = d
        .lanes()
        .iter()
        .find(|l| l.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let desc = TextureDesc {
        label: None,
        dimension: TextureDimension::D2,
        extent: [4, 4, 1],
        array_layers: 1,
        format: TextureFormat::Rgba8Unorm,
        mip_levels: 1,
        samples: 1,
        usage: TextureUsage::RENDER_ATTACHMENT,
        view_formats: vec![],
    };
    let color = d.create_texture(&desc).unwrap();
    let no_attachment = d
        .create_texture(&TextureDesc {
            usage: TextureUsage::COPY_SRC,
            ..desc.clone()
        })
        .unwrap();
    let depth = d
        .create_texture(&TextureDesc {
            format: TextureFormat::Depth32Float,
            ..desc
        })
        .unwrap();
    let float = TextureClearValue::Color(ClearColor::Float([0.0; 4]));
    let cases = [
        (&color, 0..2, 0..1, float),
        (&color, 0..1, 0..2, float),
        (&color, 0..1, 0..1, TextureClearValue::Depth(0.5)),
        (&depth, 0..1, 0..1, float),
        (
            &color,
            0..1,
            0..1,
            TextureClearValue::Color(ClearColor::Uint([0; 4])),
        ),
        (&no_attachment, 0..1, 0..1, float),
    ];
    for (texture, mips, layers, value) in cases {
        let mut e = d.encoder(lane).unwrap();
        assert!(matches!(
            e.clear_texture(texture, mips, layers, value),
            Err(Error::InvalidRange | Error::Unsupported(_))
        ));
    }
    let mut encoder = d.encoder(lane).unwrap();
    assert!(matches!(
        encoder.transition_resources(&[ResourceTransition::Texture(TextureBarrier {
            target: Subresources {
                texture: &depth,
                aspects: TextureAspects::DEPTH,
                mips: 0..1,
                layers: 0..1
            },
            usage: StateTransition {
                from: TextureUses {
                    stages: StageMask::FRAGMENT,
                    accesses: AccessMask::DEPTH_WRITE
                },
                to: TextureUses {
                    stages: StageMask::FRAGMENT,
                    accesses: AccessMask::DEPTH_WRITE
                }
            },
            layout: Some((ImageLayout::Undefined, ImageLayout::DepthStencilAttachment)),
            ownership: None,
        })]),
        Err(Error::Unsupported(_))
    ));
}

#[test]
fn depth_clear_controls_depth_test() {
    let d = block_on(VulkanInstance::new().unwrap().open_default()).unwrap();
    let lane = d
        .lanes()
        .iter()
        .find(|l| l.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let shader = |bytes: &'static [u8], name: &str, stage| {
        block_on(d.shader(ShaderDesc {
            label: None,
            format: ShaderFormat::SpirV,
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
        include_bytes!("../../rhi/examples/assets/01_triangle/triangle_vs.spv"),
        "vs_main",
        ShaderStage::Vertex,
    );
    let ps = shader(
        include_bytes!("../../rhi/examples/assets/01_triangle/triangle_ps.spv"),
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
        compare: CompareFunction::Always,
        fail: StencilOperation::Keep,
        depth_fail: StencilOperation::Keep,
        pass: StencilOperation::Keep,
    };
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
            depth_write_enabled: true,
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
    let depth = d
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [16, 16, 1],
            array_layers: 2,
            format: TextureFormat::Depth32Float,
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
    let depths = views(&depth, TextureFormat::Depth32Float, TextureAspects::DEPTH);
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
        (ImageLayout::Undefined, ImageLayout::ColorAttachment),
    )])
    .unwrap();
    e.transition_resources(&[ResourceTransition::Texture(TextureBarrier {
        target: Subresources {
            texture: &depth,
            aspects: TextureAspects::DEPTH,
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
    e.clear_texture(&depth, 0..1, 0..2, TextureClearValue::Depth(0.0))
        .unwrap();
    e.clear_texture(&depth, 0..1, 1..2, TextureClearValue::Depth(1.0))
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
            stencil: None,
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
    assert_eq!(&bytes[center..center + 4], &[0, 0, 0, 255]);
    assert!(
        bytes[4096 + center] > 100,
        "depth clear to 1 must allow triangle"
    );
}

#[test]
fn clears_only_selected_color_mip_and_layer() {
    let d = block_on(VulkanInstance::new().unwrap().open_default()).unwrap();
    let lane = d
        .lanes()
        .iter()
        .find(|l| l.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let texture = d
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [4, 4, 1],
            array_layers: 2,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 2,
            samples: 1,
            usage: TextureUsage::RENDER_ATTACHMENT.union(TextureUsage::COPY_SRC),
            view_formats: vec![],
        })
        .unwrap();
    let dst = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 4096,
            usage: BufferUsage::COPY_DST.union(BufferUsage::MAP_READ),
            memory: MemoryClass::Readback,
        })
        .unwrap();
    let mut e = d.encoder(lane).unwrap();
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
                texture: &texture,
                aspects: TextureAspects::COLOR,
                mips: 0..2,
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
        (ImageLayout::Undefined, ImageLayout::ColorAttachment),
    )])
    .unwrap();
    e.clear_texture(
        &texture,
        0..2,
        0..2,
        TextureClearValue::Color(ClearColor::Float([0.0, 0.0, 0.0, 1.0])),
    )
    .unwrap();
    e.clear_texture(
        &texture,
        1..2,
        1..2,
        TextureClearValue::Color(ClearColor::Float([1.0, 0.0, 0.0, 1.0])),
    )
    .unwrap();
    e.transition_resources(&[barrier(
        render,
        copy,
        (ImageLayout::ColorAttachment, ImageLayout::CopySrc),
    )])
    .unwrap();
    for layer in 0..2 {
        for mip in 0..2 {
            let offset = u64::from(layer * 2 + mip) * 1024;
            e.texture_to_buffer(
                TextureRegion {
                    texture: &texture,
                    mip,
                    origin: [0, 0, layer],
                    extent: [4 >> mip, 4 >> mip, 1],
                    aspect: TextureAspects::COLOR,
                },
                &dst,
                TexelCopyBufferLayout {
                    offset,
                    bytes_per_row: 256,
                    rows_per_image: 4 >> mip,
                },
            )
            .unwrap();
        }
    }
    let ticket = e.readback_buffer(&dst, 0..4096).unwrap();
    let id = PlanId {
        device: d.id(),
        ordinal: 755320,
    };
    // SAFETY: one command step uses this device's graphics lane.
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
    // SAFETY: this work is submitted exactly once on its recording lane.
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
    // SAFETY: this completion point covers the readback ticket.
    let bytes = unsafe { block_on(d.read_after(ticket, &point)).unwrap().bytes };
    for layer in 0..2usize {
        for mip in 0..2usize {
            let expected = if (layer, mip) == (1, 1) {
                [255, 0, 0, 255]
            } else {
                [0, 0, 0, 255]
            };
            let width = 4 >> mip;
            for y in 0..width {
                for x in 0..width {
                    let offset = (layer * 2 + mip) * 1024 + y * 256 + x * 4;
                    assert_eq!(&bytes[offset..offset + 4], &expected);
                }
            }
        }
    }
}
