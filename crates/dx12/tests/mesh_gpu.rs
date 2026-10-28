//! Focused GPU checks for standalone texture clear.
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
fn mesh_mrt_blend_and_stencil_control_readback() {
    let d = block_on(Dx12Instance::new().unwrap().open_default()).unwrap();
    if !d.capabilities().mesh {
        eprintln!("SKIP: mesh shader tier unavailable");
        return;
    }
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
        include_bytes!("assets/mesh/mesh.dxil"),
        "main",
        ShaderStage::Mesh,
    );
    let ps = shader(
        include_bytes!("assets/mesh/pixel.dxil"),
        "main",
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
        compare: CompareFunction::Equal,
        fail: StencilOperation::Keep,
        depth_fail: StencilOperation::Keep,
        pass: StencilOperation::Keep,
    };
    let mesh = d.mesh().unwrap();
    let cache = d.create_pipeline_cache(None).unwrap();
    let mesh_desc = |cache| MeshPipelineDesc {
        layout: layout.clone(),
        task: None,
        mesh: PipelineStage {
            module: vs.clone(),
            entry: "main".into(),
            constants: vec![],
        },
        fragment: Some(PipelineStage {
            module: ps.clone(),
            entry: "main".into(),
            constants: vec![],
        }),
        targets: vec![
            Some(ColorTargetState {
                format: TextureFormat::Rgba8Unorm,
                blend: Some(BlendState {
                    color: BlendComponent {
                        src_factor: BlendFactor::One,
                        dst_factor: BlendFactor::One,
                        operation: BlendOperation::Add,
                    },
                    alpha: BlendComponent {
                        src_factor: BlendFactor::One,
                        dst_factor: BlendFactor::Zero,
                        operation: BlendOperation::Add,
                    },
                }),
                write_mask: 15,
            }),
            Some(ColorTargetState {
                format: TextureFormat::Rgba8Unorm,
                blend: None,
                write_mask: 0,
            }),
        ],
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
            depth_write_enabled: true,
            depth_compare: CompareFunction::Less,
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
        cache: Some(cache),
    };
    block_on(mesh.create_pipeline(mesh_desc(cache.clone()))).unwrap();
    let serialized = d.serialize_pipeline_cache(&cache).unwrap();
    assert!(
        serialized.len() > 16,
        "native mesh PSO blob must be recorded"
    );
    let restored = d.create_pipeline_cache(Some(&serialized)).unwrap();
    let pipeline = block_on(mesh.create_pipeline(mesh_desc(restored))).unwrap();
    let color = d
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [16, 16, 1],
            array_layers: 4,
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
        (0..4)
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
    let depths = (0..2)
        .map(|layer| {
            d.create_view(
                &depth,
                &TextureViewDesc {
                    format: TextureFormat::Depth24PlusStencil8,
                    dimension: TextureViewDimension::D2Array,
                    aspects: TextureAspects::DEPTH | TextureAspects::STENCIL,
                    mips: 0..1,
                    layers: layer..layer + 1,
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let dst = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 16384,
            usage: BufferUsage::COPY_DST.union(BufferUsage::MAP_READ),
            memory: MemoryClass::Readback,
        })
        .unwrap();
    // A pipeline requiring depth cannot bind into a color-only pass.
    {
        let mut invalid = d.encoder(lane).unwrap();
        let color_attachment = [Some(ColorAttachment {
            view: AttachmentView::Texture(&colors[0]),
            resolve_target: None,
            load: LoadOp::Load,
            store: StoreOp::Store,
        })];
        let mut pass = invalid
            .begin_raster(&RasterPassDesc {
                label: None,
                colors: &color_attachment,
                depth_stencil: None,
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
        assert!(matches!(
            mesh.bind_pipeline(&mut pass, &pipeline),
            Err(Error::InvalidState)
        ));
    }
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
                texture: &color,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..4,
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
    for layer in 0..2 {
        let attachments = [
            Some(ColorAttachment {
                view: AttachmentView::Texture(&colors[layer]),
                resolve_target: None,
                load: LoadOp::Clear(ClearColor::Float([0.25, 0.0, 0.0, 1.0])),
                store: StoreOp::Store,
            }),
            Some(ColorAttachment {
                view: AttachmentView::Texture(&colors[layer + 2]),
                resolve_target: None,
                load: LoadOp::Clear(ClearColor::Float([0.0, 1.0, 0.0, 1.0])),
                store: StoreOp::Store,
            }),
        ];
        let depth_attachment = DepthStencilAttachment {
            view: AttachmentView::Texture(&depths[layer]),
            depth: Some((LoadOp::Clear(1.0), StoreOp::Store)),
            stencil: Some((LoadOp::Clear(layer as u32), StoreOp::Store)),
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
        pass.stencil_reference(1).unwrap();
        mesh.bind_pipeline(&mut pass, &pipeline).unwrap();
        mesh.dispatch(&mut pass, [1, 1, 1]).unwrap();
        pass.finish().unwrap();
    }
    e.transition_resources(&[barrier(
        render,
        copy,
        (ImageLayout::ColorAttachment, ImageLayout::CopySrc),
    )])
    .unwrap();
    for layer in 0..4 {
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
    let ticket = e.readback_buffer(&dst, 0..16384).unwrap();
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
    assert_eq!(&bytes[center..center + 4], &[64, 0, 0, 255]);
    assert_eq!(&bytes[4096 + center..4096 + center + 4], &[255, 0, 0, 255]);
    for layer in 2..4 {
        assert_eq!(
            &bytes[layer * 4096 + center..layer * 4096 + center + 4],
            &[0, 255, 0, 255]
        );
    }
}

#[test]
fn mesh_depth_controls_readback() {
    let d = block_on(Dx12Instance::new().unwrap().open_default()).unwrap();
    if !d.capabilities().mesh {
        eprintln!("SKIP: mesh shader tier unavailable");
        return;
    }
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
        include_bytes!("assets/mesh/mesh.dxil"),
        "main",
        ShaderStage::Mesh,
    );
    let ps = shader(
        include_bytes!("assets/mesh/pixel.dxil"),
        "main",
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
    let mesh = d.mesh().unwrap();
    let pipeline = block_on(mesh.create_pipeline(MeshPipelineDesc {
        layout,
        task: None,
        mesh: PipelineStage {
            module: vs,
            entry: "main".into(),
            constants: vec![],
        },
        fragment: Some(PipelineStage {
            module: ps,
            entry: "main".into(),
            constants: vec![],
        }),
        targets: vec![Some(ColorTargetState {
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
    let dst = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 8192,
            usage: BufferUsage::COPY_DST.union(BufferUsage::MAP_READ),
            memory: MemoryClass::Readback,
        })
        .unwrap();
    // A pipeline requiring depth cannot bind into a color-only pass.
    {
        let mut invalid = d.encoder(lane).unwrap();
        let color_attachment = [Some(ColorAttachment {
            view: AttachmentView::Texture(&colors[0]),
            resolve_target: None,
            load: LoadOp::Load,
            store: StoreOp::Store,
        })];
        let mut pass = invalid
            .begin_raster(&RasterPassDesc {
                label: None,
                colors: &color_attachment,
                depth_stencil: None,
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
        assert!(matches!(
            mesh.bind_pipeline(&mut pass, &pipeline),
            Err(Error::InvalidState)
        ));
    }
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
    // Newly allocated DX12 depth resources start in DEPTH_WRITE.
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
        mesh.bind_pipeline(&mut pass, &pipeline).unwrap();
        mesh.dispatch(&mut pass, [1, 1, 1]).unwrap();
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
    assert_eq!(&bytes[4096 + center..4096 + center + 4], &[255, 0, 0, 255]);
}

#[test]
fn mesh_multi_indirect_and_count_readback() {
    let d = block_on(Dx12Instance::new().unwrap().open_default()).unwrap();
    if !(d.capabilities().mesh_multi_indirect && d.capabilities().mesh_indirect_count) {
        eprintln!("SKIP: mesh indirect batch/count unavailable");
        return;
    }
    let lane = d
        .lanes()
        .iter()
        .find(|l| l.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let mesh_shader = block_on(d.shader(ShaderDesc {
        label: None,
        format: ShaderFormat::Dxil,
        bytes: Arc::from(include_bytes!("assets/mesh/mesh.dxil").as_slice()),
        entries: vec![ShaderEntryPoint {
            name: "main".into(),
            stage: ShaderStage::Mesh,
            workgroup_size: None,
        }],
        required_features: vec![],
    }))
    .unwrap();
    let pixel = fractional_pixel_shader(&d);
    let layout = d
        .create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &[],
            push_constants: &[],
        })
        .unwrap();
    let pipeline = block_on(d.mesh().unwrap().create_pipeline(MeshPipelineDesc {
        layout,
        task: None,
        mesh: PipelineStage {
            module: mesh_shader,
            entry: "main".into(),
            constants: vec![],
        },
        fragment: Some(PipelineStage {
            module: pixel,
            entry: "main".into(),
            constants: vec![],
        }),
        targets: vec![Some(ColorTargetState {
            format: TextureFormat::Rgba8Unorm,
            blend: Some(BlendState {
                color: BlendComponent {
                    src_factor: BlendFactor::One,
                    dst_factor: BlendFactor::One,
                    operation: BlendOperation::Add,
                },
                alpha: BlendComponent {
                    src_factor: BlendFactor::One,
                    dst_factor: BlendFactor::Zero,
                    operation: BlendOperation::Add,
                },
            }),
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
            depth_compare: CompareFunction::Always,
            stencil_front: StencilFaceState {
                compare: CompareFunction::Always,
                fail: StencilOperation::Keep,
                depth_fail: StencilOperation::Keep,
                pass: StencilOperation::Keep,
            },
            stencil_back: StencilFaceState {
                compare: CompareFunction::Always,
                fail: StencilOperation::Keep,
                depth_fail: StencilOperation::Keep,
                pass: StencilOperation::Keep,
            },
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
        cache: None,
    }))
    .unwrap();
    let color = d
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [16, 16, 1],
            array_layers: 5,
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
            array_layers: 5,
            format: TextureFormat::Depth32Float,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::RENDER_ATTACHMENT,
            view_formats: vec![],
        })
        .unwrap();
    let view = |texture: &Texture<_>, format, aspect, layer| {
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
    };
    let colors = (0..5)
        .map(|layer| {
            view(
                &color,
                TextureFormat::Rgba8Unorm,
                TextureAspects::COLOR,
                layer,
            )
        })
        .collect::<Vec<_>>();
    let depths = (0..5)
        .map(|layer| {
            view(
                &depth,
                TextureFormat::Depth32Float,
                TextureAspects::DEPTH,
                layer,
            )
        })
        .collect::<Vec<_>>();
    let args = d
        .create_buffer(&BufferDesc {
            label: None,
            // Four count words, two 12-byte dispatch records, then an
            // untouched eight-byte tail sentinel. The readback below proves
            // ExecuteIndirect consumes only the requested records.
            size: 48,
            usage: BufferUsage::INDIRECT
                .union(BufferUsage::STORAGE)
                .union(BufferUsage::COPY_SRC),
            memory: MemoryClass::Device,
        })
        .unwrap();
    // Core can check alignment and the advertised count limit, while the
    // native boundary owns usage and physical-range validation. Exercise both
    // native rejections from a valid mesh pass before recording the positive
    // GPU workload.
    let invalid_usage = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 48,
            usage: BufferUsage::STORAGE,
            memory: MemoryClass::Device,
        })
        .unwrap();
    for (buffer, offset, message) in [
        (
            &args,
            40,
            "DX12 must reject a record that extends beyond the argument buffer",
        ),
        (
            &invalid_usage,
            16,
            "DX12 must reject an indirect argument buffer without INDIRECT usage",
        ),
    ] {
        let mut invalid = d.encoder(lane).unwrap();
        let attachments = [Some(ColorAttachment {
            view: AttachmentView::Texture(&colors[0]),
            resolve_target: None,
            load: LoadOp::Clear(ClearColor::Float([0.0, 0.0, 0.0, 1.0])),
            store: StoreOp::Store,
        })];
        let mut pass = invalid
            .begin_raster(&RasterPassDesc {
                label: None,
                colors: &attachments,
                depth_stencil: Some(DepthStencilAttachment {
                    view: AttachmentView::Texture(&depths[0]),
                    depth: Some((LoadOp::Clear(1.0), StoreOp::Store)),
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
        let mesh = d.mesh().unwrap();
        mesh.bind_pipeline(&mut pass, &pipeline).unwrap();
        assert_eq!(
            mesh.dispatch_indirect_multi(&mut pass, buffer, offset, 1),
            Err(Error::InvalidRange),
            "{message}"
        );
        // A rejected native operation poisons this disposable recording, so
        // dropping it is the intended abort path rather than a second test
        // of pass finish.
    }
    for (count_buffer, count_offset, message) in [
        (
            &invalid_usage,
            0,
            "DX12 must reject a count buffer without INDIRECT usage",
        ),
        (
            &args,
            48,
            "DX12 must reject a count word extending beyond the buffer",
        ),
    ] {
        let mut invalid = d.encoder(lane).unwrap();
        let attachments = [Some(ColorAttachment {
            view: AttachmentView::Texture(&colors[0]),
            resolve_target: None,
            load: LoadOp::Clear(ClearColor::Float([0.0, 0.0, 0.0, 1.0])),
            store: StoreOp::Store,
        })];
        let mut pass = invalid
            .begin_raster(&RasterPassDesc {
                label: None,
                colors: &attachments,
                depth_stencil: Some(DepthStencilAttachment {
                    view: AttachmentView::Texture(&depths[0]),
                    depth: Some((LoadOp::Clear(1.0), StoreOp::Store)),
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
        let mesh = d.mesh().unwrap();
        mesh.bind_pipeline(&mut pass, &pipeline).unwrap();
        assert_eq!(
            mesh.dispatch_indirect_count(&mut pass, &args, 16, count_buffer, count_offset, 1),
            Err(Error::InvalidRange),
            "{message}"
        );
    }
    let producer_layout = d
        .create_bind_group_layout(&BindGroupLayoutDesc {
            label: None,
            entries: vec![BindGroupLayoutEntry {
                binding: 0,
                visibility: vec![ShaderStage::Compute],
                ty: BindingType::StorageBuffer {
                    dynamic_offset: false,
                    min_size: 48,
                    access: StorageAccess::ReadWrite,
                },
                count: BindingCount::One,
                partially_bound: false,
                nonuniform_indexing: false,
            }],
        })
        .unwrap();
    let producer_group = d
        .create_bind_group(&BindGroupDesc {
            label: None,
            layout: &producer_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::StorageBufferView {
                    buffer: &args,
                    range: 0..48,
                    view: StorageBufferView::Raw,
                },
            }],
        })
        .unwrap();
    let producer_shader = block_on(d.shader(ShaderDesc {
        label: None,
        format: ShaderFormat::Wgsl,
        bytes: Arc::from(&b"@group(0) @binding(0) var<storage, read_write> words: array<u32>; @compute @workgroup_size(1) fn main() { words[0] = 0u; words[1] = 2u; words[2] = 3u; words[3] = 1u; words[4] = 1u; words[5] = 1u; words[6] = 1u; words[7] = 1u; words[8] = 1u; words[9] = 1u; words[10] = 0xdecafbad; words[11] = 0x8badf00d; }"[..]),
        entries: vec![ShaderEntryPoint { name: "main".into(), stage: ShaderStage::Compute, workgroup_size: Some([1, 1, 1]) }],
        required_features: vec![],
    })).unwrap();
    let producer_pipeline = block_on(
        d.compute().unwrap().create_pipeline(ComputePipelineDesc {
            label: None,
            layout: d
                .create_pipeline_layout(&PipelineLayoutDesc {
                    label: None,
                    groups: &[&producer_layout],
                    push_constants: &[],
                })
                .unwrap(),
            compute: PipelineStage {
                module: producer_shader,
                entry: "main".into(),
                constants: vec![],
            },
            required_subgroup_size: None,
            cache: None,
        }),
    )
    .unwrap();
    let dst = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 20_528,
            usage: BufferUsage::COPY_DST.union(BufferUsage::MAP_READ),
            memory: MemoryClass::Readback,
        })
        .unwrap();
    let mut e = d.encoder(lane).unwrap();
    let shader_write = BufferUses {
        stages: StageMask::COMPUTE,
        accesses: AccessMask::SHADER_WRITE,
    };
    let indirect = BufferUses {
        stages: StageMask::INDIRECT,
        accesses: AccessMask::INDIRECT_READ,
    };
    e.transition_resources(&[
        ResourceTransition::Buffer(BufferBarrier {
            target: BufferSlice {
                buffer: &args,
                range: 0..48,
            },
            usage: StateTransition {
                from: BufferUses {
                    stages: StageMask::EMPTY,
                    accesses: AccessMask::EMPTY,
                },
                to: shader_write,
            },
            ownership: None,
        }),
        ResourceTransition::Texture(TextureBarrier {
            target: Subresources {
                texture: &color,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..5,
            },
            usage: StateTransition {
                from: TextureUses {
                    stages: StageMask::EMPTY,
                    accesses: AccessMask::EMPTY,
                },
                to: TextureUses {
                    stages: StageMask::COLOR_OUTPUT,
                    accesses: AccessMask::COLOR_WRITE,
                },
            },
            layout: Some((ImageLayout::Common, ImageLayout::ColorAttachment)),
            ownership: None,
        }),
        ResourceTransition::Texture(TextureBarrier {
            target: Subresources {
                texture: &depth,
                aspects: TextureAspects::DEPTH,
                mips: 0..1,
                layers: 0..5,
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
            layout: Some((ImageLayout::Common, ImageLayout::DepthStencilAttachment)),
            ownership: None,
        }),
    ])
    .unwrap();
    {
        let mut pass = d.compute().unwrap().begin(&mut e).unwrap();
        pass.set_pipeline(&producer_pipeline).unwrap();
        pass.set_bind_group(0, &producer_group, &[]).unwrap();
        pass.dispatch([1, 1, 1]).unwrap();
        pass.finish().unwrap();
    }
    e.transition_resources(&[ResourceTransition::Buffer(BufferBarrier {
        target: BufferSlice {
            buffer: &args,
            range: 0..48,
        },
        usage: StateTransition {
            from: shader_write,
            to: indirect,
        },
        ownership: None,
    })])
    .unwrap();
    for layer in 0..5 {
        let attachments = [Some(ColorAttachment {
            view: AttachmentView::Texture(&colors[layer]),
            resolve_target: None,
            load: LoadOp::Clear(ClearColor::Float([0.0, 0.0, 0.0, 1.0])),
            store: StoreOp::Store,
        })];
        let mut pass = e
            .begin_raster(&RasterPassDesc {
                label: None,
                colors: &attachments,
                depth_stencil: Some(DepthStencilAttachment {
                    view: AttachmentView::Texture(&depths[layer]),
                    depth: Some((LoadOp::Clear(1.0), StoreOp::Store)),
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
        pass.scissor(Rect {
            x: 0,
            y: 0,
            width: 16,
            height: 16,
        })
        .unwrap();
        let mesh = d.mesh().unwrap();
        mesh.bind_pipeline(&mut pass, &pipeline).unwrap();
        match layer {
            0 => mesh
                .dispatch_indirect_multi(&mut pass, &args, 16, 2)
                .unwrap(),
            1 => mesh
                .dispatch_indirect_count(&mut pass, &args, 16, &args, 4, 2)
                .unwrap(),
            2 => mesh
                .dispatch_indirect_count(&mut pass, &args, 16, &args, 0, 2)
                .unwrap(),
            3 => mesh
                .dispatch_indirect_count(&mut pass, &args, 16, &args, 8, 2)
                .unwrap(),
            4 => mesh
                .dispatch_indirect_count(&mut pass, &args, 16, &args, 12, 2)
                .unwrap(),
            _ => unreachable!(),
        }
        pass.finish().unwrap();
    }
    let render = TextureUses {
        stages: StageMask::COLOR_OUTPUT,
        accesses: AccessMask::COLOR_WRITE,
    };
    let copy = TextureUses {
        stages: StageMask::COPY,
        accesses: AccessMask::COPY_READ,
    };
    e.transition_resources(&[ResourceTransition::Texture(TextureBarrier {
        target: Subresources {
            texture: &color,
            aspects: TextureAspects::COLOR,
            mips: 0..1,
            layers: 0..5,
        },
        usage: StateTransition {
            from: render,
            to: copy,
        },
        layout: Some((ImageLayout::ColorAttachment, ImageLayout::CopySrc)),
        ownership: None,
    })])
    .unwrap();
    for layer in 0..5 {
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
    e.transition_resources(&[ResourceTransition::Buffer(BufferBarrier {
        target: BufferSlice {
            buffer: &args,
            range: 0..48,
        },
        usage: StateTransition {
            from: indirect,
            to: BufferUses {
                stages: StageMask::COPY,
                accesses: AccessMask::COPY_READ,
            },
        },
        ownership: None,
    })])
    .unwrap();
    e.copy_buffer(&args, 0, &dst, 20_480, 48).unwrap();
    let ticket = e.readback_buffer(&dst, 0..20_528).unwrap();
    let work = e.finish().unwrap();
    // The command encoder must retain the indirect resource through this
    // submission even after the public buffer and its producer group vanish.
    drop(args);
    drop(producer_group);
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: d.id(),
                ordinal: 7_553_222,
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
        d.queue().submit(SubmitPacket {
            plan,
            steps: vec![BoundStep {
                step,
                payload: StepPayload::Commands(vec![work]),
            }],
            completion: CompletionRequest::End,
        })
    } {
        SubmitOutcome::Accepted(r) => r.checkpoints.into_iter().next().unwrap().point,
        _ => panic!("submission rejected"),
    };
    block_on(d.wait_completion(&point)).unwrap();
    let bytes = unsafe { block_on(d.read_after(ticket, &point)).unwrap().bytes };
    let center = 8 * 256 + 8 * 4;
    for layer in [0, 1, 3] {
        assert!(
            (120..=136).contains(&bytes[layer * 4096 + center]),
            "layer {layer} must contain two 0.25-red records, got {}",
            bytes[layer * 4096 + center]
        );
    }
    assert_eq!(
        bytes[2 * 4096 + center],
        0,
        "GPU count zero must execute no mesh records"
    );
    assert!(
        (56..=72).contains(&bytes[4 * 4096 + center]),
        "GPU count below max must execute one 0.25-red record, got {}",
        bytes[4 * 4096 + center]
    );
    assert_eq!(
        &bytes[20_520..20_528],
        &[0xad, 0xfb, 0xca, 0xde, 0x0d, 0xf0, 0xad, 0x8b],
        "indirect execution must leave the argument buffer tail sentinel intact"
    );
}

fn fractional_pixel_shader(d: &Device<fluxel_rhi_dx12::Dx12>) -> Shader<fluxel_rhi_dx12::Dx12> {
    let dir =
        std::env::temp_dir().join(format!("fluxel-dx12-mesh-indirect-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let source = dir.join("mesh_pixel.hlsl");
    let output = dir.join("mesh_pixel.dxil");
    std::fs::write(
        &source,
        "float4 main() : SV_Target { return float4(0.25,0,0,1); }",
    )
    .unwrap();
    let result = std::process::Command::new("dxc")
        .args(["-T", "ps_6_5", "-E", "main", "-Fo"])
        .arg(&output)
        .arg(&source)
        .output()
        .expect("hardware test requires dxc in PATH");
    assert!(
        result.status.success(),
        "dxc: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    let bytes = std::fs::read(&output).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    block_on(d.shader(ShaderDesc {
        label: None,
        format: ShaderFormat::Dxil,
        bytes: Arc::from(bytes),
        entries: vec![ShaderEntryPoint {
            name: "main".into(),
            stage: ShaderStage::Fragment,
            workgroup_size: None,
        }],
        required_features: vec![],
    }))
    .unwrap()
}

#[test]
fn failed_storage_views_release_descriptor_reservations() {
    let d = block_on(Dx12Instance::new().unwrap().open_default()).unwrap();
    let buffer = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 2052,
            usage: BufferUsage::STORAGE,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let layout = d
        .create_bind_group_layout(&BindGroupLayoutDesc {
            label: None,
            entries: vec![BindGroupLayoutEntry {
                binding: 0,
                visibility: vec![ShaderStage::Mesh],
                ty: BindingType::StorageBuffer {
                    dynamic_offset: false,
                    min_size: 4,
                    access: StorageAccess::Read,
                },
                count: BindingCount::One,
                partially_bound: false,
                nonuniform_indexing: false,
            }],
        })
        .unwrap();
    for _ in 0..65_537 {
        let entries = [BindGroupEntry {
            binding: 0,
            resource: BindingResource::StorageBufferView {
                buffer: &buffer,
                range: 0..2052,
                view: StorageBufferView::Structured { stride: 2052 },
            },
        }];
        assert!(
            d.create_bind_group(&BindGroupDesc {
                label: None,
                layout: &layout,
                entries: &entries
            })
            .is_err()
        );
    }
    let entries = [BindGroupEntry {
        binding: 0,
        resource: BindingResource::StorageBufferView {
            buffer: &buffer,
            range: 0..2052,
            view: StorageBufferView::Raw,
        },
    }];
    assert!(
        d.create_bind_group(&BindGroupDesc {
            label: None,
            layout: &layout,
            entries: &entries
        })
        .is_ok()
    );
}
