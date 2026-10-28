//! Core depth, stencil and dynamic raster output checks.
use super::*;

pub(super) async fn depth_stencil_and_dynamic(d: &Device<Gl>) {
    let source = b"@vertex fn vs(@builtin(vertex_index) i:u32)->@builtin(position) vec4f { let p=array<vec2f,3>(vec2f(-1.,-1.),vec2f(3.,-1.),vec2f(-1.,3.)); return vec4f(p[i],0.5,1.); } @fragment fn fs()->@location(0) vec4f { return vec4f(1.,0.,0.,1.); }";
    let shader = |name: &str, stage| ShaderDesc {
        label: None,
        format: ShaderFormat::Wgsl,
        bytes: Arc::from(source.as_slice()),
        entries: vec![ShaderEntryPoint {
            name: name.into(),
            stage,
            workgroup_size: None,
        }],
        required_features: vec![],
    };
    let vertex = d.shader(shader("vs", ShaderStage::Vertex)).await.unwrap();
    let fragment = d.shader(shader("fs", ShaderStage::Fragment)).await.unwrap();
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
        pass: StencilOperation::Replace,
    };
    let depth_state = DepthStencilState {
        format: TextureFormat::Depth24PlusStencil8,
        depth_write_enabled: true,
        depth_compare: CompareFunction::Always,
        stencil_front: face,
        stencil_back: face,
        stencil_read_mask: 255,
        stencil_write_mask: 255,
        depth_bias_constant: 0,
        depth_bias_slope_scale: 0.,
        depth_bias_clamp: 0.,
    };
    let descriptor = |depth, fragment_stage: bool| RasterPipelineDesc {
        label: None,
        layout: layout.clone(),
        vertex: PipelineStage {
            module: vertex.clone(),
            entry: "vs".into(),
            constants: vec![],
        },
        fragment: fragment_stage.then(|| PipelineStage {
            module: fragment.clone(),
            entry: "fs".into(),
            constants: vec![],
        }),
        vertex_buffers: vec![],
        color_targets: if fragment_stage {
            vec![Some(ColorTargetState {
                format: TextureFormat::Rgba8Unorm,
                blend: Some(BlendState {
                    color: BlendComponent {
                        src_factor: BlendFactor::Constant,
                        dst_factor: BlendFactor::Zero,
                        operation: BlendOperation::Add,
                    },
                    alpha: BlendComponent {
                        src_factor: BlendFactor::Constant,
                        dst_factor: BlendFactor::Zero,
                        operation: BlendOperation::Add,
                    },
                }),
                write_mask: 15,
            })]
        } else {
            vec![]
        },
        primitive: triangle_list_state(),
        depth_stencil: Some(depth),
        multisample: single_sample_state(u64::MAX),
        multiview: None,
        cache: None,
    };
    let always = StencilFaceState {
        compare: CompareFunction::Always,
        ..face
    };
    let mut initial_state = depth_state;
    initial_state.stencil_front = always;
    initial_state.stencil_back = always;
    initial_state.stencil_write_mask = 0;
    let depth_pipe = d
        .raster_pipeline(descriptor(initial_state, false))
        .await
        .unwrap();
    let mut test_state = depth_state;
    test_state.depth_write_enabled = false;
    test_state.depth_compare = CompareFunction::Less;
    let reject_depth = d
        .raster_pipeline(descriptor(test_state, true))
        .await
        .unwrap();
    test_state.depth_compare = CompareFunction::Always;
    let pass_depth = d
        .raster_pipeline(descriptor(test_state, true))
        .await
        .unwrap();
    let texture = |format, usage| {
        d.create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [8, 8, 1],
            array_layers: 1,
            format,
            mip_levels: 1,
            samples: 1,
            usage,
            view_formats: vec![],
        })
        .unwrap()
    };
    let depth = texture(
        TextureFormat::Depth24PlusStencil8,
        TextureUsage::RENDER_ATTACHMENT,
    );
    let color = texture(
        TextureFormat::Rgba8Unorm,
        TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
    );
    let view = |t, format, aspects| {
        d.create_view(
            t,
            &TextureViewDesc {
                format,
                dimension: TextureViewDimension::D2,
                aspects,
                mips: 0..1,
                layers: 0..1,
            },
        )
        .unwrap()
    };
    let dv = view(
        &depth,
        TextureFormat::Depth24PlusStencil8,
        TextureAspects::DEPTH | TextureAspects::STENCIL,
    );
    let cv = view(&color, TextureFormat::Rgba8Unorm, TextureAspects::COLOR);
    let outputs: Vec<_> = (0..4)
        .map(|_| {
            d.create_buffer(&BufferDesc {
                label: None,
                size: 256,
                usage: BufferUsage::COPY_DST | BufferUsage::COPY_SRC,
                memory: MemoryClass::Device,
            })
            .unwrap()
        })
        .collect();
    let fixture = common::ImmediateFixture {
        device: d,
        lane: d.lanes()[0].id,
    };
    let plan = common::single_commands_plan(&fixture, 92).unwrap();
    // SAFETY: the fixture retains every referenced object until the issued work completes.
    let mut session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut run = session
        .begin_commands()
        .unwrap_or_else(|_| panic!("begin immediate commands"));
    let e = run.commands();
    let area = Rect {
        x: 0,
        y: 0,
        width: 8,
        height: 8,
    };
    let depth_attachment = |load| DepthStencilAttachment {
        view: AttachmentView::Texture(&dv),
        depth: Some((load, StoreOp::Store)),
        stencil: Some((LoadOp::Load, StoreOp::Store)),
        depth_read_only: false,
        stencil_read_only: false,
    };
    let mut initial = depth_attachment(LoadOp::Clear(1.));
    initial.stencil = Some((LoadOp::Clear(5), StoreOp::Store));
    let mut pass = e
        .begin_raster(&RasterPassDesc {
            label: None,
            colors: &[],
            depth_stencil: Some(initial),
            render_area: area,
            multiview_mask: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            secondary_inheritance: None,
        })
        .unwrap();
    pass.set_pipeline(&depth_pipe).unwrap();
    pass.draw(0..3, 0..1).unwrap();
    pass.finish().unwrap();
    e.clear_texture(&depth, 0..1, 0..1, TextureClearValue::Depth(0.75))
        .unwrap();
    let color_attachment = |load| {
        [Some(ColorAttachment {
            view: AttachmentView::Texture(&cv),
            resolve_target: None,
            load,
            store: StoreOp::Store,
        })]
    };
    let mut tickets = Vec::new();
    for (i, pipe, reference) in [
        (0, &reject_depth, 5),
        (1, &pass_depth, 4),
        (2, &pass_depth, 5),
    ] {
        let attachments = color_attachment(LoadOp::Clear(ClearColor::Float([0., 0., 0., 1.])));
        let mut pass = e
            .begin_raster(&RasterPassDesc {
                label: None,
                colors: &attachments,
                depth_stencil: Some(depth_attachment(LoadOp::Load)),
                render_area: area,
                multiview_mask: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                secondary_inheritance: None,
            })
            .unwrap();
        pass.set_pipeline(if i == 2 { &reject_depth } else { pipe })
            .unwrap();
        pass.blend_constant([0.5, 0.5, 0.5, 1.]).unwrap();
        pass.stencil_reference(reference).unwrap();
        if i == 2 {
            pass.set_pipeline(pipe).unwrap();
        }
        pass.draw(0..3, 0..1).unwrap();
        pass.finish().unwrap();
        e.texture_to_buffer(
            TextureRegion {
                texture: &color,
                mip: 0,
                origin: [0, 0, 0],
                extent: [8, 8, 1],
                aspect: TextureAspects::COLOR,
            },
            &outputs[i],
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: 32,
                rows_per_image: 8,
            },
        )
        .unwrap();
        tickets.push(e.readback_buffer(&outputs[i], 0..256).unwrap());
    }
    let full = color_attachment(LoadOp::Clear(ClearColor::Float([0., 0., 1., 1.])));
    e.begin_raster(&RasterPassDesc {
        label: None,
        colors: &full,
        depth_stencil: None,
        render_area: area,
        multiview_mask: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        secondary_inheritance: None,
    })
    .unwrap()
    .finish()
    .unwrap();
    let partial = color_attachment(LoadOp::Clear(ClearColor::Float([0., 1., 0., 1.])));
    e.begin_raster(&RasterPassDesc {
        label: None,
        colors: &partial,
        depth_stencil: None,
        render_area: Rect {
            x: 2,
            y: 2,
            width: 4,
            height: 4,
        },
        multiview_mask: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        secondary_inheritance: None,
    })
    .unwrap()
    .finish()
    .unwrap();
    e.texture_to_buffer(
        TextureRegion {
            texture: &color,
            mip: 0,
            origin: [0, 0, 0],
            extent: [8, 8, 1],
            aspect: TextureAspects::COLOR,
        },
        &outputs[3],
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: 32,
            rows_per_image: 8,
        },
    )
    .unwrap();
    tickets.push(e.readback_buffer(&outputs[3], 0..256).unwrap());
    assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
    let point = match session.finish() {
        ImmediateFinishOutcome::Issued(prefix) => {
            prefix.checkpoints.into_iter().next().unwrap().point
        }
        _ => panic!("finish immediate commands"),
    };
    d.wait_completion(&point).await.unwrap();
    for (i, ticket) in tickets.into_iter().enumerate() {
        // SAFETY: this completion covers each copy and its ticket.
        let data = unsafe { d.read_after(ticket, &point).await }.unwrap();
        let px = &data.bytes[4 * 32 + 4 * 4..4 * 32 + 4 * 4 + 4];
        if i == 1 {
            assert_eq!(px, &[0, 0, 0, 255]);
        } else if i == 0 || i == 2 {
            assert!((127..=128).contains(&px[0]));
            assert_eq!(&px[1..], &[0, 0, 255]);
        } else {
            assert_eq!(px, &[0, 255, 0, 255]);
            assert_eq!(&data.bytes[32 + 4..32 + 8], &[0, 0, 255, 255]);
        }
    }
}
