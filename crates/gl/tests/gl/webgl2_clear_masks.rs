//! Core WebGL2 depth-clear and raster-state regression entrypoint.
//!
//! `dynamic::depth_stencil_and_dynamic` is the shared dynamic-state fixture.
//! The regression below separately preserves the historical depth-mask sequence:
//! a disabled depth-write pipeline precedes an attachment `LoadOp::Clear(0.75)`.

use super::*;

#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
async fn clear_depth_ignores_old_mask_and_restores_pipeline(device: &Device<Gl>) {
    let source = |color: &str| {
        format!("@vertex fn vs(@builtin(vertex_index) i:u32)->@builtin(position) vec4f {{ let p=array<vec2f,3>(vec2f(-1.,-1.),vec2f(3.,-1.),vec2f(-1.,3.)); return vec4f(p[i],0.3,1.); }} @fragment fn fs()->@location(0) vec4f {{ return vec4f({color}); }}").into_bytes()
    };
    let shader = |bytes: Vec<u8>, name: &str, stage: ShaderStage| ShaderDesc {
        label: None,
        format: ShaderFormat::Wgsl,
        bytes: Arc::from(bytes),
        entries: vec![ShaderEntryPoint {
            name: name.into(),
            stage,
            workgroup_size: None,
        }],
        required_features: vec![],
    };
    let make_pipeline = |color: String| async move {
        let module = device
            .shader(shader(source(&color), "vs", ShaderStage::Vertex))
            .await
            .unwrap();
        let fragment = device
            .shader(shader(source(&color), "fs", ShaderStage::Fragment))
            .await
            .unwrap();
        let layout = device
            .create_pipeline_layout(&PipelineLayoutDesc {
                label: None,
                groups: &[],
                push_constants: &[],
            })
            .unwrap();
        device
            .raster_pipeline(RasterPipelineDesc {
                label: None,
                layout,
                vertex: PipelineStage {
                    module,
                    entry: "vs".into(),
                    constants: vec![],
                },
                fragment: Some(PipelineStage {
                    module: fragment,
                    entry: "fs".into(),
                    constants: vec![],
                }),
                vertex_buffers: vec![],
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
                    depth_compare: CompareFunction::Less,
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
                    depth_bias_slope_scale: 0.,
                    depth_bias_clamp: 0.,
                }),
                multisample: MultisampleState {
                    count: 1,
                    mask: u64::MAX,
                    alpha_to_coverage_enabled: false,
                },
                multiview: None,
                cache: None,
            })
            .await
            .unwrap()
    };
    let red = make_pipeline("1.,0.,0.,1.".into()).await;
    let green = make_pipeline("0.,1.,0.,1.".into()).await;
    let texture = |format, usage| {
        device
            .create_texture(&TextureDesc {
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
    let view = |texture, format, aspects| {
        device
            .create_view(
                texture,
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
    let depth_view = view(
        &depth,
        TextureFormat::Depth24PlusStencil8,
        TextureAspects::DEPTH | TextureAspects::STENCIL,
    );
    let color_view = view(&color, TextureFormat::Rgba8Unorm, TextureAspects::COLOR);
    let outputs: Vec<_> = (0..2)
        .map(|_| {
            device
                .create_buffer(&BufferDesc {
                    label: None,
                    size: 256,
                    usage: BufferUsage::COPY_DST | BufferUsage::COPY_SRC,
                    memory: MemoryClass::Device,
                })
                .unwrap()
        })
        .collect();
    let fixture = common::ImmediateFixture {
        device,
        lane: device.lanes()[0].id,
    };
    let plan = common::single_commands_plan(&fixture, 93).unwrap();
    // SAFETY: the fixture retains every referenced object until issued work completes.
    let mut session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut run = session
        .begin_commands()
        .unwrap_or_else(|_| panic!("begin immediate commands"));
    let commands = run.commands();
    let area = Rect {
        x: 0,
        y: 0,
        width: 8,
        height: 8,
    };
    let depth_attachment = |load| DepthStencilAttachment {
        view: AttachmentView::Texture(&depth_view),
        depth: Some((load, StoreOp::Store)),
        stencil: Some((LoadOp::Clear(0), StoreOp::Store)),
        depth_read_only: false,
        stencil_read_only: false,
    };
    let colors = [Some(ColorAttachment {
        view: AttachmentView::Texture(&color_view),
        resolve_target: None,
        load: LoadOp::Clear(ClearColor::Float([0., 0., 0., 1.])),
        store: StoreOp::Store,
    })];
    let desc = |colors, depth_stencil| RasterPassDesc {
        label: None,
        colors,
        depth_stencil,
        render_area: area,
        multiview_mask: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        secondary_inheritance: None,
    };
    let mut disabled = commands
        .begin_raster(&desc(&colors, Some(depth_attachment(LoadOp::Clear(0.25)))))
        .unwrap();
    disabled.set_pipeline(&red).unwrap();
    // The z=0.3 triangle fails against the initial 0.25 clear, while still
    // committing the disabled depth-write pipeline state.
    disabled.draw(0..3, 0..1).unwrap();
    disabled.finish().unwrap();
    let mut red_pass = commands
        .begin_raster(&desc(&colors, Some(depth_attachment(LoadOp::Clear(0.75)))))
        .unwrap();
    red_pass.set_pipeline(&red).unwrap();
    red_pass.draw(0..3, 0..1).unwrap();
    red_pass.finish().unwrap();
    commands
        .texture_to_buffer(
            TextureRegion {
                texture: &color,
                mip: 0,
                origin: [0, 0, 0],
                extent: [8, 8, 1],
                aspect: TextureAspects::COLOR,
            },
            &outputs[0],
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: 32,
                rows_per_image: 8,
            },
        )
        .unwrap();
    let first = commands.readback_buffer(&outputs[0], 0..256).unwrap();
    let mut green_pass = commands
        .begin_raster(&desc(&colors, Some(depth_attachment(LoadOp::Load))))
        .unwrap();
    green_pass.set_pipeline(&green).unwrap();
    green_pass.draw(0..3, 0..1).unwrap();
    green_pass.finish().unwrap();
    commands
        .texture_to_buffer(
            TextureRegion {
                texture: &color,
                mip: 0,
                origin: [0, 0, 0],
                extent: [8, 8, 1],
                aspect: TextureAspects::COLOR,
            },
            &outputs[1],
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: 32,
                rows_per_image: 8,
            },
        )
        .unwrap();
    let second = commands.readback_buffer(&outputs[1], 0..256).unwrap();
    assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
    let point = match session.finish() {
        ImmediateFinishOutcome::Issued(prefix) => {
            prefix.checkpoints.into_iter().next().unwrap().point
        }
        _ => panic!("finish immediate commands"),
    };
    device.wait_completion(&point).await.unwrap();
    // Red proves the clear overwrote 0.25 despite the old disabled mask. Green
    // at the same z proves the clear restored that disabled mask before red drew.
    for (ticket, expected) in [(first, [255, 0, 0, 255]), (second, [0, 255, 0, 255])] {
        // SAFETY: `point` covers this readback ticket.
        let data = unsafe { device.read_after(ticket, &point).await }.unwrap();
        assert_eq!(&data.bytes[4 * 32 + 4 * 4..4 * 32 + 4 * 4 + 4], expected);
    }
}

#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
pub(super) async fn run(device: &Device<Gl>) {
    clear_depth_ignores_old_mask_and_restores_pipeline(device).await;
    dynamic::depth_stencil_and_dynamic(device).await;
}
