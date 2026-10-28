//! Depth/stencil array-layer blits and comparison through the public Core path.
use super::*;

async fn stencil_blit(d: &Device<Gl>, required: bool) {
    let depth_desc = TextureDesc {
        label: None,
        dimension: TextureDimension::D2,
        extent: [4, 4, 1],
        array_layers: 5,
        format: TextureFormat::Depth24PlusStencil8,
        mip_levels: 1,
        samples: 1,
        usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC | TextureUsage::COPY_DST,
        view_formats: vec![],
    };
    if !matches!(d.texture_support(&depth_desc), Support::Supported(_)) {
        assert!(
            !required,
            "WebGL2 requires DEPTH24_STENCIL8 texture support"
        );
        eprintln!("SKIP: Stencil blit render attachment unavailable");
        return;
    }
    let source = br#"
@vertex fn vs_quarter(@builtin(vertex_index) i:u32)->@builtin(position) vec4f {
 let p=array<vec2f,3>(vec2f(-1.,-1.),vec2f(3.,-1.),vec2f(-1.,3.)); return vec4f(p[i],0.25,1.);
}
@vertex fn vs_half(@builtin(vertex_index) i:u32)->@builtin(position) vec4f {
 let p=array<vec2f,3>(vec2f(-1.,-1.),vec2f(3.,-1.),vec2f(-1.,3.)); return vec4f(p[i],0.5,1.);
}
@fragment fn fs()->@location(0) vec4f { return vec4f(1.,0.,0.,1.); }
"#;
    let shader = d
        .shader(ShaderDesc {
            label: None,
            format: ShaderFormat::Wgsl,
            bytes: Arc::from(source.as_slice()),
            entries: vec![
                ShaderEntryPoint {
                    name: "vs_quarter".into(),
                    stage: ShaderStage::Vertex,
                    workgroup_size: None,
                },
                ShaderEntryPoint {
                    name: "vs_half".into(),
                    stage: ShaderStage::Vertex,
                    workgroup_size: None,
                },
                ShaderEntryPoint {
                    name: "fs".into(),
                    stage: ShaderStage::Fragment,
                    workgroup_size: None,
                },
            ],
            required_features: vec![],
        })
        .await
        .unwrap();
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
    let pipeline = d
        .raster_pipeline(RasterPipelineDesc {
            label: None,
            layout: layout.clone(),
            vertex: PipelineStage {
                module: shader.clone(),
                entry: "vs_quarter".into(),
                constants: vec![],
            },
            fragment: Some(PipelineStage {
                module: shader.clone(),
                entry: "fs".into(),
                constants: vec![],
            }),
            vertex_buffers: vec![],
            color_targets: vec![Some(unblended_target(TextureFormat::Rgba8Unorm))],
            primitive: triangle_list_state(),
            depth_stencil: Some(DepthStencilState {
                format: TextureFormat::Depth24PlusStencil8,
                depth_write_enabled: false,
                depth_compare: CompareFunction::Equal,
                stencil_front: face.clone(),
                stencil_back: face.clone(),
                stencil_read_mask: 255,
                stencil_write_mask: 0,
                depth_bias_constant: 0,
                depth_bias_slope_scale: 0.,
                depth_bias_clamp: 0.,
            }),
            multisample: single_sample_state(u64::MAX),
            multiview: None,
            cache: None,
        })
        .await
        .unwrap();
    let half_depth_pipeline = d
        .raster_pipeline(RasterPipelineDesc {
            label: None,
            layout,
            vertex: PipelineStage {
                module: shader.clone(),
                entry: "vs_half".into(),
                constants: vec![],
            },
            fragment: Some(PipelineStage {
                module: shader,
                entry: "fs".into(),
                constants: vec![],
            }),
            vertex_buffers: vec![],
            color_targets: vec![Some(unblended_target(TextureFormat::Rgba8Unorm))],
            primitive: triangle_list_state(),
            depth_stencil: Some(DepthStencilState {
                format: TextureFormat::Depth24PlusStencil8,
                depth_write_enabled: false,
                depth_compare: CompareFunction::Equal,
                stencil_front: face.clone(),
                stencil_back: face,
                stencil_read_mask: 255,
                stencil_write_mask: 0,
                depth_bias_constant: 0,
                depth_bias_slope_scale: 0.,
                depth_bias_clamp: 0.,
            }),
            multisample: single_sample_state(u64::MAX),
            multiview: None,
            cache: None,
        })
        .await
        .unwrap();
    let depth = d.create_texture(&depth_desc).unwrap();
    let depth_source = d.create_texture(&depth_desc).unwrap();
    let color = d
        .create_texture(&TextureDesc {
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
            ..depth_desc
        })
        .unwrap();
    let view = |texture, format, aspects, layer| {
        d.create_view(
            texture,
            &TextureViewDesc {
                format,
                dimension: TextureViewDimension::D2,
                aspects,
                mips: 0..1,
                layers: layer..layer + 1,
            },
        )
        .unwrap()
    };
    let source_view = view(
        &depth_source,
        TextureFormat::Depth24PlusStencil8,
        TextureAspects::STENCIL,
        1,
    );
    let source_view_zero = view(
        &depth_source,
        TextureFormat::Depth24PlusStencil8,
        TextureAspects::STENCIL,
        0,
    );
    let source_view_three = view(
        &depth_source,
        TextureFormat::Depth24PlusStencil8,
        TextureAspects::STENCIL,
        3,
    );
    let source_depth_view = view(
        &depth_source,
        TextureFormat::Depth24PlusStencil8,
        TextureAspects::DEPTH,
        1,
    );
    let source_depth_view_zero = view(
        &depth_source,
        TextureFormat::Depth24PlusStencil8,
        TextureAspects::DEPTH,
        0,
    );
    let source_depth_view_three = view(
        &depth_source,
        TextureFormat::Depth24PlusStencil8,
        TextureAspects::DEPTH,
        3,
    );
    let source_view_second = view(
        &depth_source,
        TextureFormat::Depth24PlusStencil8,
        TextureAspects::STENCIL,
        2,
    );
    let source_depth_view_second = view(
        &depth_source,
        TextureFormat::Depth24PlusStencil8,
        TextureAspects::DEPTH,
        2,
    );
    let dv = view(
        &depth,
        TextureFormat::Depth24PlusStencil8,
        TextureAspects::DEPTH | TextureAspects::STENCIL,
        2,
    );
    let untouched_destination_view = view(
        &depth,
        TextureFormat::Depth24PlusStencil8,
        TextureAspects::DEPTH | TextureAspects::STENCIL,
        4,
    );
    let destination_second_view = view(
        &depth,
        TextureFormat::Depth24PlusStencil8,
        TextureAspects::DEPTH | TextureAspects::STENCIL,
        3,
    );
    let cv = view(&color, TextureFormat::Rgba8Unorm, TextureAspects::COLOR, 0);
    let buffer = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 320,
            usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let fixture = common::ImmediateFixture {
        device: d,
        lane: d.lanes()[0].id,
    };
    let plan = common::single_commands_plan(&fixture, 449).unwrap();
    // SAFETY: the fixture retains all resources through completion.
    let mut session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut run = session
        .begin_commands()
        .unwrap_or_else(|_| panic!("begin depth/stencil array blit"));
    let encoder = run.commands();
    for (source_view, stencil) in [
        (&source_view_zero, 13),
        (&source_view, 7),
        (&source_view_second, 9),
        (&source_view_three, 15),
    ] {
        let pass = encoder
            .begin_raster(&RasterPassDesc {
                label: None,
                colors: &[],
                depth_stencil: Some(DepthStencilAttachment {
                    view: AttachmentView::Texture(source_view),
                    depth: None,
                    stencil: Some((LoadOp::Clear(stencil), StoreOp::Store)),
                    depth_read_only: false,
                    stencil_read_only: false,
                }),
                render_area: Rect {
                    x: 0,
                    y: 0,
                    width: 4,
                    height: 4,
                },
                multiview_mask: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                secondary_inheritance: None,
            })
            .unwrap();
        pass.finish().unwrap();
    }
    for (source_view, depth) in [
        (&source_depth_view_zero, 0.75),
        (&source_depth_view, 0.25),
        (&source_depth_view_second, 0.5),
        (&source_depth_view_three, 0.125),
    ] {
        let pass = encoder
            .begin_raster(&RasterPassDesc {
                label: None,
                colors: &[],
                depth_stencil: Some(DepthStencilAttachment {
                    view: AttachmentView::Texture(source_view),
                    depth: Some((LoadOp::Clear(depth), StoreOp::Store)),
                    stencil: None,
                    depth_read_only: false,
                    stencil_read_only: false,
                }),
                render_area: Rect {
                    x: 0,
                    y: 0,
                    width: 4,
                    height: 4,
                },
                multiview_mask: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                secondary_inheritance: None,
            })
            .unwrap();
        pass.finish().unwrap();
    }
    let destination_pass = encoder
        .begin_raster(&RasterPassDesc {
            label: None,
            colors: &[],
            depth_stencil: Some(DepthStencilAttachment {
                view: AttachmentView::Texture(&dv),
                depth: Some((LoadOp::Clear(0.5), StoreOp::Store)),
                stencil: Some((LoadOp::Clear(3), StoreOp::Store)),
                depth_read_only: false,
                stencil_read_only: false,
            }),
            render_area: Rect {
                x: 0,
                y: 0,
                width: 4,
                height: 4,
            },
            multiview_mask: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            secondary_inheritance: None,
        })
        .unwrap();
    destination_pass.finish().unwrap();
    let destination_second_pass = encoder
        .begin_raster(&RasterPassDesc {
            label: None,
            colors: &[],
            depth_stencil: Some(DepthStencilAttachment {
                view: AttachmentView::Texture(&destination_second_view),
                depth: Some((LoadOp::Clear(0.25), StoreOp::Store)),
                stencil: Some((LoadOp::Clear(3), StoreOp::Store)),
                depth_read_only: false,
                stencil_read_only: false,
            }),
            render_area: Rect {
                x: 0,
                y: 0,
                width: 4,
                height: 4,
            },
            multiview_mask: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            secondary_inheritance: None,
        })
        .unwrap();
    destination_second_pass.finish().unwrap();
    let untouched_pass = encoder
        .begin_raster(&RasterPassDesc {
            label: None,
            colors: &[],
            depth_stencil: Some(DepthStencilAttachment {
                view: AttachmentView::Texture(&untouched_destination_view),
                depth: Some((LoadOp::Clear(0.5), StoreOp::Store)),
                stencil: Some((LoadOp::Clear(5), StoreOp::Store)),
                depth_read_only: false,
                stencil_read_only: false,
            }),
            render_area: Rect {
                x: 0,
                y: 0,
                width: 4,
                height: 4,
            },
            multiview_mask: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            secondary_inheritance: None,
        })
        .unwrap();
    untouched_pass.finish().unwrap();
    encoder
        .blit(
            TextureRegion {
                texture: &depth_source,
                mip: 0,
                origin: [0, 0, 1],
                extent: [4, 4, 2],
                aspect: TextureAspects::STENCIL,
            },
            TextureRegion {
                texture: &depth,
                mip: 0,
                origin: [0, 0, 2],
                extent: [4, 4, 2],
                aspect: TextureAspects::STENCIL,
            },
            FilterMode::Nearest,
        )
        .unwrap();
    let checks = [
        (&dv, &half_depth_pipeline, 7, 0),
        (&destination_second_view, &pipeline, 9, 64),
    ];
    for (view, check_pipeline, stencil_reference, offset) in checks {
        let colors = [Some(ColorAttachment {
            view: AttachmentView::Texture(&cv),
            resolve_target: None,
            load: LoadOp::Clear(ClearColor::Float([0., 0., 0., 1.])),
            store: StoreOp::Store,
        })];
        let mut pass = encoder
            .begin_raster(&RasterPassDesc {
                label: None,
                colors: &colors,
                depth_stencil: Some(DepthStencilAttachment {
                    view: AttachmentView::Texture(view),
                    depth: Some((LoadOp::Load, StoreOp::Store)),
                    stencil: Some((LoadOp::Load, StoreOp::Store)),
                    depth_read_only: true,
                    stencil_read_only: true,
                }),
                render_area: Rect {
                    x: 0,
                    y: 0,
                    width: 4,
                    height: 4,
                },
                multiview_mask: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                secondary_inheritance: None,
            })
            .unwrap();
        pass.set_pipeline(check_pipeline).unwrap();
        pass.stencil_reference(stencil_reference).unwrap();
        pass.draw(0..3, 0..1).unwrap();
        pass.finish().unwrap();
        encoder
            .texture_to_buffer(
                TextureRegion {
                    texture: &color,
                    mip: 0,
                    origin: [0, 0, 0],
                    extent: [4, 4, 1],
                    aspect: TextureAspects::COLOR,
                },
                &buffer,
                TexelCopyBufferLayout {
                    offset,
                    bytes_per_row: 16,
                    rows_per_image: 4,
                },
            )
            .unwrap();
    }
    encoder
        .blit(
            TextureRegion {
                texture: &depth_source,
                mip: 0,
                origin: [0, 0, 1],
                extent: [4, 4, 2],
                aspect: TextureAspects::DEPTH,
            },
            TextureRegion {
                texture: &depth,
                mip: 0,
                origin: [0, 0, 2],
                extent: [4, 4, 2],
                aspect: TextureAspects::DEPTH,
            },
            FilterMode::Nearest,
        )
        .unwrap();
    let checks = [
        (&dv, &pipeline, 7, 128),
        (&destination_second_view, &half_depth_pipeline, 9, 192),
        (&untouched_destination_view, &half_depth_pipeline, 5, 256),
    ];
    for (view, check_pipeline, stencil_reference, offset) in checks {
        let colors = [Some(ColorAttachment {
            view: AttachmentView::Texture(&cv),
            resolve_target: None,
            load: LoadOp::Clear(ClearColor::Float([0., 0., 0., 1.])),
            store: StoreOp::Store,
        })];
        let mut pass = encoder
            .begin_raster(&RasterPassDesc {
                label: None,
                colors: &colors,
                depth_stencil: Some(DepthStencilAttachment {
                    view: AttachmentView::Texture(view),
                    depth: Some((LoadOp::Load, StoreOp::Store)),
                    stencil: Some((LoadOp::Load, StoreOp::Store)),
                    depth_read_only: true,
                    stencil_read_only: true,
                }),
                render_area: Rect {
                    x: 0,
                    y: 0,
                    width: 4,
                    height: 4,
                },
                multiview_mask: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                secondary_inheritance: None,
            })
            .unwrap();
        pass.set_pipeline(check_pipeline).unwrap();
        pass.stencil_reference(stencil_reference).unwrap();
        pass.draw(0..3, 0..1).unwrap();
        pass.finish().unwrap();
        encoder
            .texture_to_buffer(
                TextureRegion {
                    texture: &color,
                    mip: 0,
                    origin: [0, 0, 0],
                    extent: [4, 4, 1],
                    aspect: TextureAspects::COLOR,
                },
                &buffer,
                TexelCopyBufferLayout {
                    offset,
                    bytes_per_row: 16,
                    rows_per_image: 4,
                },
            )
            .unwrap();
    }
    let ticket = encoder.readback_buffer(&buffer, 0..320).unwrap();
    assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
    let point = match session.finish() {
        ImmediateFinishOutcome::Issued(prefix) => {
            prefix.checkpoints.into_iter().next().unwrap().point
        }
        _ => panic!("finish depth/stencil array blit"),
    };
    d.wait_completion(&point).await.unwrap();
    // SAFETY: the completion point covers all aspect blits, comparisons and readback.
    let bytes = unsafe { d.read_after(ticket, &point).await }.unwrap().bytes;
    for (case, image) in bytes.chunks_exact(64).enumerate() {
        for (pixel, actual) in image.chunks_exact(4).enumerate() {
            assert_eq!(
                actual,
                [255, 0, 0, 255],
                "Depth/stencil blit case {case}, pixel {pixel}"
            );
        }
    }
    eprintln!("Depth/stencil array-layer blit and aspect comparison passed");
}

#[cfg(all(windows, feature = "native-gl-wgl"))]
#[test]
#[ignore = "requires a WGL hardware context"]
fn wgl_core_stencil_blit() {
    let window = fluxel_host::Window::new(
        fluxel_host::WindowConfig::new("GL Stencil blit fixture", 64, 64).unwrap(),
    )
    .unwrap();
    // SAFETY: the window outlives the adopted context and device.
    let native = unsafe { GlInstance::from_wgl_window(&window, [64, 64]) }.unwrap();
    block_on(stencil_blit(&block_on(open(native)), false));
}

#[cfg(feature = "native-gles-egl")]
#[test]
#[ignore = "requires an EGL GLES hardware context"]
fn gles_core_stencil_blit() {
    let native = GlInstance::gles_pbuffer([8, 8]).unwrap();
    block_on(stencil_blit(&block_on(open(native)), false));
}

#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn webgl2_core_stencil_blit() {
    use wasm_bindgen::JsCast;
    let canvas = web_sys::window()
        .unwrap()
        .document()
        .unwrap()
        .create_element("canvas")
        .unwrap()
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .unwrap();
    canvas.set_width(8);
    canvas.set_height(8);
    stencil_blit(
        &open(GlInstance::from_webgl2_canvas(canvas).unwrap()).await,
        true,
    )
    .await;
}
