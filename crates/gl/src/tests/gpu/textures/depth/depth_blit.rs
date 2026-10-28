//! Depth blit and comparison through the public Core path.
use super::*;

async fn depth_blit(d: &Device<Gl>, required: bool) {
    let depth_desc = TextureDesc {
        label: None,
        dimension: TextureDimension::D2,
        extent: [4, 4, 1],
        array_layers: 1,
        format: TextureFormat::Depth16Unorm,
        mip_levels: 1,
        samples: 1,
        usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC | TextureUsage::COPY_DST,
        view_formats: vec![],
    };
    if !matches!(d.texture_support(&depth_desc), Support::Supported(_)) {
        assert!(
            !required,
            "WebGL2 requires DEPTH_COMPONENT16 texture support"
        );
        eprintln!("SKIP: Depth blit render attachment unavailable");
        return;
    }
    let source = br#"
@vertex fn vs(@builtin(vertex_index) i:u32)->@builtin(position) vec4f {
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
                    name: "vs".into(),
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
        compare: CompareFunction::Always,
        fail: StencilOperation::Keep,
        depth_fail: StencilOperation::Keep,
        pass: StencilOperation::Keep,
    };
    let pipeline = d
        .raster_pipeline(RasterPipelineDesc {
            label: None,
            layout,
            vertex: PipelineStage {
                module: shader.clone(),
                entry: "vs".into(),
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
                format: TextureFormat::Depth16Unorm,
                depth_write_enabled: true,
                depth_compare: CompareFunction::Less,
                stencil_front: face,
                stencil_back: face,
                stencil_read_mask: 0,
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
    let view = |texture, format, aspects| {
        d.create_view(
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
    let dv = view(&depth, TextureFormat::Depth16Unorm, TextureAspects::DEPTH);
    let source_view = view(
        &depth_source,
        TextureFormat::Depth16Unorm,
        TextureAspects::DEPTH,
    );
    let cv = view(&color, TextureFormat::Rgba8Unorm, TextureAspects::COLOR);
    let buffer = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 192,
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
        .unwrap_or_else(|_| panic!("begin depth16 commands"));
    let encoder = run.commands();
    let source_pass = encoder
        .begin_raster(&RasterPassDesc {
            label: None,
            colors: &[],
            depth_stencil: Some(DepthStencilAttachment {
                view: AttachmentView::Texture(&source_view),
                depth: Some((LoadOp::Clear(0.25), StoreOp::Store)),
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
    source_pass.finish().unwrap();
    let destination_pass = encoder
        .begin_raster(&RasterPassDesc {
            label: None,
            colors: &[],
            depth_stencil: Some(DepthStencilAttachment {
                view: AttachmentView::Texture(&dv),
                depth: Some((LoadOp::Clear(0.75), StoreOp::Store)),
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
    destination_pass.finish().unwrap();
    encoder
        .blit(
            TextureRegion {
                texture: &depth_source,
                mip: 0,
                origin: [0, 0, 0],
                extent: [4, 4, 1],
                aspect: TextureAspects::DEPTH,
            },
            TextureRegion {
                texture: &depth,
                mip: 0,
                origin: [0, 0, 0],
                extent: [4, 4, 1],
                aspect: TextureAspects::DEPTH,
            },
            FilterMode::Nearest,
        )
        .unwrap();
    for (i, depth_load) in [LoadOp::Load, LoadOp::Clear(0.75), LoadOp::Load]
        .into_iter()
        .enumerate()
    {
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
                    view: AttachmentView::Texture(&dv),
                    depth: Some((depth_load, StoreOp::Store)),
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
        pass.set_pipeline(&pipeline).unwrap();
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
                    offset: i as u64 * 64,
                    bytes_per_row: 16,
                    rows_per_image: 4,
                },
            )
            .unwrap();
    }
    let ticket = encoder.readback_buffer(&buffer, 0..192).unwrap();
    assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
    let point = match session.finish() {
        ImmediateFinishOutcome::Issued(prefix) => {
            prefix.checkpoints.into_iter().next().unwrap().point
        }
        _ => panic!("finish depth16"),
    };
    d.wait_completion(&point).await.unwrap();
    // SAFETY: the completion point covers all three renders and readback.
    let bytes = unsafe { d.read_after(ticket, &point).await }.unwrap().bytes;
    for (i, expected) in [[0, 0, 0, 255], [255, 0, 0, 255], [0, 0, 0, 255]]
        .into_iter()
        .enumerate()
    {
        assert_eq!(&bytes[i * 64..i * 64 + 4], &expected, "Depth blit case {i}");
    }
    eprintln!("Depth blit/compare and clear/compare passed");
}

#[cfg(all(windows, feature = "native-gl-wgl"))]
#[test]
#[ignore = "requires a WGL hardware context"]
fn wgl_core_depth_blit() {
    let window = fluxel_host::Window::new(
        fluxel_host::WindowConfig::new("GL Depth blit fixture", 64, 64).unwrap(),
    )
    .unwrap();
    // SAFETY: the window outlives the adopted context and device.
    let native = unsafe { GlInstance::from_wgl_window(&window, [64, 64]) }.unwrap();
    block_on(depth_blit(&block_on(open(native)), false));
}

#[cfg(feature = "native-gles-egl")]
#[test]
#[ignore = "requires an EGL GLES hardware context"]
fn gles_core_depth_blit() {
    let native = GlInstance::gles_pbuffer([8, 8]).unwrap();
    block_on(depth_blit(&block_on(open(native)), false));
}

#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn webgl2_core_depth_blit() {
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
    depth_blit(
        &open(GlInstance::from_webgl2_canvas(canvas).unwrap()).await,
        true,
    )
    .await;
}
