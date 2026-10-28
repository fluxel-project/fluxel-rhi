//! Dual-source RGB and alpha blend factors on an actual color attachment.
use super::*;

async fn dual_source_blend(d: &Device<Gl>, allow_unavailable: bool) {
    let source = b"enable dual_source_blending; struct Out { @location(0) @blend_src(0) a:vec4f, @location(0) @blend_src(1) b:vec4f, } @vertex fn vs(@builtin(vertex_index) i:u32)->@builtin(position) vec4f { let p=array<vec2f,3>(vec2f(-1.,-1.),vec2f(3.,-1.),vec2f(-1.,3.)); return vec4f(p[i],0.5,1.); } @fragment fn fs()->Out { return Out(vec4f(0.8,0.4,0.2,0.7),vec4f(0.25,0.5,0.75,0.2)); }";
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
    let component = BlendComponent {
        src_factor: BlendFactor::Src1,
        dst_factor: BlendFactor::OneMinusSrc1,
        operation: BlendOperation::Add,
    };
    let descriptor = || RasterPipelineDesc {
        label: None,
        layout: layout.clone(),
        vertex: PipelineStage {
            module: vertex.clone(),
            entry: "vs".into(),
            constants: vec![],
        },
        fragment: Some(PipelineStage {
            module: fragment.clone(),
            entry: "fs".into(),
            constants: vec![],
        }),
        vertex_buffers: vec![],
        color_targets: vec![Some(ColorTargetState {
            format: TextureFormat::Rgba8Unorm,
            blend: Some(BlendState {
                color: component,
                alpha: component,
            }),
            write_mask: 15,
        })],
        primitive: triangle_list_state(),
        depth_stencil: None,
        multisample: single_sample_state(u64::MAX),
        multiview: None,
        cache: None,
    };
    let mut invalid = descriptor();
    invalid.color_targets.push(None);
    let pipeline = match d.raster_pipeline(descriptor()).await {
        Ok(p) => p,
        Err(Error::Unsupported(reason)) if allow_unavailable => {
            eprintln!("SKIP dual source: {reason}");
            #[cfg(target_arch = "wasm32")]
            web_sys::console::log_1(&format!("SKIP dual source: {reason}").into());
            return;
        }
        Err(e) => panic!("pipeline: {e:?}"),
    };
    assert!(
        d.raster_pipeline(invalid).await.is_err(),
        "dual source with multiple targets must be rejected"
    );
    let color = d
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [4, 4, 1],
            array_layers: 1,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
            view_formats: vec![],
        })
        .unwrap();
    let cv = d
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
    let output = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 64,
            usage: BufferUsage::COPY_DST | BufferUsage::COPY_SRC,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let fixture = common::ImmediateFixture {
        device: d,
        lane: d.lanes()[0].id,
    };
    let plan = common::single_commands_plan(&fixture, 320).unwrap();
    // SAFETY: resources remain alive through completion.
    let mut session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut run = session.begin_commands().unwrap_or_else(|_| panic!("begin"));
    let e = run.commands();
    let colors = [Some(ColorAttachment {
        view: AttachmentView::Texture(&cv),
        resolve_target: None,
        load: LoadOp::Clear(ClearColor::Float([0.4, 0.4, 0.4, 0.4])),
        store: StoreOp::Store,
    })];
    let mut pass = e
        .begin_raster(&RasterPassDesc {
            label: None,
            colors: &colors,
            depth_stencil: None,
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
    e.texture_to_buffer(
        TextureRegion {
            texture: &color,
            mip: 0,
            origin: [0, 0, 0],
            extent: [4, 4, 1],
            aspect: TextureAspects::COLOR,
        },
        &output,
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: 16,
            rows_per_image: 4,
        },
    )
    .unwrap();
    let ticket = e.readback_buffer(&output, 0..64).unwrap();
    assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
    let point = match session.finish() {
        ImmediateFinishOutcome::Issued(prefix) => {
            prefix.checkpoints.into_iter().next().unwrap().point
        }
        _ => panic!("finish"),
    };
    d.wait_completion(&point).await.unwrap();
    // SAFETY: completion covers the copy and readback.
    let pixels = unsafe { d.read_after(ticket, &point).await }.unwrap().bytes;
    let actual = &pixels[..4];
    eprintln!("dual source pixel: {actual:?}");
    for (&actual, expected) in actual.iter().zip([128i32, 102, 64, 117]) {
        assert!(
            (i32::from(actual) - expected).abs() <= 1,
            "actual {pixels:?}"
        );
    }
    #[cfg(target_arch = "wasm32")]
    web_sys::console::log_1(&format!("EXECUTED dual source: {actual:?}").into());
}
#[cfg(all(windows, feature = "native-gl-wgl"))]
#[test]
#[ignore = "requires a WGL hardware context"]
fn wgl_core_dual_source_blend() {
    use fluxel_host::{Window, WindowConfig};
    let window =
        Window::new(WindowConfig::new("GL raster feature fixture", 1024, 768).unwrap()).unwrap();
    // SAFETY: the window outlives the GL device and all its resources on this thread.
    let native = unsafe { GlInstance::from_wgl_window(&window, [1024, 768]) }.unwrap();
    let d = block_on(open(native));
    block_on(dual_source_blend(&d, false));
}

#[cfg(feature = "native-gles-egl")]
#[test]
#[ignore = "requires an EGL GLES hardware context"]
fn gles_core_dual_source_blend() {
    let native =
        GlInstance::gles_pbuffer([8, 8]).expect("open EGL GLES pbuffer for hardware fixture");
    let d = block_on(open(native));
    block_on(dual_source_blend(&d, true));
}

#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn webgl2_core_dual_source_blend() {
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
    let d = open(GlInstance::from_webgl2_canvas(canvas).unwrap()).await;
    dual_source_blend(&d, true).await;
}
