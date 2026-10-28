//! Core rendering into one mip and array layer.
use super::*;

async fn attachment_subresource(d: &Device<Gl>) {
    let source=br#"
@vertex fn vs(@builtin(vertex_index) i:u32)->@builtin(position) vec4f { let p=array<vec2f,3>(vec2f(-1.,-1.),vec2f(3.,-1.),vec2f(-1.,3.)); return vec4f(p[i],0.5,1.); }
@fragment fn first()->@location(0) vec4f { return vec4f(1.,0.,0.,1.); }
@fragment fn second()->@location(0) vec4f { return vec4f(1.,0.,0.,1.); }
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
                    name: "first".into(),
                    stage: ShaderStage::Fragment,
                    workgroup_size: None,
                },
                ShaderEntryPoint {
                    name: "second".into(),
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
    let desc = |entry: &str, constants| RasterPipelineDesc {
        label: None,
        layout: layout.clone(),
        vertex: PipelineStage {
            module: shader.clone(),
            entry: "vs".into(),
            constants: vec![],
        },
        fragment: Some(PipelineStage {
            module: shader.clone(),
            entry: entry.into(),
            constants,
        }),
        vertex_buffers: vec![],
        color_targets: vec![Some(unblended_target(TextureFormat::Rgba8Unorm))],
        primitive: triangle_list_state(),
        depth_stencil: None,
        multisample: single_sample_state(u64::MAX),
        multiview: None,
        cache: None,
    };
    let pipeline = d.raster_pipeline(desc("first", vec![])).await.unwrap();
    let out = d
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [4, 4, 1],
            array_layers: 2,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 2,
            samples: 1,
            usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
            view_formats: vec![],
        })
        .unwrap();
    let ov = d
        .create_view(
            &out,
            &TextureViewDesc {
                format: TextureFormat::Rgba8Unorm,
                dimension: TextureViewDimension::D2,
                aspects: TextureAspects::COLOR,
                mips: 1..2,
                layers: 1..2,
            },
        )
        .unwrap();
    let buffer = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 512,
            usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let fixture = common::ImmediateFixture {
        device: d,
        lane: d.lanes()[0].id,
    };
    let plan = common::single_commands_plan(&fixture, 160).unwrap();
    // SAFETY: context and resources survive through completion.
    let mut session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut run = session
        .begin_commands()
        .unwrap_or_else(|_| panic!("begin commands"));
    let e = run.commands();
    e.clear_texture(
        &out,
        0..2,
        0..2,
        TextureClearValue::Color(ClearColor::Float([0., 0., 1., 1.])),
    )
    .unwrap();
    let colors = [Some(ColorAttachment {
        view: AttachmentView::Texture(&ov),
        resolve_target: None,
        load: LoadOp::Clear(ClearColor::Float([0., 1., 0., 1.])),
        store: StoreOp::Store,
    })];
    let mut p = e
        .begin_raster(&RasterPassDesc {
            label: None,
            colors: &colors,
            depth_stencil: None,
            render_area: Rect {
                x: 0,
                y: 0,
                width: 2,
                height: 2,
            },
            multiview_mask: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            secondary_inheritance: None,
        })
        .unwrap();
    p.set_pipeline(&pipeline).unwrap();
    p.draw(0..3, 0..1).unwrap();
    p.finish().unwrap();
    for mip in 0..2u32 {
        for layer in 0..2u32 {
            let n = 4 >> mip;
            let offset = (mip * 2 + layer) * 64;
            e.texture_to_buffer(
                TextureRegion {
                    texture: &out,
                    mip,
                    origin: [0, 0, layer],
                    extent: [n, n, 1],
                    aspect: TextureAspects::COLOR,
                },
                &buffer,
                TexelCopyBufferLayout {
                    offset: offset as u64,
                    bytes_per_row: n * 4,
                    rows_per_image: n,
                },
            )
            .unwrap();
        }
    }
    let ticket = e.readback_buffer(&buffer, 0..512).unwrap();
    assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
    let point = match session.finish() {
        ImmediateFinishOutcome::Issued(prefix) => {
            prefix.checkpoints.into_iter().next().unwrap().point
        }
        _ => panic!("finish"),
    };
    d.wait_completion(&point).await.unwrap();
    // SAFETY: point covers the readback commands.
    let bytes = unsafe { d.read_after(ticket, &point).await }.unwrap().bytes;
    let actual: Vec<Vec<u8>> = (0..4usize)
        .map(|slot| bytes[slot * 64..slot * 64 + 4].to_vec())
        .collect();
    assert_eq!(
        actual,
        vec![
            vec![0, 0, 255, 255],
            vec![0, 0, 255, 255],
            vec![0, 0, 255, 255],
            vec![255, 0, 0, 255]
        ],
        "mip/layer slots"
    );
}
#[cfg(all(windows, feature = "native-gl-wgl"))]
#[test]
#[ignore = "requires a WGL hardware context"]
fn wgl_core_attachment_subresource() {
    use fluxel_host::{Window, WindowConfig};
    let window = Window::new(WindowConfig::new("GL MRT Core fixture", 120, 64).unwrap()).unwrap();
    // SAFETY: the window outlives the GL device and all its resources on this thread.
    let native = unsafe { GlInstance::from_wgl_window(&window, [120, 64]) }.unwrap();
    let d = block_on(open(native));
    block_on(attachment_subresource(&d));
}

#[cfg(feature = "native-gles-egl")]
#[test]
#[ignore = "requires an EGL GLES hardware context"]
fn gles_core_attachment_subresource() {
    let native =
        GlInstance::gles_pbuffer([8, 8]).expect("open EGL GLES pbuffer for hardware fixture");
    let d = block_on(open(native));
    block_on(attachment_subresource(&d));
}

#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn webgl2_core_attachment_subresource() {
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
    attachment_subresource(&d).await;
}
