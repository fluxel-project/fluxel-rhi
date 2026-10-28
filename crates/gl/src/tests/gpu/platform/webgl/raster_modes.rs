//! WebGL extension rasterization reaches observable Core output.
use super::*;

async fn draw_mode(d: &Device<Gl>, line: bool, clamp: bool) -> Option<Vec<u8>> {
    let z = if line { "0.5" } else { "-0.5" };
    let text = format!(
        "@vertex fn vs(@builtin(vertex_index) i:u32)->@builtin(position) vec4f {{ let p=array<vec2f,3>(vec2f(-0.8,-0.8),vec2f(0.8,-0.8),vec2f(0.,0.8)); return vec4f(p[i],{z},1.); }} @fragment fn fs()->@location(0) vec4f {{ return vec4f(1.,0.,0.,1.); }}"
    );
    let shader = |name: &str, stage| ShaderDesc {
        label: None,
        format: ShaderFormat::Wgsl,
        bytes: Arc::from(text.as_bytes()),
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
    let pipeline = match d
        .raster_pipeline(RasterPipelineDesc {
            label: None,
            layout,
            vertex: PipelineStage {
                module: vertex,
                entry: "vs".into(),
                constants: vec![],
            },
            fragment: Some(PipelineStage {
                module: fragment,
                entry: "fs".into(),
                constants: vec![],
            }),
            vertex_buffers: vec![],
            color_targets: vec![Some(unblended_target(TextureFormat::Rgba8Unorm))],
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: FrontFace::Ccw,
                cull_mode: CullMode::None,
                unclipped_depth: clamp,
                polygon_mode: if line {
                    PolygonMode::Line
                } else {
                    PolygonMode::Fill
                },
                conservative: false,
            },
            depth_stencil: None,
            multisample: single_sample_state(u64::MAX),
            multiview: None,
            cache: None,
        })
        .await
    {
        Ok(p) => p,
        Err(Error::Unsupported(reason)) => {
            web_sys::console::log_1(&format!("SKIP {line}: {reason}").into());
            return None;
        }
        Err(e) => panic!("pipeline {line}: {e:?}"),
    };
    let texture = d
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [16, 16, 1],
            array_layers: 1,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
            view_formats: vec![],
        })
        .unwrap();
    let view = d
        .create_view(
            &texture,
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
            size: 1024,
            usage: BufferUsage::COPY_DST | BufferUsage::COPY_SRC,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let fixture = common::ImmediateFixture {
        device: d,
        lane: d.lanes()[0].id,
    };
    let plan = common::single_commands_plan(&fixture, if line { 940 } else { 941 }).unwrap();
    // SAFETY: all resources remain live through completion.
    let mut session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut run = session.begin_commands().unwrap_or_else(|_| panic!("begin"));
    let e = run.commands();
    let colors = [Some(ColorAttachment {
        view: AttachmentView::Texture(&view),
        resolve_target: None,
        load: LoadOp::Clear(ClearColor::Float([0., 0., 0., 1.])),
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
                width: 16,
                height: 16,
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
            texture: &texture,
            mip: 0,
            origin: [0, 0, 0],
            extent: [16, 16, 1],
            aspect: TextureAspects::COLOR,
        },
        &output,
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: 64,
            rows_per_image: 16,
        },
    )
    .unwrap();
    let ticket = e.readback_buffer(&output, 0..1024).unwrap();
    assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
    let point = match session.finish() {
        ImmediateFinishOutcome::Issued(prefix) => {
            prefix.checkpoints.into_iter().next().unwrap().point
        }
        _ => panic!("finish"),
    };
    d.wait_completion(&point).await.unwrap();
    // SAFETY: completion covers the readback.
    Some(unsafe { d.read_after(ticket, &point).await }.unwrap().bytes)
}

#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn webgl_raster_modes_pixels() {
    use wasm_bindgen::JsCast;
    let canvas = web_sys::window()
        .unwrap()
        .document()
        .unwrap()
        .create_element("canvas")
        .unwrap()
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .unwrap();
    canvas.set_width(16);
    canvas.set_height(16);
    let d = open(GlInstance::from_webgl2_canvas(canvas).unwrap()).await;
    if let Some(pixels) = draw_mode(&d, true, false).await {
        let red = pixels.chunks_exact(4).filter(|p| p[0] > 200).count();
        assert!(
            red > 0 && red < 70,
            "line mode expected border only, red pixels={red}"
        );
        assert_eq!(
            &pixels[(8 * 16 + 8) * 4..(8 * 16 + 8) * 4 + 4],
            &[0, 0, 0, 255]
        );
        web_sys::console::log_1(&"EXECUTED polygon line".into());
    }
    if let Some(pixels) = draw_mode(&d, false, true).await {
        assert_eq!(
            &pixels[(8 * 16 + 8) * 4..(8 * 16 + 8) * 4 + 4],
            &[255, 0, 0, 255],
            "negative clip depth must be clamped"
        );
        web_sys::console::log_1(&"EXECUTED depth clamp".into());
        let reset = draw_mode(&d, false, false)
            .await
            .expect("default rasterization");
        assert_eq!(
            &reset[(8 * 16 + 8) * 4..(8 * 16 + 8) * 4 + 4],
            &[0, 0, 0, 255],
            "depth clamp must reset on pipeline switch"
        );
    }
}
