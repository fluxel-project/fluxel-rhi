//! WebGL2 mixed MRT surface LoadOp regression.
use super::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_test::wasm_bindgen_test;
use web_sys::WebGl2RenderingContext as GL;

#[wasm_bindgen_test(async)]
async fn webgl2_core_surface_mrt_partial_load() {
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
    let native = GlInstance::from_webgl2_canvas(canvas.clone()).unwrap();
    let d = open(native).await;
    let mut surface = common::adopted_surface(&d).unwrap();
    let caps = surface.capabilities().unwrap();
    let format = caps.formats[0];
    surface
        .configure(&SurfaceConfig {
            extent: [8, 8],
            format,
            present_mode: PresentMode::Fifo,
            alpha_mode: CompositeAlphaMode::Opaque,
            color_space: ColorSpace::Srgb,
            usage: SurfaceUsage::RENDER_ATTACHMENT,
            maximum_frame_latency: 1,
        })
        .await
        .unwrap();
    let image = surface.acquire().await.unwrap();
    let source = b"struct F { @location(0) a: vec4f, @location(2) b: vec4f }; @vertex fn vs(@builtin(vertex_index) i:u32)->@builtin(position) vec4f { let p=array<vec2f,3>(vec2f(-1.,-1.),vec2f(3.,-1.),vec2f(-1.,3.)); return vec4f(p[i],0.5,1.); } @fragment fn fs()->F { var o:F; o.a=vec4f(1.,0.,0.,1.); o.b=vec4f(0.,1.,0.,1.); return o; }";
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
    let color = |format| Some(unblended_target(format));
    let pipeline = d
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
            color_targets: vec![color(TextureFormat::Rgba8Unorm), None, color(format)],
            primitive: triangle_list_state(),
            depth_stencil: None,
            multisample: single_sample_state(u64::MAX),
            multiview: None,
            cache: None,
        })
        .await
        .unwrap();
    let texture = d
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [8, 8, 1],
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
            size: 256,
            usage: BufferUsage::COPY_DST | BufferUsage::COPY_SRC,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let fixture = common::ImmediateFixture {
        device: &d,
        lane: d.lanes()[0].id,
    };
    let plan = common::single_commands_plan(&fixture, 93).unwrap();
    // SAFETY: the context, surface image and all resources remain alive through completion.
    let mut session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut run = session
        .begin_commands()
        .unwrap_or_else(|_| panic!("begin commands"));
    let e = run.commands();
    let full = [Some(ColorAttachment {
        view: AttachmentView::Surface(&image),
        resolve_target: None,
        load: LoadOp::Clear(ClearColor::Float([0., 0., 1., 1.])),
        store: StoreOp::Store,
    })];
    e.begin_raster(&RasterPassDesc {
        label: None,
        colors: &full,
        depth_stencil: None,
        render_area: Rect {
            x: 0,
            y: 0,
            width: 8,
            height: 8,
        },
        multiview_mask: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        secondary_inheritance: None,
    })
    .unwrap()
    .finish()
    .unwrap();
    let mixed = [
        Some(ColorAttachment {
            view: AttachmentView::Texture(&view),
            resolve_target: None,
            load: LoadOp::Clear(ClearColor::Float([0., 0., 0., 1.])),
            store: StoreOp::Store,
        }),
        None,
        Some(ColorAttachment {
            view: AttachmentView::Surface(&image),
            resolve_target: None,
            load: LoadOp::Load,
            store: StoreOp::Store,
        }),
    ];
    let mut pass = e
        .begin_raster(&RasterPassDesc {
            label: None,
            colors: &mixed,
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
        .unwrap();
    pass.set_pipeline(&pipeline).unwrap();
    pass.scissor(Rect {
        x: 2,
        y: 2,
        width: 4,
        height: 4,
    })
    .unwrap();
    pass.draw(0..3, 0..1).unwrap();
    pass.finish().unwrap();
    e.texture_to_buffer(
        TextureRegion {
            texture: &texture,
            mip: 0,
            origin: [0, 0, 0],
            extent: [8, 8, 1],
            aspect: TextureAspects::COLOR,
        },
        &output,
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: 32,
            rows_per_image: 8,
        },
    )
    .unwrap();
    let ticket = e.readback_buffer(&output, 0..256).unwrap();
    assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
    let point = match session.finish() {
        ImmediateFinishOutcome::Issued(prefix) => {
            prefix.checkpoints.into_iter().next().unwrap().point
        }
        _ => panic!("finish commands"),
    };
    d.wait_completion(&point).await.unwrap();
    // SAFETY: point covers the texture copy above.
    let data = unsafe { d.read_after(ticket, &point).await }.unwrap();
    assert_eq!(
        &data.bytes[4 * 32 + 4 * 4..4 * 32 + 4 * 4 + 4],
        &[255, 0, 0, 255]
    );
    let gl = canvas
        .get_context("webgl2")
        .unwrap()
        .unwrap()
        .dyn_into::<GL>()
        .unwrap();
    gl.bind_framebuffer(GL::FRAMEBUFFER, None);
    let mut center = [0; 4];
    let mut outside = [0; 4];
    gl.read_pixels_with_opt_u8_array(4, 4, 1, 1, GL::RGBA, GL::UNSIGNED_BYTE, Some(&mut center))
        .unwrap();
    gl.read_pixels_with_opt_u8_array(0, 0, 1, 1, GL::RGBA, GL::UNSIGNED_BYTE, Some(&mut outside))
        .unwrap();
    assert_eq!(center, [0, 255, 0, 255]);
    assert_eq!(outside, [0, 0, 255, 255]);
    assert_eq!(gl.get_error(), GL::NO_ERROR);
}
