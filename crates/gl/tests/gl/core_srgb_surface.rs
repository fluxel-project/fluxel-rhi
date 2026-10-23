//! WebGL surface color-space regression.
use super::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_test::wasm_bindgen_test;
use web_sys::WebGl2RenderingContext as GL;

#[wasm_bindgen_test(async)]
async fn webgl2_surface_srgb_half_gray() {
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
    let d = open(GlInstance::from_webgl2_canvas(canvas.clone()).unwrap()).await;
    let mut surface = common::adopted_surface(&d).unwrap();
    surface
        .configure(&SurfaceConfig {
            extent: [8, 8],
            format: TextureFormat::Rgba8UnormSrgb,
            present_mode: PresentMode::Fifo,
            alpha_mode: CompositeAlphaMode::Opaque,
            color_space: ColorSpace::Srgb,
            usage: SurfaceUsage::RENDER_ATTACHMENT,
            maximum_frame_latency: 1,
        })
        .await
        .unwrap();
    let image = surface.acquire().await.unwrap();
    let fixture = common::ImmediateFixture {
        device: &d,
        lane: d.lanes()[0].id,
    };
    let plan = common::single_commands_plan(&fixture, 107).unwrap();
    // SAFETY: the surface image lives until after the submitted work completes.
    let mut session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut run = session
        .begin_commands()
        .unwrap_or_else(|_| panic!("begin commands"));
    let colors = [Some(ColorAttachment {
        view: AttachmentView::Surface(&image),
        resolve_target: None,
        load: LoadOp::Clear(ClearColor::Float([0.5, 0.5, 0.5, 1.0])),
        store: StoreOp::Store,
    })];
    run.commands()
        .begin_raster(&RasterPassDesc {
            label: None,
            colors: &colors,
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
    assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
    let point = match session.finish() {
        ImmediateFinishOutcome::Issued(prefix) => {
            prefix.checkpoints.into_iter().next().unwrap().point
        }
        _ => panic!("finish"),
    };
    d.wait_completion(&point).await.unwrap();
    let gl = canvas
        .get_context("webgl2")
        .unwrap()
        .unwrap()
        .dyn_into::<GL>()
        .unwrap();
    gl.bind_framebuffer(GL::FRAMEBUFFER, None);
    let mut pixel = [0; 4];
    gl.read_pixels_with_opt_u8_array(4, 4, 1, 1, GL::RGBA, GL::UNSIGNED_BYTE, Some(&mut pixel))
        .unwrap();
    assert!(
        (186..=189).contains(&pixel[0]),
        "sRGB half gray expected 188, got {pixel:?}"
    );
}

#[wasm_bindgen_test(async)]
async fn webgl2_surface_srgb_routes_and_load() {
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
    let d = open(GlInstance::from_webgl2_canvas(canvas.clone()).unwrap()).await;
    let mut surface = common::adopted_surface(&d).unwrap();
    let format = TextureFormat::Rgba8UnormSrgb;
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
    let make_texture = |format, samples| {
        d.create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [8, 8, 1],
            array_layers: 1,
            format,
            mip_levels: 1,
            samples,
            usage: TextureUsage::RENDER_ATTACHMENT,
            view_formats: vec![],
        })
        .unwrap()
    };
    let color = make_texture(format, 1);
    let msaa = make_texture(format, 4);
    let depth = make_texture(TextureFormat::Depth24PlusStencil8, 1);
    let make_view = |texture, format, aspects| {
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
    let color_view = make_view(&color, format, TextureAspects::COLOR);
    let msaa_view = make_view(&msaa, format, TextureAspects::COLOR);
    let depth_view = make_view(
        &depth,
        TextureFormat::Depth24PlusStencil8,
        TextureAspects::DEPTH | TextureAspects::STENCIL,
    );
    let gl = canvas
        .get_context("webgl2")
        .unwrap()
        .unwrap()
        .dyn_into::<GL>()
        .unwrap();
    let fixture = common::ImmediateFixture {
        device: &d,
        lane: d.lanes()[0].id,
    };
    let plan = common::single_commands_plan(&fixture, 108).unwrap();
    // SAFETY: all attachments and the surface image remain alive through completion.
    let mut session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut run = session
        .begin_commands()
        .unwrap_or_else(|_| panic!("begin commands"));
    let e = run.commands();
    // A previous pipeline may leave alpha-to-coverage enabled. Conversion draws
    // must still publish zero-alpha colors to the opaque canvas.
    gl.enable(GL::SAMPLE_ALPHA_TO_COVERAGE);
    let full = Rect {
        x: 0,
        y: 0,
        width: 8,
        height: 8,
    };
    let partial = Rect {
        x: 2,
        y: 2,
        width: 4,
        height: 4,
    };
    let check = |x, y, expected: u8, label: &str| {
        gl.bind_framebuffer(GL::FRAMEBUFFER, None);
        let mut pixel = [0; 4];
        gl.read_pixels_with_opt_u8_array(x, y, 1, 1, GL::RGBA, GL::UNSIGNED_BYTE, Some(&mut pixel))
            .unwrap();
        assert!(
            (expected - 2..=expected + 2).contains(&pixel[0]),
            "{label}: expected {expected}, got {pixel:?}"
        );
        assert_eq!(gl.get_error(), GL::NO_ERROR);
    };
    let pass = |colors, depth_stencil, render_area| RasterPassDesc {
        label: None,
        colors,
        depth_stencil,
        render_area,
        multiview_mask: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        secondary_inheritance: None,
    };
    let surface_half = [Some(ColorAttachment {
        view: AttachmentView::Surface(&image),
        resolve_target: None,
        load: LoadOp::Clear(ClearColor::Float([0.5, 0.5, 0.5, 0.])),
        store: StoreOp::Store,
    })];
    let depth_pass = e
        .begin_raster(&pass(
            &surface_half,
            Some(DepthStencilAttachment {
                view: AttachmentView::Texture(&depth_view),
                depth: Some((LoadOp::Clear(1.), StoreOp::Store)),
                stencil: Some((LoadOp::Clear(0), StoreOp::Store)),
                depth_read_only: false,
                stencil_read_only: false,
            }),
            full,
        ))
        .unwrap();
    gl.enable(GL::SAMPLE_ALPHA_TO_COVERAGE);
    depth_pass.finish().unwrap();
    check(4, 4, 188, "surface with depth");
    let mixed = [
        Some(ColorAttachment {
            view: AttachmentView::Texture(&color_view),
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
    e.begin_raster(&pass(&mixed, None, full))
        .unwrap()
        .finish()
        .unwrap();
    check(4, 4, 188, "mixed surface load");
    let partial_clear = [Some(ColorAttachment {
        view: AttachmentView::Surface(&image),
        resolve_target: None,
        load: LoadOp::Clear(ClearColor::Float([0.25, 0.25, 0.25, 1.])),
        store: StoreOp::Store,
    })];
    e.begin_raster(&pass(&partial_clear, None, partial))
        .unwrap()
        .finish()
        .unwrap();
    check(0, 0, 188, "partial outside preserved");
    check(4, 4, 137, "partial clear");
    let resolve = [Some(ColorAttachment {
        view: AttachmentView::Texture(&msaa_view),
        resolve_target: Some(AttachmentView::Surface(&image)),
        load: LoadOp::Clear(ClearColor::Float([0.5, 0.5, 0.5, 0.])),
        store: StoreOp::Store,
    })];
    e.begin_raster(&pass(&resolve, None, partial))
        .unwrap()
        .finish()
        .unwrap();
    check(4, 4, 188, "multisample resolve surface");
    check(0, 0, 188, "resolve outside preserved");
    assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
    let point = match session.finish() {
        ImmediateFinishOutcome::Issued(prefix) => {
            prefix.checkpoints.into_iter().next().unwrap().point
        }
        _ => panic!("finish"),
    };
    d.wait_completion(&point).await.unwrap();
}
