//! WebGL2 depth publish and scissor regression.
use super::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_test::wasm_bindgen_test;
use web_sys::WebGl2RenderingContext as GL;

#[wasm_bindgen_test(async)]
async fn webgl2_core_surface_depth_publish_ignores_scissor() {
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
    let format = TextureFormat::Rgba8Unorm;
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
    let depth = d
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [8, 8, 1],
            array_layers: 1,
            format: TextureFormat::Depth24PlusStencil8,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::RENDER_ATTACHMENT,
            view_formats: vec![],
        })
        .unwrap();
    let view = d
        .create_view(
            &depth,
            &TextureViewDesc {
                format: TextureFormat::Depth24PlusStencil8,
                dimension: TextureViewDimension::D2,
                aspects: TextureAspects::DEPTH | TextureAspects::STENCIL,
                mips: 0..1,
                layers: 0..1,
            },
        )
        .unwrap();
    let fixture = common::ImmediateFixture {
        device: &d,
        lane: d.lanes()[0].id,
    };
    let plan = common::single_commands_plan(&fixture, 97).unwrap();
    // SAFETY: device, image and depth attachment survive until the completion checkpoint.
    let mut session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut run = session
        .begin_commands()
        .unwrap_or_else(|_| panic!("begin commands"));
    let colors = [Some(ColorAttachment {
        view: AttachmentView::Surface(&image),
        resolve_target: None,
        load: LoadOp::Clear(ClearColor::Float([1., 0., 0., 1.])),
        store: StoreOp::Store,
    })];
    let mut pass = run
        .commands()
        .begin_raster(&RasterPassDesc {
            label: None,
            colors: &colors,
            depth_stencil: Some(DepthStencilAttachment {
                view: AttachmentView::Texture(&view),
                depth: Some((LoadOp::Clear(1.), StoreOp::Store)),
                stencil: Some((LoadOp::Clear(0), StoreOp::Store)),
                depth_read_only: false,
                stencil_read_only: false,
            }),
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
        .unwrap();
    pass.scissor(Rect {
        x: 2,
        y: 2,
        width: 2,
        height: 2,
    })
    .unwrap();
    pass.finish().unwrap();
    assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
    let point = match session.finish() {
        ImmediateFinishOutcome::Issued(prefix) => {
            prefix.checkpoints.into_iter().next().unwrap().point
        }
        _ => panic!("finish commands"),
    };
    d.wait_completion(&point).await.unwrap();
    let gl = canvas
        .get_context("webgl2")
        .unwrap()
        .unwrap()
        .dyn_into::<GL>()
        .unwrap();
    gl.bind_framebuffer(GL::FRAMEBUFFER, None);
    let mut outside = [0; 4];
    gl.read_pixels_with_opt_u8_array(0, 0, 1, 1, GL::RGBA, GL::UNSIGNED_BYTE, Some(&mut outside))
        .unwrap();
    assert_eq!(outside, [255, 0, 0, 255]);
    assert_eq!(gl.get_error(), GL::NO_ERROR);
}
