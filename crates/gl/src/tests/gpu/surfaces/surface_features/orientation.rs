//! WebGL2 surface orientation readback regression.
use super::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_test::wasm_bindgen_test;
use web_sys::WebGl2RenderingContext as GL;

#[wasm_bindgen_test(async)]
async fn webgl2_core_surface_upper_lower_rows() {
    let canvas: web_sys::HtmlCanvasElement = web_sys::window()
        .unwrap()
        .document()
        .unwrap()
        .create_element("canvas")
        .unwrap()
        .dyn_into()
        .unwrap();
    canvas.set_width(8);
    canvas.set_height(8);
    let d = open(GlInstance::from_webgl2_canvas(canvas.clone()).unwrap()).await;
    let mut surface = common::adopted_surface(&d).unwrap();
    let format = surface.capabilities().unwrap().formats[0];
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
    let fixture = common::ImmediateFixture {
        device: &d,
        lane: d.lanes()[0].id,
    };
    let plan = common::single_commands_plan(&fixture, 993).unwrap();
    // SAFETY: the image and device stay alive through the recorded completion.
    let mut session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut run = session
        .begin_commands()
        .unwrap_or_else(|_| panic!("begin commands"));
    let e = run.commands();
    for (color, area) in [
        (
            [0., 0., 1., 1.],
            Rect {
                x: 0,
                y: 0,
                width: 8,
                height: 8,
            },
        ),
        (
            [1., 0., 0., 1.],
            Rect {
                x: 0,
                y: 0,
                width: 8,
                height: 2,
            },
        ),
    ] {
        let colors = [Some(ColorAttachment {
            view: AttachmentView::Surface(&image),
            resolve_target: None,
            load: LoadOp::Clear(ClearColor::Float(color)),
            store: StoreOp::Store,
        })];
        e.begin_raster(&RasterPassDesc {
            label: None,
            colors: &colors,
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
    }
    assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
    let point = match session.finish() {
        ImmediateFinishOutcome::Issued(prefix) => {
            prefix.checkpoints.into_iter().next().unwrap().point
        }
        _ => panic!("finish commands"),
    };
    d.wait_completion(&point).await.unwrap();
    let gl: GL = canvas
        .get_context("webgl2")
        .unwrap()
        .unwrap()
        .dyn_into()
        .unwrap();
    gl.bind_framebuffer(GL::FRAMEBUFFER, None);
    let mut top = [0; 4];
    let mut bottom = [0; 4];
    gl.read_pixels_with_opt_u8_array(0, 7, 1, 1, GL::RGBA, GL::UNSIGNED_BYTE, Some(&mut top))
        .unwrap();
    gl.read_pixels_with_opt_u8_array(0, 0, 1, 1, GL::RGBA, GL::UNSIGNED_BYTE, Some(&mut bottom))
        .unwrap();
    assert_eq!(gl.get_error(), GL::NO_ERROR);
    assert_eq!(top, [255, 0, 0, 255]);
    assert_eq!(bottom, [0, 0, 255, 255]);
}
