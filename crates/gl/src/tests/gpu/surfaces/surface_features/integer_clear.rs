//! WebGL2 integer attachment clear regression.
use super::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_test::wasm_bindgen_test;
use web_sys::WebGl2RenderingContext as GL;

#[wasm_bindgen_test(async)]
async fn webgl2_core_integer_attachment_clears() {
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
    let gl = canvas
        .get_context("webgl2")
        .unwrap()
        .unwrap()
        .dyn_into::<GL>()
        .unwrap();
    for (index, format, value) in [
        (
            0,
            TextureFormat::Rgba8Uint,
            ClearColor::Uint([3, 7, 19, 251]),
        ),
        (
            1,
            TextureFormat::Rgba8Sint,
            ClearColor::Sint([-3, 7, -19, 101]),
        ),
    ] {
        let texture = d
            .create_texture(&TextureDesc {
                label: None,
                dimension: TextureDimension::D2,
                extent: [8, 8, 1],
                array_layers: 1,
                format,
                mip_levels: 1,
                samples: 1,
                usage: TextureUsage::RENDER_ATTACHMENT,
                view_formats: vec![],
            })
            .unwrap();
        let view = d
            .create_view(
                &texture,
                &TextureViewDesc {
                    format,
                    dimension: TextureViewDimension::D2,
                    aspects: TextureAspects::COLOR,
                    mips: 0..1,
                    layers: 0..1,
                },
            )
            .unwrap();
        let fixture = common::ImmediateFixture {
            device: &d,
            lane: d.lanes()[0].id,
        };
        let plan = common::single_commands_plan(&fixture, 95 + index).unwrap();
        // SAFETY: context and attachment stay alive until session completion.
        let mut session =
            unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
        let mut run = session
            .begin_commands()
            .unwrap_or_else(|_| panic!("begin commands"));
        let colors = [Some(ColorAttachment {
            view: AttachmentView::Texture(&view),
            resolve_target: None,
            load: LoadOp::Clear(value),
            store: StoreOp::Store,
        })];
        let pass = run
            .commands()
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
            .unwrap();
        if index == 0 {
            let pixels = js_sys::Uint32Array::new_with_length(4);
            gl.read_pixels_with_opt_array_buffer_view(
                4,
                4,
                1,
                1,
                GL::RGBA_INTEGER,
                GL::UNSIGNED_INT,
                Some(pixels.as_ref()),
            )
            .unwrap();
            assert_eq!(pixels.to_vec(), vec![3, 7, 19, 251]);
        } else {
            let pixels = js_sys::Int32Array::new_with_length(4);
            gl.read_pixels_with_opt_array_buffer_view(
                4,
                4,
                1,
                1,
                GL::RGBA_INTEGER,
                GL::INT,
                Some(pixels.as_ref()),
            )
            .unwrap();
            assert_eq!(pixels.to_vec(), vec![-3, 7, -19, 101]);
        }
        assert_eq!(gl.get_error(), GL::NO_ERROR);
        pass.finish().unwrap();
        assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
        let point = match session.finish() {
            ImmediateFinishOutcome::Issued(prefix) => {
                prefix.checkpoints.into_iter().next().unwrap().point
            }
            _ => panic!("finish commands"),
        };
        d.wait_completion(&point).await.unwrap();
    }
}
