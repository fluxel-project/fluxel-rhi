//! Shared browser setup and drawing-buffer assertions for WebGL2 examples.
#![cfg(all(target_arch = "wasm32", feature = "webgl2"))]

use wasm_bindgen::JsCast;

pub fn canvas(width: u32, height: u32) -> web_sys::HtmlCanvasElement {
    let document = web_sys::window().unwrap().document().unwrap();
    let canvas = document
        .create_element("canvas")
        .unwrap()
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .unwrap();
    canvas.set_width(width);
    canvas.set_height(height);
    let attributes = web_sys::WebGlContextAttributes::new();
    attributes.set_alpha(false);
    attributes.set_antialias(false);
    attributes.set_preserve_drawing_buffer(true);
    canvas
        .get_context_with_context_options("webgl2", attributes.as_ref())
        .unwrap()
        .expect("WebGL2 context");
    document.body().unwrap().append_child(&canvas).unwrap();
    canvas
}

/// Read back the real browser drawing buffer and reject a uniform/blank frame.
pub fn assert_canvas_rendered(canvas: &web_sys::HtmlCanvasElement) {
    let context = canvas
        .get_context("webgl2")
        .unwrap()
        .unwrap()
        .dyn_into::<web_sys::WebGl2RenderingContext>()
        .unwrap();
    let mut pixels = vec![0; canvas.width() as usize * canvas.height() as usize * 4];
    context
        .read_pixels_with_opt_u8_array(
            0,
            0,
            canvas.width() as i32,
            canvas.height() as i32,
            web_sys::WebGl2RenderingContext::RGBA,
            web_sys::WebGl2RenderingContext::UNSIGNED_BYTE,
            Some(&mut pixels),
        )
        .expect("WebGL2 drawing-buffer readback");
    let background = &pixels[..4];
    let changed = pixels
        .chunks_exact(4)
        .filter(|pixel| *pixel != background)
        .count();
    assert!(
        changed > 64,
        "WebGL2 drawing buffer is blank: changed={changed}, first={background:?}, center={:?}",
        &pixels[((canvas.height() / 2 * canvas.width() + canvas.width() / 2) * 4) as usize..][..4]
    );
}

/// Checks the opaque surface contract at all four corners after presentation.
pub fn assert_canvas_alpha_opaque(canvas: &web_sys::HtmlCanvasElement) {
    let context = canvas
        .get_context("webgl2")
        .unwrap()
        .unwrap()
        .dyn_into::<web_sys::WebGl2RenderingContext>()
        .unwrap();
    let points = [
        (0, 0),
        (canvas.width() as i32 - 1, 0),
        (0, canvas.height() as i32 - 1),
        (canvas.width() as i32 - 1, canvas.height() as i32 - 1),
    ];
    for (x, y) in points {
        let mut pixel = [0; 4];
        context
            .read_pixels_with_opt_u8_array(
                x,
                y,
                1,
                1,
                web_sys::WebGl2RenderingContext::RGBA,
                web_sys::WebGl2RenderingContext::UNSIGNED_BYTE,
                Some(&mut pixel),
            )
            .expect("WebGL2 opaque corner readback");
        assert_eq!(pixel[3], 255, "surface corner ({x}, {y}) is not opaque");
    }
}
