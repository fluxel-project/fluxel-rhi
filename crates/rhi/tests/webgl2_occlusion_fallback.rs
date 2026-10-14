//! Browser evidence for example 11's WebGL2 visual-only fallback.
#![cfg(all(target_arch = "wasm32", feature = "webgl2"))]

#[path = "webgl2_support.rs"]
mod support;

#[path = "../examples/11_occlusion_query/main.rs"]
mod example;

use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test(async)]
async fn occlusion_scene_visual_fallback_presents_three_webgl2_frames() {
    let canvas = support::canvas(1280, 720);
    example::run_webgl2_occlusion_fallback(canvas.clone(), 3)
        .await
        .expect("11 visual fallback must present three WebGL2 frames");
    support::assert_canvas_rendered(&canvas);
    support::assert_canvas_alpha_opaque(&canvas);
    canvas.remove();
}
