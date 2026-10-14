//! Browser evidence for the WebGL2 sequential per-eye stereo fallback.
#![cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[path = "webgl2_support.rs"]
mod support;

#[path = "../examples/15_multiview/main.rs"]
mod example;

use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test(async)]
async fn per_eye_fallback_draws_three_webgl2_frames() {
    let canvas = support::canvas(1280, 720);
    let result = example::run_webgl2_multiview(canvas.clone(), 3).await;
    result.expect("15 per-eye fallback must draw three WebGL2 frames");
    support::assert_canvas_rendered(&canvas);
    canvas.remove();
}
