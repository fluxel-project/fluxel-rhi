//! Browser evidence for the WebGL2 single-sample visual fallback of example 08.
#![cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[path = "webgl2_support.rs"]
mod support;

#[path = "../examples/08_msaa/main.rs"]
mod example;

use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test(async)]
async fn msaa_scene_fallback_draws_and_presents_three_webgl2_frames() {
    let canvas = support::canvas(1280, 720);
    let result = example::run_webgl2_msaa_fallback(canvas.clone(), 3).await;
    result.expect("08 single-sample visual fallback must present three WebGL2 frames");
    support::assert_canvas_rendered(&canvas);
    canvas.remove();
}
