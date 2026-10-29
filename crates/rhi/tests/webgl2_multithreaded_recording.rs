//! Browser evidence for the WebGL2 sequential primary-pass fallback.
#![cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[path = "support/webgl2.rs"]
mod support;

#[path = "../examples/13_multithreaded_recording/main.rs"]
mod example;

use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test(async)]
async fn sequential_scene_draws_and_presents_three_webgl2_frames() {
    let canvas = support::canvas(1280, 720);
    let result = example::run_webgl2_multithreaded_recording(canvas.clone(), 3).await;
    result.expect("13 sequential fallback must draw three WebGL2 frames");
    support::assert_canvas_rendered(&canvas);
    canvas.remove();
}
