//! Browser evidence for 07_offscreen.

#![cfg(all(target_arch = "wasm32", feature = "webgpu"))]

#[path = "../examples/07_offscreen/main.rs"]
mod example;

#[path = "support/webgpu.rs"]
mod support;

use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test(async)]
async fn offscreen_draws_three_webgpu_frames() {
    let canvas = support::canvas(1280, 720);
    let result = example::run_webgpu_offscreen(canvas.clone(), 3).await;
    support::remove_canvas(&canvas);
    result.expect("07 offscreen must draw three WebGPU frames");
}
