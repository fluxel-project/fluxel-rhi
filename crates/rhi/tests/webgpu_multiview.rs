//! Browser evidence for the sequential WebGPU stereo fallback of example 15.

#![cfg(all(target_arch = "wasm32", feature = "webgpu"))]

#[path = "../examples/15_multiview/main.rs"]
mod multiview;

#[path = "support/webgpu.rs"]
mod support;

use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test(async)]
async fn multiview_fallback_draws_three_webgpu_frames() {
    let canvas = support::canvas(1280, 720);
    let result = multiview::run_webgpu_multiview(canvas.clone(), 3).await;
    support::remove_canvas(&canvas);
    result.expect("15 must present three sequential WebGPU stereo frames");
}
