//! Browser proof of the offscreen render and completed GPU readback path.
#![cfg(all(target_arch = "wasm32", feature = "webgpu"))]

#[path = "../examples/12_screenshot/main.rs"]
mod screenshot;

#[path = "support/webgpu.rs"]
mod support;

use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test(async)]
async fn screenshot_renders_reads_back_and_presents_three_webgpu_frames() {
    let canvas = support::canvas(640, 480);
    let result = screenshot::run_webgpu_screenshot(canvas.clone(), 3).await;
    support::remove_canvas(&canvas);
    result.expect("12 must render, read back, and present three WebGPU frames");
}
