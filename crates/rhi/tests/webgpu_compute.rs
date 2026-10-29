//! Browser evidence for 06_compute.

#![cfg(all(target_arch = "wasm32", feature = "webgpu"))]

#[path = "../examples/06_compute/main.rs"]
mod compute;

#[path = "support/webgpu.rs"]
mod support;

use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test(async)]
async fn compute_draws_three_webgpu_frames() {
    let canvas = support::canvas(1280, 720);
    let result = compute::run_webgpu_compute(canvas.clone(), 3).await;
    support::remove_canvas(&canvas);
    result.expect("06 compute must draw three WebGPU frames");
}
