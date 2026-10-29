//! Browser evidence for the WebGPU sequential-primary fallback of example 13.
#![cfg(all(target_arch = "wasm32", feature = "webgpu"))]
#[path = "../examples/13_multithreaded_recording/main.rs"]
mod example;
#[path = "support/webgpu.rs"]
mod support;

use wasm_bindgen_test::*;
wasm_bindgen_test_configure!(run_in_browser);
#[wasm_bindgen_test(async)]
async fn sequential_primary_pass_draws_three_frames() {
    let canvas = support::canvas(640, 480);
    let result = example::run_webgpu_multithreaded_recording(canvas.clone(), 3).await;
    support::remove_canvas(&canvas);
    result.expect("13 WebGPU sequential fallback must draw three frames");
}
