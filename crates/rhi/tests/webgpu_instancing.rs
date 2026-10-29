//! Headless-browser evidence for the shared 03_instancing WebGPU entry.

#![cfg(all(target_arch = "wasm32", feature = "webgpu"))]

#[path = "../examples/03_instancing/main.rs"]
mod example;

#[path = "support/webgpu.rs"]
mod support;

use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test(async)]
async fn instancing_draws_and_presents_three_webgpu_frames() {
    let canvas = support::canvas(1280, 720);
    let result = example::run_webgpu_instancing(canvas.clone(), 3).await;
    support::remove_canvas(&canvas);
    result.expect("03_instancing must draw and present three WebGPU frames");
}
