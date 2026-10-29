//! Browser evidence for 08_msaa.

#![cfg(all(target_arch = "wasm32", feature = "webgpu"))]

#[path = "../examples/08_msaa/main.rs"]
mod example;

#[path = "support/webgpu.rs"]
mod support;

use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test(async)]
async fn msaa_draws_three_webgpu_frames() {
    let canvas = support::canvas(1280, 720);
    let result = example::run_webgpu_msaa(canvas.clone(), 3).await;
    support::remove_canvas(&canvas);
    result.expect("08 MSAA must draw three WebGPU frames");
}
