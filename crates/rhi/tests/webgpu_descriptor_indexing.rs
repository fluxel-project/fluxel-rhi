//! Browser visual-equivalence test for the 32-layer WebGPU fallback.
#![cfg(all(target_arch = "wasm32", feature = "webgpu"))]

#[path = "../examples/14_descriptor_indexing/main.rs"]
mod descriptor_indexing;

#[path = "support/webgpu.rs"]
mod support;

use wasm_bindgen_test::*;
wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test(async)]
async fn descriptor_indexing_visual_fallback_presents_three_webgpu_frames() {
    let canvas = support::canvas(640, 480);
    let result = descriptor_indexing::run_webgpu_descriptor_indexing(canvas.clone(), 3).await;
    support::remove_canvas(&canvas);
    result.expect("14 must render three visually equivalent WebGPU frames");
}
