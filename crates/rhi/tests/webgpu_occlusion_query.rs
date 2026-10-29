//! Headless-browser evidence for the native WebGPU occlusion-query workload.

#![cfg(all(target_arch = "wasm32", feature = "webgpu"))]

#[path = "../examples/11_occlusion_query/main.rs"]
mod occlusion_query;

#[path = "support/webgpu.rs"]
mod support;

use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

/// Both query slots are resolved and read before the next browser frame uses
/// their visibility values to render the teapot and sphere.
#[wasm_bindgen_test(async)]
async fn occlusion_query_draws_and_reads_three_webgpu_frames() {
    let canvas = support::canvas(640, 480);
    let result = occlusion_query::run_webgpu_occlusion_query(canvas.clone(), 3).await;
    support::remove_canvas(&canvas);
    result.expect("11 must draw, resolve, read, and present three WebGPU frames");
}
