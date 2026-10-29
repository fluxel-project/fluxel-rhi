//! Headed-browser evidence for the shared `01_triangle` WebGPU wasm entry.

#![cfg(all(target_arch = "wasm32", feature = "webgpu"))]

#[path = "../examples/01_triangle/main.rs"]
mod triangle;

#[path = "support/webgpu.rs"]
mod support;

use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

/// The exact portable triangle workload used by the desktop example must draw
/// and present three browser WebGPU frames without a JavaScript-side renderer.
#[wasm_bindgen_test(async)]
async fn triangle_draws_and_presents_three_webgpu_frames() {
    let canvas = support::canvas(640, 480);

    let result = triangle::run_webgpu_triangle(canvas.clone(), 3).await;
    support::remove_canvas(&canvas);
    result.expect("01_triangle must draw and present three WebGPU frames");
}
