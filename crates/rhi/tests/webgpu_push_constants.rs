//! Headless-browser evidence for the WebGPU dynamic-uniform equivalent of 05.

#![cfg(all(target_arch = "wasm32", feature = "webgpu"))]

#[path = "../examples/05_push_constants/main.rs"]
mod push_constants;

#[path = "support/webgpu.rs"]
mod support;

use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

/// Standard WebGPU has no push constants; the portable example must still
/// render the exact sixteen-draw sphere workload through dynamic uniforms.
#[wasm_bindgen_test(async)]
async fn push_constant_sphere_ring_draws_with_dynamic_uniforms() {
    let canvas = support::canvas(640, 480);
    let result = push_constants::run_webgpu_push_constants(canvas.clone(), 3).await;
    support::remove_canvas(&canvas);
    result.expect("05 must draw and present three WebGPU frames");
}
