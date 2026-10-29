//! Browser evidence for 04_dynamic_uniform_buffer.

#![cfg(all(target_arch = "wasm32", feature = "webgpu"))]

#[path = "../examples/04_dynamic_uniform_buffer/main.rs"]
mod dynamic_uniforms;

#[path = "support/webgpu.rs"]
mod support;

use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test(async)]
async fn dynamic_uniforms_draw_three_webgpu_frames() {
    let canvas = support::canvas(1280, 720);
    let result = dynamic_uniforms::run_webgpu_dynamic_uniforms(canvas.clone(), 3).await;
    support::remove_canvas(&canvas);
    result.expect("04 dynamic uniforms must draw three WebGPU frames");
}
