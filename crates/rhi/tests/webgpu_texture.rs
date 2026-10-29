//! Headless-browser evidence for the shared 02_texture WebGPU entry.

#![cfg(all(target_arch = "wasm32", feature = "webgpu"))]

#[path = "../examples/02_texture/main.rs"]
mod texture;

#[path = "support/webgpu.rs"]
mod support;

use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test(async)]
async fn texture_draws_and_presents_three_webgpu_frames() {
    let canvas = support::canvas(640, 480);
    let result = texture::run_webgpu_texture(canvas.clone(), 3).await;
    support::remove_canvas(&canvas);
    result.expect("02_texture must draw and present three WebGPU frames");
}
