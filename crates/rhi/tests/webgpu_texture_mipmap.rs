//! Headless-browser evidence for the shared WebGPU texture-mipmap entry.

#![cfg(all(target_arch = "wasm32", feature = "webgpu"))]

#[path = "../examples/09_texture_mipmap/main.rs"]
mod texture_mipmap;

#[path = "support/webgpu.rs"]
mod support;

use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

/// The portable tunnel workload must generate and sample its mip chain, then
/// present three browser WebGPU frames without a JavaScript-side renderer.
#[wasm_bindgen_test(async)]
async fn texture_mipmap_draws_and_presents_three_webgpu_frames() {
    let canvas = support::canvas(640, 480);
    let result = texture_mipmap::run_webgpu_texture_mipmap(canvas.clone(), 3).await;
    support::remove_canvas(&canvas);
    result.expect("09 must draw and present three WebGPU frames");
}
