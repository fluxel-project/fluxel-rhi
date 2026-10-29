//! Browser evidence for 10_indirect_draw.

#![cfg(all(target_arch = "wasm32", feature = "webgpu"))]

// Compile the example as a real module so its module documentation and
// relative sibling paths retain the same meaning as its binary target.
#[path = "../examples/10_indirect_draw/main.rs"]
mod example;

#[path = "support/webgpu.rs"]
mod support;

use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test(async)]
async fn indirect_draws_three_webgpu_frames() {
    let canvas = support::canvas(1280, 720);
    let result = example::run_webgpu_indirect(canvas.clone(), 3).await;
    support::remove_canvas(&canvas);
    result.expect("10 indirect draw must render three WebGPU frames");
}
