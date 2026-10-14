//! Browser evidence for the fixed WebGL2 texture-array visual fallback.
#![cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[path = "webgl2_support.rs"]
mod support;

#[path = "../examples/14_descriptor_indexing/main.rs"]
mod example;

use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test(async)]
async fn fixed_texture_array_draws_three_webgl2_frames() {
    let canvas = support::canvas(1280, 720);
    let result = example::run_webgl2_descriptor_indexing(canvas.clone(), 3).await;
    result.expect("14 fixed texture-array fallback must draw three WebGL2 frames");
    support::assert_canvas_rendered(&canvas);
    canvas.remove();
}
