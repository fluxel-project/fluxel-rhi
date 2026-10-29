//! Browser evidence for the immediate WebGL2 mip-generation path.
#![cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[path = "support/webgl2.rs"]
mod support;

#[path = "../examples/09_texture_mipmap/main.rs"]
mod example;

use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test(async)]
async fn texture_mipmap_draws_and_presents_three_webgl2_frames() {
    let canvas = support::canvas(1280, 720);
    let result = example::run_webgl2_texture_mipmap(canvas.clone(), 3).await;
    result.expect("09 must generate and sample mip levels, then present three WebGL2 frames");
    support::assert_canvas_rendered(&canvas);
    canvas.remove();
}
