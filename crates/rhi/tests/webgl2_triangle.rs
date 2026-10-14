//! Browser evidence for the immediate WebGL2 triangle path.
#![cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[path = "webgl2_support.rs"]
mod support;

#[path = "../examples/01_triangle/main.rs"]
mod example;

use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test(async)]
async fn triangle_draws_and_presents_three_webgl2_frames() {
    let canvas = support::canvas(640, 480);
    let result = example::run_webgl2_triangle(canvas.clone(), 3).await;
    result.expect("01_triangle must draw and present three WebGL2 frames");
    support::assert_canvas_rendered(&canvas);
    canvas.remove();
}
