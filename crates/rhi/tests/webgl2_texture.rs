//! Browser evidence for the immediate WebGL2 texture path.
#![cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[path = "support/webgl2.rs"]
mod support;

#[path = "../examples/02_texture/main.rs"]
mod example;

use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test(async)]
async fn texture_draws_and_presents_three_webgl2_frames() {
    let canvas = support::canvas(640, 480);
    let result = example::run_webgl2_texture(canvas.clone(), 3).await;
    result.expect("02_texture must draw and present three WebGL2 frames");
    support::assert_canvas_rendered(&canvas);
    canvas.remove();
}
