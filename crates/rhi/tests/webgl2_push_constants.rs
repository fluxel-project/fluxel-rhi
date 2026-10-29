//! Browser evidence for the immediate WebGL2 push-constant fallback.
#![cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[path = "support/webgl2.rs"]
mod support;

#[path = "../examples/05_push_constants/main.rs"]
mod example;

use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test(async)]
async fn push_constants_draw_three_webgl2_frames() {
    let canvas = support::canvas(640, 480);
    let result = example::run_webgl2_push_constants(canvas.clone(), 3).await;
    result.expect("05_push_constants must draw three WebGL2 frames");
    support::assert_canvas_rendered(&canvas);
    canvas.remove();
}
