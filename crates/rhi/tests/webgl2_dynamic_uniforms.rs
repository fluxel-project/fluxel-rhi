//! Browser evidence for the immediate WebGL2 dynamic-uniform path.
#![cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[path = "webgl2_support.rs"]
mod support;

#[path = "../examples/04_dynamic_uniform_buffer/main.rs"]
mod example;

use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test(async)]
async fn dynamic_uniforms_draw_three_webgl2_frames() {
    let canvas = support::canvas(640, 480);
    let result = example::run_webgl2_dynamic_uniforms(canvas.clone(), 3).await;
    result.expect("04_dynamic_uniform_buffer must draw three WebGL2 frames");
    support::assert_canvas_rendered(&canvas);
    canvas.remove();
}
