//! Browser evidence for the immediate WebGL2 offscreen mirror path.
#![cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[path = "support/webgl2.rs"]
mod support;

#[path = "../examples/07_offscreen/main.rs"]
mod example;

use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test(async)]
async fn offscreen_mirror_draws_and_presents_three_webgl2_frames() {
    let canvas = support::canvas(1280, 720);
    let result = example::run_webgl2_offscreen(canvas.clone(), 3).await;
    result.expect("07_offscreen must draw and present three WebGL2 frames");
    support::assert_canvas_rendered(&canvas);
    support::assert_canvas_alpha_opaque(&canvas);
    canvas.remove();
}
