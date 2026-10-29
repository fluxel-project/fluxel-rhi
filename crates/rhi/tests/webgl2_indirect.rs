//! Browser evidence for the WebGL2 direct-draw fallback of example 10.
#![cfg(all(target_arch = "wasm32", feature = "webgl2"))]

#[path = "support/webgl2.rs"]
mod support;

#[path = "../examples/10_indirect_draw/main.rs"]
mod example;

use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test(async)]
async fn indirect_scene_direct_draw_fallback_presents_three_webgl2_frames() {
    let canvas = support::canvas(1280, 720);
    let result = example::run_webgl2_indirect_fallback(canvas.clone(), 3).await;
    result.expect("10 direct-draw fallback must present three WebGL2 frames");
    support::assert_canvas_rendered(&canvas);
    support::assert_canvas_alpha_opaque(&canvas);
    canvas.remove();
}
