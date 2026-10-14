//! Browser evidence for the immediate WebGL2 raster fallback of example 06.
#![cfg(all(target_arch = "wasm32", feature = "webgl2"))]

#[path = "webgl2_support.rs"]
mod support;

#[path = "../examples/06_compute/main.rs"]
mod example;

use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test(async)]
async fn compute_filter_scene_has_a_webgl2_raster_fallback() {
    let canvas = support::canvas(1280, 720);
    example::run_webgl2_compute_fallback(canvas.clone(), 3)
        .await
        .expect("06_compute must show the source and raster-filter fallback");
    support::assert_canvas_rendered(&canvas);
    support::assert_canvas_alpha_opaque(&canvas);
    canvas.remove();
}
