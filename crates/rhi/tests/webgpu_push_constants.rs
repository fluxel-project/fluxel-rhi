//! Headless-browser evidence for the WebGPU dynamic-uniform equivalent of 05.

#![cfg(target_arch = "wasm32")]

#[path = "../examples/05_push_constants.rs"]
mod push_constants;

use wasm_bindgen::JsCast;
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

/// Standard WebGPU has no push constants; the portable example must still
/// render the exact sixteen-draw sphere workload through dynamic uniforms.
#[wasm_bindgen_test(async)]
async fn push_constant_sphere_ring_draws_with_dynamic_uniforms() {
    let document = web_sys::window().unwrap().document().unwrap();
    let canvas = document
        .create_element("canvas")
        .unwrap()
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .unwrap();
    canvas.set_width(640);
    canvas.set_height(480);
    document.body().unwrap().append_child(&canvas).unwrap();
    let result = push_constants::run_webgpu_push_constants(canvas.clone(), 3).await;
    canvas.remove();
    result.expect("05 must draw and present three WebGPU frames");
}
