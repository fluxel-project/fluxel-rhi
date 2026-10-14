//! Browser evidence for 06_compute.

#![cfg(all(target_arch = "wasm32", feature = "webgpu"))]

#[path = "../examples/06_compute/main.rs"]
mod compute;

use wasm_bindgen::JsCast;
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test(async)]
async fn compute_draws_three_webgpu_frames() {
    let document = web_sys::window().unwrap().document().unwrap();
    let canvas = document
        .create_element("canvas")
        .unwrap()
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .unwrap();
    canvas.set_width(1280);
    canvas.set_height(720);
    document.body().unwrap().append_child(&canvas).unwrap();
    let result = compute::run_webgpu_compute(canvas.clone(), 3).await;
    canvas.remove();
    result.expect("06 compute must draw three WebGPU frames");
}
