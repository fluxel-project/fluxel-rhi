//! Browser evidence for the sequential WebGPU stereo fallback of example 15.

#![cfg(all(target_arch = "wasm32", feature = "webgpu"))]

#[path = "../examples/15_multiview/main.rs"]
mod multiview;

use wasm_bindgen::JsCast;
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test(async)]
async fn multiview_fallback_draws_three_webgpu_frames() {
    let document = web_sys::window().unwrap().document().unwrap();
    let canvas = document
        .create_element("canvas")
        .unwrap()
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .unwrap();
    canvas.set_width(1280);
    canvas.set_height(720);
    document.body().unwrap().append_child(&canvas).unwrap();
    let result = multiview::run_webgpu_multiview(canvas.clone(), 3).await;
    canvas.remove();
    result.expect("15 must present three sequential WebGPU stereo frames");
}
