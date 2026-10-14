//! Browser evidence for 07_offscreen.

#![cfg(target_arch = "wasm32")]

#[path = "../examples/07_offscreen.rs"]
mod example;

use wasm_bindgen::JsCast;
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test(async)]
async fn offscreen_draws_three_webgpu_frames() {
    let document = web_sys::window().unwrap().document().unwrap();
    let canvas = document
        .create_element("canvas")
        .unwrap()
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .unwrap();
    canvas.set_width(1280);
    canvas.set_height(720);
    document.body().unwrap().append_child(&canvas).unwrap();
    let result = example::run_webgpu_offscreen(canvas.clone(), 3).await;
    canvas.remove();
    result.expect("07 offscreen must draw three WebGPU frames");
}
