//! Headless-browser evidence for the shared 03_instancing WebGPU entry.

#![cfg(target_arch = "wasm32")]

#[path = "../examples/03_instancing.rs"]
mod example;

use wasm_bindgen::JsCast;
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test(async)]
async fn instancing_draws_and_presents_three_webgpu_frames() {
    let document = web_sys::window().unwrap().document().unwrap();
    let canvas = document
        .create_element("canvas")
        .unwrap()
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .unwrap();
    canvas.set_width(1280);
    canvas.set_height(720);
    document.body().unwrap().append_child(&canvas).unwrap();
    let result = example::run_webgpu_instancing(canvas.clone(), 3).await;
    canvas.remove();
    result.expect("03_instancing must draw and present three WebGPU frames");
}
