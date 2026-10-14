//! Browser evidence for the WebGPU sequential-primary fallback of example 13.
#![cfg(target_arch = "wasm32")]
#[path = "../examples/13_multithreaded_recording/main.rs"]
mod example;
use wasm_bindgen::JsCast;
use wasm_bindgen_test::*;
wasm_bindgen_test_configure!(run_in_browser);
#[wasm_bindgen_test(async)]
async fn sequential_primary_pass_draws_three_frames() {
    let document = web_sys::window().unwrap().document().unwrap();
    let canvas = document
        .create_element("canvas")
        .unwrap()
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .unwrap();
    canvas.set_width(640);
    canvas.set_height(480);
    document.body().unwrap().append_child(&canvas).unwrap();
    let result = example::run_webgpu_multithreaded_recording(canvas.clone(), 3).await;
    canvas.remove();
    result.expect("13 WebGPU sequential fallback must draw three frames");
}
