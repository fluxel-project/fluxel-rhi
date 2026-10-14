//! Browser proof of the offscreen render and completed GPU readback path.
#![cfg(all(target_arch = "wasm32", feature = "webgpu"))]

#[path = "../examples/12_screenshot/main.rs"]
mod screenshot;

use wasm_bindgen::JsCast;
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test(async)]
async fn screenshot_renders_reads_back_and_presents_three_webgpu_frames() {
    let document = web_sys::window().unwrap().document().unwrap();
    let canvas = document
        .create_element("canvas")
        .unwrap()
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .unwrap();
    canvas.set_width(640);
    canvas.set_height(480);
    document.body().unwrap().append_child(&canvas).unwrap();
    let result = screenshot::run_webgpu_screenshot(canvas.clone(), 3).await;
    canvas.remove();
    result.expect("12 must render, read back, and present three WebGPU frames");
}
