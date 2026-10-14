//! Browser visual-equivalence test for the 32-layer WebGPU fallback.
#![cfg(all(target_arch = "wasm32", feature = "webgpu"))]

#[path = "../examples/14_descriptor_indexing/main.rs"]
mod descriptor_indexing;

use wasm_bindgen::JsCast;
use wasm_bindgen_test::*;
wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test(async)]
async fn descriptor_indexing_visual_fallback_presents_three_webgpu_frames() {
    let document = web_sys::window().unwrap().document().unwrap();
    let canvas = document
        .create_element("canvas")
        .unwrap()
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .unwrap();
    canvas.set_width(640);
    canvas.set_height(480);
    document.body().unwrap().append_child(&canvas).unwrap();
    let result = descriptor_indexing::run_webgpu_descriptor_indexing(canvas.clone(), 3).await;
    canvas.remove();
    result.expect("14 must render three visually equivalent WebGPU frames");
}
