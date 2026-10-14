//! Headless-browser evidence for the native WebGPU occlusion-query workload.

#![cfg(target_arch = "wasm32")]

#[path = "../examples/11_occlusion_query/main.rs"]
mod occlusion_query;

use wasm_bindgen::JsCast;
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

/// Both query slots are resolved and read before the next browser frame uses
/// their visibility values to render the teapot and sphere.
#[wasm_bindgen_test(async)]
async fn occlusion_query_draws_and_reads_three_webgpu_frames() {
    let document = web_sys::window().unwrap().document().unwrap();
    let canvas = document
        .create_element("canvas")
        .unwrap()
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .unwrap();
    canvas.set_width(640);
    canvas.set_height(480);
    document.body().unwrap().append_child(&canvas).unwrap();
    let result = occlusion_query::run_webgpu_occlusion_query(canvas.clone(), 3).await;
    canvas.remove();
    result.expect("11 must draw, resolve, read, and present three WebGPU frames");
}
