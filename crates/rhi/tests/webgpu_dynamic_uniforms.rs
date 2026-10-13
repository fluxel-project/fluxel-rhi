//! Browser evidence for 04_dynamic_uniform_buffer.

#![cfg(target_arch = "wasm32")]

#[path = "../examples/04_dynamic_uniform_buffer.rs"]
mod dynamic_uniforms;

use wasm_bindgen::JsCast;
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test(async)]
async fn dynamic_uniforms_draw_three_webgpu_frames() {
    let document = web_sys::window().unwrap().document().unwrap();
    let canvas = document
        .create_element("canvas")
        .unwrap()
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .unwrap();
    canvas.set_width(1280);
    canvas.set_height(720);
    document.body().unwrap().append_child(&canvas).unwrap();
    let result = dynamic_uniforms::run_webgpu_dynamic_uniforms(canvas.clone(), 3).await;
    canvas.remove();
    result.expect("04 dynamic uniforms must draw three WebGPU frames");
}
