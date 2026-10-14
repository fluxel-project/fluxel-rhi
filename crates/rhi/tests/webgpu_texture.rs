//! Headless-browser evidence for the shared 02_texture WebGPU entry.

#![cfg(all(target_arch = "wasm32", feature = "webgpu"))]

#[path = "../examples/02_texture/main.rs"]
mod texture;

use wasm_bindgen::JsCast;
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test(async)]
async fn texture_draws_and_presents_three_webgpu_frames() {
    let document = web_sys::window().unwrap().document().unwrap();
    let canvas = document
        .create_element("canvas")
        .unwrap()
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .unwrap();
    canvas.set_width(640);
    canvas.set_height(480);
    document.body().unwrap().append_child(&canvas).unwrap();
    let result = texture::run_webgpu_texture(canvas.clone(), 3).await;
    canvas.remove();
    result.expect("02_texture must draw and present three WebGPU frames");
}
