//! Browser evidence for 08_msaa.

#![cfg(target_arch = "wasm32")]

include!("../examples/08_msaa.rs");

use wasm_bindgen::JsCast;
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test(async)]
async fn msaa_draws_three_webgpu_frames() {
    let document = web_sys::window().unwrap().document().unwrap();
    let canvas = document
        .create_element("canvas")
        .unwrap()
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .unwrap();
    canvas.set_width(1280);
    canvas.set_height(720);
    document.body().unwrap().append_child(&canvas).unwrap();
    let result = run_webgpu_msaa(canvas.clone(), 3).await;
    canvas.remove();
    result.expect("08 MSAA must draw three WebGPU frames");
}
