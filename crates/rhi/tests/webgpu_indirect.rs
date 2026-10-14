//! Browser evidence for 10_indirect_draw.

#![cfg(target_arch = "wasm32")]

// Compile the example as a real module so its module documentation and
// relative sibling paths retain the same meaning as its binary target.
#[path = "../examples/10_indirect_draw/main.rs"]
mod example;

use wasm_bindgen::JsCast;
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test(async)]
async fn indirect_draws_three_webgpu_frames() {
    let document = web_sys::window().unwrap().document().unwrap();
    let canvas = document
        .create_element("canvas")
        .unwrap()
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .unwrap();
    canvas.set_width(1280);
    canvas.set_height(720);
    document.body().unwrap().append_child(&canvas).unwrap();
    let result = example::run_webgpu_indirect(canvas.clone(), 3).await;
    canvas.remove();
    result.expect("10 indirect draw must render three WebGPU frames");
}
