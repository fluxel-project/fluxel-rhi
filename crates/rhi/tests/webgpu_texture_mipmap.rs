//! Headless-browser evidence for the shared WebGPU texture-mipmap entry.

#![cfg(target_arch = "wasm32")]

#[path = "../examples/09_texture_mipmap/main.rs"]
mod texture_mipmap;

use wasm_bindgen::JsCast;
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

/// The portable tunnel workload must generate and sample its mip chain, then
/// present three browser WebGPU frames without a JavaScript-side renderer.
#[wasm_bindgen_test(async)]
async fn texture_mipmap_draws_and_presents_three_webgpu_frames() {
    let document = web_sys::window().unwrap().document().unwrap();
    let canvas = document
        .create_element("canvas")
        .unwrap()
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .unwrap();
    canvas.set_width(640);
    canvas.set_height(480);
    document.body().unwrap().append_child(&canvas).unwrap();
    let result = texture_mipmap::run_webgpu_texture_mipmap(canvas.clone(), 3).await;
    canvas.remove();
    result.expect("09 must draw and present three WebGPU frames");
}
