//! Headed-browser evidence for the shared `01_triangle` WebGPU wasm entry.

#![cfg(target_arch = "wasm32")]

#[path = "../examples/01_triangle.rs"]
mod triangle;

use wasm_bindgen::JsCast;
use wasm_bindgen_test::*;
use web_sys::HtmlCanvasElement;

wasm_bindgen_test_configure!(run_in_browser);

/// The exact portable triangle workload used by the desktop example must draw
/// and present three browser WebGPU frames without a JavaScript-side renderer.
#[wasm_bindgen_test(async)]
async fn triangle_draws_and_presents_three_webgpu_frames() {
    let window = web_sys::window().expect("browser window");
    let document = window.document().expect("browser document");
    let canvas = document
        .create_element("canvas")
        .expect("canvas element")
        .dyn_into::<HtmlCanvasElement>()
        .expect("HTML canvas");
    canvas.set_width(640);
    canvas.set_height(480);
    document
        .body()
        .expect("document body")
        .append_child(&canvas)
        .expect("attach canvas");

    let result = triangle::run_webgpu_triangle(canvas.clone(), 3).await;
    canvas.remove();
    result.expect("01_triangle must draw and present three WebGPU frames");
}
