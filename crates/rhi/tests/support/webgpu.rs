//! Shared canvas setup for independent WebGPU example presentation targets.

use wasm_bindgen::JsCast;
use web_sys::HtmlCanvasElement;

pub(crate) fn canvas(width: u32, height: u32) -> HtmlCanvasElement {
    let document = web_sys::window()
        .expect("browser window")
        .document()
        .expect("browser document");
    let canvas = document
        .create_element("canvas")
        .expect("canvas element")
        .dyn_into::<HtmlCanvasElement>()
        .expect("HTML canvas");
    canvas.set_width(width);
    canvas.set_height(height);
    document
        .body()
        .expect("document body")
        .append_child(&canvas)
        .expect("attach canvas");
    canvas
}

pub(crate) fn remove_canvas(canvas: &HtmlCanvasElement) {
    canvas.remove();
}
