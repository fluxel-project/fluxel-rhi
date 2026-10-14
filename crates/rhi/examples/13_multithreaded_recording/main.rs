//! Entry point for the secondary-command UFO example.
//!
//! Its scene, worker recording, and math live in a private module while this
//! file keeps the stable Cargo example target small.
mod scene;

#[allow(unused_imports)]
pub use scene::*;

/// Runs the sequential visible-scene fallback on a caller-owned WebGL2 canvas.
#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[wasm_bindgen::prelude::wasm_bindgen]
pub async fn run_webgl2_multithreaded_recording(
    canvas: web_sys::HtmlCanvasElement,
    frames: u32,
) -> std::result::Result<(), wasm_bindgen::JsValue> {
    scene::run_webgl2_multithreaded_recording(canvas, frames)
        .await
        .map_err(|error| wasm_bindgen::JsValue::from_str(&format!("{error:?}")))
}

#[cfg(windows)]
fn main() {
    #[cfg(feature = "native-gl-wgl")]
    if std::env::args()
        .skip_while(|arg| arg != "--backend")
        .nth(1)
        .is_some_and(|backend| backend == "gl")
    {
        if let Err(error) = scene::run_gl_multithreaded_recording() {
            eprintln!("13_multithreaded_recording: {error:?}");
            std::process::exit(1);
        }
        return;
    }
    scene::run();
}

#[cfg(all(target_os = "linux", feature = "native-gles-egl"))]
fn main() {
    if let Err(error) = scene::run_gl_multithreaded_recording() {
        eprintln!("13_multithreaded_recording: {error:?}");
        std::process::exit(1);
    }
}

#[cfg(not(any(windows, all(target_os = "linux", feature = "native-gles-egl"))))]
fn main() {
    scene::run();
}
