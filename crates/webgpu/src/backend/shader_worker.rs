//! Dedicated browser-worker SPIR-V to WGSL translation.
//!
//! The main WebGPU owner thread only transfers immutable SPIR-V to this
//! worker and receives WGSL back.  The worker owns no WebGPU object.  Its
//! wasm-bindgen glue URL is an explicit backend deployment setting supplied
//! by the embedding page; silently translating on the owner thread would
//! violate the async shader contract.

use std::{
    future::Future,
    pin::Pin,
    task::{Context, Poll},
};

use fluxel_rhi_core::api::{Error, Result};
use js_sys::Promise;
use wasm_bindgen::{JsValue, prelude::wasm_bindgen};
use wasm_bindgen_futures::JsFuture;

#[wasm_bindgen(module = "/src/backend/shader_worker.js")]
extern "C" {
    type Request;

    #[wasm_bindgen(js_name = startSpirvTranslation)]
    fn start_spirv_translation(bytes: &[u8]) -> Request;

    #[wasm_bindgen(method, getter)]
    fn promise(this: &Request) -> Promise;

    #[wasm_bindgen(method)]
    fn cancel(this: &Request);

    #[wasm_bindgen(catch, js_name = configureSpirvWorker)]
    fn configure_spirv_worker(url: &str) -> std::result::Result<(), JsValue>;

    #[wasm_bindgen(js_name = hasSpirvWorker)]
    fn has_spirv_worker() -> bool;

    #[cfg(test)]
    #[wasm_bindgen(js_name = resetSpirvWorkerConfigurationForTests)]
    fn reset_spirv_worker_configuration_for_tests();

    #[cfg(test)]
    #[wasm_bindgen(js_name = holdNextTranslation)]
    fn hold_next_translation();

    #[cfg(test)]
    #[wasm_bindgen(js_name = releaseActiveHold)]
    fn release_active_hold();
}

/// Installs the module URL of the separately built worker asset.  Kept
/// backend-private here; `WebGpuInstance` exposes the documented public
/// configuration method without letting this browser detail enter Core.
pub(crate) fn configure(url: &str) -> Result<()> {
    if url.is_empty() {
        return Err(Error::InvalidRange);
    }
    configure_spirv_worker(url)
        .map_err(|error| Error::Backend(error.as_string().unwrap_or_else(|| format!("{error:?}"))))
}

/// Whether this browser execution context currently has a translator module
/// configured.  The shader route uses this fact so `SpirV` is only advertised
/// when its required worker deployment asset is available to the backend.
pub(crate) fn is_configured() -> bool {
    has_spirv_worker()
}

#[cfg(test)]
pub(crate) fn reset_configuration_for_tests() {
    reset_spirv_worker_configuration_for_tests();
}

/// JS bundle bootstrap counterpart of `WebGpuInstance::configure_spirv_worker`.
/// Rust applications should use the typed method; generated browser harnesses
/// use this export before entering their Rust example function.
#[wasm_bindgen]
pub fn fluxel_webgpu_configure_spirv_worker(url: String) -> std::result::Result<(), JsValue> {
    configure(&url).map_err(|error| JsValue::from_str(&error.to_string()))
}

/// The narrow ABI invoked by the worker's own wasm instance.  This function
/// is intentionally pure: no DOM, WebGPU device, or owner-thread state is
/// reachable from the worker.
#[wasm_bindgen]
pub fn fluxel_webgpu_translate_spirv(bytes: &[u8]) -> std::result::Result<String, JsValue> {
    translate(bytes).map_err(|error| JsValue::from_str(&error.to_string()))
}

fn translate(bytes: &[u8]) -> Result<String> {
    let module = naga::front::spv::parse_u8_slice(
        bytes,
        &naga::front::spv::Options {
            // Fluxel's scene matrices already apply the framebuffer coordinate
            // convention for checked-in shaders.
            adjust_coordinate_space: false,
            ..Default::default()
        },
    )
    .map_err(|error| Error::Shader(format!("SPIR-V parse: {error}")))?;
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .map_err(|error| Error::Shader(format!("SPIR-V validation: {error}")))?;
    naga::back::wgsl::write_string(&module, &info, naga::back::wgsl::WriterFlags::empty())
        .map_err(|error| Error::Shader(format!("WGSL emission: {error}")))
}

pub(super) fn translate_async(bytes: Vec<u8>) -> TranslationFuture {
    let request = start_spirv_translation(&bytes);
    let promise = request.promise();
    TranslationFuture {
        request: Some(request),
        future: Box::pin(JsFuture::from(promise)),
    }
}

#[cfg(test)]
pub(super) fn hold_next_for_test() {
    hold_next_translation();
}
#[cfg(test)]
pub(super) fn release_hold_for_test() {
    release_active_hold();
}

pub(super) struct TranslationFuture {
    request: Option<Request>,
    future: Pin<Box<JsFuture>>,
}

impl Future for TranslationFuture {
    type Output = Result<String>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match self.future.as_mut().poll(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(Ok(value)) => {
                self.request.take();
                Poll::Ready(value.as_string().ok_or_else(|| {
                    Error::Backend("WebGPU shader worker returned a non-string result".into())
                }))
            }
            Poll::Ready(Err(error)) => {
                self.request.take();
                Poll::Ready(Err(Error::Shader(
                    error.as_string().unwrap_or_else(|| format!("{error:?}")),
                )))
            }
        }
    }
}

impl Drop for TranslationFuture {
    fn drop(&mut self) {
        if let Some(request) = self.request.take() {
            // Rust future cancellation is explicit in the worker protocol.
            // A currently executing Naga call is not preemptible, but the
            // worker is terminated and its result can neither retain nor wake
            // this dropped owner-thread future.
            request.cancel();
        }
    }
}

#[cfg(test)]
#[path = "../tests/backend/shader_worker.rs"]
mod tests;
