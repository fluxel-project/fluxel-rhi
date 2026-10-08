//! WebGPU lowering for Fluxel RHI.
//!
//! This crate owns browser-native values while exposing only Fluxel's portable
//! provider interface to callers.

#![deny(missing_docs)]

pub use fluxel_rhi_core::api;

#[cfg(target_arch = "wasm32")]
mod backend {
    pub(crate) mod webgpu;
}

/// Opens the browser WebGPU provider from the browser's `navigator.gpu` entry point.
#[cfg(target_arch = "wasm32")]
pub fn create_provider() -> api::RhiResult<api::platform::PlatformProvider> {
    use api::platform::provider::{BackendKind, PlatformProvider};

    let instance = fluxel_rhi_core::backend_spi::next_provider_instance();
    let native = backend::webgpu::WebGpuProvider::new(instance);
    Ok(PlatformProvider::new(
        BackendKind::WebGpu,
        instance,
        Box::new(native),
    ))
}
