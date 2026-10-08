//! Metal lowering for Fluxel RHI.
//!
//! Objective-C and Metal values are contained in this crate; callers receive
//! Fluxel's portable provider interface.

#![deny(missing_docs)]

pub use fluxel_rhi_core::api;

#[cfg(target_vendor = "apple")]
mod backend {
    pub(crate) mod metal;
}

/// Opens the native Metal provider on an Apple target.
#[cfg(target_vendor = "apple")]
pub fn create_provider() -> api::RhiResult<api::platform::PlatformProvider> {
    use api::platform::provider::{BackendKind, PlatformProvider};

    let instance = fluxel_rhi_core::backend_spi::next_provider_instance();
    let native = backend::metal::MetalProvider::new(instance);
    Ok(PlatformProvider::new(
        BackendKind::Metal,
        instance,
        Box::new(native),
    ))
}
