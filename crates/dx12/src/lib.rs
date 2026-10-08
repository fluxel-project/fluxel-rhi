//! Direct3D 12 lowering for the portable Fluxel RHI contract.

pub use fluxel_rhi_core::api;

#[cfg(windows)]
mod backend {
    pub(crate) mod dx12;

    #[cfg(test)]
    #[path = "../../tests/common/mod.rs"]
    pub(crate) mod conformance;

    #[cfg(test)]
    #[path = "../../tests/harness/mod.rs"]
    pub(crate) mod test_harness;
}

/// Opens the native DX12 provider owned by this process.
///
/// The returned provider exposes only Fluxel's portable API. DXGI factories,
/// adapters, and Direct3D devices remain private to this crate.
#[cfg(windows)]
pub fn create_provider() -> api::error::RhiResult<api::platform::PlatformProvider> {
    use api::platform::provider::{BackendKind, PlatformProvider};

    let instance = fluxel_rhi_core::backend_spi::next_provider_instance();
    let native = backend::dx12::Dx12Provider::new(instance)?;
    Ok(PlatformProvider::new(
        BackendKind::Dx12,
        instance,
        Box::new(native),
    ))
}
