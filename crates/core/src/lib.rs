//! Portable GPU execution API and backend implementation contract.

#![deny(missing_docs)]

pub mod api;

/// Hardware evidence and fault-injection helpers used by integration tests.
#[cfg(feature = "test-support")]
#[doc(hidden)]
pub mod test_support;

/// Implementation entry points shared by separately compiled backend crates.
#[doc(hidden)]
pub mod backend_spi {
    use std::sync::atomic::{AtomicU64, Ordering};

    use crate::api::identity::DeviceInstanceId;

    static NEXT_PUBLIC_PROVIDER: AtomicU64 = AtomicU64::new(0xF1_0000_0000_0000);

    /// Reserves a process-wide provider identity for one backend provider.
    pub fn next_provider_instance() -> DeviceInstanceId {
        DeviceInstanceId::new(NEXT_PUBLIC_PROVIDER.fetch_add(1, Ordering::Relaxed))
    }
}
