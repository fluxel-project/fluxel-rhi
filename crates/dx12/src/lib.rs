//! Direct3D 12 lowering for Fluxel's statically typed execution API.

#[cfg(windows)]
mod backend;
#[cfg(windows)]
pub use backend::{Dx12, Dx12ExternalHandle, Dx12ExternalSync, Dx12Heap, Dx12Instance};

#[cfg(all(windows, feature = "hardware-tests"))]
#[doc(hidden)]
pub mod diagnostics;
#[cfg(all(windows, not(feature = "hardware-tests")))]
mod diagnostics;
