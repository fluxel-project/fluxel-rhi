//! Direct3D 12 lowering for Fluxel's statically typed execution API.

#[cfg(windows)]
mod backend;
#[cfg(windows)]
pub use backend::{Dx12, Dx12Instance};
