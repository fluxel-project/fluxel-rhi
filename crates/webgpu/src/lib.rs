//! Browser WebGPU lowering for Fluxel's statically typed execution API.
//!
//! Browser objects remain private to this crate and its owning thread.

#[cfg(target_arch = "wasm32")]
mod backend;

#[cfg(target_arch = "wasm32")]
pub use backend::{WebGpu, WebGpuInstance};
