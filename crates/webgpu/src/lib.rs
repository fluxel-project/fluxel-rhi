//! Browser WebGPU lowering for Fluxel's statically typed execution API.
//!
//! Browser objects remain private to this crate and its wasm owner thread.
//! Queue completion, surface generations, and device-loss observation stay in
//! the backend; Core receives only typed native handles and completion facts.

#[cfg(target_arch = "wasm32")]
mod backend;

#[cfg(target_arch = "wasm32")]
pub use backend::{WebGpu, WebGpuInstance};
