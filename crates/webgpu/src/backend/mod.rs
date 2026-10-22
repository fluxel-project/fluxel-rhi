//! Browser-owned WebGPU objects and native command lowering.
//!
//! This module is the native boundary for one wasm browser owner thread. It
//! keeps JavaScript objects private, records directly into `GPUCommandEncoder`,
//! and retains only the native objects needed until queue completion. Surface
//! leases and device loss are handled by their dedicated modules; no renderer
//! scheduling or CPU command replay lives here.

mod bindings;
mod blit;
mod command;
mod compute;
mod immediate;
mod immediate_shader;
mod indirect;
mod indirect_count;
mod instance;
mod js;
mod mapping;
mod partial;
mod pass_split;
mod pipeline;
mod query;
mod query_segments;
mod raster;
mod readback;
mod resource;
mod resource_validation;
mod subgroup;
mod submit;
mod surface;

pub use instance::{WebGpu, WebGpuInstance};

#[cfg(test)]
mod conformance;
#[cfg(test)]
mod tests;
