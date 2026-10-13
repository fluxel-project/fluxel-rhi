//! Browser-owned WebGPU objects and native command lowering.

mod bindings;
mod blit;
mod command;
mod compute;
mod indirect;
mod instance;
mod js;
mod pipeline;
mod query;
mod raster;
mod readback;
mod resource;
mod resource_validation;
mod submit;
mod surface;

pub use instance::{WebGpu, WebGpuInstance};

#[cfg(test)]
mod tests;
