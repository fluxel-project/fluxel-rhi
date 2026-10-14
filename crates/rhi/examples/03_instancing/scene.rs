//! Shared types and backend contract for the instanced rock-field example.
mod pipeline;
mod resources;
mod workload;

use fluxel_rhi::api::*;

pub use workload::Instancing;

pub trait InstancingBackend:
    Backend + ResourceBackend + ShaderBackend + PipelineBackend + RasterBackend
{
}
impl<T> InstancingBackend for T where
    T: Backend + ResourceBackend + ShaderBackend + PipelineBackend + RasterBackend
{
}

fn binding(binding: u32, visibility: Vec<ShaderStage>, ty: BindingType) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility,
        ty,
        count: BindingCount::One,
        partially_bound: false,
        nonuniform_indexing: false,
    }
}

fn buffer(label: &str, size: u64, usage: BufferUsage) -> BufferDesc {
    BufferDesc {
        label: Some(label.into()),
        size,
        usage,
        memory: MemoryClass::Device,
    }
}
