//! Compute-filter example implementation, separated from the platform entry point.
mod ktx;
mod pipelines;
mod workload;

use fluxel_rhi::api::*;

pub use workload::Compute;

pub trait ComputeBackend:
    Backend + ResourceBackend + ShaderBackend + PipelineBackend + CommandBackend + RasterBackend
{
}

impl<T> ComputeBackend for T where
    T: Backend + ResourceBackend + ShaderBackend + PipelineBackend + CommandBackend + RasterBackend
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
