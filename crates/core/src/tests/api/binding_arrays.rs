//! Core bind-group array, storage-view, and transient-texture contract tests.
use super::{type_contract_tests::NativeShared, *};
use std::cell::Cell;

thread_local! {
    static LAYOUT_CALLS: Cell<usize> = const { Cell::new(0) };
    static GROUP_CALLS: Cell<usize> = const { Cell::new(0) };
    static TEXTURE_CALLS: Cell<usize> = const { Cell::new(0) };
}

impl ResourceBackend for NativeShared {
    fn buffer_support(_: &Arc<()>, _: &BufferDesc) -> Support<BufferSupport> {
        unreachable!()
    }
    fn texture_support(_: &Arc<()>, _: &TextureDesc) -> Support<TextureSupport> {
        Support::Supported(TextureSupport {
            max_extent: [4096, 4096, 1],
            max_array_layers: 1,
            max_mips: 1,
            sample_counts: vec![1, 4],
            compatible_view_formats: vec![TextureFormat::Rgba8Unorm],
            filterable: true,
            blendable: true,
            storage: false,
        })
    }
    fn create_buffer(_: &Arc<()>, _: &BufferDesc) -> Result<Arc<()>> {
        unreachable!()
    }
    fn create_texture(_: &Arc<()>, _: &TextureDesc) -> Result<Arc<()>> {
        TEXTURE_CALLS.with(|calls| calls.set(calls.get() + 1));
        Ok(Arc::new(()))
    }
    fn create_view(_: &Arc<()>, _: &Arc<()>, _: &TextureViewDesc) -> Result<Arc<()>> {
        Ok(Arc::new(()))
    }
    fn create_sampler(_: &Arc<()>, _: &SamplerDesc) -> Result<Arc<()>> {
        Ok(Arc::new(()))
    }
    fn create_bind_group_layout(_: &Arc<()>, _: &BindGroupLayoutDesc) -> Result<Arc<()>> {
        LAYOUT_CALLS.with(|calls| calls.set(calls.get() + 1));
        Ok(Arc::new(()))
    }
    fn create_bind_group(_: &Arc<()>, _: &BindGroupDesc<'_, Self>) -> Result<Arc<()>> {
        GROUP_CALLS.with(|calls| calls.set(calls.get() + 1));
        Ok(Arc::new(()))
    }
    fn create_pipeline_layout(_: &Arc<()>, _: &PipelineLayoutDesc<'_, Self>) -> Result<Arc<()>> {
        unreachable!()
    }
}

fn device(runtime_arrays: bool) -> Device<NativeShared> {
    let id = DeviceId::allocate().unwrap();
    let mut caps = DeviceCapabilities::default();
    caps.runtime_sized_binding_arrays = runtime_arrays;
    caps.nonuniform_sampler_indexing = true;
    caps.limits.max_bindings_per_group = 4;
    // SAFETY: this fixture supplies a native handle and capabilities for the same mock device.
    unsafe { Device::from_native(id, Arc::new(()), caps) }
}

fn entry(binding: u32, count: BindingCount, partially_bound: bool) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: vec![],
        ty: BindingType::Sampler {
            filtering: false,
            comparison: false,
        },
        count,
        partially_bound,
        nonuniform_indexing: false,
    }
}

#[path = "binding_arrays/acceleration_arrays.rs"]
mod acceleration_arrays;
#[path = "binding_arrays/array_capabilities.rs"]
mod array_capabilities;
#[path = "binding_arrays/array_instances.rs"]
mod array_instances;
#[path = "binding_arrays/storage_views.rs"]
mod storage_views;
#[path = "binding_arrays/texture_atomic.rs"]
mod texture_atomic;
#[path = "binding_arrays/transient.rs"]
mod transient;
