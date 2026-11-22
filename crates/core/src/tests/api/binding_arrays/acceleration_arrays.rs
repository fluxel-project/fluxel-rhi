//! Acceleration-structure binding arrays require their dedicated capability.
use super::LAYOUT_CALLS;
use crate::api::tests::ray_validation::Mock;
use crate::{api::*, backend::*};

impl ResourceBackend for Mock {
    fn buffer_support(_: &(), _: &BufferDesc) -> Support<BufferSupport> {
        unreachable!()
    }
    fn texture_support(_: &(), _: &TextureDesc) -> Support<TextureSupport> {
        unreachable!()
    }
    fn create_buffer(_: &(), _: &BufferDesc) -> Result<()> {
        unreachable!()
    }
    fn create_texture(_: &(), _: &TextureDesc) -> Result<()> {
        unreachable!()
    }
    fn create_view(_: &(), _: &(), _: &TextureViewDesc) -> Result<()> {
        unreachable!()
    }
    fn create_sampler(_: &(), _: &SamplerDesc) -> Result<()> {
        unreachable!()
    }
    fn create_bind_group_layout(_: &(), _: &BindGroupLayoutDesc) -> Result<()> {
        LAYOUT_CALLS.with(|calls| calls.set(calls.get() + 1));
        Ok(())
    }
    fn create_bind_group(_: &(), _: &BindGroupDesc<'_, Self>) -> Result<()> {
        unreachable!()
    }
    fn create_pipeline_layout(_: &(), _: &PipelineLayoutDesc<'_, Self>) -> Result<()> {
        unreachable!()
    }
}

#[test]
fn acceleration_structure_arrays_require_their_own_capability() {
    LAYOUT_CALLS.with(|calls| calls.set(0));
    let layout = BindGroupLayoutDesc {
        label: None,
        entries: vec![BindGroupLayoutEntry {
            binding: 0,
            visibility: vec![ShaderStage::Compute],
            ty: BindingType::AccelerationStructure,
            count: BindingCount::Fixed(2),
            partially_bound: false,
            nonuniform_indexing: false,
        }],
    };
    let id = DeviceId::allocate().unwrap();
    let mut caps = DeviceCapabilities::default();
    caps.binding_arrays = true;
    caps.acceleration_structure = true;
    caps.limits.max_bindings_per_group = 4;
    // SAFETY: this mock supports scalar acceleration bindings without arrays.
    let device = unsafe { Device::<Mock>::from_native(id, (), caps) };
    assert!(matches!(
        device.create_bind_group_layout(&layout),
        Err(Error::Unsupported("acceleration structure binding arrays"))
    ));
    LAYOUT_CALLS.with(|calls| assert_eq!(calls.get(), 0));

    let mut runtime_layout = layout.clone();
    runtime_layout.entries[0].count = BindingCount::RuntimeSized { max_count: 2 };
    let id = DeviceId::allocate().unwrap();
    let mut caps = DeviceCapabilities::default();
    caps.binding_arrays = true;
    caps.runtime_sized_binding_arrays = true;
    caps.acceleration_structure = true;
    caps.acceleration_structure_binding_arrays = true;
    caps.limits.max_bindings_per_group = 4;
    // SAFETY: this mock supports fixed acceleration arrays and generic runtime arrays.
    let device = unsafe { Device::<Mock>::from_native(id, (), caps) };
    assert!(matches!(
        device.create_bind_group_layout(&runtime_layout),
        Err(Error::Unsupported(
            "runtime-sized acceleration structure arrays"
        ))
    ));
    LAYOUT_CALLS.with(|calls| assert_eq!(calls.get(), 0));
}
