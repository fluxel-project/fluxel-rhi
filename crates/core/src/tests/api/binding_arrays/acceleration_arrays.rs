//! Acceleration-structure binding arrays require their dedicated capability.
use super::{GROUP_CALLS, LAYOUT_CALLS, TEXTURE_CALLS, device, entry};
use crate::api::tests::type_contract_tests::NativeShared;
use crate::{api::*, backend::*};
use std::sync::Arc;

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
    caps.limits.max_bindings_per_group = 4;
    // SAFETY: this fixture represents a device without the optional ray domain.
    let device = unsafe { Device::<NativeShared>::from_native(id, Arc::new(()), caps) };
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
    caps.limits.max_bindings_per_group = 4;
    // SAFETY: this mock has generic runtime arrays but no acceleration-array route.
    let device = unsafe { Device::<NativeShared>::from_native(id, Arc::new(()), caps) };
    assert!(matches!(
        device.create_bind_group_layout(&runtime_layout),
        Err(Error::Unsupported(
            "runtime-sized acceleration structure arrays"
        ))
    ));
    LAYOUT_CALLS.with(|calls| assert_eq!(calls.get(), 0));
}
