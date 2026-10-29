//! Binding-array shape and nonuniform resource capability admission.
use super::{GROUP_CALLS, LAYOUT_CALLS, TEXTURE_CALLS, device, entry};
use crate::api::tests::type_contract_tests::NativeShared;
use crate::{api::*, backend::*};
use std::sync::Arc;

#[test]
fn runtime_array_layout_validates_capacity_capability_and_position() {
    LAYOUT_CALLS.with(|calls| calls.set(0));
    let device = device(true);
    let layout = |entries| BindGroupLayoutDesc {
        label: None,
        entries,
    };
    assert!(matches!(
        device.create_bind_group_layout(&layout(vec![entry(
            0,
            BindingCount::RuntimeSized { max_count: 0 },
            false
        )])),
        Err(Error::InvalidRange)
    ));
    assert!(matches!(
        device.create_bind_group_layout(&layout(vec![
            entry(0, BindingCount::RuntimeSized { max_count: 2 }, false),
            entry(1, BindingCount::One, false)
        ])),
        Err(Error::InvalidState)
    ));
    assert!(matches!(
        self::device(false).create_bind_group_layout(&layout(vec![entry(
            0,
            BindingCount::RuntimeSized { max_count: 2 },
            false
        )])),
        Err(Error::Unsupported("runtime-sized binding arrays"))
    ));
    assert!(
        device
            .create_bind_group_layout(&layout(vec![
                entry(0, BindingCount::One, false),
                entry(1, BindingCount::RuntimeSized { max_count: 2 }, false)
            ]))
            .is_ok()
    );
    LAYOUT_CALLS.with(|calls| assert_eq!(calls.get(), 1));
}

#[test]
fn storage_texture_nonuniform_indexing_uses_its_own_capability() {
    LAYOUT_CALLS.with(|calls| calls.set(0));
    let layout = BindGroupLayoutDesc {
        label: None,
        entries: vec![BindGroupLayoutEntry {
            binding: 0,
            visibility: vec![ShaderStage::Compute],
            ty: BindingType::StorageTexture {
                format: TextureFormat::Rgba8Unorm,
                dimension: TextureViewDimension::D2,
                access: StorageAccess::Write,
            },
            count: BindingCount::Fixed(2),
            partially_bound: false,
            nonuniform_indexing: true,
        }],
    };
    let id = DeviceId::allocate().unwrap();
    let mut caps = DeviceCapabilities::default();
    caps.binding_arrays = true;
    // The compatibility aggregate must not admit a storage texture by itself.
    caps.nonuniform_sampled_texture_and_storage_buffer_indexing = true;
    caps.limits.max_bindings_per_group = 4;
    // SAFETY: this fixture supplies matching capabilities for its mock device.
    let device_without_storage_texture_indexing =
        unsafe { Device::<NativeShared>::from_native(id, Arc::new(()), caps) };
    assert!(matches!(
        device_without_storage_texture_indexing.create_bind_group_layout(&layout),
        Err(Error::Unsupported("nonuniform binding indexing"))
    ));

    let id = DeviceId::allocate().unwrap();
    let mut caps = DeviceCapabilities::default();
    caps.binding_arrays = true;
    caps.nonuniform_storage_texture_indexing = true;
    caps.limits.max_bindings_per_group = 4;
    // SAFETY: this fixture supplies matching capabilities for its mock device.
    let device = unsafe { Device::<NativeShared>::from_native(id, Arc::new(()), caps) };
    assert!(device.create_bind_group_layout(&layout).is_ok());
    LAYOUT_CALLS.with(|calls| assert_eq!(calls.get(), 1));
}

#[test]
fn uniform_buffer_nonuniform_indexing_uses_its_own_capability() {
    LAYOUT_CALLS.with(|calls| calls.set(0));
    let layout = BindGroupLayoutDesc {
        label: None,
        entries: vec![BindGroupLayoutEntry {
            binding: 0,
            visibility: vec![ShaderStage::Compute],
            ty: BindingType::UniformBuffer {
                dynamic_offset: false,
                min_size: 4,
            },
            count: BindingCount::Fixed(2),
            partially_bound: false,
            nonuniform_indexing: true,
        }],
    };
    let id = DeviceId::allocate().unwrap();
    let mut caps = DeviceCapabilities::default();
    caps.binding_arrays = true;
    caps.nonuniform_sampled_texture_and_storage_buffer_indexing = true;
    caps.limits.max_bindings_per_group = 4;
    // The compatibility aggregate must not admit a uniform buffer by itself.
    // SAFETY: this mock reports only the compatibility aggregate, not UBO indexing.
    let without = unsafe { Device::<NativeShared>::from_native(id, Arc::new(()), caps) };
    assert!(matches!(
        without.create_bind_group_layout(&layout),
        Err(Error::Unsupported("nonuniform binding indexing"))
    ));

    let id = DeviceId::allocate().unwrap();
    let mut caps = DeviceCapabilities::default();
    caps.binding_arrays = true;
    caps.nonuniform_uniform_buffer_indexing = true;
    caps.limits.max_bindings_per_group = 4;
    // SAFETY: this mock explicitly reports the UBO indexing capability.
    let with = unsafe { Device::<NativeShared>::from_native(id, Arc::new(()), caps) };
    assert!(with.create_bind_group_layout(&layout).is_ok());
    LAYOUT_CALLS.with(|calls| assert_eq!(calls.get(), 1));
}

#[test]
fn storage_buffer_nonuniform_indexing_uses_its_own_capability() {
    LAYOUT_CALLS.with(|calls| calls.set(0));
    let layout = BindGroupLayoutDesc {
        label: None,
        entries: vec![BindGroupLayoutEntry {
            binding: 0,
            visibility: vec![ShaderStage::Compute],
            ty: BindingType::StorageBuffer {
                dynamic_offset: false,
                min_size: 4,
                access: StorageAccess::Read,
            },
            count: BindingCount::Fixed(2),
            partially_bound: false,
            nonuniform_indexing: true,
        }],
    };
    let id = DeviceId::allocate().unwrap();
    let mut caps = DeviceCapabilities::default();
    caps.binding_arrays = true;
    caps.nonuniform_sampled_texture_and_storage_buffer_indexing = true;
    caps.limits.max_bindings_per_group = 4;
    // The compatibility aggregate must not admit a storage buffer by itself.
    // SAFETY: this mock reports only the compatibility aggregate, not storage-buffer indexing.
    let without = unsafe { Device::<NativeShared>::from_native(id, Arc::new(()), caps) };
    assert!(matches!(
        without.create_bind_group_layout(&layout),
        Err(Error::Unsupported("nonuniform binding indexing"))
    ));

    let id = DeviceId::allocate().unwrap();
    let mut caps = DeviceCapabilities::default();
    caps.binding_arrays = true;
    caps.nonuniform_storage_buffer_indexing = true;
    caps.limits.max_bindings_per_group = 4;
    // SAFETY: this mock explicitly reports storage-buffer indexing.
    let with = unsafe { Device::<NativeShared>::from_native(id, Arc::new(()), caps) };
    assert!(with.create_bind_group_layout(&layout).is_ok());
    LAYOUT_CALLS.with(|calls| assert_eq!(calls.get(), 1));
}

#[test]
fn sampled_texture_nonuniform_indexing_uses_its_own_capability() {
    LAYOUT_CALLS.with(|calls| calls.set(0));
    let layout = BindGroupLayoutDesc {
        label: None,
        entries: vec![BindGroupLayoutEntry {
            binding: 0,
            visibility: vec![ShaderStage::Compute],
            ty: BindingType::SampledTexture {
                sample: TextureSampleType::UnfilterableFloat,
                dimension: TextureViewDimension::D2,
                multisampled: false,
            },
            count: BindingCount::Fixed(2),
            partially_bound: false,
            nonuniform_indexing: true,
        }],
    };
    let id = DeviceId::allocate().unwrap();
    let mut caps = DeviceCapabilities::default();
    caps.binding_arrays = true;
    caps.nonuniform_sampled_texture_and_storage_buffer_indexing = true;
    caps.limits.max_bindings_per_group = 4;
    // The compatibility aggregate must not admit sampled textures by itself.
    // SAFETY: this mock reports only the legacy aggregate, not sampled-texture indexing.
    let without = unsafe { Device::<NativeShared>::from_native(id, Arc::new(()), caps) };
    assert!(matches!(
        without.create_bind_group_layout(&layout),
        Err(Error::Unsupported("nonuniform binding indexing"))
    ));

    let id = DeviceId::allocate().unwrap();
    let mut caps = DeviceCapabilities::default();
    caps.binding_arrays = true;
    caps.nonuniform_sampled_texture_indexing = true;
    caps.limits.max_bindings_per_group = 4;
    // SAFETY: this mock explicitly reports sampled-texture indexing.
    let with = unsafe { Device::<NativeShared>::from_native(id, Arc::new(()), caps) };
    assert!(with.create_bind_group_layout(&layout).is_ok());
    LAYOUT_CALLS.with(|calls| assert_eq!(calls.get(), 1));
}
