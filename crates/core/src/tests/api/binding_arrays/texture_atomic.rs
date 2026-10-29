//! Format, shader stage, and per-operation texture atomic admission.
use super::{GROUP_CALLS, LAYOUT_CALLS, TEXTURE_CALLS, device, entry};
use crate::api::tests::type_contract_tests::NativeShared;
use crate::{api::*, backend::*};
use std::sync::Arc;

#[test]
fn texture_atomic_layout_is_format_stage_and_capability_scoped() {
    LAYOUT_CALLS.with(|calls| calls.set(0));
    let layout = |format, visibility| BindGroupLayoutDesc {
        label: None,
        entries: vec![BindGroupLayoutEntry {
            binding: 0,
            visibility,
            ty: BindingType::StorageTexture {
                format,
                dimension: TextureViewDimension::D2,
                access: StorageAccess::Atomic,
            },
            count: BindingCount::One,
            partially_bound: false,
            nonuniform_indexing: false,
        }],
    };
    let r64_unavailable = device(false);
    assert!(matches!(
        r64_unavailable
            .create_bind_group_layout(&layout(TextureFormat::R64Uint, vec![ShaderStage::Compute])),
        Err(Error::Unsupported("R64Uint compute image atomic binding"))
    ));
    let unsupported = device(false);
    assert!(matches!(
        unsupported
            .create_bind_group_layout(&layout(TextureFormat::R32Uint, vec![ShaderStage::Compute])),
        Err(Error::Unsupported("R32Uint compute texture atomic binding"))
    ));
    let id = DeviceId::allocate().unwrap();
    let mut caps = DeviceCapabilities::default();
    caps.texture_atomic_r32uint_compute = true;
    caps.limits.max_bindings_per_group = 4;
    // SAFETY: the fixture supplies an atomic capability for its mock device.
    let supported = unsafe { Device::<NativeShared>::from_native(id, Arc::new(()), caps) };
    assert!(matches!(
        supported
            .create_bind_group_layout(&layout(TextureFormat::R32Sint, vec![ShaderStage::Compute])),
        Err(Error::Unsupported(_))
    ));
    assert!(matches!(
        supported.create_bind_group_layout(&layout(
            TextureFormat::R32Uint,
            vec![ShaderStage::Compute, ShaderStage::Fragment]
        )),
        Err(Error::Unsupported(_))
    ));
    assert!(
        supported
            .create_bind_group_layout(&layout(TextureFormat::R32Uint, vec![ShaderStage::Compute]))
            .is_ok()
    );
    assert!(matches!(
        supported
            .create_bind_group_layout(&layout(TextureFormat::R32Sint, vec![ShaderStage::Compute])),
        Err(Error::Unsupported(_))
    ));
    let sint_id = DeviceId::allocate().unwrap();
    let mut sint_caps = DeviceCapabilities::default();
    sint_caps.texture_atomic_r32sint_compute = true;
    sint_caps.limits.max_bindings_per_group = 4;
    // SAFETY: the fixture supplies the signed atomic capability for its mock device.
    let sint_supported =
        unsafe { Device::<NativeShared>::from_native(sint_id, Arc::new(()), sint_caps) };
    let sint_result = sint_supported
        .create_bind_group_layout(&layout(TextureFormat::R32Sint, vec![ShaderStage::Compute]));
    if let Err(error) = sint_result {
        panic!("{error:?}");
    }
    let r64_id = DeviceId::allocate().unwrap();
    let mut r64_caps = DeviceCapabilities::default();
    r64_caps.shader_int64_atomic_image_r64uint_umin = true;
    r64_caps.limits.max_bindings_per_group = 4;
    // SAFETY: the fixture supplies the R64Uint image atomic capability.
    let r64_supported =
        unsafe { Device::<NativeShared>::from_native(r64_id, Arc::new(()), r64_caps) };
    assert!(
        r64_supported
            .create_bind_group_layout(&layout(TextureFormat::R64Uint, vec![ShaderStage::Compute]))
            .is_ok()
    );
    let mut array_image_layout = layout(TextureFormat::R64Uint, vec![ShaderStage::Compute]);
    array_image_layout.entries[0].ty = BindingType::StorageTexture {
        format: TextureFormat::R64Uint,
        dimension: TextureViewDimension::D2Array,
        access: StorageAccess::Atomic,
    };
    assert!(matches!(
        r64_supported.create_bind_group_layout(&array_image_layout),
        Err(Error::Unsupported("R64Uint compute image atomic binding"))
    ));
    let r64_umax_id = DeviceId::allocate().unwrap();
    let mut r64_umax_caps = DeviceCapabilities::default();
    r64_umax_caps.shader_int64_atomic_image_r64uint_umax = true;
    r64_umax_caps.limits.max_bindings_per_group = 4;
    // SAFETY: this fixture exposes only the R64Uint image unsigned-max route.
    let r64_umax_supported =
        unsafe { Device::<NativeShared>::from_native(r64_umax_id, Arc::new(()), r64_umax_caps) };
    assert!(
        r64_umax_supported
            .create_bind_group_layout(&layout(TextureFormat::R64Uint, vec![ShaderStage::Compute]))
            .is_ok()
    );
    assert!(matches!(
        r64_supported.create_bind_group_layout(&layout(
            TextureFormat::R64Uint,
            vec![ShaderStage::Compute, ShaderStage::Fragment]
        )),
        Err(Error::Unsupported("R64Uint compute image atomic binding"))
    ));
    let min_id = DeviceId::allocate().unwrap();
    let mut min_caps = DeviceCapabilities::default();
    min_caps.texture_atomic_r32uint_min_compute = true;
    min_caps.limits.max_bindings_per_group = 4;
    // SAFETY: this fixture exposes only the unsigned min atomic route.
    let min_supported =
        unsafe { Device::<NativeShared>::from_native(min_id, Arc::new(()), min_caps) };
    assert!(
        min_supported
            .create_bind_group_layout(&layout(TextureFormat::R32Uint, vec![ShaderStage::Compute]))
            .is_ok()
    );
    let max_id = DeviceId::allocate().unwrap();
    let mut max_caps = DeviceCapabilities::default();
    max_caps.texture_atomic_r32uint_max_compute = true;
    max_caps.limits.max_bindings_per_group = 4;
    // SAFETY: this fixture exposes only the unsigned max atomic route.
    let max_supported =
        unsafe { Device::<NativeShared>::from_native(max_id, Arc::new(()), max_caps) };
    assert!(
        max_supported
            .create_bind_group_layout(&layout(TextureFormat::R32Uint, vec![ShaderStage::Compute]))
            .is_ok()
    );
    let sint_min_id = DeviceId::allocate().unwrap();
    let mut sint_min_caps = DeviceCapabilities::default();
    sint_min_caps.texture_atomic_r32sint_min_compute = true;
    sint_min_caps.limits.max_bindings_per_group = 4;
    // SAFETY: this fixture exposes only the signed min atomic route.
    let sint_min_supported =
        unsafe { Device::<NativeShared>::from_native(sint_min_id, Arc::new(()), sint_min_caps) };
    assert!(
        sint_min_supported
            .create_bind_group_layout(&layout(TextureFormat::R32Sint, vec![ShaderStage::Compute]))
            .is_ok()
    );
    let sint_max_id = DeviceId::allocate().unwrap();
    let mut sint_max_caps = DeviceCapabilities::default();
    sint_max_caps.texture_atomic_r32sint_max_compute = true;
    sint_max_caps.limits.max_bindings_per_group = 4;
    // SAFETY: this fixture exposes only the signed max atomic route.
    let sint_max_supported =
        unsafe { Device::<NativeShared>::from_native(sint_max_id, Arc::new(()), sint_max_caps) };
    assert!(
        sint_max_supported
            .create_bind_group_layout(&layout(TextureFormat::R32Sint, vec![ShaderStage::Compute]))
            .is_ok()
    );
    let xor_id = DeviceId::allocate().unwrap();
    let mut xor_caps = DeviceCapabilities::default();
    xor_caps.texture_atomic_r32uint_xor_compute = true;
    xor_caps.limits.max_bindings_per_group = 4;
    // SAFETY: this fixture exposes only the unsigned xor atomic route.
    let xor_supported =
        unsafe { Device::<NativeShared>::from_native(xor_id, Arc::new(()), xor_caps) };
    assert!(
        xor_supported
            .create_bind_group_layout(&layout(TextureFormat::R32Uint, vec![ShaderStage::Compute]))
            .is_ok()
    );
    LAYOUT_CALLS.with(|calls| assert_eq!(calls.get(), 9));
}
