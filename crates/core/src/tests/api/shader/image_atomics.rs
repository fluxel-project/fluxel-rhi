//! Image atomic shader feature admission and route-count checks.
use super::*;

#[test]
fn r32uint_texture_atomic_requirement_is_compute_only_and_capability_gated() {
    ROUTES.with(|v| v.set(0));
    let device = |supported| unsafe {
        Device::<Mock>::from_native(
            DeviceId::allocate().unwrap(),
            (),
            DeviceCapabilities {
                texture_atomic_r32uint_compute: supported,
                ..Default::default()
            },
        )
    };
    let desc = |stage| ShaderDesc {
        label: None,
        format: ShaderFormat::Wgsl,
        bytes: Arc::from(b"@compute @workgroup_size(1) fn main() {}".as_slice()),
        entries: vec![ShaderEntryPoint {
            name: "main".into(),
            stage,
            workgroup_size: Some([1, 1, 1]),
        }],
        required_features: vec![SHADER_FEATURE_TEXTURE_ATOMIC_R32UINT_ADD.into()],
    };
    assert!(matches!(
        device(false).shader_route(&desc(ShaderStage::Compute)),
        ShaderRoute::Unsupported
    ));
    let supported = device(true);
    assert!(matches!(
        supported.shader_route(&desc(ShaderStage::Fragment)),
        ShaderRoute::Unsupported
    ));
    assert!(matches!(
        supported.shader_route(&desc(ShaderStage::Compute)),
        ShaderRoute::Direct(_)
    ));
    ROUTES.with(|v| assert_eq!(v.get(), 1));
}

#[test]
fn r32sint_texture_atomic_requirement_is_compute_only_and_capability_gated() {
    ROUTES.with(|v| v.set(0));
    let device = |supported| unsafe {
        Device::<Mock>::from_native(
            DeviceId::allocate().unwrap(),
            (),
            DeviceCapabilities {
                texture_atomic_r32sint_compute: supported,
                ..Default::default()
            },
        )
    };
    let desc = |stage| ShaderDesc {
        label: None,
        format: ShaderFormat::Wgsl,
        bytes: Arc::from(b"@compute @workgroup_size(1) fn main() {}".as_slice()),
        entries: vec![ShaderEntryPoint {
            name: "main".into(),
            stage,
            workgroup_size: Some([1, 1, 1]),
        }],
        required_features: vec![SHADER_FEATURE_TEXTURE_ATOMIC_R32SINT_ADD.into()],
    };
    assert!(matches!(
        device(false).shader_route(&desc(ShaderStage::Compute)),
        ShaderRoute::Unsupported
    ));
    let supported = device(true);
    assert!(matches!(
        supported.shader_route(&desc(ShaderStage::Fragment)),
        ShaderRoute::Unsupported
    ));
    assert!(matches!(
        supported.shader_route(&desc(ShaderStage::Compute)),
        ShaderRoute::Direct(_)
    ));
    ROUTES.with(|v| assert_eq!(v.get(), 1));
}

#[test]
fn r32sint_texture_atomic_min_requirement_is_compute_only_and_capability_gated() {
    ROUTES.with(|v| v.set(0));
    let device = |supported| unsafe {
        Device::<Mock>::from_native(
            DeviceId::allocate().unwrap(),
            (),
            DeviceCapabilities {
                texture_atomic_r32sint_min_compute: supported,
                ..Default::default()
            },
        )
    };
    let desc = |stage| ShaderDesc {
        label: None,
        format: ShaderFormat::Wgsl,
        bytes: Arc::from(b"@compute @workgroup_size(1) fn main() {}".as_slice()),
        entries: vec![ShaderEntryPoint {
            name: "main".into(),
            stage,
            workgroup_size: Some([1, 1, 1]),
        }],
        required_features: vec![SHADER_FEATURE_TEXTURE_ATOMIC_R32SINT_MIN.into()],
    };
    assert!(matches!(
        device(false).shader_route(&desc(ShaderStage::Compute)),
        ShaderRoute::Unsupported
    ));
    let supported = device(true);
    assert!(matches!(
        supported.shader_route(&desc(ShaderStage::Fragment)),
        ShaderRoute::Unsupported
    ));
    assert!(matches!(
        supported.shader_route(&desc(ShaderStage::Compute)),
        ShaderRoute::Direct(_)
    ));
    ROUTES.with(|v| assert_eq!(v.get(), 1));
}

#[test]
fn r32sint_texture_atomic_max_requirement_is_compute_only_and_capability_gated() {
    ROUTES.with(|v| v.set(0));
    let device = |supported| unsafe {
        Device::<Mock>::from_native(
            DeviceId::allocate().unwrap(),
            (),
            DeviceCapabilities {
                texture_atomic_r32sint_max_compute: supported,
                ..Default::default()
            },
        )
    };
    let desc = |stage| ShaderDesc {
        label: None,
        format: ShaderFormat::Wgsl,
        bytes: Arc::from(b"@compute @workgroup_size(1) fn main() {}".as_slice()),
        entries: vec![ShaderEntryPoint {
            name: "main".into(),
            stage,
            workgroup_size: Some([1, 1, 1]),
        }],
        required_features: vec![SHADER_FEATURE_TEXTURE_ATOMIC_R32SINT_MAX.into()],
    };
    assert!(matches!(
        device(false).shader_route(&desc(ShaderStage::Compute)),
        ShaderRoute::Unsupported
    ));
    let supported = device(true);
    assert!(matches!(
        supported.shader_route(&desc(ShaderStage::Fragment)),
        ShaderRoute::Unsupported
    ));
    assert!(matches!(
        supported.shader_route(&desc(ShaderStage::Compute)),
        ShaderRoute::Direct(_)
    ));
    ROUTES.with(|v| assert_eq!(v.get(), 1));
}

#[test]
fn r32uint_texture_atomic_min_requirement_is_compute_only_and_capability_gated() {
    ROUTES.with(|v| v.set(0));
    let device = |supported| unsafe {
        Device::<Mock>::from_native(
            DeviceId::allocate().unwrap(),
            (),
            DeviceCapabilities {
                texture_atomic_r32uint_min_compute: supported,
                ..Default::default()
            },
        )
    };
    let desc = |stage| ShaderDesc {
        label: None,
        format: ShaderFormat::Wgsl,
        bytes: Arc::from(b"@compute @workgroup_size(1) fn main() {}".as_slice()),
        entries: vec![ShaderEntryPoint {
            name: "main".into(),
            stage,
            workgroup_size: Some([1, 1, 1]),
        }],
        required_features: vec![SHADER_FEATURE_TEXTURE_ATOMIC_R32UINT_MIN.into()],
    };
    assert!(matches!(
        device(false).shader_route(&desc(ShaderStage::Compute)),
        ShaderRoute::Unsupported
    ));
    let supported = device(true);
    assert!(matches!(
        supported.shader_route(&desc(ShaderStage::Fragment)),
        ShaderRoute::Unsupported
    ));
    assert!(matches!(
        supported.shader_route(&desc(ShaderStage::Compute)),
        ShaderRoute::Direct(_)
    ));
    ROUTES.with(|v| assert_eq!(v.get(), 1));
}

#[test]
fn r32uint_texture_atomic_max_requirement_is_compute_only_and_capability_gated() {
    ROUTES.with(|v| v.set(0));
    let device = |supported| unsafe {
        Device::<Mock>::from_native(
            DeviceId::allocate().unwrap(),
            (),
            DeviceCapabilities {
                texture_atomic_r32uint_min_compute: true,
                texture_atomic_r32uint_max_compute: supported,
                ..Default::default()
            },
        )
    };
    let desc = |stage| ShaderDesc {
        label: None,
        format: ShaderFormat::Wgsl,
        bytes: Arc::from(b"@compute @workgroup_size(1) fn main() {}".as_slice()),
        entries: vec![ShaderEntryPoint {
            name: "main".into(),
            stage,
            workgroup_size: Some([1, 1, 1]),
        }],
        required_features: vec![SHADER_FEATURE_TEXTURE_ATOMIC_R32UINT_MAX.into()],
    };
    assert!(matches!(
        device(false).shader_route(&desc(ShaderStage::Compute)),
        ShaderRoute::Unsupported
    ));
    let supported = device(true);
    assert!(matches!(
        supported.shader_route(&desc(ShaderStage::Fragment)),
        ShaderRoute::Unsupported
    ));
    assert!(matches!(
        supported.shader_route(&desc(ShaderStage::Compute)),
        ShaderRoute::Direct(_)
    ));
    ROUTES.with(|v| assert_eq!(v.get(), 1));
}

#[test]
fn r32uint_texture_atomic_xor_requirement_is_compute_only_and_capability_gated() {
    ROUTES.with(|v| v.set(0));
    let device = |supported| unsafe {
        Device::<Mock>::from_native(
            DeviceId::allocate().unwrap(),
            (),
            DeviceCapabilities {
                texture_atomic_r32uint_xor_compute: supported,
                ..Default::default()
            },
        )
    };
    let desc = |stage| ShaderDesc {
        label: None,
        format: ShaderFormat::Wgsl,
        bytes: Arc::from(b"@compute @workgroup_size(1) fn main() {}".as_slice()),
        entries: vec![ShaderEntryPoint {
            name: "main".into(),
            stage,
            workgroup_size: Some([1, 1, 1]),
        }],
        required_features: vec![SHADER_FEATURE_TEXTURE_ATOMIC_R32UINT_XOR.into()],
    };
    assert!(matches!(
        device(false).shader_route(&desc(ShaderStage::Compute)),
        ShaderRoute::Unsupported
    ));
    let supported = device(true);
    assert!(matches!(
        supported.shader_route(&desc(ShaderStage::Fragment)),
        ShaderRoute::Unsupported
    ));
    assert!(matches!(
        supported.shader_route(&desc(ShaderStage::Compute)),
        ShaderRoute::Direct(_)
    ));
    ROUTES.with(|v| assert_eq!(v.get(), 1));
}

#[test]
fn r32uint_texture_atomic_and_requirement_is_compute_only_and_capability_gated() {
    ROUTES.with(|v| v.set(0));
    let make_device = |supported| unsafe {
        Device::<Mock>::from_native(
            DeviceId::allocate().unwrap(),
            (),
            DeviceCapabilities {
                texture_atomic_r32uint_and_compute: supported,
                ..Default::default()
            },
        )
    };
    let desc = |stage| ShaderDesc {
        label: None,
        format: ShaderFormat::Wgsl,
        bytes: Arc::from(b"@compute @workgroup_size(1) fn main() {}".as_slice()),
        entries: vec![ShaderEntryPoint {
            name: "main".into(),
            stage,
            workgroup_size: Some([1, 1, 1]),
        }],
        required_features: vec![SHADER_FEATURE_TEXTURE_ATOMIC_R32UINT_AND.into()],
    };
    assert!(matches!(
        make_device(false).shader_route(&desc(ShaderStage::Compute)),
        ShaderRoute::Unsupported
    ));
    let supported = make_device(true);
    assert!(matches!(
        supported.shader_route(&desc(ShaderStage::Fragment)),
        ShaderRoute::Unsupported
    ));
    assert!(matches!(
        supported.shader_route(&desc(ShaderStage::Compute)),
        ShaderRoute::Direct(_)
    ));
    ROUTES.with(|v| assert_eq!(v.get(), 1));
}

#[test]
fn r32uint_texture_atomic_or_requirement_is_compute_only_and_capability_gated() {
    ROUTES.with(|v| v.set(0));
    let make_device = |supported| unsafe {
        Device::<Mock>::from_native(
            DeviceId::allocate().unwrap(),
            (),
            DeviceCapabilities {
                texture_atomic_r32uint_or_compute: supported,
                ..Default::default()
            },
        )
    };
    let desc = |stage| ShaderDesc {
        label: None,
        format: ShaderFormat::Wgsl,
        bytes: Arc::from(b"@compute @workgroup_size(1) fn main() {}".as_slice()),
        entries: vec![ShaderEntryPoint {
            name: "main".into(),
            stage,
            workgroup_size: Some([1, 1, 1]),
        }],
        required_features: vec![SHADER_FEATURE_TEXTURE_ATOMIC_R32UINT_OR.into()],
    };
    assert!(matches!(
        make_device(false).shader_route(&desc(ShaderStage::Compute)),
        ShaderRoute::Unsupported
    ));
    let supported = make_device(true);
    assert!(matches!(
        supported.shader_route(&desc(ShaderStage::Fragment)),
        ShaderRoute::Unsupported
    ));
    assert!(matches!(
        supported.shader_route(&desc(ShaderStage::Compute)),
        ShaderRoute::Direct(_)
    ));
    ROUTES.with(|v| assert_eq!(v.get(), 1));
}

#[test]
fn int64_r64uint_image_atomic_requirement_is_compute_only_and_capability_gated() {
    ROUTES.with(|v| v.set(0));
    let device = |supported| unsafe {
        Device::<Mock>::from_native(
            DeviceId::allocate().unwrap(),
            (),
            DeviceCapabilities {
                shader_int64_atomic_image_r64uint_add: supported,
                ..Default::default()
            },
        )
    };
    let desc = |stage| ShaderDesc {
        label: None,
        format: ShaderFormat::SpirV,
        bytes: Arc::from([0u8; 20].as_slice()),
        entries: vec![ShaderEntryPoint {
            name: "main".into(),
            stage,
            workgroup_size: Some([37, 1, 1]),
        }],
        required_features: vec![SHADER_FEATURE_INT64_ATOMIC_IMAGE_R64UINT_ADD.into()],
    };
    assert!(matches!(
        device(false).shader_route(&desc(ShaderStage::Compute)),
        ShaderRoute::Unsupported
    ));
    let supported = device(true);
    assert!(matches!(
        supported.shader_route(&desc(ShaderStage::Vertex)),
        ShaderRoute::Unsupported
    ));
    assert!(matches!(
        supported.shader_route(&desc(ShaderStage::Compute)),
        ShaderRoute::Direct(_)
    ));
    ROUTES.with(|v| assert_eq!(v.get(), 1));
}

#[test]
fn int64_r64uint_image_unsigned_min_requirement_is_compute_only_and_capability_gated() {
    ROUTES.with(|v| v.set(0));
    let device = |supported| unsafe {
        Device::<Mock>::from_native(
            DeviceId::allocate().unwrap(),
            (),
            DeviceCapabilities {
                shader_int64_atomic_image_r64uint_umin: supported,
                ..Default::default()
            },
        )
    };
    let desc = |stage| ShaderDesc {
        label: None,
        format: ShaderFormat::SpirV,
        bytes: Arc::from([0u8; 20].as_slice()),
        entries: vec![ShaderEntryPoint {
            name: "main".into(),
            stage,
            workgroup_size: Some([37, 1, 1]),
        }],
        required_features: vec![SHADER_FEATURE_INT64_ATOMIC_IMAGE_R64UINT_UMIN.into()],
    };
    assert!(matches!(
        device(false).shader_route(&desc(ShaderStage::Compute)),
        ShaderRoute::Unsupported
    ));
    let supported = device(true);
    assert!(matches!(
        supported.shader_route(&desc(ShaderStage::Fragment)),
        ShaderRoute::Unsupported
    ));
    assert!(matches!(
        supported.shader_route(&desc(ShaderStage::Compute)),
        ShaderRoute::Direct(_)
    ));
    ROUTES.with(|v| assert_eq!(v.get(), 1));
}

#[test]
fn int64_r64uint_image_unsigned_max_requirement_is_compute_only_and_capability_gated() {
    ROUTES.with(|v| v.set(0));
    let device = |supported| unsafe {
        Device::<Mock>::from_native(
            DeviceId::allocate().unwrap(),
            (),
            DeviceCapabilities {
                shader_int64_atomic_image_r64uint_umax: supported,
                ..Default::default()
            },
        )
    };
    let desc = |stage| ShaderDesc {
        label: None,
        format: ShaderFormat::SpirV,
        bytes: Arc::from([0u8; 20].as_slice()),
        entries: vec![ShaderEntryPoint {
            name: "main".into(),
            stage,
            workgroup_size: Some([37, 1, 1]),
        }],
        required_features: vec![SHADER_FEATURE_INT64_ATOMIC_IMAGE_R64UINT_UMAX.into()],
    };
    assert!(matches!(
        device(false).shader_route(&desc(ShaderStage::Compute)),
        ShaderRoute::Unsupported
    ));
    let supported = device(true);
    assert!(matches!(
        supported.shader_route(&desc(ShaderStage::Fragment)),
        ShaderRoute::Unsupported
    ));
    assert!(matches!(
        supported.shader_route(&desc(ShaderStage::Compute)),
        ShaderRoute::Direct(_)
    ));
    ROUTES.with(|v| assert_eq!(v.get(), 1));
}
