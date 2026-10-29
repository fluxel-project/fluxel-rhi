//! Float32 atomic shader feature admission and route-count checks.
use super::*;

#[test]
fn float32_atomic_buffer_add_requirement_is_compute_only_and_capability_gated() {
    ROUTES.with(|v| v.set(0));
    let device = |supported| unsafe {
        Device::<Mock>::from_native(
            DeviceId::allocate().unwrap(),
            (),
            DeviceCapabilities {
                shader_float32_atomic_buffer_add: supported,
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
            workgroup_size: Some([1, 1, 1]),
        }],
        required_features: vec![SHADER_FEATURE_FLOAT32_ATOMIC_BUFFER_ADD.into()],
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
fn float32_atomic_workgroup_add_requirement_is_compute_only_and_capability_gated() {
    ROUTES.with(|v| v.set(0));
    let device = |supported| unsafe {
        Device::<Mock>::from_native(
            DeviceId::allocate().unwrap(),
            (),
            DeviceCapabilities {
                shader_float32_atomic_workgroup_add: supported,
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
        required_features: vec![SHADER_FEATURE_FLOAT32_ATOMIC_WORKGROUP_ADD.into()],
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
fn float32_atomic_workgroup_min_max_require_independent_capabilities() {
    ROUTES.with(|v| v.set(0));
    let device = |min, max| unsafe {
        Device::<Mock>::from_native(
            DeviceId::allocate().unwrap(),
            (),
            DeviceCapabilities {
                shader_float32_atomic_workgroup_min: min,
                shader_float32_atomic_workgroup_max: max,
                ..Default::default()
            },
        )
    };
    let desc = |stage: ShaderStage, feature: &str| ShaderDesc {
        label: None,
        format: ShaderFormat::SpirV,
        bytes: Arc::from([0u8; 20].as_slice()),
        entries: vec![ShaderEntryPoint {
            name: "main".into(),
            stage,
            workgroup_size: Some([37, 1, 1]),
        }],
        required_features: vec![feature.into()],
    };
    assert!(matches!(
        device(false, true).shader_route(&desc(
            ShaderStage::Compute,
            SHADER_FEATURE_FLOAT32_ATOMIC_WORKGROUP_MIN
        )),
        ShaderRoute::Unsupported
    ));
    assert!(matches!(
        device(true, false).shader_route(&desc(
            ShaderStage::Compute,
            SHADER_FEATURE_FLOAT32_ATOMIC_WORKGROUP_MAX
        )),
        ShaderRoute::Unsupported
    ));
    let supported = device(true, true);
    assert!(matches!(
        supported.shader_route(&desc(
            ShaderStage::Fragment,
            SHADER_FEATURE_FLOAT32_ATOMIC_WORKGROUP_MIN
        )),
        ShaderRoute::Unsupported
    ));
    assert!(matches!(
        supported.shader_route(&desc(
            ShaderStage::Compute,
            SHADER_FEATURE_FLOAT32_ATOMIC_WORKGROUP_MIN
        )),
        ShaderRoute::Direct(_)
    ));
    assert!(matches!(
        supported.shader_route(&desc(
            ShaderStage::Compute,
            SHADER_FEATURE_FLOAT32_ATOMIC_WORKGROUP_MAX
        )),
        ShaderRoute::Direct(_)
    ));
    ROUTES.with(|v| assert_eq!(v.get(), 2));
}

#[test]
fn float32_atomic_buffer_min_requirement_is_compute_only_and_capability_gated() {
    ROUTES.with(|v| v.set(0));
    let device = |supported| unsafe {
        Device::<Mock>::from_native(
            DeviceId::allocate().unwrap(),
            (),
            DeviceCapabilities {
                shader_float32_atomic_buffer_min: supported,
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
            workgroup_size: Some([1, 1, 1]),
        }],
        required_features: vec![SHADER_FEATURE_FLOAT32_ATOMIC_BUFFER_MIN.into()],
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
fn float32_atomic_buffer_max_requirement_is_compute_only_and_capability_gated() {
    ROUTES.with(|v| v.set(0));
    let device = |supported| unsafe {
        Device::<Mock>::from_native(
            DeviceId::allocate().unwrap(),
            (),
            DeviceCapabilities {
                shader_float32_atomic_buffer_max: supported,
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
            workgroup_size: Some([1, 1, 1]),
        }],
        required_features: vec![SHADER_FEATURE_FLOAT32_ATOMIC_BUFFER_MAX.into()],
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
