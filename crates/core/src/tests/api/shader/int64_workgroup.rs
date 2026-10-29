//! Int64 Workgroup atomic shader feature admission and route-count checks.
use super::*;

#[test]
fn int64_atomic_workgroup_add_requirement_is_compute_only_and_capability_gated() {
    ROUTES.with(|v| v.set(0));
    let device = |supported| unsafe {
        Device::<Mock>::from_native(
            DeviceId::allocate().unwrap(),
            (),
            DeviceCapabilities {
                shader_int64_atomic_workgroup_add: supported,
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
        required_features: vec![SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_ADD.into()],
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
fn int64_atomic_workgroup_min_max_require_independent_capabilities() {
    ROUTES.with(|v| v.set(0));
    let device = |min, max| unsafe {
        Device::<Mock>::from_native(
            DeviceId::allocate().unwrap(),
            (),
            DeviceCapabilities {
                shader_int64_atomic_workgroup_min: min,
                shader_int64_atomic_workgroup_max: max,
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
            SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_MIN
        )),
        ShaderRoute::Unsupported
    ));
    assert!(matches!(
        device(true, false).shader_route(&desc(
            ShaderStage::Compute,
            SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_MAX
        )),
        ShaderRoute::Unsupported
    ));
    let supported = device(true, true);
    assert!(matches!(
        supported.shader_route(&desc(
            ShaderStage::Fragment,
            SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_MIN
        )),
        ShaderRoute::Unsupported
    ));
    assert!(matches!(
        supported.shader_route(&desc(
            ShaderStage::Compute,
            SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_MIN
        )),
        ShaderRoute::Direct(_)
    ));
    assert!(matches!(
        supported.shader_route(&desc(
            ShaderStage::Compute,
            SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_MAX
        )),
        ShaderRoute::Direct(_)
    ));
    ROUTES.with(|v| assert_eq!(v.get(), 2));
}

#[test]
fn int64_atomic_workgroup_xor_is_compute_only_and_independently_gated() {
    ROUTES.with(|v| v.set(0));
    let device = |supported| unsafe {
        Device::<Mock>::from_native(
            DeviceId::allocate().unwrap(),
            (),
            DeviceCapabilities {
                shader_int64_atomic_workgroup_xor: supported,
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
        required_features: vec![SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_XOR.into()],
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
fn int64_atomic_workgroup_and_or_require_independent_capabilities() {
    ROUTES.with(|v| v.set(0));
    let device = |and, or| unsafe {
        Device::<Mock>::from_native(
            DeviceId::allocate().unwrap(),
            (),
            DeviceCapabilities {
                shader_int64_atomic_workgroup_and: and,
                shader_int64_atomic_workgroup_or: or,
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
            SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_AND
        )),
        ShaderRoute::Unsupported
    ));
    assert!(matches!(
        device(true, false).shader_route(&desc(
            ShaderStage::Compute,
            SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_OR
        )),
        ShaderRoute::Unsupported
    ));
    let supported = device(true, true);
    assert!(matches!(
        supported.shader_route(&desc(
            ShaderStage::Fragment,
            SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_AND
        )),
        ShaderRoute::Unsupported
    ));
    assert!(matches!(
        supported.shader_route(&desc(
            ShaderStage::Compute,
            SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_AND
        )),
        ShaderRoute::Direct(_)
    ));
    assert!(matches!(
        supported.shader_route(&desc(
            ShaderStage::Compute,
            SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_OR
        )),
        ShaderRoute::Direct(_)
    ));
    ROUTES.with(|v| assert_eq!(v.get(), 2));
}

#[test]
fn int64_atomic_workgroup_unsigned_min_max_are_compute_only_and_independent() {
    ROUTES.with(|v| v.set(0));
    let device = |umin, umax| unsafe {
        Device::<Mock>::from_native(
            DeviceId::allocate().unwrap(),
            (),
            DeviceCapabilities {
                shader_int64_atomic_workgroup_umin: umin,
                shader_int64_atomic_workgroup_umax: umax,
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
            SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_UMIN
        )),
        ShaderRoute::Unsupported
    ));
    assert!(matches!(
        device(true, false).shader_route(&desc(
            ShaderStage::Compute,
            SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_UMAX
        )),
        ShaderRoute::Unsupported
    ));
    let supported = device(true, true);
    assert!(matches!(
        supported.shader_route(&desc(
            ShaderStage::Vertex,
            SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_UMIN
        )),
        ShaderRoute::Unsupported
    ));
    assert!(matches!(
        supported.shader_route(&desc(
            ShaderStage::Compute,
            SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_UMIN
        )),
        ShaderRoute::Direct(_)
    ));
    assert!(matches!(
        supported.shader_route(&desc(
            ShaderStage::Compute,
            SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_UMAX
        )),
        ShaderRoute::Direct(_)
    ));
    ROUTES.with(|v| assert_eq!(v.get(), 2));
}

#[test]
fn int64_atomic_workgroup_exchange_routes_are_compute_only_and_independent() {
    ROUTES.with(|v| v.set(0));
    let device = |exchange, compare_exchange| unsafe {
        Device::<Mock>::from_native(
            DeviceId::allocate().unwrap(),
            (),
            DeviceCapabilities {
                shader_int64_atomic_workgroup_exchange: exchange,
                shader_int64_atomic_workgroup_compare_exchange: compare_exchange,
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
            SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_EXCHANGE
        )),
        ShaderRoute::Unsupported
    ));
    assert!(matches!(
        device(true, false).shader_route(&desc(
            ShaderStage::Compute,
            SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_COMPARE_EXCHANGE
        )),
        ShaderRoute::Unsupported
    ));
    let supported = device(true, true);
    assert!(matches!(
        supported.shader_route(&desc(
            ShaderStage::Fragment,
            SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_EXCHANGE
        )),
        ShaderRoute::Unsupported
    ));
    assert!(matches!(
        supported.shader_route(&desc(
            ShaderStage::Compute,
            SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_EXCHANGE
        )),
        ShaderRoute::Direct(_)
    ));
    assert!(matches!(
        supported.shader_route(&desc(
            ShaderStage::Compute,
            SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_COMPARE_EXCHANGE
        )),
        ShaderRoute::Direct(_)
    ));
    ROUTES.with(|v| assert_eq!(v.get(), 2));
}
