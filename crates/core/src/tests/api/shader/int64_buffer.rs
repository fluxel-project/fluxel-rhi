//! Int64 StorageBuffer atomic shader feature admission and route-count checks.
use super::*;

#[test]
fn int64_atomic_buffer_add_requirement_is_compute_only_and_capability_gated() {
    ROUTES.with(|v| v.set(0));
    let device = |supported| unsafe {
        Device::<Mock>::from_native(
            DeviceId::allocate().unwrap(),
            (),
            DeviceCapabilities {
                shader_int64_atomic_buffer_add: supported,
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
        required_features: vec![SHADER_FEATURE_INT64_ATOMIC_BUFFER_ADD.into()],
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
fn int64_atomic_buffer_min_requirement_is_compute_only_and_capability_gated() {
    ROUTES.with(|v| v.set(0));
    let device = |supported| unsafe {
        Device::<Mock>::from_native(
            DeviceId::allocate().unwrap(),
            (),
            DeviceCapabilities {
                shader_int64_atomic_buffer_min: supported,
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
        required_features: vec![SHADER_FEATURE_INT64_ATOMIC_BUFFER_MIN.into()],
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
fn int64_atomic_buffer_max_requirement_is_compute_only_and_capability_gated() {
    ROUTES.with(|v| v.set(0));
    let device = |supported| unsafe {
        Device::<Mock>::from_native(
            DeviceId::allocate().unwrap(),
            (),
            DeviceCapabilities {
                shader_int64_atomic_buffer_max: supported,
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
        required_features: vec![SHADER_FEATURE_INT64_ATOMIC_BUFFER_MAX.into()],
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
fn int64_atomic_buffer_xor_requirement_is_compute_only_and_capability_gated() {
    ROUTES.with(|v| v.set(0));
    let device = |supported| unsafe {
        Device::<Mock>::from_native(
            DeviceId::allocate().unwrap(),
            (),
            DeviceCapabilities {
                shader_int64_atomic_buffer_xor: supported,
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
        required_features: vec![SHADER_FEATURE_INT64_ATOMIC_BUFFER_XOR.into()],
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
fn int64_atomic_buffer_compare_exchange_requirement_is_compute_only_and_capability_gated() {
    ROUTES.with(|v| v.set(0));
    let device = |supported| unsafe {
        Device::<Mock>::from_native(
            DeviceId::allocate().unwrap(),
            (),
            DeviceCapabilities {
                shader_int64_atomic_buffer_compare_exchange: supported,
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
        required_features: vec![SHADER_FEATURE_INT64_ATOMIC_BUFFER_COMPARE_EXCHANGE.into()],
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
fn int64_atomic_buffer_exchange_requirement_is_compute_only_and_capability_gated() {
    ROUTES.with(|v| v.set(0));
    let device = |supported| unsafe {
        Device::<Mock>::from_native(
            DeviceId::allocate().unwrap(),
            (),
            DeviceCapabilities {
                shader_int64_atomic_buffer_exchange: supported,
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
        required_features: vec![SHADER_FEATURE_INT64_ATOMIC_BUFFER_EXCHANGE.into()],
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
fn int64_atomic_buffer_bitwise_requirements_are_compute_only_and_independent() {
    for (feature, is_and) in [
        (SHADER_FEATURE_INT64_ATOMIC_BUFFER_AND, true),
        (SHADER_FEATURE_INT64_ATOMIC_BUFFER_OR, false),
    ] {
        ROUTES.with(|v| v.set(0));
        let device = |and, or| unsafe {
            Device::<Mock>::from_native(
                DeviceId::allocate().unwrap(),
                (),
                DeviceCapabilities {
                    shader_int64_atomic_buffer_and: and,
                    shader_int64_atomic_buffer_or: or,
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
            required_features: vec![feature.into()],
        };
        let (and, or) = if is_and { (true, false) } else { (false, true) };
        assert!(matches!(
            device(false, false).shader_route(&desc(ShaderStage::Compute)),
            ShaderRoute::Unsupported
        ));
        assert!(matches!(
            device(and, or).shader_route(&desc(ShaderStage::Fragment)),
            ShaderRoute::Unsupported
        ));
        assert!(matches!(
            device(and, or).shader_route(&desc(ShaderStage::Compute)),
            ShaderRoute::Direct(_)
        ));
        let only_other = if is_and {
            device(false, true)
        } else {
            device(true, false)
        };
        assert!(matches!(
            only_other.shader_route(&desc(ShaderStage::Compute)),
            ShaderRoute::Unsupported
        ));
        ROUTES.with(|v| assert_eq!(v.get(), 1));
    }
}

#[test]
fn int64_atomic_buffer_umin_requirement_is_compute_only_and_capability_gated() {
    ROUTES.with(|v| v.set(0));
    let device = |supported| unsafe {
        Device::<Mock>::from_native(
            DeviceId::allocate().unwrap(),
            (),
            DeviceCapabilities {
                shader_int64_atomic_buffer_umin: supported,
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
        required_features: vec![SHADER_FEATURE_INT64_ATOMIC_BUFFER_UMIN.into()],
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
fn int64_atomic_buffer_umax_requirement_is_compute_only_and_capability_gated() {
    ROUTES.with(|v| v.set(0));
    let device = |supported| unsafe {
        Device::<Mock>::from_native(
            DeviceId::allocate().unwrap(),
            (),
            DeviceCapabilities {
                shader_int64_atomic_buffer_umax: supported,
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
        required_features: vec![SHADER_FEATURE_INT64_ATOMIC_BUFFER_UMAX.into()],
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
