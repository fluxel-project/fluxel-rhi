//! Stage-scoped shader feature admission and route-count checks.
use super::*;

#[test]
fn ray_query_requirement_is_checked_before_shader_backend() {
    ROUTES.with(|v| v.set(0));
    let device = |supported| unsafe {
        Device::<Mock>::from_native(
            DeviceId::allocate().unwrap(),
            (),
            DeviceCapabilities {
                ray_query: supported,
                ..Default::default()
            },
        )
    };
    let desc = |required| ShaderDesc {
        label: None,
        format: ShaderFormat::Dxil,
        bytes: Arc::from([1]),
        entries: vec![],
        required_features: required,
    };
    let unavailable = device(false);
    let required = desc(vec![SHADER_FEATURE_RAY_QUERY.into()]);
    assert!(matches!(
        unavailable.shader_route(&required),
        ShaderRoute::Unsupported
    ));
    let mut cx = Context::from_waker(Waker::noop());
    let mut failed = std::pin::pin!(unavailable.shader(required.clone()));
    assert!(matches!(
        failed.as_mut().poll(&mut cx),
        Poll::Ready(Err(Error::Unsupported(_)))
    ));
    ROUTES.with(|v| assert_eq!(v.get(), 0));
    let supported = device(true);
    let mut accepted = std::pin::pin!(supported.shader(required));
    assert!(matches!(
        accepted.as_mut().poll(&mut cx),
        Poll::Ready(Ok(_))
    ));
    assert!(matches!(
        unavailable.shader_route(&desc(vec![])),
        ShaderRoute::Direct(_)
    ));
    ROUTES.with(|v| assert_eq!(v.get(), 2));
}

#[test]
fn early_depth_test_requirement_is_fragment_only_and_checked_before_backend() {
    ROUTES.with(|v| v.set(0));
    let device = |supported| unsafe {
        Device::<Mock>::from_native(
            DeviceId::allocate().unwrap(),
            (),
            DeviceCapabilities {
                shader_early_depth_test: supported,
                ..Default::default()
            },
        )
    };
    let desc = |stage| ShaderDesc {
        label: None,
        format: ShaderFormat::Wgsl,
        bytes: Arc::from(b"@fragment fn fs() {}".as_slice()),
        entries: vec![ShaderEntryPoint {
            name: "main".into(),
            stage,
            workgroup_size: None,
        }],
        required_features: vec![SHADER_FEATURE_EARLY_DEPTH_TEST.into()],
    };
    let unavailable = device(false);
    assert!(matches!(
        unavailable.shader_route(&desc(ShaderStage::Fragment)),
        ShaderRoute::Unsupported
    ));
    let supported = device(true);
    assert!(matches!(
        supported.shader_route(&desc(ShaderStage::Vertex)),
        ShaderRoute::Unsupported
    ));
    assert!(matches!(
        supported.shader_route(&desc(ShaderStage::Fragment)),
        ShaderRoute::Direct(_)
    ));
    ROUTES.with(|v| assert_eq!(v.get(), 1));
}

#[test]
fn primitive_index_requirement_is_fragment_only_and_checked_before_backend() {
    ROUTES.with(|v| v.set(0));
    let device = |supported| unsafe {
        Device::<Mock>::from_native(
            DeviceId::allocate().unwrap(),
            (),
            DeviceCapabilities {
                shader_primitive_index: supported,
                ..Default::default()
            },
        )
    };
    let desc = |stage| ShaderDesc {
        label: None,
        format: ShaderFormat::Wgsl,
        bytes: Arc::from(b"@fragment fn fs() {}".as_slice()),
        entries: vec![ShaderEntryPoint {
            name: "main".into(),
            stage,
            workgroup_size: None,
        }],
        required_features: vec![SHADER_FEATURE_PRIMITIVE_INDEX.into()],
    };
    let unavailable = device(false);
    assert!(matches!(
        unavailable.shader_route(&desc(ShaderStage::Fragment)),
        ShaderRoute::Unsupported
    ));
    let supported = device(true);
    assert!(matches!(
        supported.shader_route(&desc(ShaderStage::Vertex)),
        ShaderRoute::Unsupported
    ));
    assert!(matches!(
        supported.shader_route(&desc(ShaderStage::Fragment)),
        ShaderRoute::Direct(_)
    ));
    ROUTES.with(|v| assert_eq!(v.get(), 1));
}

#[test]
fn draw_index_requirement_is_vertex_only_and_checked_before_backend() {
    ROUTES.with(|v| v.set(0));
    let device = |supported| unsafe {
        Device::<Mock>::from_native(
            DeviceId::allocate().unwrap(),
            (),
            DeviceCapabilities {
                shader_draw_index: supported,
                ..Default::default()
            },
        )
    };
    let desc = |stage| ShaderDesc {
        label: None,
        format: ShaderFormat::Wgsl,
        bytes: Arc::from(
            b"@vertex fn vs() -> @builtin(position) vec4f { return vec4f(0.0); }".as_slice(),
        ),
        entries: vec![ShaderEntryPoint {
            name: "main".into(),
            stage,
            workgroup_size: None,
        }],
        required_features: vec![SHADER_FEATURE_DRAW_INDEX.into()],
    };
    let unavailable = device(false);
    assert!(matches!(
        unavailable.shader_route(&desc(ShaderStage::Vertex)),
        ShaderRoute::Unsupported
    ));
    let supported = device(true);
    assert!(matches!(
        supported.shader_route(&desc(ShaderStage::Fragment)),
        ShaderRoute::Unsupported
    ));
    assert!(matches!(
        supported.shader_route(&desc(ShaderStage::Vertex)),
        ShaderRoute::Direct(_)
    ));
    ROUTES.with(|v| assert_eq!(v.get(), 1));
}

#[test]
fn barycentric_requirements_are_fragment_only_and_extension_scoped() {
    ROUTES.with(|v| v.set(0));
    let device = |standard, amd| unsafe {
        Device::<Mock>::from_native(
            DeviceId::allocate().unwrap(),
            (),
            DeviceCapabilities {
                shader_barycentrics: standard,
                shader_barycentrics_amd: amd,
                ..Default::default()
            },
        )
    };
    let desc = |stage: ShaderStage, requirement: &str| ShaderDesc {
        label: None,
        format: ShaderFormat::SpirV,
        bytes: Arc::from([1]),
        entries: vec![ShaderEntryPoint {
            name: "main".into(),
            stage,
            workgroup_size: None,
        }],
        required_features: vec![requirement.into()],
    };
    let no_barycentrics = device(false, false);
    assert!(matches!(
        no_barycentrics.shader_route(&desc(ShaderStage::Fragment, SHADER_FEATURE_BARYCENTRICS)),
        ShaderRoute::Unsupported
    ));
    assert!(matches!(
        no_barycentrics.shader_route(&desc(
            ShaderStage::Fragment,
            SHADER_FEATURE_BARYCENTRICS_AMD
        )),
        ShaderRoute::Unsupported
    ));
    let standard = device(true, false);
    assert!(matches!(
        standard.shader_route(&desc(ShaderStage::Vertex, SHADER_FEATURE_BARYCENTRICS)),
        ShaderRoute::Unsupported
    ));
    assert!(matches!(
        standard.shader_route(&desc(ShaderStage::Fragment, SHADER_FEATURE_BARYCENTRICS)),
        ShaderRoute::Direct(_)
    ));
    assert!(matches!(
        standard.shader_route(&desc(
            ShaderStage::Fragment,
            SHADER_FEATURE_BARYCENTRICS_AMD
        )),
        ShaderRoute::Unsupported
    ));
    let amd = device(false, true);
    assert!(matches!(
        amd.shader_route(&desc(
            ShaderStage::Fragment,
            SHADER_FEATURE_BARYCENTRICS_AMD
        )),
        ShaderRoute::Direct(_)
    ));
    ROUTES.with(|v| assert_eq!(v.get(), 2));
}
