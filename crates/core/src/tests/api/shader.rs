//! Unit tests for the parent module.

use super::super::tests::ray_validation::Mock;
use super::*;
use std::{
    cell::Cell,
    future::{Ready, ready},
    task::{Context, Poll, Waker},
};
thread_local! { static ROUTES: Cell<usize> = const { Cell::new(0) }; }
impl ShaderBackend for Mock {
    type DirectFuture<'a> = Ready<Result<()>>;
    type NagaFuture<'a> = Ready<Result<()>>;
    fn shader_route(_: &(), _: &ShaderDesc) -> ShaderRoute {
        ROUTES.with(|v| v.set(v.get() + 1));
        ShaderRoute::Direct(ShaderTarget {
            format: ShaderFormat::Dxil,
            environment: String::new(),
            abi_key: String::new(),
        })
    }
    fn compile_direct<'a>(
        _: &'a (),
        _: &'a ShaderDesc,
        _: &'a ShaderTarget,
    ) -> Self::DirectFuture<'a> {
        ready(Ok(()))
    }
    fn compile_naga<'a>(_: &'a (), _: &'a ShaderDesc, _: &'a ShaderTarget) -> Self::NagaFuture<'a> {
        unreachable!()
    }
}
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

#[test]
fn spirv_memory_decoration_requirements_are_format_stage_and_capability_gated() {
    ROUTES.with(|v| v.set(0));
    let device = |coherent, volatile| unsafe {
        Device::<Mock>::from_native(
            DeviceId::allocate().unwrap(),
            (),
            DeviceCapabilities {
                shader_memory_coherent_spirv: coherent,
                shader_memory_volatile_spirv: volatile,
                ..Default::default()
            },
        )
    };
    let desc = |format, stage, feature: &str| ShaderDesc {
        label: None,
        format,
        bytes: Arc::from([0u8; 20].as_slice()),
        entries: vec![ShaderEntryPoint {
            name: "main".into(),
            stage,
            workgroup_size: Some([37, 1, 1]),
        }],
        required_features: vec![feature.into()],
    };
    assert!(matches!(
        device(false, false).shader_route(&desc(
            ShaderFormat::SpirV,
            ShaderStage::Compute,
            SHADER_FEATURE_MEMORY_COHERENT_SPIRV,
        )),
        ShaderRoute::Unsupported
    ));
    assert!(matches!(
        device(true, false).shader_route(&desc(
            ShaderFormat::SpirV,
            ShaderStage::Compute,
            SHADER_FEATURE_MEMORY_VOLATILE_SPIRV,
        )),
        ShaderRoute::Unsupported
    ));
    let supported = device(true, true);
    assert!(matches!(
        supported.shader_route(&desc(
            ShaderFormat::Wgsl,
            ShaderStage::Compute,
            SHADER_FEATURE_MEMORY_COHERENT_SPIRV,
        )),
        ShaderRoute::Unsupported
    ));
    assert!(matches!(
        supported.shader_route(&desc(
            ShaderFormat::SpirV,
            ShaderStage::Vertex,
            SHADER_FEATURE_MEMORY_VOLATILE_SPIRV,
        )),
        ShaderRoute::Unsupported
    ));
    assert!(matches!(
        device(true, false).shader_route(&desc(
            ShaderFormat::SpirV,
            ShaderStage::Compute,
            SHADER_FEATURE_MEMORY_COHERENT_SPIRV,
        )),
        ShaderRoute::Direct(_)
    ));
    assert!(matches!(
        device(false, true).shader_route(&desc(
            ShaderFormat::SpirV,
            ShaderStage::Compute,
            SHADER_FEATURE_MEMORY_VOLATILE_SPIRV,
        )),
        ShaderRoute::Direct(_)
    ));
    let empty = ShaderDesc {
        label: None,
        format: ShaderFormat::SpirV,
        bytes: Arc::from([0u8; 20].as_slice()),
        entries: vec![],
        required_features: vec![SHADER_FEATURE_MEMORY_COHERENT_SPIRV.into()],
    };
    assert!(matches!(
        supported.shader_route(&empty),
        ShaderRoute::Unsupported
    ));
    ROUTES.with(|v| assert_eq!(v.get(), 2));
}

#[test]
fn cooperative_matrix_shader_requirement_is_compute_only_and_configuration_gated() {
    ROUTES.with(|v| v.set(0));
    let device = |supported| unsafe {
        Device::<Mock>::from_native(
            DeviceId::allocate().unwrap(),
            (),
            DeviceCapabilities {
                cooperative_matrix_configurations: if supported {
                    vec![CooperativeMatrixConfiguration {
                        m: 16,
                        n: 16,
                        k: 16,
                        a_type: CooperativeMatrixComponentType::Float16,
                        b_type: CooperativeMatrixComponentType::Float16,
                        c_type: CooperativeMatrixComponentType::Float32,
                        result_type: CooperativeMatrixComponentType::Float32,
                        saturating_accumulation: false,
                        scope: CooperativeMatrixScope::Subgroup,
                    }]
                } else {
                    vec![]
                },
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
            workgroup_size: Some([64, 1, 1]),
        }],
        required_features: vec![SHADER_FEATURE_EXPERIMENTAL_COOPERATIVE_MATRIX.into()],
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
