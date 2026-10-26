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
