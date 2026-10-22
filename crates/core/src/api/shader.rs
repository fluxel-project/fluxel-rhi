//! Shader-module creation and source artifact contract.
//!
//! Backend-specific compilation remains behind the SPI. This module does not
//! select platform shader dialects or introduce a cross-backend compiler.
#![allow(missing_docs, dead_code, unused_variables, unused_imports)]
use super::*;
use std::{
    future::Future,
    marker::PhantomData,
    ops::Range,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShaderFormat {
    Glsl(u16),
    GlslEs(u16),
    SpirV,
    Wgsl,
    Hlsl(u16),
    Dxil,
    MetalSource,
    /// A complete precompiled Metal library supplied as bytes, not a source file.
    /// Entry names and resource binding ABI must match the consuming backend.
    MetalLibrary,
}
#[derive(Clone, Debug)]
pub struct ShaderEntryPoint {
    pub name: String,
    pub stage: ShaderStage,
    pub workgroup_size: Option<[u32; 3]>,
}
/// A shader performs inline acceleration-structure traversal.
///
/// Declaring this requirement makes shader creation conditional on the device's
/// `ray_query` capability. The caller must declare features used by opaque shader
/// artifacts; Core does not inspect native bytecode.
pub const SHADER_FEATURE_RAY_QUERY: &str = "ray-query";

#[derive(Clone, Debug)]
pub struct ShaderDesc {
    pub label: Option<String>,
    pub format: ShaderFormat,
    pub bytes: Arc<[u8]>,
    pub entries: Vec<ShaderEntryPoint>,
    /// Portable requirements use the `SHADER_FEATURE_*` constants. Other names
    /// are passed to the backend, which must reject requirements it cannot honor.
    pub required_features: Vec<String>,
}
#[derive(Clone, Debug)]
pub struct ShaderTarget {
    pub format: ShaderFormat,
    pub environment: String,
    pub abi_key: String,
}
#[derive(Clone, Debug)]
pub enum ShaderRoute {
    Direct(ShaderTarget),
    Naga(ShaderTarget),
    Unsupported,
}

/// Cold shader operations: native direct route first, actual Naga frontend/output
/// pair second, else error. No retry after a selected compiler fails.
pub trait ShaderBackend: Backend {
    type DirectFuture<'a>: Future<Output = Result<Self::NativeShader>> + 'a
    where
        Self: 'a;
    type NagaFuture<'a>: Future<Output = Result<Self::NativeShader>> + 'a
    where
        Self: 'a;
    fn shader_route(device: &Self::NativeDevice, desc: &ShaderDesc) -> ShaderRoute;
    fn compile_direct<'a>(
        device: &'a Self::NativeDevice,
        desc: &'a ShaderDesc,
        target: &'a ShaderTarget,
    ) -> Self::DirectFuture<'a>;
    fn compile_naga<'a>(
        device: &'a Self::NativeDevice,
        desc: &'a ShaderDesc,
        target: &'a ShaderTarget,
    ) -> Self::NagaFuture<'a>;
}
impl<B: ShaderBackend> Device<B> {
    pub fn shader_route(&self, desc: &ShaderDesc) -> ShaderRoute {
        if !self.caps.ray_query
            && desc
                .required_features
                .iter()
                .any(|feature| feature == SHADER_FEATURE_RAY_QUERY)
        {
            return ShaderRoute::Unsupported;
        }
        B::shader_route(&self.native, desc)
    }
    pub async fn shader(&self, desc: ShaderDesc) -> Result<Shader<B>> {
        let native = match self.shader_route(&desc) {
            ShaderRoute::Direct(target) => B::compile_direct(&self.native, &desc, &target).await?,
            ShaderRoute::Naga(target) => B::compile_naga(&self.native, &desc, &target).await?,
            ShaderRoute::Unsupported => {
                return Err(Error::Unsupported("shader format/profile/ABI"));
            }
        };
        // Do not try another route after the selected compiler fails. A retry
        // could silently compile a different ABI than the backend advertised.
        // SAFETY: the SPI created this shader for this device.
        Ok(unsafe { Shader::from_native(self.id, native) })
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::type_contract_tests::NativeShared;
    use super::*;
    use std::{
        cell::Cell,
        future::{Ready, ready},
        task::{Context, Poll, Waker},
    };
    thread_local! { static ROUTES: Cell<usize> = const { Cell::new(0) }; }
    impl ShaderBackend for NativeShared {
        type DirectFuture<'a> = Ready<Result<Arc<()>>>;
        type NagaFuture<'a> = Ready<Result<Arc<()>>>;
        fn shader_route(_: &Arc<()>, _: &ShaderDesc) -> ShaderRoute {
            ROUTES.with(|v| v.set(v.get() + 1));
            ShaderRoute::Direct(ShaderTarget {
                format: ShaderFormat::Dxil,
                environment: String::new(),
                abi_key: String::new(),
            })
        }
        fn compile_direct<'a>(
            _: &'a Arc<()>,
            _: &'a ShaderDesc,
            _: &'a ShaderTarget,
        ) -> Self::DirectFuture<'a> {
            ready(Ok(Arc::new(())))
        }
        fn compile_naga<'a>(
            _: &'a Arc<()>,
            _: &'a ShaderDesc,
            _: &'a ShaderTarget,
        ) -> Self::NagaFuture<'a> {
            unreachable!()
        }
    }
    #[test]
    fn ray_query_requirement_is_checked_before_shader_backend() {
        ROUTES.with(|v| v.set(0));
        let device = |supported| unsafe {
            Device::<NativeShared>::from_native(
                DeviceId::allocate().unwrap(),
                Arc::new(()),
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
}
