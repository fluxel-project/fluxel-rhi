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
    MetalLibrary,
}
#[derive(Clone, Debug)]
pub struct ShaderEntryPoint {
    pub name: String,
    pub stage: ShaderStage,
    pub workgroup_size: Option<[u32; 3]>,
}
#[derive(Clone, Debug)]
pub struct ShaderDesc {
    pub label: Option<String>,
    pub format: ShaderFormat,
    pub bytes: Arc<[u8]>,
    pub entries: Vec<ShaderEntryPoint>,
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
        B::shader_route(&self.native, desc)
    }
    pub async fn shader(&self, desc: ShaderDesc) -> Result<Shader<B>> {
        let native = match B::shader_route(&self.native, &desc) {
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
