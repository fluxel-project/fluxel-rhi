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
#[path = "../tests/api/shader.rs"]
mod tests;
