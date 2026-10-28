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
/// A fragment shader requires the API's early depth-test semantics.
///
/// This requirement is checked against `DeviceCapabilities::shader_early_depth_test`.
/// Shaders that use the WGSL `@early_depth_test` attribute or Direct-SPIR-V
/// `EarlyFragmentTests` execution mode must declare it.
pub const SHADER_FEATURE_EARLY_DEPTH_TEST: &str = "shader-early-depth-test";
/// A fragment shader reads the primitive index assigned by rasterization.
pub const SHADER_FEATURE_PRIMITIVE_INDEX: &str = "primitive-index";
/// A vertex shader reads the index of its draw within a multi-draw command.
pub const SHADER_FEATURE_DRAW_INDEX: &str = "draw-index";
/// A fragment shader reads standard perspective or linear barycentric
/// coordinates. Vulkan KHR and NV SPIR-V variants have matching semantics.
pub const SHADER_FEATURE_BARYCENTRICS: &str = "barycentrics";
/// A fragment shader uses the additional AMD explicit interpolation
/// barycentric built-ins. This is separate because its sample/centroid and
/// pull-model modes do not share the standard KHR/NV semantics.
pub const SHADER_FEATURE_BARYCENTRICS_AMD: &str = "barycentrics-amd";
/// WGSL compute `textureAtomicAdd` on an R32Uint storage texture.
pub const SHADER_FEATURE_TEXTURE_ATOMIC_R32UINT_ADD: &str = "texture-atomic-r32uint-add";
/// WGSL compute `textureAtomicAdd` on an R32Sint storage texture.
pub const SHADER_FEATURE_TEXTURE_ATOMIC_R32SINT_ADD: &str = "texture-atomic-r32sint-add";
/// WGSL compute `textureAtomicMin` on an R32Sint storage texture.
pub const SHADER_FEATURE_TEXTURE_ATOMIC_R32SINT_MIN: &str = "texture-atomic-r32sint-min";
/// WGSL compute `textureAtomicMax` on an R32Sint storage texture.
pub const SHADER_FEATURE_TEXTURE_ATOMIC_R32SINT_MAX: &str = "texture-atomic-r32sint-max";
/// WGSL compute `textureAtomicMin` on an R32Uint storage texture.
pub const SHADER_FEATURE_TEXTURE_ATOMIC_R32UINT_MIN: &str = "texture-atomic-r32uint-min";
/// WGSL compute `textureAtomicMax` on an R32Uint storage texture.
pub const SHADER_FEATURE_TEXTURE_ATOMIC_R32UINT_MAX: &str = "texture-atomic-r32uint-max";
/// WGSL compute `textureAtomicXor` on an R32Uint storage texture.
pub const SHADER_FEATURE_TEXTURE_ATOMIC_R32UINT_XOR: &str = "texture-atomic-r32uint-xor";
/// WGSL compute `textureAtomicAnd` on an R32Uint storage texture.
pub const SHADER_FEATURE_TEXTURE_ATOMIC_R32UINT_AND: &str = "texture-atomic-r32uint-and";
/// WGSL compute `textureAtomicOr` on an R32Uint storage texture.
pub const SHADER_FEATURE_TEXTURE_ATOMIC_R32UINT_OR: &str = "texture-atomic-r32uint-or";
/// Direct SPIR-V compute Float32 `OpAtomicFAddEXT` on storage-buffer memory.
pub const SHADER_FEATURE_FLOAT32_ATOMIC_BUFFER_ADD: &str = "float32-atomic-buffer-add";
/// Direct SPIR-V compute Float32 `OpAtomicFAddEXT` on workgroup/shared memory.
pub const SHADER_FEATURE_FLOAT32_ATOMIC_WORKGROUP_ADD: &str = "float32-atomic-workgroup-add";
/// Direct SPIR-V compute Float32 `OpAtomicFMinEXT` on workgroup/shared memory.
pub const SHADER_FEATURE_FLOAT32_ATOMIC_WORKGROUP_MIN: &str = "float32-atomic-workgroup-min";
/// Direct SPIR-V compute Float32 `OpAtomicFMaxEXT` on workgroup/shared memory.
pub const SHADER_FEATURE_FLOAT32_ATOMIC_WORKGROUP_MAX: &str = "float32-atomic-workgroup-max";
/// Direct SPIR-V compute Float32 storage-buffer atomic minimum.
pub const SHADER_FEATURE_FLOAT32_ATOMIC_BUFFER_MIN: &str = "float32-atomic-buffer-min";
/// Direct SPIR-V compute Float32 storage-buffer atomic maximum.
pub const SHADER_FEATURE_FLOAT32_ATOMIC_BUFFER_MAX: &str = "float32-atomic-buffer-max";
/// Direct SPIR-V compute Int64 storage-buffer atomic add.
pub const SHADER_FEATURE_INT64_ATOMIC_BUFFER_ADD: &str = "int64-atomic-buffer-add";
/// Direct SPIR-V compute Int64 signed atomic min on storage-buffer memory.
pub const SHADER_FEATURE_INT64_ATOMIC_BUFFER_MIN: &str = "int64-atomic-buffer-min";
/// Direct SPIR-V compute Int64 storage-buffer signed atomic maximum.
pub const SHADER_FEATURE_INT64_ATOMIC_BUFFER_MAX: &str = "int64-atomic-buffer-max";
/// Direct SPIR-V compute Int64 storage-buffer unsigned atomic minimum.
pub const SHADER_FEATURE_INT64_ATOMIC_BUFFER_UMIN: &str = "int64-atomic-buffer-umin";
/// Direct SPIR-V compute Int64 storage-buffer unsigned atomic maximum.
pub const SHADER_FEATURE_INT64_ATOMIC_BUFFER_UMAX: &str = "int64-atomic-buffer-umax";
/// Direct SPIR-V compute Int64 storage-buffer atomic bitwise XOR.
pub const SHADER_FEATURE_INT64_ATOMIC_BUFFER_XOR: &str = "int64-atomic-buffer-xor";
/// Direct SPIR-V compute Int64 storage-buffer atomic compare-exchange.
pub const SHADER_FEATURE_INT64_ATOMIC_BUFFER_COMPARE_EXCHANGE: &str =
    "int64-atomic-buffer-compare-exchange";
pub const SHADER_FEATURE_INT64_ATOMIC_BUFFER_EXCHANGE: &str = "int64-atomic-buffer-exchange";
pub const SHADER_FEATURE_INT64_ATOMIC_BUFFER_AND: &str = "int64-atomic-buffer-and";
pub const SHADER_FEATURE_INT64_ATOMIC_BUFFER_OR: &str = "int64-atomic-buffer-or";
/// Direct SPIR-V compute Int64 workgroup/shared atomic add.
pub const SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_ADD: &str = "int64-atomic-workgroup-add";
/// Direct SPIR-V compute Int64 workgroup/shared signed atomic min.
pub const SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_MIN: &str = "int64-atomic-workgroup-min";
/// Direct SPIR-V compute Int64 workgroup/shared signed atomic max.
pub const SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_MAX: &str = "int64-atomic-workgroup-max";
/// 64-bit integer atomic XOR in workgroup memory.
pub const SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_XOR: &str = "int64-atomic-workgroup-xor";
/// Direct SPIR-V compute Int64 workgroup/shared atomic AND.
pub const SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_AND: &str = "int64-atomic-workgroup-and";
/// Direct SPIR-V compute Int64 workgroup/shared atomic OR.
pub const SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_OR: &str = "int64-atomic-workgroup-or";
/// Direct SPIR-V compute Int64 workgroup/shared unsigned atomic minimum.
pub const SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_UMIN: &str = "int64-atomic-workgroup-umin";
/// Direct SPIR-V compute Int64 workgroup/shared unsigned atomic maximum.
pub const SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_UMAX: &str = "int64-atomic-workgroup-umax";
/// Direct SPIR-V compute Int64 workgroup/shared atomic exchange.
pub const SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_EXCHANGE: &str = "int64-atomic-workgroup-exchange";
/// Direct SPIR-V compute Int64 workgroup/shared atomic compare-exchange.
pub const SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_COMPARE_EXCHANGE: &str =
    "int64-atomic-workgroup-compare-exchange";
/// Direct SPIR-V compute atomic add on a non-array, single-sample 2D R64Uint image.
pub const SHADER_FEATURE_INT64_ATOMIC_IMAGE_R64UINT_ADD: &str = "int64-atomic-image-r64uint-add";
/// Direct SPIR-V compute unsigned atomic minimum on a non-array, single-sample 2D R64Uint image.
pub const SHADER_FEATURE_INT64_ATOMIC_IMAGE_R64UINT_UMIN: &str = "int64-atomic-image-r64uint-umin";
/// Direct SPIR-V compute unsigned atomic maximum on a non-array, single-sample 2D R64Uint image.
pub const SHADER_FEATURE_INT64_ATOMIC_IMAGE_R64UINT_UMAX: &str = "int64-atomic-image-r64uint-umax";
/// Direct-SPIR-V GLSL450 compute `StorageBuffer` route using the SPIR-V
/// `Coherent` decoration. This requests validated shader admission only; it
/// neither queries a native feature nor supplies a synchronization barrier.
pub const SHADER_FEATURE_MEMORY_COHERENT_SPIRV: &str = "memory-coherent-spirv";
/// Direct-SPIR-V GLSL450 compute `StorageBuffer` route using the SPIR-V
/// `Volatile` decoration. This requests validated shader admission only; it
/// neither queries a native feature nor supplies a synchronization barrier.
pub const SHADER_FEATURE_MEMORY_VOLATILE_SPIRV: &str = "memory-volatile-spirv";
/// Experimental compute shader using a device-reported cooperative-matrix tuple.
pub const SHADER_FEATURE_EXPERIMENTAL_COOPERATIVE_MATRIX: &str = "experimental-cooperative-matrix";

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
        if desc.required_features.iter().any(|feature| {
            (feature == SHADER_FEATURE_RAY_QUERY && !self.caps.ray_query)
                || (feature == SHADER_FEATURE_EARLY_DEPTH_TEST
                    && (!self.caps.shader_early_depth_test
                        || !desc
                            .entries
                            .iter()
                            .any(|entry| entry.stage == ShaderStage::Fragment)))
                || (feature == SHADER_FEATURE_PRIMITIVE_INDEX
                    && (!self.caps.shader_primitive_index
                        || !desc
                            .entries
                            .iter()
                            .any(|entry| entry.stage == ShaderStage::Fragment)))
                || (feature == SHADER_FEATURE_DRAW_INDEX
                    && (!self.caps.shader_draw_index
                        || !desc
                            .entries
                            .iter()
                            .any(|entry| entry.stage == ShaderStage::Vertex)))
                || ((feature == SHADER_FEATURE_BARYCENTRICS
                    || feature == SHADER_FEATURE_BARYCENTRICS_AMD)
                    && (!if feature == SHADER_FEATURE_BARYCENTRICS {
                        self.caps.shader_barycentrics
                    } else {
                        self.caps.shader_barycentrics_amd
                    } || !desc
                        .entries
                        .iter()
                        .any(|entry| entry.stage == ShaderStage::Fragment)))
                || (feature == SHADER_FEATURE_TEXTURE_ATOMIC_R32UINT_ADD
                    && (!self.caps.texture_atomic_r32uint_compute
                        || desc.entries.is_empty()
                        || !desc
                            .entries
                            .iter()
                            .all(|entry| entry.stage == ShaderStage::Compute)))
                || (feature == SHADER_FEATURE_TEXTURE_ATOMIC_R32SINT_ADD
                    && (!self.caps.texture_atomic_r32sint_compute
                        || desc.entries.is_empty()
                        || !desc
                            .entries
                            .iter()
                            .all(|entry| entry.stage == ShaderStage::Compute)))
                || (feature == SHADER_FEATURE_TEXTURE_ATOMIC_R32SINT_MIN
                    && (!self.caps.texture_atomic_r32sint_min_compute
                        || desc.entries.is_empty()
                        || !desc
                            .entries
                            .iter()
                            .all(|entry| entry.stage == ShaderStage::Compute)))
                || (feature == SHADER_FEATURE_TEXTURE_ATOMIC_R32SINT_MAX
                    && (!self.caps.texture_atomic_r32sint_max_compute
                        || desc.entries.is_empty()
                        || !desc
                            .entries
                            .iter()
                            .all(|entry| entry.stage == ShaderStage::Compute)))
                || (feature == SHADER_FEATURE_TEXTURE_ATOMIC_R32UINT_MIN
                    && (!self.caps.texture_atomic_r32uint_min_compute
                        || desc.entries.is_empty()
                        || !desc
                            .entries
                            .iter()
                            .all(|entry| entry.stage == ShaderStage::Compute)))
                || (feature == SHADER_FEATURE_TEXTURE_ATOMIC_R32UINT_MAX
                    && (!self.caps.texture_atomic_r32uint_max_compute
                        || desc.entries.is_empty()
                        || !desc
                            .entries
                            .iter()
                            .all(|entry| entry.stage == ShaderStage::Compute)))
                || (feature == SHADER_FEATURE_TEXTURE_ATOMIC_R32UINT_XOR
                    && (!self.caps.texture_atomic_r32uint_xor_compute
                        || desc.entries.is_empty()
                        || !desc
                            .entries
                            .iter()
                            .all(|entry| entry.stage == ShaderStage::Compute)))
                || (feature == SHADER_FEATURE_TEXTURE_ATOMIC_R32UINT_AND
                    && (!self.caps.texture_atomic_r32uint_and_compute
                        || desc.entries.is_empty()
                        || !desc
                            .entries
                            .iter()
                            .all(|entry| entry.stage == ShaderStage::Compute)))
                || (feature == SHADER_FEATURE_TEXTURE_ATOMIC_R32UINT_OR
                    && (!self.caps.texture_atomic_r32uint_or_compute
                        || desc.entries.is_empty()
                        || !desc
                            .entries
                            .iter()
                            .all(|entry| entry.stage == ShaderStage::Compute)))
                || (feature == SHADER_FEATURE_FLOAT32_ATOMIC_BUFFER_ADD
                    && (!self.caps.shader_float32_atomic_buffer_add
                        || desc.entries.is_empty()
                        || !desc
                            .entries
                            .iter()
                            .all(|entry| entry.stage == ShaderStage::Compute)))
                || (feature == SHADER_FEATURE_FLOAT32_ATOMIC_WORKGROUP_ADD
                    && (!self.caps.shader_float32_atomic_workgroup_add
                        || desc.entries.is_empty()
                        || !desc
                            .entries
                            .iter()
                            .all(|entry| entry.stage == ShaderStage::Compute)))
                || (feature == SHADER_FEATURE_FLOAT32_ATOMIC_WORKGROUP_MIN
                    && (!self.caps.shader_float32_atomic_workgroup_min
                        || desc.entries.is_empty()
                        || !desc
                            .entries
                            .iter()
                            .all(|entry| entry.stage == ShaderStage::Compute)))
                || (feature == SHADER_FEATURE_FLOAT32_ATOMIC_WORKGROUP_MAX
                    && (!self.caps.shader_float32_atomic_workgroup_max
                        || desc.entries.is_empty()
                        || !desc
                            .entries
                            .iter()
                            .all(|entry| entry.stage == ShaderStage::Compute)))
                || (feature == SHADER_FEATURE_FLOAT32_ATOMIC_BUFFER_MIN
                    && (!self.caps.shader_float32_atomic_buffer_min
                        || desc.entries.is_empty()
                        || !desc
                            .entries
                            .iter()
                            .all(|entry| entry.stage == ShaderStage::Compute)))
                || (feature == SHADER_FEATURE_FLOAT32_ATOMIC_BUFFER_MAX
                    && (!self.caps.shader_float32_atomic_buffer_max
                        || desc.entries.is_empty()
                        || !desc
                            .entries
                            .iter()
                            .all(|entry| entry.stage == ShaderStage::Compute)))
                || (feature == SHADER_FEATURE_INT64_ATOMIC_BUFFER_ADD
                    && (!self.caps.shader_int64_atomic_buffer_add
                        || desc.entries.is_empty()
                        || !desc
                            .entries
                            .iter()
                            .all(|entry| entry.stage == ShaderStage::Compute)))
                || (feature == SHADER_FEATURE_INT64_ATOMIC_BUFFER_MIN
                    && (!self.caps.shader_int64_atomic_buffer_min
                        || desc.entries.is_empty()
                        || !desc
                            .entries
                            .iter()
                            .all(|entry| entry.stage == ShaderStage::Compute)))
                || (feature == SHADER_FEATURE_INT64_ATOMIC_BUFFER_MAX
                    && (!self.caps.shader_int64_atomic_buffer_max
                        || desc.entries.is_empty()
                        || !desc
                            .entries
                            .iter()
                            .all(|entry| entry.stage == ShaderStage::Compute)))
                || (feature == SHADER_FEATURE_INT64_ATOMIC_BUFFER_UMIN
                    && (!self.caps.shader_int64_atomic_buffer_umin
                        || desc.entries.is_empty()
                        || !desc
                            .entries
                            .iter()
                            .all(|entry| entry.stage == ShaderStage::Compute)))
                || (feature == SHADER_FEATURE_INT64_ATOMIC_BUFFER_UMAX
                    && (!self.caps.shader_int64_atomic_buffer_umax
                        || desc.entries.is_empty()
                        || !desc
                            .entries
                            .iter()
                            .all(|entry| entry.stage == ShaderStage::Compute)))
                || (feature == SHADER_FEATURE_INT64_ATOMIC_BUFFER_XOR
                    && (!self.caps.shader_int64_atomic_buffer_xor
                        || desc.entries.is_empty()
                        || !desc
                            .entries
                            .iter()
                            .all(|entry| entry.stage == ShaderStage::Compute)))
                || (feature == SHADER_FEATURE_INT64_ATOMIC_BUFFER_COMPARE_EXCHANGE
                    && (!self.caps.shader_int64_atomic_buffer_compare_exchange
                        || desc.entries.is_empty()
                        || !desc
                            .entries
                            .iter()
                            .all(|entry| entry.stage == ShaderStage::Compute)))
                || (feature == SHADER_FEATURE_INT64_ATOMIC_BUFFER_EXCHANGE
                    && (!self.caps.shader_int64_atomic_buffer_exchange
                        || desc.entries.is_empty()
                        || !desc
                            .entries
                            .iter()
                            .all(|entry| entry.stage == ShaderStage::Compute)))
                || (feature == SHADER_FEATURE_INT64_ATOMIC_BUFFER_AND
                    && (!self.caps.shader_int64_atomic_buffer_and
                        || desc.entries.is_empty()
                        || !desc
                            .entries
                            .iter()
                            .all(|entry| entry.stage == ShaderStage::Compute)))
                || (feature == SHADER_FEATURE_INT64_ATOMIC_BUFFER_OR
                    && (!self.caps.shader_int64_atomic_buffer_or
                        || desc.entries.is_empty()
                        || !desc
                            .entries
                            .iter()
                            .all(|entry| entry.stage == ShaderStage::Compute)))
                || (feature == SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_ADD
                    && (!self.caps.shader_int64_atomic_workgroup_add
                        || desc.entries.is_empty()
                        || !desc
                            .entries
                            .iter()
                            .all(|entry| entry.stage == ShaderStage::Compute)))
                || (feature == SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_MIN
                    && (!self.caps.shader_int64_atomic_workgroup_min
                        || desc.entries.is_empty()
                        || !desc
                            .entries
                            .iter()
                            .all(|entry| entry.stage == ShaderStage::Compute)))
                || (feature == SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_MAX
                    && (!self.caps.shader_int64_atomic_workgroup_max
                        || desc.entries.is_empty()
                        || !desc
                            .entries
                            .iter()
                            .all(|entry| entry.stage == ShaderStage::Compute)))
                || (feature == SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_XOR
                    && (!self.caps.shader_int64_atomic_workgroup_xor
                        || desc.entries.is_empty()
                        || !desc
                            .entries
                            .iter()
                            .all(|entry| entry.stage == ShaderStage::Compute)))
                || (feature == SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_AND
                    && (!self.caps.shader_int64_atomic_workgroup_and
                        || desc.entries.is_empty()
                        || !desc
                            .entries
                            .iter()
                            .all(|entry| entry.stage == ShaderStage::Compute)))
                || (feature == SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_OR
                    && (!self.caps.shader_int64_atomic_workgroup_or
                        || desc.entries.is_empty()
                        || !desc
                            .entries
                            .iter()
                            .all(|entry| entry.stage == ShaderStage::Compute)))
                || (feature == SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_UMIN
                    && (!self.caps.shader_int64_atomic_workgroup_umin
                        || desc.entries.is_empty()
                        || !desc
                            .entries
                            .iter()
                            .all(|entry| entry.stage == ShaderStage::Compute)))
                || (feature == SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_UMAX
                    && (!self.caps.shader_int64_atomic_workgroup_umax
                        || desc.entries.is_empty()
                        || !desc
                            .entries
                            .iter()
                            .all(|entry| entry.stage == ShaderStage::Compute)))
                || (feature == SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_EXCHANGE
                    && (!self.caps.shader_int64_atomic_workgroup_exchange
                        || desc.entries.is_empty()
                        || !desc
                            .entries
                            .iter()
                            .all(|entry| entry.stage == ShaderStage::Compute)))
                || (feature == SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_COMPARE_EXCHANGE
                    && (!self.caps.shader_int64_atomic_workgroup_compare_exchange
                        || desc.entries.is_empty()
                        || !desc
                            .entries
                            .iter()
                            .all(|entry| entry.stage == ShaderStage::Compute)))
                || (feature == SHADER_FEATURE_INT64_ATOMIC_IMAGE_R64UINT_ADD
                    && (!self.caps.shader_int64_atomic_image_r64uint_add
                        || desc.entries.is_empty()
                        || !desc
                            .entries
                            .iter()
                            .all(|entry| entry.stage == ShaderStage::Compute)))
                || (feature == SHADER_FEATURE_INT64_ATOMIC_IMAGE_R64UINT_UMIN
                    && (!self.caps.shader_int64_atomic_image_r64uint_umin
                        || desc.entries.is_empty()
                        || !desc
                            .entries
                            .iter()
                            .all(|entry| entry.stage == ShaderStage::Compute)))
                || (feature == SHADER_FEATURE_INT64_ATOMIC_IMAGE_R64UINT_UMAX
                    && (!self.caps.shader_int64_atomic_image_r64uint_umax
                        || desc.entries.is_empty()
                        || !desc
                            .entries
                            .iter()
                            .all(|entry| entry.stage == ShaderStage::Compute)))
                || (feature == SHADER_FEATURE_MEMORY_COHERENT_SPIRV
                    && (!self.caps.shader_memory_coherent_spirv
                        || desc.format != ShaderFormat::SpirV
                        || desc.entries.is_empty()
                        || !desc
                            .entries
                            .iter()
                            .all(|entry| entry.stage == ShaderStage::Compute)))
                || (feature == SHADER_FEATURE_MEMORY_VOLATILE_SPIRV
                    && (!self.caps.shader_memory_volatile_spirv
                        || desc.format != ShaderFormat::SpirV
                        || desc.entries.is_empty()
                        || !desc
                            .entries
                            .iter()
                            .all(|entry| entry.stage == ShaderStage::Compute)))
                || (feature == SHADER_FEATURE_EXPERIMENTAL_COOPERATIVE_MATRIX
                    && (self.caps.cooperative_matrix_configurations.is_empty()
                        || desc.entries.is_empty()
                        || !desc
                            .entries
                            .iter()
                            .all(|entry| entry.stage == ShaderStage::Compute)))
        }) {
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
