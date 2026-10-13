//! Optional ray-tracing pipeline and acceleration-structure contract.
//!
//! This module carries device-local operations only. Scene scheduling and
//! memory residency decisions remain renderer or backend concerns.
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

#[derive(Clone, Debug)]
pub struct AccelerationSizes {
    pub storage_bytes: u64,
    pub scratch_bytes: u64,
    pub scratch_alignment: u64,
}
#[derive(Clone)]
pub enum AccelerationGeometry<B: Backend> {
    Triangles {
        vertices: Buffer<B>,
        stride: u64,
        indices: Option<(Buffer<B>, IndexFormat)>,
        opaque: bool,
    },
    Aabbs {
        data: Buffer<B>,
        stride: u64,
        count: u32,
        opaque: bool,
    },
}
#[derive(Clone)]
pub struct AccelerationInstance<B: Backend> {
    pub blas: AccelerationStructure<B>,
    pub transform: [[f32; 4]; 3],
    pub mask: u8,
    pub custom_index: u32,
    pub shader_table_offset: u32,
    pub flags: u32,
}
#[derive(Clone)]
pub enum AccelerationInput<B: Backend> {
    Blas(Vec<AccelerationGeometry<B>>),
    Tlas(Vec<AccelerationInstance<B>>),
}
#[derive(Clone)]
pub struct AccelerationDesc<B: Backend> {
    pub input: AccelerationInput<B>,
    pub fast_trace: bool,
    pub allow_update: bool,
}
#[derive(Clone)]
pub struct RayPipelineDesc<B: Backend> {
    pub layout: PipelineLayout<B>,
    pub shaders: Vec<PipelineStage<B>>,
    pub groups: Vec<RayGroup>,
    pub max_recursion: u32,
    pub payload_bytes: u32,
    pub hit_attribute_bytes: u32,
    pub cache: Option<PipelineCache<B>>,
}
#[derive(Clone, Debug)]
pub struct RayGroup {
    pub general: Option<u32>,
    pub closest_hit: Option<u32>,
    pub any_hit: Option<u32>,
    pub intersection: Option<u32>,
}
#[derive(Clone, Debug)]
pub struct ShaderRecord {
    pub group_index: u32,
    pub local_data: Vec<u8>,
}

pub trait RayOps<B: Backend> {
    const AVAILABLE: bool;
    type PipelineFuture<'a>: Future<Output = Result<B::NativeRayPipeline>> + 'a
    where
        B: 'a;
    fn create_pipeline<'a>(
        device: &'a B::NativeDevice,
        desc: &'a RayPipelineDesc<B>,
    ) -> Self::PipelineFuture<'a>;
    fn acceleration_sizes(
        device: &B::NativeDevice,
        desc: &AccelerationDesc<B>,
    ) -> Result<AccelerationSizes>;
    fn create_acceleration(
        device: &B::NativeDevice,
        desc: &AccelerationDesc<B>,
    ) -> Result<B::NativeAcceleration>;
    fn build_acceleration(
        encoder: &mut B::NativeEncoder,
        dst: &B::NativeAcceleration,
        scratch: &B::NativeBuffer,
        input: &AccelerationDesc<B>,
    ) -> Result<()>;
    fn trace(
        encoder: &mut B::NativeEncoder,
        pipeline: &B::NativeRayPipeline,
        shader_table: &B::NativeBuffer,
        dims: [u32; 3],
    ) -> Result<()>;
}
impl<B: Backend> RayOps<B> for Unavailable {
    const AVAILABLE: bool = false;
    type PipelineFuture<'a>
        = std::future::Ready<Result<B::NativeRayPipeline>>
    where
        B: 'a;
    fn create_pipeline<'a>(
        _: &'a B::NativeDevice,
        _: &'a RayPipelineDesc<B>,
    ) -> Self::PipelineFuture<'a> {
        std::future::ready(Err(Error::Unsupported("ray tracing")))
    }
    fn acceleration_sizes(
        _: &B::NativeDevice,
        _: &AccelerationDesc<B>,
    ) -> Result<AccelerationSizes> {
        Err(Error::Unsupported("acceleration structure"))
    }
    fn create_acceleration(
        _: &B::NativeDevice,
        _: &AccelerationDesc<B>,
    ) -> Result<B::NativeAcceleration> {
        Err(Error::Unsupported("acceleration structure"))
    }
    fn build_acceleration(
        _: &mut B::NativeEncoder,
        _: &B::NativeAcceleration,
        _: &B::NativeBuffer,
        _: &AccelerationDesc<B>,
    ) -> Result<()> {
        Err(Error::Unsupported("acceleration structure"))
    }
    fn trace(
        _: &mut B::NativeEncoder,
        _: &B::NativeRayPipeline,
        _: &B::NativeBuffer,
        _: [u32; 3],
    ) -> Result<()> {
        Err(Error::Unsupported("ray tracing"))
    }
}
pub struct RayDomain<'a, B: Backend> {
    device: &'a Device<B>,
}
impl<B: Backend> Device<B> {
    pub fn ray(&self) -> Option<RayDomain<'_, B>> {
        (self.caps.ray && B::Ray::AVAILABLE).then_some(RayDomain { device: self })
    }
}
impl<B: Backend> RayDomain<'_, B> {
    pub async fn create_pipeline(&self, desc: RayPipelineDesc<B>) -> Result<RayPipeline<B>> {
        // Shader tables refer to the executable pipeline's shader identities;
        // accepting a foreign module would defer a deterministic ownership
        // error until a backend-specific trace operation.
        if desc.layout.device != self.device.id
            || desc
                .shaders
                .iter()
                .any(|s| s.module.device != self.device.id)
            || desc
                .cache
                .as_ref()
                .is_some_and(|c| c.device != self.device.id)
        {
            return Err(Error::WrongDevice);
        }
        let native = B::Ray::create_pipeline(&self.device.native, &desc).await?;
        // SAFETY: the SPI created this pipeline for this exact native device.
        Ok(unsafe { RayPipeline::from_native(self.device.id, native) })
    }
    pub fn sizes(&self, desc: &AccelerationDesc<B>) -> Result<AccelerationSizes> {
        B::Ray::acceleration_sizes(&self.device.native, desc)
    }
    pub fn create_acceleration(
        &self,
        desc: &AccelerationDesc<B>,
    ) -> Result<AccelerationStructure<B>> {
        let native = B::Ray::create_acceleration(&self.device.native, desc)?;
        // SAFETY: the SPI created this acceleration structure for this device.
        Ok(unsafe { AccelerationStructure::from_native(self.device.id, native) })
    }
    pub fn build(
        &self,
        encoder: &mut CommandEncoder<B>,
        dst: &AccelerationStructure<B>,
        scratch: &Buffer<B>,
        desc: &AccelerationDesc<B>,
    ) -> Result<()>
    where
        B: CommandBackend,
    {
        if encoder.device != self.device.id
            || dst.device != self.device.id
            || scratch.device != self.device.id
        {
            return Err(Error::WrongDevice);
        }
        encoder.call(|e| B::Ray::build_acceleration(e, &dst.native, &scratch.native, desc))
    }
    pub fn trace(
        &self,
        encoder: &mut CommandEncoder<B>,
        pipeline: &RayPipeline<B>,
        table: &Buffer<B>,
        dims: [u32; 3],
    ) -> Result<()>
    where
        B: CommandBackend,
    {
        if encoder.device != self.device.id
            || pipeline.device != self.device.id
            || table.device != self.device.id
        {
            return Err(Error::WrongDevice);
        }
        encoder.call(|e| B::Ray::trace(e, &pipeline.native, &table.native, dims))
    }
}
