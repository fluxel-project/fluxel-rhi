//! Optional mesh-shader pipeline and draw contract.
//!
//! Capability checks occur before creation; graph scheduling and fallback
//! selection remain with the renderer.
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

#[derive(Clone)]
pub struct MeshPipelineDesc<B: Backend> {
    pub layout: PipelineLayout<B>,
    pub task: Option<PipelineStage<B>>,
    pub mesh: PipelineStage<B>,
    pub fragment: Option<PipelineStage<B>>,
    pub primitive: PrimitiveState,
    pub targets: Vec<Option<ColorTargetState>>,
    pub depth_stencil: Option<DepthStencilState>,
    pub multisample: MultisampleState,
    pub cache: Option<PipelineCache<B>>,
}
impl<B: Backend> MeshOps<B> for Unavailable {
    const AVAILABLE: bool = false;
    type PipelineFuture<'a>
        = std::future::Ready<Result<B::NativeMeshPipeline>>
    where
        B: 'a;
    fn create_pipeline<'a>(
        _: &'a B::NativeDevice,
        _: &'a MeshPipelineDesc<B>,
    ) -> Self::PipelineFuture<'a> {
        std::future::ready(Err(Error::Unsupported("mesh")))
    }
    fn bind_pipeline(_: &mut B::NativeEncoder, _: &B::NativeMeshPipeline) -> Result<()> {
        Err(Error::Unsupported("mesh"))
    }
    fn dispatch(_: &mut B::NativeEncoder, _: [u32; 3]) -> Result<()> {
        Err(Error::Unsupported("mesh"))
    }
    fn dispatch_indirect(_: &mut B::NativeEncoder, _: &B::NativeBuffer, _: u64) -> Result<()> {
        Err(Error::Unsupported("mesh"))
    }
}
pub struct MeshDomain<'a, B: Backend> {
    device: &'a Device<B>,
}
impl<B: Backend> Device<B> {
    pub fn mesh(&self) -> Option<MeshDomain<'_, B>> {
        (self.caps.mesh && B::Mesh::AVAILABLE).then_some(MeshDomain { device: self })
    }
}
impl<B: Backend> MeshDomain<'_, B> {
    pub async fn create_pipeline(&self, desc: MeshPipelineDesc<B>) -> Result<MeshPipeline<B>> {
        // Task, mesh, fragment, layout, and cache must share one device even
        // when an optional stage is omitted from the executable pipeline.
        if desc.layout.device != self.device.id
            || desc.mesh.module.device != self.device.id
            || desc
                .task
                .as_ref()
                .is_some_and(|s| s.module.device != self.device.id)
            || desc
                .fragment
                .as_ref()
                .is_some_and(|s| s.module.device != self.device.id)
            || desc
                .cache
                .as_ref()
                .is_some_and(|c| c.device != self.device.id)
        {
            return Err(Error::WrongDevice);
        }
        let native = B::Mesh::create_pipeline(&self.device.native, &desc).await?;
        // SAFETY: the SPI created this pipeline for this exact native device.
        Ok(unsafe { MeshPipeline::from_native(self.device.id, native) })
    }
    pub fn bind_pipeline(&self, pass: &mut RasterPass<'_, B>, p: &MeshPipeline<B>) -> Result<()>
    where
        B: RasterBackend,
    {
        if p.device != self.device.id || pass.encoder.device != self.device.id {
            return Err(Error::WrongDevice);
        }
        pass.call(|e| B::Mesh::bind_pipeline(e, &p.native))
    }
    pub fn dispatch(&self, pass: &mut RasterPass<'_, B>, groups: [u32; 3]) -> Result<()>
    where
        B: RasterBackend,
    {
        if pass.encoder.device != self.device.id {
            return Err(Error::WrongDevice);
        }
        let result = pass.call(|e| B::Mesh::dispatch(e, groups));
        if result.is_ok() {
            pass.encoder.stats.record_dispatch();
        }
        result
    }
    pub fn dispatch_indirect(
        &self,
        pass: &mut RasterPass<'_, B>,
        buffer: &Buffer<B>,
        offset: u64,
    ) -> Result<()>
    where
        B: RasterBackend,
    {
        if pass.encoder.device != self.device.id || buffer.device != self.device.id {
            return Err(Error::WrongDevice);
        }
        let result = pass.call(|e| B::Mesh::dispatch_indirect(e, &buffer.native, offset));
        if result.is_ok() {
            pass.encoder.stats.record_dispatch();
        }
        result
    }
}
