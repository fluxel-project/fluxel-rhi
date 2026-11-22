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
    fn dispatch_indirect_multi(
        _: &mut B::NativeEncoder,
        _: &B::NativeBuffer,
        _: u64,
        _: u32,
    ) -> Result<()> {
        Err(Error::Unsupported("mesh multi-indirect"))
    }
    fn dispatch_indirect_count(
        _: &mut B::NativeEncoder,
        _: &B::NativeBuffer,
        _: u64,
        _: &B::NativeBuffer,
        _: u64,
        _: u32,
    ) -> Result<()> {
        Err(Error::Unsupported("mesh indirect count"))
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
    /// Queries the output topology independently of a specific shader.
    pub fn topology_support(&self, topology: PrimitiveTopology) -> Support<()> {
        B::Mesh::topology_support(&self.device.native, topology)
    }

    /// Checks an ordinary mesh pipeline descriptor using the selected device
    /// and declared shader entries. Native artifact and linkage checks continue
    /// during pipeline creation.
    pub fn pipeline_support(&self, desc: &MeshPipelineDesc<B>) -> Result<()> {
        self.validate_pipeline_desc(desc)?;
        B::Mesh::pipeline_support(&self.device.native, desc)
    }

    pub async fn create_pipeline(&self, desc: MeshPipelineDesc<B>) -> Result<MeshPipeline<B>> {
        self.create_pipeline_inner(desc, None, None).await
    }

    /// Creates a mesh raster pipeline with independent front and back stencil masks.
    pub async fn create_pipeline_with_stencil_masks(
        &self,
        desc: MeshPipelineDesc<B>,
        masks: StencilMasks,
    ) -> Result<MeshPipeline<B>> {
        self.create_pipeline_inner(desc, None, Some(masks)).await
    }

    /// Creates a mesh pipeline covering the declared view ordinals.
    /// The raster pass selects the active views.
    pub async fn create_multiview_pipeline(
        &self,
        desc: MeshPipelineDesc<B>,
        views: u32,
        stencil_masks: Option<StencilMasks>,
    ) -> Result<MeshPipeline<B>> {
        self.create_pipeline_inner(desc, Some(views), stencil_masks)
            .await
    }

    async fn create_pipeline_inner(
        &self,
        desc: MeshPipelineDesc<B>,
        views: Option<u32>,
        stencil_masks: Option<StencilMasks>,
    ) -> Result<MeshPipeline<B>> {
        self.validate_pipeline_desc(&desc)?;
        if let Some(views) = views {
            if !self.device.caps.mesh_multiview {
                return Err(Error::Unsupported("mesh multiview"));
            }
            if views == 0 || views > self.device.caps.limits.max_multiview_views {
                return Err(Error::InvalidRange);
            }
        }
        if stencil_masks.is_some() {
            self.device.validate_stencil_masks(desc.depth_stencil)?;
        }
        let native = match (views, stencil_masks) {
            (Some(views), masks) => {
                B::Mesh::create_multiview_pipeline(&self.device.native, &desc, views, masks).await?
            }
            (None, Some(masks)) => {
                B::Mesh::create_pipeline_with_stencil_masks(&self.device.native, &desc, masks)
                    .await?
            }
            (None, None) => B::Mesh::create_pipeline(&self.device.native, &desc).await?,
        };
        // SAFETY: the SPI created this pipeline for this exact native device.
        let mut pipeline = unsafe { MeshPipeline::from_native(self.device.id, native) };
        pipeline.layout = Some(desc.layout);
        Ok(pipeline)
    }

    fn validate_pipeline_desc(&self, desc: &MeshPipelineDesc<B>) -> Result<()> {
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
        if desc.targets.len() > self.device.caps.limits.max_color_attachments as usize {
            return Err(Error::InvalidRange);
        }
        if desc.primitive.conservative && self.device.caps.conservative_raster == Some(false) {
            return Err(Error::Unsupported("conservative rasterization"));
        }
        Ok(())
    }
    pub fn bind_pipeline(&self, pass: &mut RasterPass<'_, B>, p: &MeshPipeline<B>) -> Result<()>
    where
        B: RasterBackend,
    {
        if p.device != self.device.id || pass.encoder.device != self.device.id {
            return Err(Error::WrongDevice);
        }
        let result = pass.call(|e| B::Mesh::bind_pipeline(e, &p.native));
        if result.is_ok() {
            pass.encoder.stats.record_pipeline_bind();
        }
        result
    }
    pub fn dispatch(&self, pass: &mut RasterPass<'_, B>, groups: [u32; 3]) -> Result<()>
    where
        B: RasterBackend,
    {
        if pass.encoder.device != self.device.id {
            return Err(Error::WrongDevice);
        }
        let limits = self.device.caps.limits;
        if (limits.max_mesh_workgroups_per_dimension != 0
            && groups
                .iter()
                .any(|&axis| axis == 0 || axis > limits.max_mesh_workgroups_per_dimension))
            || (limits.max_mesh_workgroup_total_count != 0
                && groups
                    .into_iter()
                    .try_fold(1u64, |count, axis| count.checked_mul(u64::from(axis)))
                    .is_none_or(|count| count > u64::from(limits.max_mesh_workgroup_total_count)))
        {
            return Err(Error::InvalidRange);
        }
        let result = pass.call(|e| B::Mesh::dispatch(e, groups));
        if result.is_ok() {
            pass.encoder.stats.record_dispatch();
        }
        result
    }
    /// Executes one 12-byte `[u32; 3]` mesh dispatch record at a four-byte
    /// aligned offset. The caller transitions the argument range to
    /// `INDIRECT_READ` and provides valid dispatch group counts.
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
        if !offset.is_multiple_of(4) {
            return Err(Error::InvalidRange);
        }
        let result = pass.call(|e| B::Mesh::dispatch_indirect(e, &buffer.native, offset));
        if result.is_ok() {
            pass.encoder.stats.record_dispatch();
        }
        result
    }

    /// Executes a tightly packed batch of `[u32; 3]` mesh dispatch records.
    /// `offset` is four-byte aligned; every record occupies 12 bytes. The
    /// caller must transition the argument range to `INDIRECT_READ` first.
    pub fn dispatch_indirect_multi(
        &self,
        pass: &mut RasterPass<'_, B>,
        buffer: &Buffer<B>,
        offset: u64,
        count: u32,
    ) -> Result<()>
    where
        B: RasterBackend,
    {
        if pass.encoder.device != self.device.id || buffer.device != self.device.id {
            return Err(Error::WrongDevice);
        }
        if !self.device.caps.mesh_multi_indirect {
            return Err(Error::Unsupported("mesh multi-indirect"));
        }
        if count == 0 {
            return Ok(());
        }
        if count > self.device.caps.limits.max_mesh_indirect_count || offset % 4 != 0 {
            return Err(Error::InvalidRange);
        }
        let result =
            pass.call(|e| B::Mesh::dispatch_indirect_multi(e, &buffer.native, offset, count));
        if result.is_ok() {
            pass.encoder.stats.record_dispatch();
        }
        result
    }

    /// Executes up to `max_count` packed mesh dispatch records. The GPU reads
    /// the actual record count from a four-byte count buffer at `count_offset`.
    /// The caller must transition both buffers to `INDIRECT_READ` first.
    pub fn dispatch_indirect_count(
        &self,
        pass: &mut RasterPass<'_, B>,
        buffer: &Buffer<B>,
        offset: u64,
        count_buffer: &Buffer<B>,
        count_offset: u64,
        max_count: u32,
    ) -> Result<()>
    where
        B: RasterBackend,
    {
        if pass.encoder.device != self.device.id
            || buffer.device != self.device.id
            || count_buffer.device != self.device.id
        {
            return Err(Error::WrongDevice);
        }
        if !self.device.caps.mesh_indirect_count {
            return Err(Error::Unsupported("mesh indirect count"));
        }
        if max_count == 0 {
            return Ok(());
        }
        if max_count > self.device.caps.limits.max_mesh_indirect_count
            || offset % 4 != 0
            || count_offset % 4 != 0
        {
            return Err(Error::InvalidRange);
        }
        let result = pass.call(|e| {
            B::Mesh::dispatch_indirect_count(
                e,
                &buffer.native,
                offset,
                &count_buffer.native,
                count_offset,
                max_count,
            )
        });
        if result.is_ok() {
            pass.encoder.stats.record_dispatch();
        }
        result
    }
}
