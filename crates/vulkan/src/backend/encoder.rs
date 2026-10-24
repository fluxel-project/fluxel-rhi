//! Direct Vulkan command-buffer ownership.
//!
//! One Core encoder owns one transient primary command pool and command buffer.
//! It records native commands immediately; it does not retain an interpreted
//! CPU command stream for later replay.

use super::{
    DeviceState, VulkanBindGroup, VulkanBuffer, VulkanComputePipeline, VulkanDevice,
    VulkanRasterPipeline, VulkanTexture, VulkanTextureView, command_pool::CommandPoolLease,
    vk_error,
};
use ash::vk;
use fluxel_rhi_core::api::{Error, Result};
use std::sync::Arc;

pub(super) struct FrameCapture {
    pub(super) buffer: VulkanBuffer,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) path: std::path::PathBuf,
}

pub struct VulkanEncoder {
    pub(super) owner: VulkanDevice,
    pub(super) lane: fluxel_rhi_core::api::LaneId,
    pub(super) device: Arc<DeviceState>,
    pub(super) pool: Option<CommandPoolLease>,
    pub(super) command: vk::CommandBuffer,
    /// Closed primary buffers separated by an explicit external queue
    /// synchronization boundary.  The active `command` is the final segment.
    /// This retains native work only; it is never a CPU command replay stream.
    pub(super) external_segments: Vec<ExternalSegment>,
    pub(super) raster: bool,
    pub(super) raster_depth_read_only: bool,
    pub(super) raster_stencil_read_only: bool,
    pub(super) pending_resolves: Vec<super::raster::resolve::PendingResolve>,
    pub(super) compute: bool,
    pub(super) color_surface_image: Option<vk::Image>,
    pub(super) capture: Option<FrameCapture>,
    pub(super) raster_layout: Option<vk::PipelineLayout>,
    pub(super) raster_bound_groups: std::collections::BTreeMap<u32, (VulkanBindGroup, Vec<u32>)>,
    pub(super) raster_index_bound: bool,
    /// The dynamic-rendering view mask of the active primary pass. Pipelines
    /// carry their own creation-time mask and must match it before binding.
    pub(super) raster_multiview_mask: Option<u32>,
    pub(super) raster_signature: Option<(Vec<vk::Format>, Option<vk::Format>, u32)>,
    pub(super) raster_layout_state: Option<super::pipeline::VulkanPipelineLayout>,
    pub(super) raster_push_constants: Option<super::pipeline::PushConstants>,
    /// Set only for a primary dynamic-rendering pass that accepts inherited
    /// secondary command buffers.  The signature is validated again by the
    /// secondary executor before `vkCmdExecuteCommands`.
    pub(super) secondary_signature: Option<fluxel_rhi_core::api::SecondaryRasterDesc>,
    pub(super) compute_bound_groups: std::collections::BTreeMap<u32, (VulkanBindGroup, Vec<u32>)>,
    pub(super) ray_bound_groups: std::collections::BTreeMap<u32, (VulkanBindGroup, Vec<u32>)>,
    pub(super) compute_layout: Option<vk::PipelineLayout>,
    pub(super) compute_layout_state: Option<super::pipeline::VulkanPipelineLayout>,
    pub(super) compute_push_constants: Option<super::pipeline::PushConstants>,
    pub(super) mesh_active: Option<bool>,
    pub(super) mesh_pipelines: Vec<super::VulkanMeshPipeline>,
    pub(super) accelerations: Vec<super::VulkanAcceleration>,
    pub(crate) ray_builds: Vec<super::ray::RecordedTlasBuild>,
    pub(super) ray_pipelines: Vec<super::VulkanRayPipeline>,
    pub(super) ray_layout_state: Option<super::pipeline::VulkanPipelineLayout>,
    pub(super) pipelines: Vec<VulkanRasterPipeline>,
    pub(super) compute_pipelines: Vec<VulkanComputePipeline>,
    pub(super) groups: Vec<VulkanBindGroup>,
    pub(super) attachments: Vec<VulkanTextureView>,
    pub(super) uploads: Vec<VulkanBuffer>,
    pub(crate) upload_slices: Vec<super::upload::UploadSlice>,
    pub(crate) readback_slices: Vec<super::readback::ReadbackSlice>,
    pub(super) buffers: Vec<VulkanBuffer>,
    pub(super) textures: Vec<VulkanTexture>,
    pub(super) queries: Vec<super::VulkanQuerySet>,
    pub(super) active_occlusion: Option<(super::VulkanQuerySet, u32)>,
    pub(super) active_statistics: Option<(super::VulkanQuerySet, u32)>,
    pub(super) pass_timestamp_end: Option<(super::VulkanQuerySet, u32, bool)>,
    pub(super) query_ops: Vec<super::query::QueryOp>,
    pub(super) secondary: Vec<super::VulkanSecondaryWork>,
}

pub(super) enum ExternalBoundary {
    Wait(Arc<super::external::ExternalSemaphore>),
    Signal(Arc<super::external::ExternalSemaphore>),
}

pub(super) struct ExternalSegment {
    pub(super) command: vk::CommandBuffer,
    pub(super) boundary: ExternalBoundary,
}

fn begin_primary(pool: &mut CommandPoolLease) -> Result<vk::CommandBuffer> {
    pool.begin()
}
impl VulkanEncoder {
    pub(super) fn new(device: &VulkanDevice, lane: fluxel_rhi_core::api::LaneId) -> Result<Self> {
        let family = if lane.ordinal == 1 {
            device.0.compute_family
        } else {
            device.0.graphics_family
        };
        let mut pool = device
            .0
            .command_pools
            .lease(family, vk::CommandBufferLevel::PRIMARY)?;
        let command = begin_primary(&mut pool)?;
        Ok(Self {
            owner: device.clone(),
            lane,
            device: device.0.clone(),
            pool: Some(pool),
            command,
            external_segments: Vec::new(),
            raster: false,
            raster_depth_read_only: false,
            raster_stencil_read_only: false,
            pending_resolves: Vec::new(),
            compute: false,
            color_surface_image: None,
            capture: None,
            raster_layout: None,
            raster_bound_groups: Default::default(),
            raster_index_bound: false,
            raster_multiview_mask: None,
            raster_signature: None,
            raster_layout_state: None,
            raster_push_constants: None,
            secondary_signature: None,
            compute_bound_groups: Default::default(),
            ray_bound_groups: Default::default(),
            compute_layout: None,
            compute_layout_state: None,
            compute_push_constants: None,
            mesh_active: None,
            mesh_pipelines: Vec::new(),
            accelerations: Vec::new(),
            ray_builds: Vec::new(),
            ray_pipelines: Vec::new(),
            ray_layout_state: None,
            pipelines: Vec::new(),
            compute_pipelines: Vec::new(),
            groups: Vec::new(),
            attachments: Vec::new(),
            uploads: Vec::new(),
            upload_slices: Vec::new(),
            readback_slices: Vec::new(),
            buffers: Vec::new(),
            textures: Vec::new(),
            queries: Vec::new(),
            active_occlusion: None,
            active_statistics: None,
            pass_timestamp_end: None,
            query_ops: Vec::new(),
            secondary: Vec::new(),
        })
    }
}
impl VulkanEncoder {
    /// Ends the current primary buffer at an explicit Core external boundary
    /// and begins the following one from the same transient pool.  It does
    /// not derive dependencies, schedule work, or preserve a CPU command log.
    pub(super) fn split_external_boundary(&mut self, boundary: ExternalBoundary) -> Result<()> {
        if self.raster
            || self.compute
            || self.active_occlusion.is_some()
            || self.active_statistics.is_some()
        {
            return Err(Error::InvalidState);
        }
        // Allocate and begin first so an allocation failure leaves the active
        // command buffer recording and no externally visible boundary exists.
        let next = begin_primary(self.pool.as_mut().ok_or(Error::InvalidState)?)?;
        // SAFETY: this encoder exclusively owns the active recording buffer.
        unsafe { self.device.device.end_command_buffer(self.command) }.map_err(vk_error)?;
        let prior = std::mem::replace(&mut self.command, next);
        self.external_segments.push(ExternalSegment {
            command: prior,
            boundary,
        });
        // Core admits boundaries only outside raster/compute passes.  Ray
        // bindings are legal outside a pass, however, and Vulkan binding state
        // is command-buffer-local, so restore that native state only.
        super::ray::restore_after_external_boundary(self)
    }
}
impl Drop for VulkanEncoder {
    fn drop(&mut self) {
        // Raw staging leases must return while this encoder still owns its
        // DeviceState, including failed recordings with no retained target.
        self.upload_slices.clear();
        self.readback_slices.clear();
        // Reset/return the exclusive native page before releasing the final
        // DeviceState owner. Submitted encoders remain fence-retained.
        drop(self.pool.take());
    }
}
