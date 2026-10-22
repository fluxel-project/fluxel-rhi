//! Direct Vulkan command-buffer ownership.
//!
//! One Core encoder owns one transient primary command pool and command buffer.
//! It records native commands immediately; it does not retain an interpreted
//! CPU command stream for later replay.

use super::{
    DeviceState, VulkanBindGroup, VulkanBuffer, VulkanComputePipeline, VulkanDevice,
    VulkanRasterPipeline, VulkanTexture, VulkanTextureView, vk_error,
};
use ash::vk;
use fluxel_rhi_core::api::Result;
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
    pub(super) pool: vk::CommandPool,
    pub(super) command: vk::CommandBuffer,
    pub(super) raster: bool,
    pub(super) compute: bool,
    pub(super) color_surface_image: Option<vk::Image>,
    pub(super) capture: Option<FrameCapture>,
    pub(super) raster_layout: Option<vk::PipelineLayout>,
    pub(super) raster_index_bound: bool,
    /// The dynamic-rendering view mask of the active primary pass. Pipelines
    /// carry their own creation-time mask and must match it before binding.
    pub(super) raster_multiview_mask: Option<u32>,
    pub(super) raster_signature: Option<(vk::Format, Option<vk::Format>, u32)>,
    pub(super) raster_layout_state: Option<super::pipeline::VulkanPipelineLayout>,
    pub(super) raster_push_constants: Option<super::pipeline::PushConstants>,
    /// Set only for a primary dynamic-rendering pass that accepts inherited
    /// secondary command buffers.  The signature is validated again by the
    /// secondary executor before `vkCmdExecuteCommands`.
    pub(super) secondary_signature: Option<fluxel_rhi_core::api::SecondaryRasterDesc>,
    pub(super) compute_layout: Option<vk::PipelineLayout>,
    pub(super) compute_layout_state: Option<super::pipeline::VulkanPipelineLayout>,
    pub(super) compute_push_constants: Option<super::pipeline::PushConstants>,
    pub(super) mesh_active: Option<bool>,
    pub(super) mesh_pipelines: Vec<super::VulkanMeshPipeline>,
    pub(super) accelerations: Vec<super::VulkanAcceleration>,
    pub(super) ray_pipelines: Vec<super::VulkanRayPipeline>,
    pub(super) ray_layout_state: Option<super::pipeline::VulkanPipelineLayout>,
    pub(super) pipelines: Vec<VulkanRasterPipeline>,
    pub(super) compute_pipelines: Vec<VulkanComputePipeline>,
    pub(super) groups: Vec<VulkanBindGroup>,
    pub(super) attachments: Vec<VulkanTextureView>,
    pub(super) uploads: Vec<VulkanBuffer>,
    pub(super) buffers: Vec<VulkanBuffer>,
    pub(super) textures: Vec<VulkanTexture>,
    pub(super) queries: Vec<super::VulkanQuerySet>,
    pub(super) active_occlusion: Option<(super::VulkanQuerySet, u32)>,
    pub(super) active_statistics: Option<(super::VulkanQuerySet, u32)>,
    pub(super) query_ops: Vec<super::query::QueryOp>,
    pub(super) secondary: Vec<super::VulkanSecondaryWork>,
}
impl VulkanEncoder {
    pub(super) fn new(device: &VulkanDevice, lane: fluxel_rhi_core::api::LaneId) -> Result<Self> {
        let info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(device.0.graphics_family)
            .flags(vk::CommandPoolCreateFlags::TRANSIENT);
        // SAFETY: DeviceState owns a live logical device and this pool is
        // restricted to its graphics family; VulkanEncoder becomes its owner.
        let pool = unsafe { device.0.device.create_command_pool(&info, None) }.map_err(vk_error)?;
        let alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        // SAFETY: `pool` was just created on this device and has not been
        // destroyed or reset, so it can allocate one primary command buffer.
        let command = match unsafe { device.0.device.allocate_command_buffers(&alloc) } {
            Ok(v) => v[0],
            Err(e) => {
                // SAFETY: allocation returned no command buffer, leaving this
                // local pool unused and exclusively owned by this error path.
                unsafe { device.0.device.destroy_command_pool(pool, None) };
                return Err(vk_error(e));
            }
        };
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: `command` is a newly allocated primary buffer and no thread
        // can access this private pool before the encoder is returned.
        if let Err(e) = unsafe { device.0.device.begin_command_buffer(command, &begin) } {
            // SAFETY: a failed begin leaves the locally owned pool safe to
            // destroy; it has not been submitted.
            unsafe { device.0.device.destroy_command_pool(pool, None) };
            return Err(vk_error(e));
        };
        Ok(Self {
            owner: device.clone(),
            lane,
            device: device.0.clone(),
            pool,
            command,
            raster: false,
            compute: false,
            color_surface_image: None,
            capture: None,
            raster_layout: None,
            raster_index_bound: false,
            raster_multiview_mask: None,
            raster_signature: None,
            raster_layout_state: None,
            raster_push_constants: None,
            secondary_signature: None,
            compute_layout: None,
            compute_layout_state: None,
            compute_push_constants: None,
            mesh_active: None,
            mesh_pipelines: Vec::new(),
            accelerations: Vec::new(),
            ray_pipelines: Vec::new(),
            ray_layout_state: None,
            pipelines: Vec::new(),
            compute_pipelines: Vec::new(),
            groups: Vec::new(),
            attachments: Vec::new(),
            uploads: Vec::new(),
            buffers: Vec::new(),
            textures: Vec::new(),
            queries: Vec::new(),
            active_occlusion: None,
            active_statistics: None,
            query_ops: Vec::new(),
            secondary: Vec::new(),
        })
    }
}
impl Drop for VulkanEncoder {
    fn drop(&mut self) {
        if self.pool != vk::CommandPool::null() {
            // SAFETY: submitted encoders move into RecordedWork and remain
            // retained until their fence; this Drop runs only for unsubmitted
            // or completed work, and owns the pool and all its command buffers.
            unsafe { self.device.device.destroy_command_pool(self.pool, None) }
        }
    }
}
