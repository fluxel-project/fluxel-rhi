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
    pub(super) raster_layout_state: Option<super::pipeline::VulkanPipelineLayout>,
    pub(super) raster_push_constants: Option<super::pipeline::PushConstants>,
    /// Set only for a primary dynamic-rendering pass that accepts inherited
    /// secondary command buffers.  The signature is validated again by the
    /// secondary executor before `vkCmdExecuteCommands`.
    pub(super) secondary_signature: Option<fluxel_rhi_core::api::SecondaryRasterDesc>,
    pub(super) compute_layout: Option<vk::PipelineLayout>,
    pub(super) compute_layout_state: Option<super::pipeline::VulkanPipelineLayout>,
    pub(super) compute_push_constants: Option<super::pipeline::PushConstants>,
    pub(super) pipelines: Vec<VulkanRasterPipeline>,
    pub(super) compute_pipelines: Vec<VulkanComputePipeline>,
    pub(super) groups: Vec<VulkanBindGroup>,
    pub(super) attachments: Vec<VulkanTextureView>,
    pub(super) uploads: Vec<VulkanBuffer>,
    pub(super) buffers: Vec<VulkanBuffer>,
    pub(super) textures: Vec<VulkanTexture>,
    pub(super) queries: Vec<super::VulkanQuerySet>,
    pub(super) active_occlusion: Option<(super::VulkanQuerySet, u32)>,
    pub(super) secondary: Vec<super::VulkanSecondaryWork>,
}
impl VulkanEncoder {
    pub(super) fn new(device: &VulkanDevice) -> Result<Self> {
        let info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(device.0.graphics_family)
            .flags(vk::CommandPoolCreateFlags::TRANSIENT);
        let pool = unsafe { device.0.device.create_command_pool(&info, None) }.map_err(vk_error)?;
        let alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command = match unsafe { device.0.device.allocate_command_buffers(&alloc) } {
            Ok(v) => v[0],
            Err(e) => {
                unsafe { device.0.device.destroy_command_pool(pool, None) };
                return Err(vk_error(e));
            }
        };
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        if let Err(e) = unsafe { device.0.device.begin_command_buffer(command, &begin) } {
            unsafe { device.0.device.destroy_command_pool(pool, None) };
            return Err(vk_error(e));
        };
        Ok(Self {
            owner: device.clone(),
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
            raster_layout_state: None,
            raster_push_constants: None,
            secondary_signature: None,
            compute_layout: None,
            compute_layout_state: None,
            compute_push_constants: None,
            pipelines: Vec::new(),
            compute_pipelines: Vec::new(),
            groups: Vec::new(),
            attachments: Vec::new(),
            uploads: Vec::new(),
            buffers: Vec::new(),
            textures: Vec::new(),
            queries: Vec::new(),
            active_occlusion: None,
            secondary: Vec::new(),
        })
    }
}
impl Drop for VulkanEncoder {
    fn drop(&mut self) {
        if self.pool != vk::CommandPool::null() {
            unsafe { self.device.device.destroy_command_pool(self.pool, None) }
        }
    }
}
