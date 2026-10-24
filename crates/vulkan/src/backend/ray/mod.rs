//! Optional Vulkan acceleration structures, ray pipelines, and ray query support.
mod acceleration;
mod pipeline;
pub(super) use pipeline::restore_after_external_boundary;
pub(crate) mod probe;
use super::{Vulkan, VulkanBindGroup, VulkanBuffer, VulkanDevice, VulkanEncoder};
pub use acceleration::VulkanAcceleration;
use fluxel_rhi_core::api::{self, Error, Result};
pub use pipeline::VulkanRayPipeline;
use std::future::{Ready, ready};

pub struct VulkanRay;
impl api::RayOps<Vulkan> for VulkanRay {
    const AVAILABLE: bool = true;
    type PipelineFuture<'a> = Ready<Result<VulkanRayPipeline>>;
    fn create_pipeline<'a>(
        device: &'a VulkanDevice,
        desc: &'a api::RayPipelineDesc<Vulkan>,
    ) -> Self::PipelineFuture<'a> {
        ready(pipeline::create(device, desc))
    }
    fn set_pipeline(encoder: &mut VulkanEncoder, pipeline: &VulkanRayPipeline) -> Result<()> {
        pipeline::bind(encoder, pipeline)
    }
    fn set_bind_group(
        encoder: &mut VulkanEncoder,
        slot: u32,
        group: &VulkanBindGroup,
        offsets: &[u32],
    ) -> Result<()> {
        pipeline::bind_group(encoder, slot, group, offsets)
    }
    fn shader_table_properties(device: &VulkanDevice) -> Result<api::ShaderTableProperties> {
        let p = device.0.ray_support;
        if !p.pipeline {
            return Err(Error::Unsupported("Vulkan ray pipeline"));
        }
        Ok(api::ShaderTableProperties {
            identifier_size: p.shader_group_handle_size,
            record_alignment: p.shader_group_handle_alignment,
            table_alignment: p.shader_group_base_alignment,
            max_record_stride: p.max_shader_group_stride,
        })
    }
    fn shader_group_identifier(
        _: &VulkanDevice,
        p: &VulkanRayPipeline,
        index: u32,
    ) -> Result<Vec<u8>> {
        pipeline::identifier(p, index)
    }
    fn acceleration_sizes(
        device: &VulkanDevice,
        desc: &api::AccelerationDesc<Vulkan>,
    ) -> Result<api::AccelerationSizes> {
        acceleration::sizes(device, desc)
    }
    fn create_acceleration(
        device: &VulkanDevice,
        desc: &api::AccelerationDesc<Vulkan>,
    ) -> Result<VulkanAcceleration> {
        acceleration::create(device, desc)
    }
    fn build_acceleration(
        encoder: &mut VulkanEncoder,
        dst: &VulkanAcceleration,
        scratch: &VulkanBuffer,
        desc: &api::AccelerationDesc<Vulkan>,
    ) -> Result<()> {
        acceleration::build(encoder, dst, scratch, desc)
    }
    fn trace(
        encoder: &mut VulkanEncoder,
        pipeline: &VulkanRayPipeline,
        table: &api::ShaderTable<Vulkan>,
        dims: [u32; 3],
    ) -> Result<()> {
        pipeline::trace(encoder, pipeline, table, dims)
    }
}
