//! Immediate Vulkan compute command lowering.

use super::*;
use std::future::{Ready, ready};

/// Static ComputeOps implementation for Vulkan.
pub struct VulkanCompute;

impl api::ComputeOps<Vulkan> for VulkanCompute {
    const AVAILABLE: bool = true;
    type PipelineFuture<'a>
        = Ready<Result<VulkanComputePipeline>>
    where
        Vulkan: 'a;

    fn create_pipeline<'a>(
        device: &'a VulkanDevice,
        desc: &'a api::ComputePipelineDesc<Vulkan>,
    ) -> Self::PipelineFuture<'a> {
        ready(pipeline::create_compute(device, desc))
    }

    fn begin(e: &mut VulkanEncoder) -> Result<()> {
        if e.raster || e.compute {
            return Err(Error::InvalidState);
        }
        e.compute = true;
        Ok(())
    }

    fn set_pipeline(e: &mut VulkanEncoder, pipeline: &VulkanComputePipeline) -> Result<()> {
        if !e.compute {
            return Err(Error::InvalidState);
        }
        // SAFETY: `e` owns a recording primary command buffer; the pipeline
        // retains this device and was created for its COMPUTE bind point.
        unsafe {
            e.device.device.cmd_bind_pipeline(
                e.command,
                vk::PipelineBindPoint::COMPUTE,
                pipeline.native(),
            )
        };
        e.compute_layout = Some(pipeline.layout());
        let layout = pipeline.layout_state();
        e.compute_push_constants = Some(layout.push_constants());
        e.compute_layout_state = Some(layout);
        e.compute_pipelines.push(pipeline.clone());
        Ok(())
    }

    fn set_bind_group(
        e: &mut VulkanEncoder,
        slot: u32,
        group: &VulkanBindGroup,
        offsets: &[u32],
    ) -> Result<()> {
        if !e.compute {
            return Err(Error::InvalidState);
        }
        group.validate_dynamic_offsets(offsets)?;
        let layout = e.compute_layout.ok_or(Error::InvalidState)?;
        e.compute_layout_state
            .as_ref()
            .ok_or(Error::InvalidState)?
            .validate_group(slot, group)?;
        // SAFETY: the validated group/layout pair belongs to this pipeline and
        // encoder; offsets were checked against retained buffer ranges above.
        unsafe {
            e.device.device.cmd_bind_descriptor_sets(
                e.command,
                vk::PipelineBindPoint::COMPUTE,
                layout,
                slot,
                &[group.native()],
                offsets,
            );
        }
        e.groups.push(group.clone());
        Ok(())
    }

    fn dispatch(e: &mut VulkanEncoder, groups: [u32; 3]) -> Result<()> {
        if !e.compute || e.compute_layout.is_none() {
            return Err(Error::InvalidState);
        }
        if groups.contains(&0) {
            return Err(Error::InvalidRange);
        }
        // SAFETY: an active compute pass owns this recording command buffer and
        // the nonzero dispatch counts were validated against the Core contract.
        unsafe {
            e.device
                .device
                .cmd_dispatch(e.command, groups[0], groups[1], groups[2])
        };
        Ok(())
    }

    fn dispatch_indirect(_: &mut VulkanEncoder, _: &VulkanBuffer, _: u64) -> Result<()> {
        Err(Error::Unsupported("Vulkan compute indirect dispatch"))
    }

    fn set_push_constants(e: &mut VulkanEncoder, offset: u32, bytes: &[u8]) -> Result<()> {
        let layout = e.compute_layout.ok_or(Error::InvalidState)?;
        let ranges = e
            .compute_push_constants
            .as_ref()
            .ok_or(Error::InvalidState)?;
        let stages = ranges.validate(api::StageMask::COMPUTE, offset, bytes)?;
        // SAFETY: PushConstants validates the supplied bytes against the live
        // pipeline layout's declared COMPUTE ranges before this native record.
        unsafe {
            e.device
                .device
                .cmd_push_constants(e.command, layout, stages, offset, bytes)
        };
        Ok(())
    }

    fn end(e: &mut VulkanEncoder) -> Result<()> {
        if !e.compute {
            return Err(Error::InvalidState);
        }
        e.compute = false;
        e.compute_layout = None;
        e.compute_layout_state = None;
        e.compute_push_constants = None;
        Ok(())
    }

    fn abort(e: &mut VulkanEncoder) {
        e.compute = false;
        e.compute_layout = None;
        e.compute_layout_state = None;
        e.compute_push_constants = None;
    }
}
