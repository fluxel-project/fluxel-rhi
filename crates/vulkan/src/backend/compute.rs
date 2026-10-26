//! Immediate Vulkan compute command lowering.

use super::*;

/// Static ComputeOps implementation for Vulkan.
pub struct VulkanCompute;

impl fluxel_rhi_core::backend::ComputeOps<Vulkan> for VulkanCompute {
    const AVAILABLE: bool = true;
    type PipelineFuture<'a>
        = super::cold_worker::ColdFuture<VulkanComputePipeline>
    where
        Vulkan: 'a;

    fn create_pipeline<'a>(
        device: &'a VulkanDevice,
        desc: &'a api::ComputePipelineDesc<Vulkan>,
    ) -> Self::PipelineFuture<'a> {
        let device = device.clone();
        let desc = desc.clone();
        super::cold_worker::submit(move || pipeline::create_compute(&device, &desc))
    }

    fn begin(e: &mut VulkanEncoder) -> Result<()> {
        if e.raster || e.compute {
            return Err(Error::InvalidState);
        }
        if e.lane.ordinal == 0 && !e.device.graphics_compute {
            return Err(Error::Unsupported("Vulkan queue lacks compute capability"));
        }
        e.compute = true;
        Ok(())
    }

    fn set_pipeline(e: &mut VulkanEncoder, pipeline: &VulkanComputePipeline) -> Result<()> {
        if !e.compute {
            return Err(Error::InvalidState);
        }
        let layout = pipeline.layout_state();
        let bindings = super::raster::prepare_groups(&layout, &e.compute_bound_groups)?;
        // SAFETY: `e` owns a recording primary command buffer; the pipeline
        // retains this device and was created for its COMPUTE bind point.
        unsafe {
            e.device.device.cmd_bind_pipeline(
                e.command,
                vk::PipelineBindPoint::COMPUTE,
                pipeline.native(),
            )
        };
        for (slot, set, offsets) in &bindings {
            // SAFETY: preparation retained and validated each descriptor set.
            unsafe {
                e.device.device.cmd_bind_descriptor_sets(
                    e.command,
                    vk::PipelineBindPoint::COMPUTE,
                    layout.native(),
                    *slot,
                    &[*set],
                    offsets,
                );
            }
        }
        e.compute_layout = Some(pipeline.layout());
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
        let offsets = group.validate_dynamic_offsets(offsets)?;
        let layout = e.compute_layout.ok_or(Error::InvalidState)?;
        let descriptor = e
            .compute_layout_state
            .as_ref()
            .ok_or(Error::InvalidState)?
            .native_group(slot, group)?;
        // SAFETY: the validated group/layout pair belongs to this pipeline and
        // encoder; offsets were checked against retained buffer ranges above.
        unsafe {
            e.device.device.cmd_bind_descriptor_sets(
                e.command,
                vk::PipelineBindPoint::COMPUTE,
                layout,
                slot,
                &[descriptor],
                &offsets,
            );
        }
        e.compute_bound_groups
            .insert(slot, (group.clone(), offsets));
        e.groups.push(group.clone());
        Ok(())
    }

    fn dispatch(e: &mut VulkanEncoder, groups: [u32; 3]) -> Result<()> {
        if !e.compute || e.compute_layout.is_none() {
            return Err(Error::InvalidState);
        }
        if groups.contains(&0)
            || groups
                .iter()
                .zip(e.device.max_compute_workgroups)
                .any(|(&count, max)| count > max)
        {
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

    fn dispatch_indirect(
        e: &mut VulkanEncoder,
        arguments: &VulkanBuffer,
        offset: u64,
    ) -> Result<()> {
        if !e.compute || e.compute_layout.is_none() {
            return Err(Error::InvalidState);
        }
        if !arguments.0.usage.contains(api::BufferUsage::INDIRECT)
            || offset % 4 != 0
            || offset
                .checked_add(12)
                .is_none_or(|end| end > arguments.0.size)
        {
            return Err(Error::InvalidRange);
        }
        // SAFETY: the command buffer is recording a compute pass, the argument
        // range is aligned and in bounds, and the buffer is retained below.
        unsafe {
            e.device
                .device
                .cmd_dispatch_indirect(e.command, arguments.0.buffer, offset)
        };
        e.buffers.push(arguments.clone());
        Ok(())
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
        for (stage, native_offset) in stages {
            unsafe {
                e.device
                    .device
                    .cmd_push_constants(e.command, layout, stage, native_offset, bytes);
            }
        }
        Ok(())
    }

    fn end(e: &mut VulkanEncoder) -> Result<()> {
        if !e.compute {
            return Err(Error::InvalidState);
        }
        e.compute = false;
        e.compute_bound_groups.clear();
        e.compute_layout = None;
        e.compute_layout_state = None;
        e.compute_push_constants = None;
        Ok(())
    }

    fn abort(e: &mut VulkanEncoder) {
        e.compute = false;
        e.compute_bound_groups.clear();
        e.compute_layout = None;
        e.compute_layout_state = None;
        e.compute_push_constants = None;
    }
}
