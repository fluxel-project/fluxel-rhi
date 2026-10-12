//! Explicit Core resource-transition lowering for Vulkan.
//!
//! Vulkan itself has no `COMMON` image layout.  Fluxel's portable `Common`
//! state is represented by `GENERAL`; callers still supply every dependency.

use super::*;

pub(super) fn record(
    encoder: &mut VulkanEncoder,
    ops: &[api::ResourceTransition<'_, Vulkan>],
) -> Result<()> {
    for op in ops {
        let api::ResourceTransition::Texture(barrier) = op else {
            return Err(Error::Unsupported("Vulkan resource transition"));
        };
        if barrier.ownership.is_some() {
            return Err(Error::Unsupported("Vulkan queue ownership transition"));
        }
        let Some((old_layout, new_layout)) = barrier.layout else {
            return Err(Error::Unsupported("Vulkan memory-only texture transition"));
        };
        let texture = unsafe { barrier.target.texture.native_ref() };
        let range = vk::ImageSubresourceRange::default()
            .aspect_mask(aspects(barrier.target.aspects)?)
            .base_mip_level(barrier.target.mips.start)
            .level_count(
                barrier
                    .target
                    .mips
                    .end
                    .checked_sub(barrier.target.mips.start)
                    .ok_or(Error::InvalidRange)?,
            )
            .base_array_layer(barrier.target.layers.start)
            .layer_count(
                barrier
                    .target
                    .layers
                    .end
                    .checked_sub(barrier.target.layers.start)
                    .ok_or(Error::InvalidRange)?,
            );
        if range.level_count == 0
            || range.layer_count == 0
            || barrier.target.mips.end > texture.desc().mip_levels
            || barrier.target.layers.end > texture.desc().array_layers
        {
            return Err(Error::InvalidRange);
        }
        let native = vk::ImageMemoryBarrier::default()
            .src_access_mask(accesses(barrier.usage.from.accesses)?)
            .dst_access_mask(accesses(barrier.usage.to.accesses)?)
            .old_layout(layout(old_layout))
            .new_layout(layout(new_layout))
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(texture.native())
            .subresource_range(range);
        unsafe {
            encoder.device.device.cmd_pipeline_barrier(
                encoder.command,
                stages(barrier.usage.from.stages)?,
                stages(barrier.usage.to.stages)?,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[native],
            );
        }
        if new_layout != api::ImageLayout::Undefined {
            texture.mark_initialized(barrier.target.mips.clone(), barrier.target.layers.clone())?;
        }
        encoder.textures.push(texture.clone());
    }
    Ok(())
}

fn layout(layout: api::ImageLayout) -> vk::ImageLayout {
    match layout {
        api::ImageLayout::Undefined => vk::ImageLayout::UNDEFINED,
        api::ImageLayout::Common | api::ImageLayout::General => vk::ImageLayout::GENERAL,
        api::ImageLayout::ShaderReadOnly => vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        api::ImageLayout::CopySrc => vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        api::ImageLayout::CopyDst => vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        api::ImageLayout::ColorAttachment => vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        api::ImageLayout::DepthStencilAttachment => {
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
        }
        api::ImageLayout::DepthReadOnly => vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
        api::ImageLayout::Present => vk::ImageLayout::PRESENT_SRC_KHR,
    }
}

fn stages(mask: api::StageMask) -> Result<vk::PipelineStageFlags> {
    if mask == api::StageMask::ALL_COMMANDS {
        return Ok(vk::PipelineStageFlags::ALL_COMMANDS);
    }
    let known = api::StageMask::HOST.0
        | api::StageMask::COPY.0
        | api::StageMask::INDIRECT.0
        | api::StageMask::VERTEX_INPUT.0
        | api::StageMask::VERTEX.0
        | api::StageMask::FRAGMENT.0
        | api::StageMask::COLOR_OUTPUT.0
        | api::StageMask::COMPUTE.0;
    if mask.0 & !known != 0 {
        return Err(Error::Unsupported("Vulkan transition stage"));
    }
    let mut result = vk::PipelineStageFlags::empty();
    if mask.0 & api::StageMask::HOST.0 != 0 {
        result |= vk::PipelineStageFlags::HOST;
    }
    if mask.0 & api::StageMask::COPY.0 != 0 {
        result |= vk::PipelineStageFlags::TRANSFER;
    }
    if mask.0 & api::StageMask::INDIRECT.0 != 0 {
        result |= vk::PipelineStageFlags::DRAW_INDIRECT;
    }
    if mask.0 & api::StageMask::VERTEX_INPUT.0 != 0 {
        result |= vk::PipelineStageFlags::VERTEX_INPUT;
    }
    if mask.0 & api::StageMask::VERTEX.0 != 0 {
        result |= vk::PipelineStageFlags::VERTEX_SHADER;
    }
    if mask.0 & api::StageMask::FRAGMENT.0 != 0 {
        result |= vk::PipelineStageFlags::FRAGMENT_SHADER;
    }
    if mask.0 & api::StageMask::COLOR_OUTPUT.0 != 0 {
        result |= vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT;
    }
    if mask.0 & api::StageMask::COMPUTE.0 != 0 {
        result |= vk::PipelineStageFlags::COMPUTE_SHADER;
    }
    if result.is_empty() {
        return Err(Error::InvalidRange);
    }
    Ok(result)
}

fn accesses(mask: api::AccessMask) -> Result<vk::AccessFlags> {
    let known = api::AccessMask::HOST_READ.0
        | api::AccessMask::HOST_WRITE.0
        | api::AccessMask::COPY_READ.0
        | api::AccessMask::COPY_WRITE.0
        | api::AccessMask::VERTEX_READ.0
        | api::AccessMask::INDEX_READ.0
        | api::AccessMask::UNIFORM_READ.0
        | api::AccessMask::SHADER_READ.0
        | api::AccessMask::SHADER_WRITE.0
        | api::AccessMask::INDIRECT_READ.0
        | api::AccessMask::COLOR_READ.0
        | api::AccessMask::COLOR_WRITE.0
        | api::AccessMask::DEPTH_READ.0
        | api::AccessMask::DEPTH_WRITE.0;
    if mask.0 & !known != 0 && mask != api::AccessMask::ALL {
        return Err(Error::Unsupported("Vulkan transition access"));
    }
    if mask == api::AccessMask::ALL {
        return Ok(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
    }
    let mut result = vk::AccessFlags::empty();
    if mask.0 & api::AccessMask::HOST_READ.0 != 0 {
        result |= vk::AccessFlags::HOST_READ;
    }
    if mask.0 & api::AccessMask::HOST_WRITE.0 != 0 {
        result |= vk::AccessFlags::HOST_WRITE;
    }
    if mask.0 & api::AccessMask::COPY_READ.0 != 0 {
        result |= vk::AccessFlags::TRANSFER_READ;
    }
    if mask.0 & api::AccessMask::COPY_WRITE.0 != 0 {
        result |= vk::AccessFlags::TRANSFER_WRITE;
    }
    if mask.0 & api::AccessMask::VERTEX_READ.0 != 0 {
        result |= vk::AccessFlags::VERTEX_ATTRIBUTE_READ;
    }
    if mask.0 & api::AccessMask::INDEX_READ.0 != 0 {
        result |= vk::AccessFlags::INDEX_READ;
    }
    if mask.0 & api::AccessMask::UNIFORM_READ.0 != 0 {
        result |= vk::AccessFlags::UNIFORM_READ;
    }
    if mask.0 & api::AccessMask::SHADER_READ.0 != 0 {
        result |= vk::AccessFlags::SHADER_READ;
    }
    if mask.0 & api::AccessMask::SHADER_WRITE.0 != 0 {
        result |= vk::AccessFlags::SHADER_WRITE;
    }
    if mask.0 & api::AccessMask::INDIRECT_READ.0 != 0 {
        result |= vk::AccessFlags::INDIRECT_COMMAND_READ;
    }
    if mask.0 & api::AccessMask::COLOR_READ.0 != 0 {
        result |= vk::AccessFlags::COLOR_ATTACHMENT_READ;
    }
    if mask.0 & api::AccessMask::COLOR_WRITE.0 != 0 {
        result |= vk::AccessFlags::COLOR_ATTACHMENT_WRITE;
    }
    if mask.0 & api::AccessMask::DEPTH_READ.0 != 0 {
        result |= vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ;
    }
    if mask.0 & api::AccessMask::DEPTH_WRITE.0 != 0 {
        result |= vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE;
    }
    Ok(result)
}

fn aspects(aspects: api::TextureAspects) -> Result<vk::ImageAspectFlags> {
    if aspects == api::TextureAspects::COLOR {
        Ok(vk::ImageAspectFlags::COLOR)
    } else if aspects == api::TextureAspects::DEPTH {
        Ok(vk::ImageAspectFlags::DEPTH)
    } else {
        Err(Error::Unsupported("Vulkan transition texture aspect"))
    }
}
