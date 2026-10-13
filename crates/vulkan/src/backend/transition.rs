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
        match op {
            api::ResourceTransition::Global { before, after } => global(encoder, *before, *after)?,
            api::ResourceTransition::Buffer(barrier) => buffer(encoder, barrier)?,
            api::ResourceTransition::Texture(barrier) => texture(encoder, barrier)?,
            api::ResourceTransition::Acceleration { .. }
            | api::ResourceTransition::Aliasing { .. } => {
                return Err(Error::Unsupported("Vulkan resource transition"));
            }
        }
    }
    Ok(())
}

fn global(
    encoder: &mut VulkanEncoder,
    before: api::AccessScope,
    after: api::AccessScope,
) -> Result<()> {
    let barrier = vk::MemoryBarrier::default()
        .src_access_mask(accesses(before.accesses)?)
        .dst_access_mask(accesses(after.accesses)?);
    pipeline_barrier(
        encoder,
        stages(before.stages)?,
        stages(after.stages)?,
        &[barrier],
        &[],
        &[],
    );
    Ok(())
}

fn buffer(encoder: &mut VulkanEncoder, barrier: &api::BufferBarrier<'_, Vulkan>) -> Result<()> {
    if barrier.ownership.is_some() {
        return Err(Error::Unsupported("Vulkan queue ownership transition"));
    }
    let buffer = unsafe { barrier.target.buffer.native_ref() };
    let range = &barrier.target.range;
    if range.start >= range.end || range.end > buffer.0.size {
        return Err(Error::InvalidRange);
    }
    validate_buffer_use(buffer, barrier.usage.from)?;
    validate_buffer_use(buffer, barrier.usage.to)?;
    let native = vk::BufferMemoryBarrier::default()
        .src_access_mask(accesses(barrier.usage.from.accesses)?)
        .dst_access_mask(accesses(barrier.usage.to.accesses)?)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .buffer(buffer.0.buffer)
        .offset(range.start)
        .size(range.end - range.start);
    pipeline_barrier(
        encoder,
        stages(barrier.usage.from.stages)?,
        stages(barrier.usage.to.stages)?,
        &[],
        &[native],
        &[],
    );
    encoder.buffers.push(buffer.clone());
    Ok(())
}

fn texture(encoder: &mut VulkanEncoder, barrier: &api::TextureBarrier<'_, Vulkan>) -> Result<()> {
    if barrier.ownership.is_some() {
        return Err(Error::Unsupported("Vulkan queue ownership transition"));
    }
    let (old_layout, new_layout) = barrier.layout.ok_or(Error::Unsupported(
        "Vulkan texture transition requires explicit layouts",
    ))?;
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
    validate_texture_use(texture, old_layout, barrier.usage.from)?;
    validate_texture_use(texture, new_layout, barrier.usage.to)?;
    let native = vk::ImageMemoryBarrier::default()
        .src_access_mask(accesses(barrier.usage.from.accesses)?)
        .dst_access_mask(accesses(barrier.usage.to.accesses)?)
        .old_layout(layout(old_layout))
        .new_layout(layout(new_layout))
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(texture.native())
        .subresource_range(range);
    let source_stages = if old_layout == api::ImageLayout::Undefined
        && barrier.usage.from.stages == api::StageMask::EMPTY
        && barrier.usage.from.accesses == api::AccessMask::EMPTY
    {
        vk::PipelineStageFlags::TOP_OF_PIPE
    } else {
        stages(barrier.usage.from.stages)?
    };
    pipeline_barrier(
        encoder,
        source_stages,
        stages(barrier.usage.to.stages)?,
        &[],
        &[],
        &[native],
    );
    encoder.textures.push(texture.clone());
    Ok(())
}

fn pipeline_barrier(
    encoder: &VulkanEncoder,
    source_stages: vk::PipelineStageFlags,
    destination_stages: vk::PipelineStageFlags,
    memory: &[vk::MemoryBarrier],
    buffers: &[vk::BufferMemoryBarrier],
    images: &[vk::ImageMemoryBarrier],
) {
    unsafe {
        encoder.device.device.cmd_pipeline_barrier(
            encoder.command,
            source_stages,
            destination_stages,
            vk::DependencyFlags::empty(),
            memory,
            buffers,
            images,
        );
    }
}

fn validate_buffer_use(buffer: &VulkanBuffer, uses: api::BufferUses) -> Result<()> {
    let accesses = uses.accesses;
    let usage = buffer.0.usage;
    let required = [
        (api::AccessMask::COPY_READ, api::BufferUsage::COPY_SRC),
        (api::AccessMask::COPY_WRITE, api::BufferUsage::COPY_DST),
        (api::AccessMask::VERTEX_READ, api::BufferUsage::VERTEX),
        (api::AccessMask::INDEX_READ, api::BufferUsage::INDEX),
        (api::AccessMask::UNIFORM_READ, api::BufferUsage::UNIFORM),
        (api::AccessMask::SHADER_READ, api::BufferUsage::STORAGE),
        (api::AccessMask::SHADER_WRITE, api::BufferUsage::STORAGE),
        (api::AccessMask::INDIRECT_READ, api::BufferUsage::INDIRECT),
        (
            api::AccessMask::QUERY_RESOLVE_WRITE,
            api::BufferUsage::QUERY_RESOLVE,
        ),
    ];
    for (access, expected_usage) in required {
        if accesses.contains(access) && !usage.contains(expected_usage) {
            return Err(Error::Unsupported("Vulkan buffer transition usage"));
        }
    }
    require_stage(
        accesses,
        uses.stages,
        api::AccessMask::HOST_READ,
        api::StageMask::HOST,
    )?;
    require_stage(
        accesses,
        uses.stages,
        api::AccessMask::HOST_WRITE,
        api::StageMask::HOST,
    )?;
    require_stage(
        accesses,
        uses.stages,
        api::AccessMask::COPY_READ,
        api::StageMask::COPY,
    )?;
    require_stage(
        accesses,
        uses.stages,
        api::AccessMask::COPY_WRITE,
        api::StageMask::COPY,
    )?;
    require_stage(
        accesses,
        uses.stages,
        api::AccessMask::VERTEX_READ,
        api::StageMask::VERTEX_INPUT,
    )?;
    require_stage(
        accesses,
        uses.stages,
        api::AccessMask::INDEX_READ,
        api::StageMask::VERTEX_INPUT,
    )?;
    require_stage(
        accesses,
        uses.stages,
        api::AccessMask::INDIRECT_READ,
        api::StageMask::INDIRECT,
    )?;
    require_stage(
        accesses,
        uses.stages,
        api::AccessMask::QUERY_RESOLVE_WRITE,
        api::StageMask::COPY,
    )?;
    Ok(())
}

fn validate_texture_use(
    texture: &VulkanTexture,
    layout: api::ImageLayout,
    uses: api::TextureUses,
) -> Result<()> {
    if layout == api::ImageLayout::Undefined && uses.accesses != api::AccessMask::EMPTY {
        return Err(Error::Unsupported("Vulkan Undefined layout access"));
    }
    let usage = texture.desc().usage;
    let accesses = uses.accesses;
    let required = [
        (api::AccessMask::COPY_READ, api::TextureUsage::COPY_SRC),
        (api::AccessMask::COPY_WRITE, api::TextureUsage::COPY_DST),
        (
            api::AccessMask::SHADER_WRITE,
            api::TextureUsage::STORAGE_BINDING,
        ),
        (
            api::AccessMask::COLOR_READ,
            api::TextureUsage::RENDER_ATTACHMENT,
        ),
        (
            api::AccessMask::COLOR_WRITE,
            api::TextureUsage::RENDER_ATTACHMENT,
        ),
        (
            api::AccessMask::DEPTH_READ,
            api::TextureUsage::RENDER_ATTACHMENT,
        ),
        (
            api::AccessMask::DEPTH_WRITE,
            api::TextureUsage::RENDER_ATTACHMENT,
        ),
    ];
    for (access, expected_usage) in required {
        if accesses.contains(access) && !usage.contains(expected_usage) {
            return Err(Error::Unsupported("Vulkan texture transition usage"));
        }
    }
    if accesses.contains(api::AccessMask::SHADER_READ)
        && !usage.contains(api::TextureUsage::TEXTURE_BINDING)
        && !usage.contains(api::TextureUsage::STORAGE_BINDING)
    {
        return Err(Error::Unsupported("Vulkan texture transition usage"));
    }
    match layout {
        api::ImageLayout::CopySrc if !accesses.contains(api::AccessMask::COPY_READ) => {
            return Err(Error::Unsupported("Vulkan CopySrc layout access"));
        }
        api::ImageLayout::CopyDst if !accesses.contains(api::AccessMask::COPY_WRITE) => {
            return Err(Error::Unsupported("Vulkan CopyDst layout access"));
        }
        api::ImageLayout::ShaderReadOnly if !accesses.contains(api::AccessMask::SHADER_READ) => {
            return Err(Error::Unsupported("Vulkan ShaderReadOnly layout access"));
        }
        api::ImageLayout::ColorAttachment
            if !(accesses.contains(api::AccessMask::COLOR_READ)
                || accesses.contains(api::AccessMask::COLOR_WRITE)) =>
        {
            return Err(Error::Unsupported("Vulkan ColorAttachment layout access"));
        }
        api::ImageLayout::DepthStencilAttachment
            if !(accesses.contains(api::AccessMask::DEPTH_READ)
                || accesses.contains(api::AccessMask::DEPTH_WRITE)) =>
        {
            return Err(Error::Unsupported(
                "Vulkan DepthStencilAttachment layout access",
            ));
        }
        _ => {}
    }
    Ok(())
}

fn require_stage(
    accesses: api::AccessMask,
    stages: api::StageMask,
    access: api::AccessMask,
    required_stage: api::StageMask,
) -> Result<()> {
    if accesses.contains(access)
        && !stages.contains(required_stage)
        && stages != api::StageMask::ALL_COMMANDS
    {
        return Err(Error::Unsupported("Vulkan transition access stage"));
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
        | api::AccessMask::DEPTH_WRITE.0
        | api::AccessMask::STENCIL_READ.0
        | api::AccessMask::STENCIL_WRITE.0
        | api::AccessMask::QUERY_RESOLVE_WRITE.0;
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
    if mask.0 & api::AccessMask::STENCIL_READ.0 != 0 {
        result |= vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ;
    }
    if mask.0 & api::AccessMask::STENCIL_WRITE.0 != 0 {
        result |= vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE;
    }
    if mask.0 & api::AccessMask::QUERY_RESOLVE_WRITE.0 != 0 {
        result |= vk::AccessFlags::TRANSFER_WRITE;
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
