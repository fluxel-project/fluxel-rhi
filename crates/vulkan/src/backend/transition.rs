//! Explicit Core resource-transition lowering for Vulkan.
//!
//! Vulkan itself has no `COMMON` image layout.  Fluxel's portable `Common`
//! state is represented by `GENERAL`; callers still supply every dependency.
//!
//! No state is cached here.  Each barrier is emitted from the immutable Core
//! transition selected by RenderGraph, and retained resource clones keep every
//! referenced Vulkan handle live until the enclosing encoder completes.

use super::*;

pub(super) fn record(
    encoder: &mut VulkanEncoder,
    ops: &[api::ResourceTransition<'_, Vulkan>],
) -> Result<()> {
    for op in ops {
        let (from, to) = match op {
            api::ResourceTransition::Global { before, after }
            | api::ResourceTransition::Acceleration { before, after, .. } => {
                (before.stages, after.stages)
            }
            api::ResourceTransition::Buffer(b) => (b.usage.from.stages, b.usage.to.stages),
            api::ResourceTransition::Texture(b) => (b.usage.from.stages, b.usage.to.stages),
            api::ResourceTransition::Aliasing { .. } => {
                (api::StageMask::EMPTY, api::StageMask::EMPTY)
            }
        };
        let used = from | to;
        if used != api::StageMask::ALL_COMMANDS {
            if used.contains(api::StageMask::MESH) && encoder.device.mesh.is_none() {
                return Err(Error::Unsupported("Vulkan mesh transition stage"));
            }
            if used.contains(api::StageMask::TASK)
                && !encoder.device.mesh.as_ref().is_some_and(|m| m.task)
            {
                return Err(Error::Unsupported("Vulkan task transition stage"));
            }
            if used.contains(api::StageMask::ACCELERATION)
                && !encoder.device.ray_support.acceleration
            {
                return Err(Error::Unsupported("Vulkan acceleration transition stage"));
            }
            if used.contains(api::StageMask::RAY) && !encoder.device.ray_support.pipeline {
                return Err(Error::Unsupported("Vulkan ray transition stage"));
            }
        }
        match op {
            api::ResourceTransition::Global { before, after } => global(encoder, *before, *after)?,
            api::ResourceTransition::Buffer(barrier) => buffer(encoder, barrier)?,
            api::ResourceTransition::Texture(barrier) => texture(encoder, barrier)?,
            api::ResourceTransition::Acceleration {
                target,
                before,
                after,
            } => {
                global(encoder, *before, *after)?;
                // SAFETY: Core checked identity and the encoder retains the AS.
                encoder
                    .accelerations
                    .push(unsafe { target.native_ref() }.clone());
            }
            api::ResourceTransition::Aliasing { before, after } => {
                let memory = vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)
                    .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
                pipeline_barrier(
                    encoder,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    &[memory],
                    &[],
                    &[],
                );
                for resource in [before.as_ref(), after.as_ref()].into_iter().flatten() {
                    // SAFETY: Core validated ownership and each clone is retained
                    // through completion of this memory dependency.
                    unsafe {
                        match resource {
                            api::AliasedResource::Buffer(b) => {
                                encoder.buffers.push(b.native_ref().clone())
                            }
                            api::AliasedResource::Texture(t) => {
                                encoder.textures.push(t.native_ref().clone())
                            }
                            api::AliasedResource::Acceleration(a) => {
                                encoder.accelerations.push(a.native_ref().clone())
                            }
                        }
                    }
                }
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
    let transfer = ownership(encoder, barrier.ownership)?;
    // SAFETY: Core checked device identity before CommandBackend::transition;
    // this backend bridge borrows the buffer only while recording its barrier.
    let buffer = unsafe { barrier.target.buffer.native_ref() };
    let range = &barrier.target.range;
    if range.start >= range.end || range.end > buffer.0.size {
        return Err(Error::InvalidRange);
    }
    validate_buffer_use(buffer, barrier.usage.from)?;
    validate_buffer_use(buffer, barrier.usage.to)?;
    let buffer_accesses = |mask: api::AccessMask| -> Result<vk::AccessFlags> {
        let mut result = accesses(mask)?;
        // Vulkan acceleration build inputs are shader reads; scratch memory
        // uses acceleration-structure accesses. A buffer may have both usages.
        if mask.contains(api::AccessMask::ACCEL_READ)
            && buffer.0.usage.contains(api::BufferUsage::ACCEL_BUILD_INPUT)
        {
            result |= vk::AccessFlags::SHADER_READ;
        }
        Ok(result)
    };
    let native = vk::BufferMemoryBarrier::default()
        .src_access_mask(if transfer.is_some_and(|t| !t.release) {
            vk::AccessFlags::empty()
        } else {
            buffer_accesses(barrier.usage.from.accesses)?
        })
        .dst_access_mask(if transfer.is_some_and(|t| t.release) {
            vk::AccessFlags::empty()
        } else {
            buffer_accesses(barrier.usage.to.accesses)?
        })
        .src_queue_family_index(transfer.map_or(vk::QUEUE_FAMILY_IGNORED, |t| t.src))
        .dst_queue_family_index(transfer.map_or(vk::QUEUE_FAMILY_IGNORED, |t| t.dst))
        .buffer(buffer.0.buffer)
        .offset(range.start)
        .size(range.end - range.start);
    pipeline_barrier(
        encoder,
        if transfer.is_some_and(|t| !t.release)
            || (barrier.usage.from.stages == api::StageMask::EMPTY
                && barrier.usage.from.accesses == api::AccessMask::EMPTY)
        {
            vk::PipelineStageFlags::TOP_OF_PIPE
        } else {
            stages(barrier.usage.from.stages)?
        },
        if transfer.is_some_and(|t| t.release) {
            vk::PipelineStageFlags::BOTTOM_OF_PIPE
        } else {
            stages(barrier.usage.to.stages)?
        },
        &[],
        &[native],
        &[],
    );
    encoder.buffers.push(buffer.clone());
    Ok(())
}

fn texture(encoder: &mut VulkanEncoder, barrier: &api::TextureBarrier<'_, Vulkan>) -> Result<()> {
    let transfer = ownership(encoder, barrier.ownership)?;
    // SAFETY: Core has checked device identity. The encoder retains a clone
    // after recording, so this temporary native borrow cannot outlive it.
    let texture = unsafe { barrier.target.texture.native_ref() };
    if barrier.layout.is_none() {
        if transfer.is_some() {
            return Err(Error::InvalidRange);
        }
        let allowed = match texture.desc().format {
            api::TextureFormat::Depth16Unorm
            | api::TextureFormat::Depth24Plus
            | api::TextureFormat::Depth32Float => api::TextureAspects::DEPTH,
            api::TextureFormat::Stencil8 => api::TextureAspects::STENCIL,
            api::TextureFormat::Depth24PlusStencil8 | api::TextureFormat::Depth32FloatStencil8 => {
                api::TextureAspects::DEPTH | api::TextureAspects::STENCIL
            }
            _ => api::TextureAspects::COLOR,
        };
        if barrier.target.aspects.0 == 0 || barrier.target.aspects.0 & !allowed.0 != 0 {
            return Err(Error::InvalidRange);
        }
        if barrier.target.mips.start >= barrier.target.mips.end
            || barrier.target.mips.end > texture.desc().mip_levels
            || barrier.target.layers.start >= barrier.target.layers.end
            || barrier.target.layers.end > texture.desc().array_layers
        {
            return Err(Error::InvalidRange);
        }
        validate_texture_use(texture, api::ImageLayout::General, barrier.usage.from)?;
        validate_texture_use(texture, api::ImageLayout::General, barrier.usage.to)?;
        let native = vk::MemoryBarrier::default()
            .src_access_mask(accesses(barrier.usage.from.accesses)?)
            .dst_access_mask(accesses(barrier.usage.to.accesses)?);
        pipeline_barrier(
            encoder,
            stages(barrier.usage.from.stages)?,
            stages(barrier.usage.to.stages)?,
            &[native],
            &[],
            &[],
        );
        encoder.textures.push(texture.clone());
        return Ok(());
    }
    let (old_layout, new_layout) = barrier.layout.expect("checked above");
    let combined_depth_stencil = matches!(
        texture.desc().format,
        api::TextureFormat::Depth24PlusStencil8 | api::TextureFormat::Depth32FloatStencil8
    );
    let valid_aspects = match texture.desc().format {
        api::TextureFormat::Depth16Unorm
        | api::TextureFormat::Depth24Plus
        | api::TextureFormat::Depth32Float => api::TextureAspects::DEPTH,
        api::TextureFormat::Stencil8 => api::TextureAspects::STENCIL,
        api::TextureFormat::Depth24PlusStencil8 | api::TextureFormat::Depth32FloatStencil8 => {
            api::TextureAspects::DEPTH | api::TextureAspects::STENCIL
        }
        _ => api::TextureAspects::COLOR,
    };
    if barrier.target.aspects.0 & !valid_aspects.0 != 0 {
        return Err(Error::InvalidRange);
    }
    if combined_depth_stencil
        && barrier.target.aspects != valid_aspects
        && !encoder.device.separate_depth_stencil_layouts
        && old_layout != new_layout
    {
        return Err(Error::Unsupported("Vulkan separate depth/stencil layouts"));
    }
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
    let native_range = if texture.desc().dimension == api::TextureDimension::D3 {
        vk::ImageSubresourceRange {
            layer_count: vk::REMAINING_ARRAY_LAYERS,
            ..range
        }
    } else {
        range
    };
    let resolve_scope = resolve_scope(old_layout) || resolve_scope(new_layout);
    let native = vk::ImageMemoryBarrier::default()
        .src_access_mask(if transfer.is_some_and(|t| !t.release) {
            vk::AccessFlags::empty()
        } else if resolve_scope {
            vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE
        } else {
            accesses(barrier.usage.from.accesses)?
        })
        .dst_access_mask(if transfer.is_some_and(|t| t.release) {
            vk::AccessFlags::empty()
        } else if resolve_scope {
            vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE
        } else {
            accesses(barrier.usage.to.accesses)?
        })
        .old_layout(layout(
            if transfer.is_some_and(|t| !t.release && t.src == vk::QUEUE_FAMILY_IGNORED) {
                new_layout
            } else {
                old_layout
            },
        ))
        .new_layout(layout(new_layout))
        .src_queue_family_index(transfer.map_or(vk::QUEUE_FAMILY_IGNORED, |t| t.src))
        .dst_queue_family_index(transfer.map_or(vk::QUEUE_FAMILY_IGNORED, |t| t.dst))
        .image(texture.native())
        .subresource_range(native_range);
    let source_stages = if resolve_scope {
        vk::PipelineStageFlags::ALL_COMMANDS
    } else if transfer.is_some_and(|t| !t.release)
        || (barrier.usage.from.stages == api::StageMask::EMPTY
            && barrier.usage.from.accesses == api::AccessMask::EMPTY)
    {
        vk::PipelineStageFlags::TOP_OF_PIPE
    } else {
        stages(barrier.usage.from.stages)?
    };
    pipeline_barrier(
        encoder,
        source_stages,
        if resolve_scope {
            vk::PipelineStageFlags::ALL_COMMANDS
        } else if transfer.is_some_and(|t| t.release) {
            vk::PipelineStageFlags::BOTTOM_OF_PIPE
        } else {
            stages(barrier.usage.to.stages)?
        },
        &[],
        &[],
        &[native],
    );
    encoder.textures.push(texture.clone());
    Ok(())
}

#[derive(Clone, Copy)]
struct Transfer {
    release: bool,
    src: u32,
    dst: u32,
}

// A matched pair uses the caller's semaphore dependency. Distinct queue
// families require identical old/new layouts on both halves; the same-family
// acquire instead sees the layout already changed by the release.
fn ownership(
    encoder: &VulkanEncoder,
    transfer: Option<api::OwnershipTransfer>,
) -> Result<Option<Transfer>> {
    let Some(transfer) = transfer else {
        return Ok(None);
    };
    let (from, to, release) = match transfer {
        api::OwnershipTransfer::Release { from, to } => (from, to, true),
        api::OwnershipTransfer::Acquire { from, to } => (from, to, false),
    };
    if from.device != encoder.lane.device
        || to.device != encoder.lane.device
        || from.ordinal > 1
        || to.ordinal > 1
        || from == to
        || encoder.device.compute_queue.is_none()
        || (release && encoder.lane != from)
        || (!release && encoder.lane != to)
    {
        return Err(Error::WrongLane);
    }
    let family = |lane: api::LaneId| {
        if lane.ordinal == 0 {
            encoder.device.graphics_family
        } else {
            encoder.device.compute_family
        }
    };
    let (mut src, mut dst) = (family(from), family(to));
    if src == dst {
        src = vk::QUEUE_FAMILY_IGNORED;
        dst = vk::QUEUE_FAMILY_IGNORED;
    }
    Ok(Some(Transfer { release, src, dst }))
}

fn pipeline_barrier(
    encoder: &VulkanEncoder,
    source_stages: vk::PipelineStageFlags,
    destination_stages: vk::PipelineStageFlags,
    memory: &[vk::MemoryBarrier],
    buffers: &[vk::BufferMemoryBarrier],
    images: &[vk::ImageMemoryBarrier],
) {
    // SAFETY: validated Core scopes determine stages/accesses and every native
    // resource handle is retained by the encoder until submission completion.
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
    if accesses.contains(api::AccessMask::ACCEL_READ)
        && !usage.contains(api::BufferUsage::ACCEL_BUILD_INPUT)
        && !usage.contains(api::BufferUsage::ACCEL_SCRATCH)
    {
        return Err(Error::Unsupported("Vulkan acceleration buffer read usage"));
    }
    if accesses.contains(api::AccessMask::ACCEL_WRITE)
        && !usage.contains(api::BufferUsage::ACCEL_SCRATCH)
    {
        return Err(Error::Unsupported("Vulkan acceleration buffer write usage"));
    }
    if accesses.contains(api::AccessMask::RAY_TRACING_SHADER_DATA_READ)
        && !usage.contains(api::BufferUsage::SHADER_TABLE)
    {
        return Err(Error::Unsupported("Vulkan shader table buffer usage"));
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

pub(super) fn validate_texture_use(
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
        (
            api::AccessMask::STENCIL_READ,
            api::TextureUsage::RENDER_ATTACHMENT,
        ),
        (
            api::AccessMask::STENCIL_WRITE,
            api::TextureUsage::RENDER_ATTACHMENT,
        ),
    ];
    for (access, expected_usage) in required {
        let resolve_copy_usage = (layout == api::ImageLayout::ResolveSrc
            && access == api::AccessMask::COLOR_READ
            && usage.contains(api::TextureUsage::COPY_SRC))
            || (layout == api::ImageLayout::ResolveDst
                && access == api::AccessMask::COLOR_WRITE
                && usage.contains(api::TextureUsage::COPY_DST));
        if accesses.contains(access) && !usage.contains(expected_usage) && !resolve_copy_usage {
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
        api::ImageLayout::ResolveSrc
            if !accesses.contains(api::AccessMask::COLOR_READ)
                || !(usage.contains(api::TextureUsage::RENDER_ATTACHMENT)
                    || usage.contains(api::TextureUsage::COPY_SRC)) =>
        {
            return Err(Error::Unsupported("Vulkan ResolveSrc layout access"));
        }
        api::ImageLayout::ResolveDst
            if !accesses.contains(api::AccessMask::COLOR_WRITE)
                || !(usage.contains(api::TextureUsage::RENDER_ATTACHMENT)
                    || usage.contains(api::TextureUsage::COPY_DST)) =>
        {
            return Err(Error::Unsupported("Vulkan ResolveDst layout access"));
        }
        api::ImageLayout::DepthStencilAttachment
            if !(accesses.contains(api::AccessMask::DEPTH_READ)
                || accesses.contains(api::AccessMask::DEPTH_WRITE)
                || accesses.contains(api::AccessMask::STENCIL_READ)
                || accesses.contains(api::AccessMask::STENCIL_WRITE)) =>
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

pub(super) fn layout(layout: api::ImageLayout) -> vk::ImageLayout {
    match layout {
        api::ImageLayout::Undefined => vk::ImageLayout::UNDEFINED,
        api::ImageLayout::Common | api::ImageLayout::General => vk::ImageLayout::GENERAL,
        api::ImageLayout::ShaderReadOnly => vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        api::ImageLayout::CopySrc => vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        api::ImageLayout::CopyDst => vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        api::ImageLayout::ColorAttachment => vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        // GENERAL is legal for both dynamic-rendering attachment resolves and
        // standalone vkCmdResolveImage. It avoids guessing which path follows.
        api::ImageLayout::ResolveSrc | api::ImageLayout::ResolveDst => vk::ImageLayout::GENERAL,
        api::ImageLayout::DepthStencilAttachment => {
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
        }
        api::ImageLayout::DepthReadOnly => vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
        api::ImageLayout::Present => vk::ImageLayout::PRESENT_SRC_KHR,
    }
}

/// Resolve layouts deliberately use `GENERAL`, so their command scope cannot
/// be inferred from an attachment-only access mask.
pub(super) const fn resolve_scope(layout: api::ImageLayout) -> bool {
    matches!(
        layout,
        api::ImageLayout::ResolveSrc | api::ImageLayout::ResolveDst
    )
}

#[cfg(test)]
#[path = "../tests/backend/transition.rs"]
mod tests;

pub(super) fn stages(mask: api::StageMask) -> Result<vk::PipelineStageFlags> {
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
        | api::StageMask::COMPUTE.0
        | api::StageMask::TASK.0
        | api::StageMask::MESH.0
        | api::StageMask::ACCELERATION.0
        | api::StageMask::RAY.0;
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
        result |= vk::PipelineStageFlags::FRAGMENT_SHADER
            | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
            | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS;
    }
    if mask.0 & api::StageMask::COLOR_OUTPUT.0 != 0 {
        result |= vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT;
    }
    if mask.0 & api::StageMask::COMPUTE.0 != 0 {
        result |= vk::PipelineStageFlags::COMPUTE_SHADER;
    }
    if mask.0 & api::StageMask::TASK.0 != 0 {
        result |= vk::PipelineStageFlags::TASK_SHADER_EXT;
    }
    if mask.0 & api::StageMask::MESH.0 != 0 {
        result |= vk::PipelineStageFlags::MESH_SHADER_EXT;
    }
    if mask.0 & api::StageMask::ACCELERATION.0 != 0 {
        result |= vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR;
    }
    if mask.0 & api::StageMask::RAY.0 != 0 {
        result |= vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR;
    }
    if result.is_empty() {
        return Err(Error::InvalidRange);
    }
    Ok(result)
}

pub(super) fn accesses(mask: api::AccessMask) -> Result<vk::AccessFlags> {
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
        | api::AccessMask::ACCEL_READ.0
        | api::AccessMask::ACCEL_WRITE.0
        | api::AccessMask::RAY_TRACING_SHADER_DATA_READ.0
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
    if mask.0 & api::AccessMask::ACCEL_READ.0 != 0 {
        result |= vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR;
    }
    if mask.0 & api::AccessMask::ACCEL_WRITE.0 != 0 {
        result |= vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR;
    }
    if mask.0 & api::AccessMask::RAY_TRACING_SHADER_DATA_READ.0 != 0 {
        result |= vk::AccessFlags::SHADER_READ;
    }
    Ok(result)
}

pub(super) fn aspects(aspects: api::TextureAspects) -> Result<vk::ImageAspectFlags> {
    if aspects == api::TextureAspects::COLOR {
        Ok(vk::ImageAspectFlags::COLOR)
    } else if aspects == api::TextureAspects::DEPTH {
        Ok(vk::ImageAspectFlags::DEPTH)
    } else if aspects == api::TextureAspects::STENCIL {
        Ok(vk::ImageAspectFlags::STENCIL)
    } else if aspects == (api::TextureAspects::DEPTH | api::TextureAspects::STENCIL) {
        Ok(vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL)
    } else {
        Err(Error::Unsupported("Vulkan transition texture aspect"))
    }
}
