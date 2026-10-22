//! Image transfer, upload, readback, and blit commands.

use super::super::{VulkanBuffer, VulkanEncoder, VulkanTexture, resource, vk_error};
use ash::vk;
use fluxel_rhi_core::api::{self, Error, Result};

pub(crate) fn copy_texture(
    encoder: &mut VulkanEncoder,
    src: api::TextureRegion<'_, super::super::Vulkan>,
    dst: api::TextureRegion<'_, super::super::Vulkan>,
) -> Result<()> {
    // SAFETY: the Core command contract has checked device identity. These
    // borrowed native handles remain retained by the encoder through submit.
    let source = unsafe { src.texture.native_ref() };
    let target = unsafe { dst.texture.native_ref() };
    validate_image_region(source, &src)?;
    validate_image_region(target, &dst)?;
    if !source.desc().usage.contains(api::TextureUsage::COPY_SRC)
        || !target.desc().usage.contains(api::TextureUsage::COPY_DST)
        || source.desc().format != target.desc().format
        || source.desc().samples != target.desc().samples
        || source.native() == target.native()
        || src.extent != dst.extent
        || (source.desc().samples > 1
            && (src.origin[0] != 0
                || src.origin[1] != 0
                || dst.origin[0] != 0
                || dst.origin[1] != 0
                || src.extent != mip_extent(source.desc(), src.mip)
                || dst.extent != mip_extent(target.desc(), dst.mip)))
    {
        return Err(Error::Unsupported(
            "Vulkan texture copy format, usage or overlap",
        ));
    }
    let aspect = image_aspect(src.aspect)?;
    let subresource = |region: &api::TextureRegion<'_, super::super::Vulkan>| {
        vk::ImageSubresourceLayers::default()
            .aspect_mask(aspect)
            .mip_level(region.mip)
            .base_array_layer(region.origin[2])
            .layer_count(1)
    };
    let offset = |region: &api::TextureRegion<'_, super::super::Vulkan>| vk::Offset3D {
        x: region.origin[0] as i32,
        y: region.origin[1] as i32,
        z: 0,
    };
    let region = vk::ImageCopy::default()
        .src_subresource(subresource(&src))
        .src_offset(offset(&src))
        .dst_subresource(subresource(&dst))
        .dst_offset(offset(&dst))
        .extent(vk::Extent3D {
            width: src.extent[0],
            height: src.extent[1],
            depth: 1,
        });
    // SAFETY: region validation bounds every subresource and extent; Core has
    // selected the transfer layouts before this direct command is recorded.
    unsafe {
        encoder.device.device.cmd_copy_image(
            encoder.command,
            source.native(),
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            target.native(),
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[region],
        );
    }
    encoder.textures.extend([source.clone(), target.clone()]);
    Ok(())
}

pub(crate) fn buffer_to_texture(
    encoder: &mut VulkanEncoder,
    src: &VulkanBuffer,
    layout: api::TexelCopyBufferLayout,
    dst: api::TextureRegion<'_, super::super::Vulkan>,
) -> Result<()> {
    // SAFETY: Core has checked the texture belongs to this command device; the
    // encoder retains it after recording.
    let target = unsafe { dst.texture.native_ref() };
    validate_image_region(target, &dst)?;
    if !src.0.usage.contains(api::BufferUsage::COPY_SRC)
        || !target.desc().usage.contains(api::TextureUsage::COPY_DST)
        || !matches!(
            target.desc().format,
            api::TextureFormat::Rgba8Unorm | api::TextureFormat::Bgra8Unorm
        )
        || target.desc().samples != 1
    {
        return Err(Error::Unsupported(
            "Vulkan RGBA8/BGRA8 buffer-to-texture copy",
        ));
    }
    let tight = dst.extent[0].checked_mul(4).ok_or(Error::InvalidRange)?;
    if layout.offset % 4 != 0
        || layout.bytes_per_row < tight
        || layout.bytes_per_row % 4 != 0
        || layout.rows_per_image < dst.extent[1]
    {
        return Err(Error::InvalidRange);
    }
    let end = layout
        .offset
        .checked_add(u64::from(layout.bytes_per_row) * u64::from(dst.extent[1] - 1))
        .and_then(|value| value.checked_add(u64::from(tight)))
        .ok_or(Error::InvalidRange)?;
    if end > src.0.size {
        return Err(Error::InvalidRange);
    }
    let region = vk::BufferImageCopy::default()
        .buffer_offset(layout.offset)
        .buffer_row_length(layout.bytes_per_row / 4)
        .buffer_image_height(layout.rows_per_image)
        .image_subresource(
            vk::ImageSubresourceLayers::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .mip_level(dst.mip)
                .base_array_layer(dst.origin[2])
                .layer_count(1),
        )
        .image_offset(vk::Offset3D {
            x: dst.origin[0] as i32,
            y: dst.origin[1] as i32,
            z: 0,
        })
        .image_extent(vk::Extent3D {
            width: dst.extent[0],
            height: dst.extent[1],
            depth: 1,
        });
    // SAFETY: validation covers buffer range, texel layout, and image region.
    // The caller supplies the required CopyDst layout through Core transitions.
    unsafe {
        encoder.device.device.cmd_copy_buffer_to_image(
            encoder.command,
            src.0.buffer,
            target.native(),
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[region],
        );
    }
    encoder.buffers.push(src.clone());
    encoder.textures.push(target.clone());
    Ok(())
}

fn image_aspect(aspect: api::TextureAspects) -> Result<vk::ImageAspectFlags> {
    if aspect == api::TextureAspects::COLOR {
        Ok(vk::ImageAspectFlags::COLOR)
    } else if aspect == api::TextureAspects::DEPTH {
        Ok(vk::ImageAspectFlags::DEPTH)
    } else {
        Err(Error::Unsupported("Vulkan copy texture aspect"))
    }
}

fn validate_image_region(
    texture: &VulkanTexture,
    region: &api::TextureRegion<'_, super::super::Vulkan>,
) -> Result<()> {
    let desc = texture.desc();
    let expected = if desc.format == api::TextureFormat::Depth32Float {
        api::TextureAspects::DEPTH
    } else {
        api::TextureAspects::COLOR
    };
    if desc.dimension != api::TextureDimension::D2
        || region.aspect != expected
        || region.mip >= desc.mip_levels
        || region.origin[2] >= desc.array_layers
        || region.extent[0] == 0
        || region.extent[1] == 0
        || region.extent[2] != 1
        || region.origin[0]
            .checked_add(region.extent[0])
            .is_none_or(|end| end > (desc.extent[0] >> region.mip).max(1))
        || region.origin[1]
            .checked_add(region.extent[1])
            .is_none_or(|end| end > (desc.extent[1] >> region.mip).max(1))
    {
        return Err(Error::InvalidRange);
    }
    Ok(())
}

pub(crate) fn write_texture(
    encoder: &mut VulkanEncoder,
    region: api::TextureRegion<'_, super::super::Vulkan>,
    layout: api::TexelCopyBufferLayout,
    bytes: &[u8],
) -> Result<()> {
    // SAFETY: Core checked identity and this encoder retains the image until
    // submission completion.
    let texture = unsafe { region.texture.native_ref() };
    validate_rgba_upload(texture, &region, layout, bytes.len())?;
    let upload = resource::create_buffer(
        &encoder.owner,
        &api::BufferDesc {
            label: Some("Vulkan texture staging".into()),
            size: bytes.len() as u64,
            usage: api::BufferUsage::COPY_SRC,
            memory: api::MemoryClass::Upload,
        },
    )?;
    // SAFETY: `create_buffer` selected host-visible coherent upload memory and
    // this exact, fresh allocation has no concurrent GPU use.
    let mapped = unsafe {
        encoder.device.device.map_memory(
            upload.0.memory,
            0,
            bytes.len() as u64,
            vk::MemoryMapFlags::empty(),
        )
    }
    .map_err(vk_error)?;
    // SAFETY: the mapping covers `bytes.len()` bytes at offset zero. The copy
    // completes and memory is unmapped before any submitted GPU command uses it.
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), mapped.cast(), bytes.len());
        encoder.device.device.unmap_memory(upload.0.memory);
    }
    let image = texture.native();
    // Full-subresource uploads replace every texel. UNDEFINED is valid even
    // when the image was used before; it discards old contents without keeping
    // recording-time layout history on the shared texture object.
    let old_layout = vk::ImageLayout::UNDEFINED;
    let range = vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_mip_level(region.mip)
        .level_count(1)
        .base_array_layer(region.origin[2])
        .layer_count(1);
    let to_copy = vk::ImageMemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)
        .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .old_layout(old_layout)
        .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(range);
    // SAFETY: validation proves a full color subresource upload. This owned
    // command buffer retains upload and image handles through fence retirement.
    unsafe {
        encoder.device.device.cmd_pipeline_barrier(
            encoder.command,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[to_copy],
        );
        encoder.device.device.cmd_copy_buffer_to_image(
            encoder.command,
            upload.0.buffer,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[vk::BufferImageCopy::default()
                .buffer_offset(layout.offset)
                .buffer_row_length(layout.bytes_per_row / 4)
                .buffer_image_height(layout.rows_per_image)
                .image_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .mip_level(region.mip)
                        .base_array_layer(region.origin[2])
                        .layer_count(1),
                )
                .image_extent(vk::Extent3D {
                    width: region.extent[0],
                    height: region.extent[1],
                    depth: 1,
                })],
        );
        let common = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(range);
        encoder.device.device.cmd_pipeline_barrier(
            encoder.command,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[common],
        );
    }
    encoder.uploads.push(upload);
    encoder.textures.push(texture.clone());
    Ok(())
}

/// Records a full RGBA8 texture readback into a host-readable transfer buffer.
///
/// The caller owns image state selection: this operation requires the source
/// to have been explicitly transitioned to `CopySrc`.  It only lowers the
/// requested native copy and retains the two native resources until submission
/// completes; it does not add a backend image-state tracker.
pub(crate) fn texture_to_buffer(
    encoder: &mut VulkanEncoder,
    source: api::TextureRegion<'_, super::super::Vulkan>,
    destination: &VulkanBuffer,
    layout: api::TexelCopyBufferLayout,
) -> Result<()> {
    // SAFETY: Core checked identity; command submission retains both the image
    // and destination buffer through the completion point.
    let texture = unsafe { source.texture.native_ref() };
    validate_rgba_readback(texture, &source, destination, layout)?;
    // SAFETY: validated regions and layouts match the transfer command; the
    // source's CopySrc state is an explicit Core prerequisite.
    unsafe {
        encoder.device.device.cmd_copy_image_to_buffer(
            encoder.command,
            texture.native(),
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            destination.0.buffer,
            &[vk::BufferImageCopy::default()
                .buffer_offset(layout.offset)
                .buffer_row_length(layout.bytes_per_row / 4)
                .buffer_image_height(layout.rows_per_image)
                .image_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .mip_level(source.mip)
                        .base_array_layer(source.origin[2])
                        .layer_count(1),
                )
                .image_offset(vk::Offset3D {
                    x: source.origin[0] as i32,
                    y: source.origin[1] as i32,
                    z: 0,
                })
                .image_extent(vk::Extent3D {
                    width: source.extent[0],
                    height: source.extent[1],
                    depth: 1,
                })],
        );
    }
    let barrier = vk::BufferMemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::HOST_READ)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .buffer(destination.0.buffer)
        .offset(layout.offset)
        .size(vk::WHOLE_SIZE);
    // SAFETY: this barrier belongs to the same recording command buffer and
    // makes its preceding transfer visible to host reads after completion.
    unsafe {
        encoder.device.device.cmd_pipeline_barrier(
            encoder.command,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(),
            &[],
            &[barrier],
            &[],
        );
    }
    encoder.textures.push(texture.clone());
    encoder.buffers.push(destination.clone());
    Ok(())
}

/// Records one native filtered image blit.
///
/// The caller must have explicitly transitioned `src` to `CopySrc` and `dst`
/// to `CopyDst`.  Keeping that state selection at the Core call site avoids a
/// backend-local image-state tracker and makes successive mip dependencies
/// visible in the recorded workload.
pub(crate) fn blit(
    encoder: &mut VulkanEncoder,
    src: api::TextureRegion<'_, super::super::Vulkan>,
    dst: api::TextureRegion<'_, super::super::Vulkan>,
    filter: api::FilterMode,
) -> Result<()> {
    // SAFETY: Core checked both borrowed textures against this device, and the
    // encoder retains their native handles until the submission retires.
    let source = unsafe { src.texture.native_ref() };
    let target = unsafe { dst.texture.native_ref() };
    validate_rgba_blit(source, &src, target, &dst, filter)?;

    // SAFETY: DeviceState retains the Vulkan instance and selected physical
    // device for this query; it passes no application-owned pointers.
    let properties = unsafe {
        encoder
            .device
            .instance
            .instance
            .get_physical_device_format_properties(
                encoder.device.physical,
                vk::Format::R8G8B8A8_UNORM,
            )
    }
    .optimal_tiling_features;
    let required = vk::FormatFeatureFlags::BLIT_SRC | vk::FormatFeatureFlags::BLIT_DST;
    if !properties.contains(required)
        || (filter == api::FilterMode::Linear
            && !properties.contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR))
    {
        return Err(Error::Unsupported("Vulkan RGBA8 texture blit format"));
    }

    let layers = |region: &api::TextureRegion<'_, super::super::Vulkan>| {
        vk::ImageSubresourceLayers::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .mip_level(region.mip)
            .base_array_layer(region.origin[2])
            .layer_count(1)
    };
    let end = |region: &api::TextureRegion<'_, super::super::Vulkan>| vk::Offset3D {
        x: region.extent[0] as i32,
        y: region.extent[1] as i32,
        z: 1,
    };
    let region = vk::ImageBlit::default()
        .src_subresource(layers(&src))
        .src_offsets([vk::Offset3D::default(), end(&src)])
        .dst_subresource(layers(&dst))
        .dst_offsets([vk::Offset3D::default(), end(&dst)]);
    // SAFETY: validation proves non-overlapping compatible full subresources;
    // explicit Core transitions established the two transfer layouts.
    unsafe {
        encoder.device.device.cmd_blit_image(
            encoder.command,
            source.native(),
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            target.native(),
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            std::slice::from_ref(&region),
            match filter {
                api::FilterMode::Nearest => vk::Filter::NEAREST,
                api::FilterMode::Linear => vk::Filter::LINEAR,
            },
        );
    }
    encoder.textures.extend([source.clone(), target.clone()]);
    Ok(())
}

fn validate_rgba_blit(
    source: &VulkanTexture,
    src: &api::TextureRegion<'_, super::super::Vulkan>,
    target: &VulkanTexture,
    dst: &api::TextureRegion<'_, super::super::Vulkan>,
    filter: api::FilterMode,
) -> Result<()> {
    if !matches!(filter, api::FilterMode::Nearest | api::FilterMode::Linear)
        || source.desc().format != api::TextureFormat::Rgba8Unorm
        || target.desc().format != api::TextureFormat::Rgba8Unorm
        || source.desc().dimension != api::TextureDimension::D2
        || target.desc().dimension != api::TextureDimension::D2
        || source.desc().array_layers != 1
        || target.desc().array_layers != 1
        || source.desc().samples != 1
        || target.desc().samples != 1
        || !source.desc().usage.contains(api::TextureUsage::COPY_SRC)
        || !target.desc().usage.contains(api::TextureUsage::COPY_DST)
        || src.aspect != api::TextureAspects::COLOR
        || dst.aspect != api::TextureAspects::COLOR
        || src.origin != [0, 0, 0]
        || dst.origin != [0, 0, 0]
        || src.mip >= source.desc().mip_levels
        || dst.mip >= target.desc().mip_levels
        || (source.native() == target.native() && src.mip == dst.mip)
        || src.extent != mip_extent(source.desc(), src.mip)
        || dst.extent != mip_extent(target.desc(), dst.mip)
    {
        return Err(Error::Unsupported("Vulkan full RGBA8 texture blit"));
    }
    Ok(())
}

fn mip_extent(desc: &api::TextureDesc, mip: u32) -> [u32; 3] {
    [
        (desc.extent[0] >> mip).max(1),
        (desc.extent[1] >> mip).max(1),
        1,
    ]
}

fn validate_rgba_upload(
    texture: &VulkanTexture,
    region: &api::TextureRegion<'_, super::super::Vulkan>,
    layout: api::TexelCopyBufferLayout,
    len: usize,
) -> Result<()> {
    let d = texture.desc();
    if region.mip >= d.mip_levels || region.mip >= 32 {
        return Err(Error::InvalidRange);
    }
    let expected = [
        (d.extent[0] >> region.mip).max(1),
        (d.extent[1] >> region.mip).max(1),
        1,
    ];
    if d.format != api::TextureFormat::Rgba8Unorm
        || !d.usage.contains(api::TextureUsage::COPY_DST)
        || region.origin[0] != 0
        || region.origin[1] != 0
        || region.origin[2] >= d.array_layers
        || region.extent != expected
        || region.aspect != api::TextureAspects::COLOR
    {
        return Err(Error::Unsupported("Vulkan full RGBA8 texture upload"));
    }
    let row_bytes = expected[0].checked_mul(4).ok_or(Error::InvalidRange)?;
    if layout.bytes_per_row < row_bytes
        || layout.bytes_per_row % 4 != 0
        || layout.rows_per_image < expected[1]
        || layout.offset % 4 != 0
    {
        return Err(Error::InvalidRange);
    }
    let required = u64::from(layout.offset)
        .checked_add(u64::from(layout.bytes_per_row) * u64::from(expected[1] - 1))
        .and_then(|n| n.checked_add(u64::from(row_bytes)))
        .ok_or(Error::InvalidRange)?;
    if required > len as u64 {
        return Err(Error::InvalidRange);
    }
    Ok(())
}

fn validate_rgba_readback(
    texture: &VulkanTexture,
    source: &api::TextureRegion<'_, super::super::Vulkan>,
    destination: &VulkanBuffer,
    layout: api::TexelCopyBufferLayout,
) -> Result<()> {
    let desc = texture.desc();
    if desc.format != api::TextureFormat::Rgba8Unorm
        || desc.dimension != api::TextureDimension::D2
        || desc.samples != 1
        || !desc.usage.contains(api::TextureUsage::COPY_SRC)
        || !destination.0.usage.contains(api::BufferUsage::COPY_DST)
        || source.aspect != api::TextureAspects::COLOR
        || source.mip >= desc.mip_levels
        || source.origin[0..2] != [0, 0]
        || source.origin[2] >= desc.array_layers
        || source.extent[0..2] != mip_extent(desc, source.mip)[0..2]
        || source.extent[2] != 1
        || layout.offset % 4 != 0
    {
        return Err(Error::Unsupported("Vulkan full RGBA8 texture readback"));
    }
    let row_bytes = source.extent[0].checked_mul(4).ok_or(Error::InvalidRange)?;
    if layout.bytes_per_row < row_bytes
        || layout.bytes_per_row % 4 != 0
        || layout.rows_per_image < source.extent[1]
    {
        return Err(Error::InvalidRange);
    }
    let required = u64::from(layout.offset)
        .checked_add(u64::from(layout.bytes_per_row) * u64::from(source.extent[1] - 1))
        .and_then(|n| n.checked_add(u64::from(row_bytes)))
        .ok_or(Error::InvalidRange)?;
    if required > destination.0.size {
        return Err(Error::InvalidRange);
    }
    Ok(())
}
