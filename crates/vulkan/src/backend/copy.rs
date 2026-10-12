//! Native transfer commands used by the current texture examples.

use super::{VulkanBuffer, VulkanEncoder, VulkanTexture, resource, vk_error};
use ash::vk;
use fluxel_rhi_core::api::{self, Error, Result};

pub(super) fn copy_buffer(
    encoder: &mut VulkanEncoder,
    src: &VulkanBuffer,
    src_offset: u64,
    dst: &VulkanBuffer,
    dst_offset: u64,
    bytes: u64,
) -> Result<()> {
    let src_end = src_offset.checked_add(bytes).ok_or(Error::InvalidRange)?;
    let dst_end = dst_offset.checked_add(bytes).ok_or(Error::InvalidRange)?;
    if bytes == 0
        || !src.0.usage.contains(api::BufferUsage::COPY_SRC)
        || !dst.0.usage.contains(api::BufferUsage::COPY_DST)
        || src_end > src.0.size
        || dst_end > dst.0.size
        || (src.0.buffer == dst.0.buffer && src_offset < dst_end && dst_offset < src_end)
    {
        return Err(Error::InvalidRange);
    }
    // Explicit buffer transitions are not available in this Vulkan slice.
    // Preserve native ordering/visibility for the supported direct copy path.
    before_transfer(encoder);
    unsafe {
        encoder.device.device.cmd_copy_buffer(
            encoder.command,
            src.0.buffer,
            dst.0.buffer,
            &[vk::BufferCopy {
                src_offset,
                dst_offset,
                size: bytes,
            }],
        );
    }
    after_transfer(encoder, dst, dst_offset, bytes);
    encoder.buffers.extend([src.clone(), dst.clone()]);
    Ok(())
}

pub(super) fn write_buffer(
    encoder: &mut VulkanEncoder,
    dst: &VulkanBuffer,
    offset: u64,
    bytes: &[u8],
) -> Result<()> {
    if bytes.is_empty()
        || !dst.0.usage.contains(api::BufferUsage::COPY_DST)
        || offset
            .checked_add(bytes.len() as u64)
            .is_none_or(|end| end > dst.0.size)
    {
        return Err(Error::InvalidRange);
    }
    let upload = resource::create_buffer(
        &encoder.owner,
        &api::BufferDesc {
            label: Some("Vulkan buffer staging".into()),
            size: bytes.len() as u64,
            usage: api::BufferUsage::COPY_SRC,
            memory: api::MemoryClass::Upload,
        },
    )?;
    let mapped = unsafe {
        encoder.device.device.map_memory(
            upload.0.memory,
            0,
            bytes.len() as u64,
            vk::MemoryMapFlags::empty(),
        )
    }
    .map_err(vk_error)?;
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), mapped.cast(), bytes.len());
        encoder.device.device.unmap_memory(upload.0.memory);
    }
    before_transfer(encoder);
    unsafe {
        encoder.device.device.cmd_copy_buffer(
            encoder.command,
            upload.0.buffer,
            dst.0.buffer,
            &[vk::BufferCopy {
                src_offset: 0,
                dst_offset: offset,
                size: bytes.len() as u64,
            }],
        );
    }
    after_transfer(encoder, dst, offset, bytes.len() as u64);
    encoder.uploads.push(upload);
    encoder.buffers.push(dst.clone());
    Ok(())
}

fn before_transfer(encoder: &VulkanEncoder) {
    let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
        .dst_access_mask(vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE);
    unsafe {
        encoder.device.device.cmd_pipeline_barrier(
            encoder.command,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[barrier],
            &[],
            &[],
        );
    }
}

fn after_transfer(encoder: &VulkanEncoder, dst: &VulkanBuffer, offset: u64, bytes: u64) {
    let mut stages = vk::PipelineStageFlags::empty();
    let mut access = vk::AccessFlags::empty();
    if dst.0.usage.contains(api::BufferUsage::INDEX) {
        stages |= vk::PipelineStageFlags::VERTEX_INPUT;
        access |= vk::AccessFlags::INDEX_READ;
    }
    if dst.0.usage.contains(api::BufferUsage::VERTEX) {
        stages |= vk::PipelineStageFlags::VERTEX_INPUT;
        access |= vk::AccessFlags::VERTEX_ATTRIBUTE_READ;
    }
    if dst.0.usage.contains(api::BufferUsage::UNIFORM) {
        stages |= vk::PipelineStageFlags::VERTEX_SHADER
            | vk::PipelineStageFlags::FRAGMENT_SHADER
            | vk::PipelineStageFlags::COMPUTE_SHADER;
        access |= vk::AccessFlags::UNIFORM_READ;
    }
    if dst.0.usage.contains(api::BufferUsage::STORAGE) {
        stages |= vk::PipelineStageFlags::VERTEX_SHADER
            | vk::PipelineStageFlags::FRAGMENT_SHADER
            | vk::PipelineStageFlags::COMPUTE_SHADER;
        access |= vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE;
    }
    if dst.0.usage.contains(api::BufferUsage::COPY_SRC)
        || dst.0.usage.contains(api::BufferUsage::COPY_DST)
    {
        stages |= vk::PipelineStageFlags::TRANSFER;
        access |= vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE;
    }
    let barrier = vk::BufferMemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(access)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .buffer(dst.0.buffer)
        .offset(offset)
        .size(bytes);
    unsafe {
        encoder.device.device.cmd_pipeline_barrier(
            encoder.command,
            vk::PipelineStageFlags::TRANSFER,
            stages,
            vk::DependencyFlags::empty(),
            &[],
            &[barrier],
            &[],
        );
    }
}

pub(super) fn write_texture(
    encoder: &mut VulkanEncoder,
    region: api::TextureRegion<'_, super::Vulkan>,
    layout: api::TexelCopyBufferLayout,
    bytes: &[u8],
) -> Result<()> {
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
    let mapped = unsafe {
        encoder.device.device.map_memory(
            upload.0.memory,
            0,
            bytes.len() as u64,
            vk::MemoryMapFlags::empty(),
        )
    }
    .map_err(vk_error)?;
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), mapped.cast(), bytes.len());
        encoder.device.device.unmap_memory(upload.0.memory);
    }
    let image = texture.native();
    let old_layout = if texture.take_initial_layout(region.mip, region.origin[2])? {
        vk::ImageLayout::UNDEFINED
    } else {
        // `write_texture` requires the caller-visible idle `Common` state.
        // Vulkan represents that state as GENERAL after the first use.
        vk::ImageLayout::GENERAL
    };
    let range = vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_mip_level(region.mip)
        .level_count(1)
        .base_array_layer(region.origin[2])
        .layer_count(1);
    let to_copy = vk::ImageMemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::empty())
        .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .old_layout(old_layout)
        .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(range);
    unsafe {
        encoder.device.device.cmd_pipeline_barrier(
            encoder.command,
            vk::PipelineStageFlags::TOP_OF_PIPE,
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
            .dst_access_mask(vk::AccessFlags::empty())
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(range);
        encoder.device.device.cmd_pipeline_barrier(
            encoder.command,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
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

fn validate_rgba_upload(
    texture: &VulkanTexture,
    region: &api::TextureRegion<'_, super::Vulkan>,
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
