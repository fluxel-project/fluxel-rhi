//! Buffer transfer commands and staging uploads.

use super::super::{VulkanBuffer, VulkanEncoder, resource, vk_error};
use ash::vk;
use fluxel_rhi_core::api::{self, Error, Result};

pub(crate) fn copy_buffer(
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
    // SAFETY: both retained buffers belong to this recording device; Core has
    // selected CopySrc/CopyDst states and the validated ranges fit allocations.
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
    encoder.buffers.extend([src.clone(), dst.clone()]);
    Ok(())
}

pub(crate) fn write_buffer(
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
    // SAFETY: this freshly allocated upload buffer owns host-visible coherent
    // memory and no GPU submission can observe it before the copy is recorded.
    let mapped = unsafe {
        encoder.device.device.map_memory(
            upload.0.memory,
            0,
            bytes.len() as u64,
            vk::MemoryMapFlags::empty(),
        )
    }
    .map_err(vk_error)?;
    // SAFETY: the mapping covers `bytes.len()` from offset zero; copy then
    // unmap completes before this allocation is used by any recorded command.
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), mapped.cast(), bytes.len());
        encoder.device.device.unmap_memory(upload.0.memory);
    }
    before_transfer(encoder);
    // SAFETY: encoder owns a recording command buffer; upload and destination
    // are retained by it through fence completion and ranges were checked above.
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
    // The upload helper owns its temporary CopyDst state. Return to its
    // documented idle boundary without guessing the caller's next usage.
    after_upload(encoder);
    encoder.uploads.push(upload);
    encoder.buffers.push(dst.clone());
    Ok(())
}

fn before_transfer(encoder: &VulkanEncoder) {
    let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
        .dst_access_mask(vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE);
    // SAFETY: this barrier is recorded into the owned command buffer and names
    // no resource handle, only the explicit Core transfer ordering boundary.
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

fn after_upload(encoder: &VulkanEncoder) {
    let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
    // SAFETY: this post-copy barrier remains within the same recording command
    // buffer and does not infer a later resource state.
    unsafe {
        encoder.device.device.cmd_pipeline_barrier(
            encoder.command,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::DependencyFlags::empty(),
            &[barrier],
            &[],
            &[],
        );
    }
}
