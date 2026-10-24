//! Buffer transfer commands and staging uploads.

use super::super::{VulkanBuffer, VulkanEncoder};
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
    let family = if encoder.lane.ordinal == 1 {
        encoder.device.compute_family
    } else {
        encoder.device.graphics_family
    };
    let upload = encoder.device.upload_pool.write(
        &encoder.device,
        family,
        &mut encoder.upload_slices,
        bytes,
        encoder.device.upload_copy_offset_alignment,
        0,
    )?;
    // SAFETY: encoder owns a recording command buffer; upload and destination
    // are retained by it through fence completion and ranges were checked above.
    unsafe {
        encoder.device.device.cmd_copy_buffer(
            encoder.command,
            upload.buffer,
            dst.0.buffer,
            &[vk::BufferCopy {
                src_offset: upload.offset,
                dst_offset: offset,
                size: bytes.len() as u64,
            }],
        );
    }
    // The upload helper owns its temporary CopyDst state. Return to its
    // documented idle boundary without guessing the caller's next usage.
    after_upload(encoder, dst, offset, bytes.len() as u64);
    encoder.buffers.push(dst.clone());
    Ok(())
}

fn after_upload(encoder: &VulkanEncoder, dst: &VulkanBuffer, offset: u64, size: u64) {
    let barrier = vk::BufferMemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .buffer(dst.0.buffer)
        .offset(offset)
        .size(size);
    // SAFETY: write_buffer's documented idle boundary is scoped to the copied
    // destination range. Later GPU dependencies are still explicit Core
    // buffer barriers; no usage tracker is introduced here.
    unsafe {
        encoder.device.device.cmd_pipeline_barrier(
            encoder.command,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::DependencyFlags::empty(),
            &[],
            &[barrier],
            &[],
        );
    }
}
