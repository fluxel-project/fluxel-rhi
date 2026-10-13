//! Completion-gated mapping of Vulkan host-readable buffers.
//!
//! Core exposes a ticket only after the caller has supplied completion proof.
//! For a device-local source the encoder retains a host-visible staging buffer
//! and records the transfer plus host-read barrier.  Mapping borrows no native
//! pointer after unmap: bytes are copied into the portable result first.

use std::{
    future::{Ready, ready},
    ops::Range,
};

use ash::vk;
use fluxel_rhi_core::api::{self, Error, ReadbackData, Result};

use super::{Vulkan, VulkanBuffer, VulkanDevice, VulkanEncoder, resource, vk_error};

pub struct VulkanReadback {
    buffer: VulkanBuffer,
    range: Range<u64>,
}

impl api::ReadbackBackend for Vulkan {
    type NativeReadback = VulkanReadback;
    type ReadbackFuture<'a>
        = Ready<Result<ReadbackData>>
    where
        Self: 'a;

    fn encode_readback(
        encoder: &mut VulkanEncoder,
        source: &VulkanBuffer,
        range: Range<u64>,
    ) -> Result<VulkanReadback> {
        if range.start > range.end || range.end > source.0.size {
            return Err(Error::InvalidRange);
        }
        if !source.0.usage.contains(api::BufferUsage::MAP_READ) {
            if !source.0.usage.contains(api::BufferUsage::COPY_SRC) {
                return Err(Error::Unsupported("Vulkan readback requires COPY_SRC"));
            }
            let size = range.end - range.start;
            if size == 0 {
                return Err(Error::InvalidRange);
            }
            let staging = resource::create_buffer(
                &encoder.owner,
                &api::BufferDesc {
                    label: Some("Vulkan readback staging".into()),
                    size,
                    usage: api::BufferUsage::COPY_DST.union(api::BufferUsage::MAP_READ),
                    memory: api::MemoryClass::Readback,
                },
            )?;
            let ready = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(source.0.buffer)
                .offset(range.start)
                .size(size);
            // SAFETY: encoder owns a recording command buffer; source and
            // staging retain this device and the caller supplied transition
            // contract for source visibility before readback.
            unsafe {
                encoder.device.device.cmd_pipeline_barrier(
                    encoder.command,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[ready],
                    &[],
                );
                encoder.device.device.cmd_copy_buffer(
                    encoder.command,
                    source.0.buffer,
                    staging.0.buffer,
                    &[vk::BufferCopy {
                        src_offset: range.start,
                        dst_offset: 0,
                        size,
                    }],
                );
            }
            let host = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(staging.0.buffer)
                .offset(0)
                .size(size);
            // SAFETY: this records a host-read dependency for the staging
            // buffer in the same still-recording command buffer.
            unsafe {
                encoder.device.device.cmd_pipeline_barrier(
                    encoder.command,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::HOST,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[host],
                    &[],
                );
            }
            encoder.buffers.extend([source.clone(), staging.clone()]);
            return Ok(VulkanReadback {
                buffer: staging,
                range: 0..size,
            });
        }
        encoder.buffers.push(source.clone());
        Ok(VulkanReadback {
            buffer: source.clone(),
            range,
        })
    }

    fn map_readback<'a>(_: &'a VulkanDevice, ticket: VulkanReadback) -> Self::ReadbackFuture<'a> {
        ready(map(ticket))
    }
}

fn map(ticket: VulkanReadback) -> Result<ReadbackData> {
    let size = ticket.range.end - ticket.range.start;
    if size == 0 {
        return Ok(ReadbackData {
            bytes: Vec::new(),
            row_pitch: 0,
            slice_pitch: 0,
        });
    }
    let size_usize = usize::try_from(size).map_err(|_| Error::InvalidRange)?;
    let start = usize::try_from(ticket.range.start).map_err(|_| Error::InvalidRange)?;
    // SAFETY: the readback ticket is consumed only after its Core completion
    // point, and this buffer owns host-visible, coherent allocation memory.
    let mapped = unsafe {
        ticket.buffer.0.device.device.map_memory(
            ticket.buffer.0.memory,
            0,
            ticket.buffer.0.size,
            vk::MemoryMapFlags::empty(),
        )
    }
    .map_err(vk_error)?;
    if mapped.is_null() {
        return Err(Error::Backend("Vulkan readback Map returned null".into()));
    }
    // SAFETY: map_memory returned a non-null range covering `buffer.size`;
    // range validation above proves start..start+size stays inside it.
    let bytes =
        unsafe { std::slice::from_raw_parts(mapped.cast::<u8>().add(start), size_usize).to_vec() };
    // SAFETY: this is the matching unmap for the successful mapping above;
    // `bytes` owns its copy and retains no mapped pointer.
    unsafe {
        ticket
            .buffer
            .0
            .device
            .device
            .unmap_memory(ticket.buffer.0.memory)
    };
    Ok(ReadbackData {
        bytes,
        row_pitch: size,
        slice_pitch: size,
    })
}
