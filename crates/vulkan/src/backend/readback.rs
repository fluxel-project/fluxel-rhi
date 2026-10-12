//! Completion-gated mapping of Vulkan host-readable buffers.

use std::{
    future::{Ready, ready},
    ops::Range,
};

use ash::vk;
use fluxel_rhi_core::api::{self, Error, ReadbackData, Result};

use super::{Vulkan, VulkanBuffer, VulkanDevice, VulkanEncoder, vk_error};

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
            return Err(Error::Unsupported(
                "Vulkan readback requires MAP_READ buffer",
            ));
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
    let bytes =
        unsafe { std::slice::from_raw_parts(mapped.cast::<u8>().add(start), size_usize).to_vec() };
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
