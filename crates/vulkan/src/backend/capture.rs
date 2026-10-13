//! Opt-in GPU frame readback for local visual conformance checks.

use super::{VulkanEncoder, encoder::FrameCapture, resource, vk_error};
use ash::vk;
use fluxel_rhi_core::api::{self, Error, Result};
use std::sync::atomic::{AtomicBool, Ordering};

static CAPTURED: AtomicBool = AtomicBool::new(false);

pub(super) fn prepare(encoder: &mut VulkanEncoder, width: u32, height: u32) -> Result<()> {
    let Some(path) = std::env::var_os("FLUXEL_VULKAN_CAPTURE") else {
        return Ok(());
    };
    if CAPTURED.swap(true, Ordering::AcqRel) {
        return Ok(());
    }
    let size = u64::from(width)
        .checked_mul(u64::from(height))
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or(Error::InvalidRange)?;
    let buffer = resource::create_buffer(
        &encoder.owner,
        &api::BufferDesc {
            label: Some("Vulkan visual capture".into()),
            size,
            usage: api::BufferUsage::COPY_DST,
            memory: api::MemoryClass::Upload,
        },
    )
    .inspect_err(|_| CAPTURED.store(false, Ordering::Release))?;
    encoder.capture = Some(FrameCapture {
        buffer,
        width,
        height,
        path: path.into(),
    });
    Ok(())
}

pub(super) fn finish(encoder: &mut VulkanEncoder, image: vk::Image) {
    let range = vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .level_count(1)
        .layer_count(1);
    let copying = encoder.capture.is_some();
    let first = vk::ImageMemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
        .dst_access_mask(if copying {
            vk::AccessFlags::TRANSFER_READ
        } else {
            vk::AccessFlags::empty()
        })
        .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .new_layout(if copying {
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL
        } else {
            vk::ImageLayout::PRESENT_SRC_KHR
        })
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(range);
    // SAFETY: the encoder owns this still-recording command buffer and the
    // active surface image is in the color-attachment layout established by
    // the matching raster pass.
    unsafe {
        encoder.device.device.cmd_pipeline_barrier(
            encoder.command,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            if copying {
                vk::PipelineStageFlags::TRANSFER
            } else {
                vk::PipelineStageFlags::BOTTOM_OF_PIPE
            },
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[first],
        );
    }
    if let Some(capture) = &encoder.capture {
        let region = vk::BufferImageCopy::default()
            .image_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .layer_count(1),
            )
            .image_extent(vk::Extent3D {
                width: capture.width,
                height: capture.height,
                depth: 1,
            });
        // SAFETY: capture retains its destination buffer and the preceding
        // barrier made the image readable as TRANSFER_SRC in this command buffer.
        unsafe {
            encoder.device.device.cmd_copy_image_to_buffer(
                encoder.command,
                image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                capture.buffer.0.buffer,
                &[region],
            );
        }
        let present = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_READ)
            .dst_access_mask(vk::AccessFlags::empty())
            .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(range);
        // SAFETY: the copy above is ordered before this barrier, which restores
        // the present layout required by the same surface image lease.
        unsafe {
            encoder.device.device.cmd_pipeline_barrier(
                encoder.command,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[present],
            );
        }
    }
}

pub(super) fn write_completed(work: &[super::submit::RecordedWork]) -> Result<()> {
    for item in work {
        let Some(capture) = &item.encoder.capture else {
            continue;
        };
        let size = u64::from(capture.width) * u64::from(capture.height) * 4;
        // SAFETY: the fence waiter invokes this only after GPU completion;
        // capture owns a host-visible allocation large enough for `size`.
        let mapped = unsafe {
            item.encoder.device.device.map_memory(
                capture.buffer.0.memory,
                0,
                size,
                vk::MemoryMapFlags::empty(),
            )
        }
        .map_err(vk_error)?;
        // SAFETY: map_memory returned the exact `size` byte range requested.
        let bytes = unsafe { std::slice::from_raw_parts(mapped.cast::<u8>(), size as usize) };
        let result = std::fs::write(&capture.path, bytes)
            .map_err(|e| Error::Backend(format!("Vulkan visual capture: {e}")));
        // SAFETY: this balances the successful map after the synchronous file
        // write has consumed the borrowed byte slice.
        unsafe {
            item.encoder
                .device
                .device
                .unmap_memory(capture.buffer.0.memory)
        };
        result?;
    }
    Ok(())
}
