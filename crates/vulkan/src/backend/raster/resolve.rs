//! Attachment resolve workaround for nonzero image-view base layers.
//!
//! Some Vulkan drivers fail to resolve a single nonzero array layer inside
//! dynamic rendering. The attachment is stored, then resolved natively after
//! the rendering scope, restoring the explicit attachment layouts afterward.
use super::*;

pub(crate) struct PendingResolve {
    pub(super) source: vk::Image,
    pub(super) destination: vk::Image,
    pub(super) source_mip: u32,
    pub(super) destination_mip: u32,
    pub(super) source_layer: u32,
    pub(super) destination_layer: u32,
    pub(super) area: api::Rect,
}

pub(super) fn finish(e: &VulkanEncoder, resolves: &[PendingResolve]) {
    for r in resolves {
        let range = |mip, layer| {
            vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(mip)
                .level_count(1)
                .base_array_layer(layer)
                .layer_count(1)
        };
        let barrier = |image, range, old, new, from, to| {
            vk::ImageMemoryBarrier::default()
                .src_access_mask(from)
                .dst_access_mask(to)
                .old_layout(old)
                .new_layout(new)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(range)
        };
        let sr = range(r.source_mip, r.source_layer);
        let dr = range(r.destination_mip, r.destination_layer);
        let before = [
            barrier(
                r.source,
                sr,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                vk::AccessFlags::TRANSFER_READ,
            ),
            barrier(
                r.destination,
                dr,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                vk::AccessFlags::TRANSFER_WRITE,
            ),
        ];
        let after = [
            barrier(
                r.source,
                sr,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::AccessFlags::TRANSFER_READ,
                vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            ),
            barrier(
                r.destination,
                dr,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::AccessFlags::TRANSFER_WRITE,
                vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            ),
        ];
        let sub = |mip, layer| {
            vk::ImageSubresourceLayers::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .mip_level(mip)
                .base_array_layer(layer)
                .layer_count(1)
        };
        let offset = vk::Offset3D {
            x: r.area.x as i32,
            y: r.area.y as i32,
            z: 0,
        };
        let region = vk::ImageResolve::default()
            .src_subresource(sub(r.source_mip, r.source_layer))
            .dst_subresource(sub(r.destination_mip, r.destination_layer))
            .src_offset(offset)
            .dst_offset(offset)
            .extent(vk::Extent3D {
                width: r.area.width,
                height: r.area.height,
                depth: 1,
            });
        // SAFETY: the encoder retains the validated attachment images, ranges and render area.
        unsafe {
            e.device.device.cmd_pipeline_barrier(
                e.command,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &before,
            );
            e.device.device.cmd_resolve_image(
                e.command,
                r.source,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                r.destination,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
            );
            e.device.device.cmd_pipeline_barrier(
                e.command,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &after,
            );
        }
    }
}
