//! Attachment resolve workaround for nonzero image-view base layers.
//!
//! Some Vulkan drivers fail to resolve a single nonzero array layer inside
//! dynamic rendering. The source attachment is stored and temporarily moved to
//! `GENERAL`; the target remains in the explicit `ResolveDst` (`GENERAL`) state.
use super::*;

pub(crate) struct PendingResolve {
    pub(super) source: vk::Image,
    pub(super) destination: vk::Image,
    pub(super) source_mip: u32,
    pub(super) destination_mip: u32,
    pub(super) source_layer: u32,
    pub(super) destination_layer: u32,
    /// Surface images are temporarily placed in the color-attachment layout
    /// by the WSI pass setup, while ordinary resolve targets already arrive in
    /// their Core-selected `GENERAL` layout.
    pub(super) destination_is_surface: bool,
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
        let before = [barrier(
            r.source,
            sr,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::ImageLayout::GENERAL,
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            vk::AccessFlags::MEMORY_READ,
        )];
        let after = [barrier(
            r.source,
            sr,
            vk::ImageLayout::GENERAL,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::AccessFlags::MEMORY_READ,
            vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
        )];
        let destination_before = [barrier(
            r.destination,
            dr,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::ImageLayout::GENERAL,
            vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            vk::AccessFlags::TRANSFER_WRITE,
        )];
        let destination_after = [barrier(
            r.destination,
            dr,
            vk::ImageLayout::GENERAL,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
        )];
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
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &before,
            );
            // A fallback resolve to a surface cannot use dynamic rendering's
            // resolve attachment.  The surface setup put it in
            // COLOR_ATTACHMENT_OPTIMAL for the common direct-attachment path;
            // make the actual `vkCmdResolveImage` layout and access explicit.
            if r.destination_is_surface {
                e.device.device.cmd_pipeline_barrier(
                    e.command,
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &destination_before,
                );
            }
            e.device.device.cmd_resolve_image(
                e.command,
                r.source,
                vk::ImageLayout::GENERAL,
                r.destination,
                vk::ImageLayout::GENERAL,
                &[region],
            );
            e.device.device.cmd_pipeline_barrier(
                e.command,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &after,
            );
            if r.destination_is_surface {
                // `capture::finish` owns the common surface exit and expects
                // this image in the color-attachment layout.
                e.device.device.cmd_pipeline_barrier(
                    e.command,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &destination_after,
                );
            }
        }
    }
}
