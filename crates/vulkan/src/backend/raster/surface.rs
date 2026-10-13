//! WSI attachment extraction and local dynamic-rendering layout setup.
//!
//! Acquired images have a distinct lease lifetime from ordinary textures. This
//! helper exposes only the image data needed by the raster recorder and keeps
//! final present/capture layout selection with the surface/capture path.

use super::*;

pub(super) struct SurfaceAttachment {
    pub(super) view: vk::ImageView,
    pub(super) image: vk::Image,
    pub(super) extent: vk::Extent2D,
    pub(super) format: vk::Format,
}

/// Acquired images form their own WSI boundary. Every raster pass that writes
/// one begins from discarded contents and leaves final present/capture layout
/// selection to `capture::finish` after dynamic rendering has ended.
pub(super) fn surface_attachment(image: &api::SurfaceImage<Vulkan>) -> SurfaceAttachment {
    // SAFETY: the public surface-image lease is valid for the caller's raster
    // description; this helper reads only its private Vulkan attachment data.
    let native = unsafe { image.native_ref() };
    let (view, image, extent, format) = native.attachment();
    SurfaceAttachment {
        view,
        image,
        extent,
        format,
    }
}

pub(super) fn transition_surface_to_color(encoder: &mut VulkanEncoder, image: vk::Image) {
    let range = vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .level_count(1)
        .layer_count(1);
    let to_color = vk::ImageMemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::empty())
        .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
        .old_layout(vk::ImageLayout::UNDEFINED)
        .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(range);
    // SAFETY: an acquired surface image is used only by this recording encoder;
    // its lease is retained until present/capture selects its final layout.
    unsafe {
        encoder.device.device.cmd_pipeline_barrier(
            encoder.command,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[to_color],
        );
    }
}
