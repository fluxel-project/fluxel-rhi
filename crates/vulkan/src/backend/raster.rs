//! Native Vulkan dynamic rendering and draw lowering.

use super::*;

pub struct RasterExtras;

struct SurfaceAttachment {
    view: vk::ImageView,
    image: vk::Image,
    extent: vk::Extent2D,
    format: vk::Format,
}

impl api::RasterExtrasOps<Vulkan> for RasterExtras {
    const AVAILABLE: bool = true;

    fn draw_indirect(_: &mut VulkanEncoder, _: &VulkanBuffer, _: u64, _: bool) -> Result<()> {
        Err(Error::Unsupported("Vulkan indirect draw"))
    }
    fn multi_draw(
        _: &mut VulkanEncoder,
        _: &VulkanBuffer,
        _: u64,
        _: u32,
        _: u32,
        _: bool,
    ) -> Result<()> {
        Err(Error::Unsupported("Vulkan multi draw"))
    }
    fn multi_draw_count(
        _: &mut VulkanEncoder,
        _: &VulkanBuffer,
        _: u64,
        _: &VulkanBuffer,
        _: u64,
        _: u32,
        _: u32,
        _: bool,
    ) -> Result<()> {
        Err(Error::Unsupported("Vulkan multi draw count"))
    }
    fn set_push_constants(
        encoder: &mut VulkanEncoder,
        stages: api::StageMask,
        offset: u32,
        bytes: &[u8],
    ) -> Result<()> {
        let layout = encoder.raster_layout.ok_or(Error::InvalidState)?;
        let push_constants = encoder
            .raster_push_constants
            .as_ref()
            .ok_or(Error::InvalidState)?;
        let native_stages = push_constants.validate(stages, offset, bytes)?;
        unsafe {
            encoder.device.device.cmd_push_constants(
                encoder.command,
                layout,
                native_stages,
                offset,
                bytes,
            );
        }
        Ok(())
    }
}

impl api::RasterBackend for Vulkan {
    fn begin_raster(e: &mut VulkanEncoder, d: &api::RasterPassDesc<'_, Self>) -> Result<()> {
        if e.raster
            || e.compute
            || d.colors.len() != 1
            || d.multiview_mask.is_some()
            || d.timestamp_writes.is_some()
            || d.secondary_inheritance.is_some()
        {
            return Err(Error::Unsupported("Vulkan raster pass configuration"));
        }
        let color = d
            .colors
            .first()
            .and_then(Option::as_ref)
            .ok_or(Error::Unsupported("Vulkan no-color raster pass"))?;
        if color.resolve_target.is_none() && !matches!(color.store, api::StoreOp::Store) {
            return Err(Error::Unsupported("Vulkan raster resolve/store operation"));
        }
        let (color_view, color_samples, mut surface, color_format, color_attachment) =
            match &color.view {
                api::AttachmentView::Surface(image) => {
                    let surface = surface_attachment(image);
                    if d.render_area.x != 0
                        || d.render_area.y != 0
                        || surface.extent
                            != (vk::Extent2D {
                                width: d.render_area.width,
                                height: d.render_area.height,
                            })
                    {
                        return Err(Error::InvalidRange);
                    }
                    (surface.view, 1, Some(surface), None, None)
                }
                api::AttachmentView::Texture(view) => {
                    let native = unsafe { view.native_ref() };
                    if d.render_area.x != 0
                        || d.render_area.y != 0
                        || native.attachment_extent(api::TextureAspects::COLOR)
                            != Some([d.render_area.width, d.render_area.height])
                    {
                        return Err(Error::InvalidRange);
                    }
                    (
                        native.native(),
                        native.samples(),
                        None,
                        Some(
                            super::resource::format(native.format())
                                .ok_or(Error::Unsupported("Vulkan color attachment format"))?,
                        ),
                        Some(native.clone()),
                    )
                }
            };
        let resolve = match color.resolve_target.as_ref() {
            None => {
                if color_samples != 1 {
                    return Err(Error::Unsupported(
                        "Vulkan multisampled color without resolve",
                    ));
                }
                None
            }
            Some(api::AttachmentView::Surface(image)) => {
                if !matches!(color.store, api::StoreOp::Discard) || color_samples == 1 {
                    return Err(Error::Unsupported("Vulkan MSAA resolve source/store"));
                }
                if surface.is_some() {
                    return Err(Error::Unsupported("Vulkan surface resolve source"));
                }
                let resolve = surface_attachment(image);
                if resolve.extent
                    != (vk::Extent2D {
                        width: d.render_area.width,
                        height: d.render_area.height,
                    })
                    || color_format != Some(resolve.format)
                {
                    return Err(Error::InvalidRange);
                }
                let view = resolve.view;
                surface = Some(resolve);
                Some(view)
            }
            Some(api::AttachmentView::Texture(_)) => {
                return Err(Error::Unsupported("Vulkan texture resolve target"));
            }
        };
        let clear = match color.load {
            api::LoadOp::Clear(api::ClearColor::Float(v)) => vk::ClearValue {
                color: vk::ClearColorValue { float32: v },
            },
            _ => return Err(Error::Unsupported("Vulkan raster load operation")),
        };
        let mut attachment = vk::RenderingAttachmentInfo::default()
            .image_view(color_view)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(if resolve.is_some() {
                vk::AttachmentStoreOp::DONT_CARE
            } else {
                vk::AttachmentStoreOp::STORE
            })
            .clear_value(clear);
        if let Some(resolve_view) = resolve {
            attachment = attachment
                .resolve_mode(vk::ResolveModeFlags::AVERAGE)
                .resolve_image_view(resolve_view)
                .resolve_image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        }
        let depth_attachment = if let Some(depth) = &d.depth_stencil {
            if depth.stencil.is_some() || depth.depth_read_only || depth.stencil_read_only {
                return Err(Error::Unsupported("Vulkan depth/stencil pass state"));
            }
            let Some((api::LoadOp::Clear(value), store)) = depth.depth else {
                return Err(Error::Unsupported("Vulkan depth load operation"));
            };
            if !matches!(store, api::StoreOp::Discard) {
                return Err(Error::Unsupported("Vulkan depth store operation"));
            }
            let api::AttachmentView::Texture(view) = &depth.view else {
                return Err(Error::Unsupported("Vulkan depth surface attachment"));
            };
            let native = unsafe { view.native_ref() };
            if native.attachment_extent(api::TextureAspects::DEPTH)
                != Some([d.render_area.width, d.render_area.height])
                || native.samples() != color_samples
            {
                return Err(Error::InvalidRange);
            }
            // 07 clears then discards its depth attachments every pass.  The
            // old depth contents therefore carry no meaning, so Vulkan may
            // discard them with UNDEFINED instead of tracking a cross-frame
            // layout. Load/Store depth attachments need an explicit state
            // contract before this backend can support them.
            let to_depth = vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(native.image())
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::DEPTH)
                        .level_count(1)
                        .layer_count(1),
                );
            unsafe {
                e.device.device.cmd_pipeline_barrier(
                    e.command,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[to_depth],
                );
            }
            e.attachments.push(native.clone());
            Some(
                vk::RenderingAttachmentInfo::default()
                    .image_view(native.native())
                    .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::DONT_CARE)
                    .clear_value(vk::ClearValue {
                        depth_stencil: vk::ClearDepthStencilValue {
                            depth: value,
                            stencil: 0,
                        },
                    }),
            )
        } else {
            None
        };
        let mut info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D {
                    x: d.render_area.x as i32,
                    y: d.render_area.y as i32,
                },
                extent: vk::Extent2D {
                    width: d.render_area.width,
                    height: d.render_area.height,
                },
            })
            .layer_count(1)
            .color_attachments(std::slice::from_ref(&attachment));
        if let Some(depth) = &depth_attachment {
            info = info.depth_attachment(depth);
        }
        if let Some(color_attachment) = color_attachment {
            e.attachments.push(color_attachment);
        }
        if let Some(surface) = surface.as_ref() {
            transition_surface_to_color(e, surface.image);
            capture::prepare(e, d.render_area.width, d.render_area.height)?;
        }
        unsafe { e.device.device.cmd_begin_rendering(e.command, &info) };
        e.color_surface_image = surface.map(|surface| surface.image);
        e.raster = true;
        Ok(())
    }
    fn set_pipeline(e: &mut VulkanEncoder, p: &VulkanRasterPipeline) -> Result<()> {
        unsafe {
            e.device.device.cmd_bind_pipeline(
                e.command,
                vk::PipelineBindPoint::GRAPHICS,
                p.native(),
            )
        };
        e.raster_layout = Some(p.layout());
        e.raster_layout_state = Some(p.layout_state());
        e.raster_push_constants = Some(p.push_constants());
        e.pipelines.push(p.clone());
        Ok(())
    }
    fn set_bind_group(
        e: &mut VulkanEncoder,
        index: u32,
        group: &VulkanBindGroup,
        offsets: &[u32],
    ) -> Result<()> {
        group.validate_dynamic_offsets(offsets)?;
        let layout = e.raster_layout.ok_or(Error::InvalidState)?;
        e.raster_layout_state
            .as_ref()
            .ok_or(Error::InvalidState)?
            .validate_group(index, group)?;
        unsafe {
            e.device.device.cmd_bind_descriptor_sets(
                e.command,
                vk::PipelineBindPoint::GRAPHICS,
                layout,
                index,
                &[group.native()],
                offsets,
            );
        }
        e.groups.push(group.clone());
        Ok(())
    }
    fn set_vertex_buffer(
        e: &mut VulkanEncoder,
        slot: u32,
        b: &VulkanBuffer,
        offset: u64,
        size: u64,
    ) -> Result<()> {
        if size == 0
            || !b.0.usage.contains(api::BufferUsage::VERTEX)
            || offset.checked_add(size).is_none_or(|end| end > b.0.size)
        {
            return Err(Error::InvalidRange);
        }
        unsafe {
            e.device
                .device
                .cmd_bind_vertex_buffers(e.command, slot, &[b.0.buffer], &[offset])
        };
        e.buffers.push(b.clone());
        Ok(())
    }
    fn set_index_buffer(
        e: &mut VulkanEncoder,
        b: &VulkanBuffer,
        offset: u64,
        size: u64,
        f: api::IndexFormat,
    ) -> Result<()> {
        let stride = if matches!(f, api::IndexFormat::Uint16) {
            2
        } else {
            4
        };
        if size == 0
            || offset % stride != 0
            || size % stride != 0
            || !b.0.usage.contains(api::BufferUsage::INDEX)
            || offset.checked_add(size).is_none_or(|end| end > b.0.size)
        {
            return Err(Error::InvalidRange);
        }
        unsafe {
            e.device.device.cmd_bind_index_buffer(
                e.command,
                b.0.buffer,
                offset,
                if matches!(f, api::IndexFormat::Uint16) {
                    vk::IndexType::UINT16
                } else {
                    vk::IndexType::UINT32
                },
            )
        };
        e.buffers.push(b.clone());
        Ok(())
    }
    fn viewport(e: &mut VulkanEncoder, v: api::Viewport) -> Result<()> {
        unsafe {
            e.device.device.cmd_set_viewport(
                e.command,
                0,
                &[vk::Viewport {
                    x: v.x,
                    y: v.y + v.height,
                    width: v.width,
                    height: -v.height,
                    min_depth: v.min_depth,
                    max_depth: v.max_depth,
                }],
            )
        };
        Ok(())
    }
    fn scissor(e: &mut VulkanEncoder, r: api::Rect) -> Result<()> {
        unsafe {
            e.device.device.cmd_set_scissor(
                e.command,
                0,
                &[vk::Rect2D {
                    offset: vk::Offset2D {
                        x: r.x as i32,
                        y: r.y as i32,
                    },
                    extent: vk::Extent2D {
                        width: r.width,
                        height: r.height,
                    },
                }],
            )
        };
        Ok(())
    }
    fn blend_constant(_: &mut VulkanEncoder, _: [f32; 4]) -> Result<()> {
        Err(Error::Unsupported("Vulkan dynamic blend constant"))
    }
    fn stencil_reference(_: &mut VulkanEncoder, _: u32) -> Result<()> {
        Err(Error::Unsupported("Vulkan dynamic stencil reference"))
    }
    fn draw(e: &mut VulkanEncoder, v: std::ops::Range<u32>, i: std::ops::Range<u32>) -> Result<()> {
        unsafe {
            e.device.device.cmd_draw(
                e.command,
                v.end - v.start,
                i.end - i.start,
                v.start,
                i.start,
            )
        };
        Ok(())
    }
    fn draw_indexed(
        e: &mut VulkanEncoder,
        x: std::ops::Range<u32>,
        base: i32,
        i: std::ops::Range<u32>,
    ) -> Result<()> {
        unsafe {
            e.device.device.cmd_draw_indexed(
                e.command,
                x.end - x.start,
                i.end - i.start,
                x.start,
                base,
                i.start,
            )
        };
        Ok(())
    }
    fn end_raster(e: &mut VulkanEncoder) -> Result<()> {
        if !e.raster {
            return Err(Error::InvalidState);
        }
        let image = e.color_surface_image.take();
        unsafe { e.device.device.cmd_end_rendering(e.command) };
        if let Some(image) = image {
            capture::finish(e, image);
        }
        e.raster = false;
        e.raster_layout = None;
        e.raster_layout_state = None;
        e.raster_push_constants = None;
        Ok(())
    }
    fn abort_raster(e: &mut VulkanEncoder) {
        if e.raster {
            unsafe { e.device.device.cmd_end_rendering(e.command) };
            e.raster = false;
            e.color_surface_image = None;
            e.raster_layout = None;
            e.raster_layout_state = None;
            e.raster_push_constants = None;
        }
    }
}

/// Acquired images form their own WSI boundary. Every raster pass that writes
/// one begins from discarded contents and leaves final present/capture layout
/// selection to `capture::finish` after dynamic rendering has ended.
fn surface_attachment(image: &api::SurfaceImage<Vulkan>) -> SurfaceAttachment {
    let native = unsafe { image.native_ref() };
    let (view, image, extent, format) = native.attachment();
    SurfaceAttachment {
        view,
        image,
        extent,
        format,
    }
}

fn transition_surface_to_color(encoder: &mut VulkanEncoder, image: vk::Image) {
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
