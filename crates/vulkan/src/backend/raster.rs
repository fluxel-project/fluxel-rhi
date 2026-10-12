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

    fn draw_indirect(
        encoder: &mut VulkanEncoder,
        arguments: &VulkanBuffer,
        offset: u64,
        indexed: bool,
    ) -> Result<()> {
        validate_indirect(
            encoder,
            arguments,
            offset,
            1,
            indirect_stride(indexed),
            indexed,
            false,
        )?;
        unsafe {
            if indexed {
                encoder.device.device.cmd_draw_indexed_indirect(
                    encoder.command,
                    arguments.0.buffer,
                    offset,
                    1,
                    indirect_stride(true),
                );
            } else {
                encoder.device.device.cmd_draw_indirect(
                    encoder.command,
                    arguments.0.buffer,
                    offset,
                    1,
                    indirect_stride(false),
                );
            }
        }
        encoder.buffers.push(arguments.clone());
        Ok(())
    }
    fn multi_draw(
        encoder: &mut VulkanEncoder,
        arguments: &VulkanBuffer,
        offset: u64,
        count: u32,
        stride: u32,
        indexed: bool,
    ) -> Result<()> {
        if count == 0 {
            return Ok(());
        }
        validate_indirect(encoder, arguments, offset, count, stride, indexed, true)?;
        unsafe {
            if indexed {
                encoder.device.device.cmd_draw_indexed_indirect(
                    encoder.command,
                    arguments.0.buffer,
                    offset,
                    count,
                    stride,
                );
            } else {
                encoder.device.device.cmd_draw_indirect(
                    encoder.command,
                    arguments.0.buffer,
                    offset,
                    count,
                    stride,
                );
            }
        }
        encoder.buffers.push(arguments.clone());
        Ok(())
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

fn indirect_stride(indexed: bool) -> u32 {
    if indexed { 20 } else { 16 }
}

fn validate_indirect(
    encoder: &VulkanEncoder,
    arguments: &VulkanBuffer,
    offset: u64,
    count: u32,
    stride: u32,
    indexed: bool,
    multi: bool,
) -> Result<()> {
    if !encoder.raster
        || encoder.raster_layout.is_none()
        || (indexed && !encoder.raster_index_bound)
        || !arguments.0.usage.contains(api::BufferUsage::INDIRECT)
        || offset % 4 != 0
    {
        return Err(Error::InvalidState);
    }
    let command_size = indirect_stride(indexed);
    if (count != 0 && (stride < command_size || stride % 4 != 0))
        || (multi
            && count > 1
            && (!encoder.device.multi_draw_indirect
                || count > encoder.device.max_draw_indirect_count))
    {
        return Err(Error::InvalidRange);
    }
    let bytes = if count == 0 {
        offset
    } else {
        u64::from(count - 1)
            .checked_mul(u64::from(stride))
            .and_then(|tail| offset.checked_add(tail))
            .and_then(|last| last.checked_add(u64::from(command_size)))
            .ok_or(Error::InvalidRange)?
    };
    if bytes > arguments.0.size {
        return Err(Error::InvalidRange);
    }
    Ok(())
}

impl api::RasterBackend for Vulkan {
    fn begin_raster(e: &mut VulkanEncoder, d: &api::RasterPassDesc<'_, Self>) -> Result<()> {
        if e.raster || e.compute || d.colors.len() != 1 || d.timestamp_writes.is_some() {
            return Err(Error::Unsupported("Vulkan raster pass configuration"));
        }
        let multiview_layers = match d.multiview_mask {
            None => 1,
            Some(mask) if e.device.multiview && mask != 0 => 32 - mask.leading_zeros(),
            Some(_) => return Err(Error::Unsupported("Vulkan multiview")),
        };
        let color = d
            .colors
            .first()
            .and_then(Option::as_ref)
            .ok_or(Error::Unsupported("Vulkan no-color raster pass"))?;
        if color.resolve_target.is_none() && !matches!(color.store, api::StoreOp::Store) {
            return Err(Error::Unsupported("Vulkan raster resolve/store operation"));
        }
        let (color_view, color_samples, mut surface, color_format, color_attachment) = match &color
            .view
        {
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
                    || native
                        .attachment_extent_for_layers(api::TextureAspects::COLOR, multiview_layers)
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
            if native.attachment_extent_for_layers(api::TextureAspects::DEPTH, multiview_layers)
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
                        .layer_count(multiview_layers),
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
        let secondary_signature = d.secondary_inheritance.as_ref();
        if let Some(signature) = secondary_signature {
            let formats = super::secondary::native_signature(signature)?;
            let expected_color =
                color_format.unwrap_or_else(|| surface.as_ref().expect("surface").format);
            let expected_depth = depth_attachment.as_ref().map(|_| vk::Format::D32_SFLOAT);
            if formats.as_slice() != [expected_color]
                || signature.samples != color_samples
                || signature.multiview_mask != d.multiview_mask
                || signature
                    .depth_stencil_format
                    .and_then(super::resource::format)
                    != expected_depth
            {
                return Err(Error::InvalidState);
            }
        }
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
        if let Some(mask) = d.multiview_mask {
            // A nonzero view mask broadcasts the one draw to its selected
            // attachment layers. Dynamic rendering requires layer_count=1.
            info = info.view_mask(mask);
        }
        if secondary_signature.is_some() {
            info = info.flags(vk::RenderingFlags::CONTENTS_SECONDARY_COMMAND_BUFFERS);
        }
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
        e.raster_index_bound = false;
        e.raster_multiview_mask = d.multiview_mask;
        e.secondary_signature = secondary_signature.cloned();
        Ok(())
    }
    fn set_pipeline(e: &mut VulkanEncoder, p: &VulkanRasterPipeline) -> Result<()> {
        if e.secondary_signature.is_some() || e.raster_multiview_mask != p.multiview_mask() {
            return Err(Error::InvalidState);
        }
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
        if e.secondary_signature.is_some() {
            return Err(Error::InvalidState);
        }
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
        if e.secondary_signature.is_some() {
            return Err(Error::InvalidState);
        }
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
        if e.secondary_signature.is_some() {
            return Err(Error::InvalidState);
        }
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
        e.raster_index_bound = true;
        Ok(())
    }
    fn viewport(e: &mut VulkanEncoder, v: api::Viewport) -> Result<()> {
        if e.secondary_signature.is_some() {
            return Err(Error::InvalidState);
        }
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
        if e.secondary_signature.is_some() {
            return Err(Error::InvalidState);
        }
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
        if e.secondary_signature.is_some() {
            return Err(Error::InvalidState);
        }
        if !e.raster || e.raster_layout.is_none() || v.start > v.end || i.start > i.end {
            return Err(Error::InvalidState);
        }
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
        if e.secondary_signature.is_some() {
            return Err(Error::InvalidState);
        }
        if !e.raster
            || e.raster_layout.is_none()
            || !e.raster_index_bound
            || x.start > x.end
            || i.start > i.end
        {
            return Err(Error::InvalidState);
        }
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
        if !e.raster || e.active_occlusion.is_some() {
            return Err(Error::InvalidState);
        }
        let image = e.color_surface_image.take();
        unsafe { e.device.device.cmd_end_rendering(e.command) };
        if let Some(image) = image {
            capture::finish(e, image);
        }
        e.raster = false;
        e.raster_index_bound = false;
        e.raster_multiview_mask = None;
        e.raster_layout = None;
        e.raster_layout_state = None;
        e.raster_push_constants = None;
        e.secondary_signature = None;
        Ok(())
    }
    fn abort_raster(e: &mut VulkanEncoder) {
        if e.raster {
            if let Some((set, slot)) = e.active_occlusion.take() {
                unsafe { e.device.device.cmd_end_query(e.command, set.pool(), slot) };
            }
            unsafe { e.device.device.cmd_end_rendering(e.command) };
            e.raster = false;
            e.raster_index_bound = false;
            e.raster_multiview_mask = None;
            e.color_surface_image = None;
            e.raster_layout = None;
            e.raster_layout_state = None;
            e.raster_push_constants = None;
            e.secondary_signature = None;
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
