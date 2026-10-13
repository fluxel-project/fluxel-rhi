//! Native Vulkan dynamic rendering and draw lowering.
//!
//! This module records directly into a live primary command buffer.  It owns
//! no render graph and does not infer resource state.  Texture attachments and
//! surface-image leases are retained by the encoder through submission; a
//! surface image is deliberately never converted into a durable TextureView.
//! The state machine is `idle -> raster -> idle`; all draw setters require the
//! middle state and abort closes any native scope it opened. RenderGraph owns
//! cross-pass hazards and layouts, while this module handles only attachment
//! setup that Vulkan requires inside the pass.

use super::*;

mod extras;
mod surface;

pub use extras::RasterExtras;
use surface::{surface_attachment, transition_surface_to_color};

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
                // SAFETY: `AttachmentView::Texture` was constructed from this
                // backend and Core keeps the typed view alive for this call.
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
        let (color_load, clear) = match color.load {
            api::LoadOp::Clear(api::ClearColor::Float(v)) => (
                vk::AttachmentLoadOp::CLEAR,
                vk::ClearValue {
                    color: vk::ClearColorValue { float32: v },
                },
            ),
            api::LoadOp::Load => (vk::AttachmentLoadOp::LOAD, vk::ClearValue::default()),
            api::LoadOp::Discard => (vk::AttachmentLoadOp::DONT_CARE, vk::ClearValue::default()),
            _ => return Err(Error::Unsupported("Vulkan raster load operation")),
        };
        if color_load == vk::AttachmentLoadOp::LOAD && surface.is_some() {
            return Err(Error::Unsupported("Vulkan surface attachment Load"));
        }
        let mut attachment = vk::RenderingAttachmentInfo::default()
            .image_view(color_view)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(color_load)
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
            let Some((load, store)) = depth.depth else {
                return Err(Error::Unsupported("Vulkan depth load operation"));
            };
            let (depth_load, value) = match load {
                api::LoadOp::Clear(value) => (vk::AttachmentLoadOp::CLEAR, value),
                api::LoadOp::Load => (vk::AttachmentLoadOp::LOAD, 0.0),
                api::LoadOp::Discard => (vk::AttachmentLoadOp::DONT_CARE, 0.0),
            };
            let api::AttachmentView::Texture(view) = &depth.view else {
                return Err(Error::Unsupported("Vulkan depth surface attachment"));
            };
            // SAFETY: this typed attachment view belongs to Vulkan and remains
            // borrowed by the pass description while native metadata is read.
            let native = unsafe { view.native_ref() };
            if native.attachment_extent_for_layers(api::TextureAspects::DEPTH, multiview_layers)
                != Some([d.render_area.width, d.render_area.height])
                || native.samples() != color_samples
            {
                return Err(Error::InvalidRange);
            }
            // Discarding the old contents is valid only for Clear/Discard.
            // Load uses the caller's explicit DepthStencilAttachment layout.
            if depth_load != vk::AttachmentLoadOp::LOAD {
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
                // SAFETY: the encoder still owns this recording command buffer;
                // Clear/Discard makes UNDEFINED a valid old layout for this
                // attachment, while the retained view keeps the image alive.
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
            }
            e.attachments.push(native.clone());
            Some(
                vk::RenderingAttachmentInfo::default()
                    .image_view(native.native())
                    .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                    .load_op(depth_load)
                    .store_op(match store {
                        api::StoreOp::Store => vk::AttachmentStoreOp::STORE,
                        api::StoreOp::Discard => vk::AttachmentStoreOp::DONT_CARE,
                    })
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
        // SAFETY: all attachment views above are validated and retained by the
        // encoder. `info` only borrows stack data for this recording call.
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
        // SAFETY: the pipeline matches this pass's multiview state and is
        // retained through submission before its native handle is recorded.
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
        // SAFETY: layout compatibility and dynamic-offset alignment were
        // checked above; retaining the group preserves its descriptor set.
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
        // SAFETY: range and VERTEX usage were validated, and the cloned buffer
        // remains owned by the encoder until the GPU has executed this bind.
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
        // SAFETY: index type alignment, extent and INDEX usage were validated;
        // the encoder retains the buffer for this recorded command.
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
        // SAFETY: this is a live primary raster command buffer. Negative height
        // intentionally maps the public top-left viewport convention to Vulkan.
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
        // SAFETY: a live primary command buffer owns this dynamic state and the
        // nonzero extent was checked before it is lowered to Vulkan.
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
        // SAFETY: raster and pipeline state are active, and checked ranges avoid
        // arithmetic underflow when converted into Vulkan draw counts.
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
        // SAFETY: an index buffer and compatible pipeline are bound; validated
        // ranges are converted directly to the native indexed draw parameters.
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
        // SAFETY: `e.raster` proves this primary command buffer has a matching
        // dynamic-rendering begin which has not yet been ended.
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
                // SAFETY: the encoder owns the still-active query and command
                // buffer; ending it makes abort leave native state balanced.
                unsafe { e.device.device.cmd_end_query(e.command, set.pool(), slot) };
            }
            // SAFETY: the branch is entered only for a begun rendering scope.
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
