//! Dynamic-rendering attachment validation and pass setup.
use super::*;

pub(super) fn begin_raster(
    e: &mut VulkanEncoder,
    d: &api::RasterPassDesc<'_, Vulkan>,
) -> Result<()> {
    if e.raster || e.compute {
        return Err(Error::InvalidState);
    }
    // SAFETY: the encoder retains the instance and selected physical device.
    let max_colors = unsafe {
        e.device
            .instance
            .instance
            .get_physical_device_properties(e.device.physical)
    }
    .limits
    .max_color_attachments;
    if d.colors.len() > max_colors as usize {
        return Err(Error::Unsupported("Vulkan maxColorAttachments"));
    }
    let layers = match d.multiview_mask {
        None => 1,
        Some(mask)
            if e.device.multiview
                && mask != 0
                && (32 - mask.leading_zeros()) <= e.device.max_multiview_views =>
        {
            32 - mask.leading_zeros()
        }
        Some(_) => return Err(Error::Unsupported("Vulkan multiview")),
    };
    let extent = vk::Extent2D {
        width: d.render_area.width,
        height: d.render_area.height,
    };
    let mut surfaces = Vec::new();
    let mut formats = Vec::with_capacity(d.colors.len());
    let mut attachments = Vec::with_capacity(d.colors.len());
    let mut pending_resolves = Vec::new();
    let mut samples = None;
    for color in d.colors {
        let Some(color) = color else {
            formats.push(vk::Format::UNDEFINED);
            attachments.push(vk::RenderingAttachmentInfo::default());
            continue;
        };
        let (view, format, count) =
            attachment_view(e, &color.view, layers, d.render_area, &mut surfaces)?;
        if samples.is_some_and(|s| s != count) {
            return Err(Error::InvalidRange);
        }
        samples = Some(count);
        formats.push(format);
        let (load, clear) = match color.load {
            api::LoadOp::Load => (vk::AttachmentLoadOp::LOAD, vk::ClearValue::default()),
            api::LoadOp::Discard => (vk::AttachmentLoadOp::DONT_CARE, vk::ClearValue::default()),
            api::LoadOp::Clear(c) => (
                vk::AttachmentLoadOp::CLEAR,
                vk::ClearValue {
                    color: match c {
                        api::ClearColor::Float(v) => vk::ClearColorValue { float32: v },
                        api::ClearColor::Uint(v) => vk::ClearColorValue { uint32: v },
                        api::ClearColor::Sint(v) => vk::ClearColorValue { int32: v },
                    },
                },
            ),
        };
        let mut attachment = vk::RenderingAttachmentInfo::default()
            .image_view(view)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(load)
            .store_op(store_op(color.store))
            .clear_value(clear);
        if let Some(resolve) = &color.resolve_target {
            let (resolve_view, resolve_format, resolve_samples) =
                attachment_view(e, resolve, layers, d.render_area, &mut surfaces)?;
            if count == 1 || resolve_samples != 1 || format != resolve_format {
                return Err(Error::InvalidRange);
            }
            let fallback = if let api::AttachmentView::Texture(source) = &color.view {
                // SAFETY: attachment_view validated and retained this typed view.
                let source = unsafe { source.native_ref() };
                let source_layer = source.layer_range().start;
                let destination = match resolve {
                    api::AttachmentView::Texture(view) => {
                        // SAFETY: attachment_view validated and retained this typed view.
                        let view = unsafe { view.native_ref() };
                        Some((
                            view.image(),
                            view.mip_range().start,
                            view.layer_range().start,
                        ))
                    }
                    api::AttachmentView::Surface(image) => {
                        let surface = surface_attachment(image);
                        surface.transfer_dst.then_some((surface.image, 0, 0))
                    }
                };
                if let Some((destination, destination_mip, destination_layer)) = destination {
                    if source_layer != 0 || destination_layer != 0 {
                        let mask = d.multiview_mask.unwrap_or(1);
                        for index in 0..32 {
                            if mask & (1 << index) != 0 {
                                pending_resolves.push(resolve::PendingResolve {
                                    source: source.image(),
                                    destination,
                                    source_mip: source.mip_range().start,
                                    destination_mip,
                                    source_layer: source_layer + index,
                                    destination_layer: destination_layer + index,
                                    area: d.render_area,
                                });
                            }
                        }
                        true
                    } else {
                        false
                    }
                } else {
                    false
                }
            } else {
                false
            };
            if fallback {
                // The source must survive until the immediately following native resolve.
                attachment = attachment.store_op(vk::AttachmentStoreOp::STORE);
            } else {
                attachment = attachment
                    .resolve_mode(resolve_mode(format))
                    .resolve_image_view(resolve_view)
                    .resolve_image_layout(vk::ImageLayout::GENERAL);
            }
        }
        attachments.push(attachment);
    }
    let mut depth_format = None;
    let mut depth_attachment = None;
    let mut stencil_attachment = None;
    if let Some(depth) = &d.depth_stencil {
        let api::AttachmentView::Texture(view) = &depth.view else {
            return Err(Error::InvalidRange);
        };
        // SAFETY: Core checked the typed view's device identity.
        let native = unsafe { view.native_ref() };
        let aspects = native.aspects();
        let has_depth = aspects.0 & api::TextureAspects::DEPTH.0 != 0;
        let has_stencil = aspects.0 & api::TextureAspects::STENCIL.0 != 0;
        if (!has_depth && depth.depth.is_some())
            || (!has_stencil && depth.stencil.is_some())
            || (depth.depth_read_only && matches!(depth.depth, Some((api::LoadOp::Clear(_), _))))
            || (depth.stencil_read_only
                && matches!(depth.stencil, Some((api::LoadOp::Clear(_), _))))
        {
            return Err(Error::InvalidState);
        }
        if !native
            .attachment_extent_for_layers(aspects, layers)
            .is_some_and(|size| contains_area(size, d.render_area))
            || samples.is_some_and(|s| s != native.samples())
        {
            return Err(Error::InvalidRange);
        }
        samples = Some(native.samples());
        depth_format = Some(
            super::resource::native_format(&e.device, native.format())
                .ok_or(Error::Unsupported("Vulkan depth format"))?,
        );
        if has_depth
            && has_stencil
            && depth.depth_read_only != depth.stencil_read_only
            && !e.device.separate_depth_stencil_layouts
        {
            return Err(Error::Unsupported("Vulkan separate depth/stencil layouts"));
        }
        let layout = match (
            has_depth && depth.depth_read_only,
            has_stencil && depth.stencil_read_only,
        ) {
            (true, true) => vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
            (true, false) if !has_stencil => vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
            (false, true) if !has_depth => vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
            (true, false) => vk::ImageLayout::DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL,
            (false, true) => vk::ImageLayout::DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL,
            _ => vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        };
        if has_depth {
            let (load, store) = depth
                .depth
                .unwrap_or((api::LoadOp::Load, api::StoreOp::Store));
            let (op, value) = match load {
                api::LoadOp::Load => (vk::AttachmentLoadOp::LOAD, 0.0),
                api::LoadOp::Discard => (vk::AttachmentLoadOp::DONT_CARE, 0.0),
                api::LoadOp::Clear(v) => (vk::AttachmentLoadOp::CLEAR, v),
            };
            depth_attachment = Some(
                vk::RenderingAttachmentInfo::default()
                    .image_view(native.native())
                    .image_layout(layout)
                    .load_op(op)
                    .store_op(store_op(store))
                    .clear_value(vk::ClearValue {
                        depth_stencil: vk::ClearDepthStencilValue {
                            depth: value,
                            stencil: 0,
                        },
                    }),
            );
        }
        if has_stencil {
            let (load, store) = depth
                .stencil
                .unwrap_or((api::LoadOp::Load, api::StoreOp::Store));
            let (op, value) = match load {
                api::LoadOp::Load => (vk::AttachmentLoadOp::LOAD, 0),
                api::LoadOp::Discard => (vk::AttachmentLoadOp::DONT_CARE, 0),
                api::LoadOp::Clear(v) => (vk::AttachmentLoadOp::CLEAR, v),
            };
            stencil_attachment = Some(
                vk::RenderingAttachmentInfo::default()
                    .image_view(native.native())
                    .image_layout(layout)
                    .load_op(op)
                    .store_op(store_op(store))
                    .clear_value(vk::ClearValue {
                        depth_stencil: vk::ClearDepthStencilValue {
                            depth: 0.0,
                            stencil: value,
                        },
                    }),
            );
        }
        e.attachments.push(native.clone());
    }
    let samples = samples.unwrap_or(0);
    let secondary_signature = d.secondary_inheritance.as_ref();
    if let Some(signature) = secondary_signature {
        if super::secondary::native_signature(&e.device, signature)? != formats
            || (samples != 0 && signature.samples != samples)
            || signature.multiview_mask != d.multiview_mask
            || signature
                .depth_stencil_format
                .and_then(|f| super::resource::native_format(&e.device, f))
                != depth_format
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
            extent,
        })
        .layer_count(1)
        .color_attachments(&attachments)
        .view_mask(d.multiview_mask.unwrap_or(0));
    if secondary_signature.is_some() {
        info = info.flags(vk::RenderingFlags::CONTENTS_SECONDARY_COMMAND_BUFFERS);
    }
    if let Some(depth) = &depth_attachment {
        info = info.depth_attachment(depth);
    }
    if let Some(stencil) = &stencil_attachment {
        info = info.stencil_attachment(stencil);
    }
    if surfaces.len() > 1 {
        return Err(Error::InvalidRange);
    }
    if let Some(surface) = surfaces.first() {
        let load = d.colors.iter().flatten().any(|color| {
            matches!(color.view, api::AttachmentView::Surface(_))
                && matches!(color.load, api::LoadOp::Load)
        });
        transition_surface_to_color(e, surface.image, load);
        capture::prepare(e, extent.width, extent.height)?;
    }
    super::query::begin_pass_timestamps(e, d.timestamp_writes.as_ref())?;
    // SAFETY: attachment handles are retained by the encoder and the borrowed arrays live for this call.
    unsafe {
        e.device
            .device
            .cmd_set_blend_constants(e.command, &[0.0; 4]);
        e.device.device.cmd_set_stencil_reference(
            e.command,
            vk::StencilFaceFlags::FRONT_AND_BACK,
            0,
        );
        e.device.device.cmd_begin_rendering(e.command, &info);
    };
    e.raster_signature = Some((formats, depth_format, samples));
    e.pending_resolves = pending_resolves;
    e.color_surface_image = surfaces.first().map(|surface| surface.image);
    e.raster_depth_read_only = d.depth_stencil.as_ref().is_some_and(|d| d.depth_read_only);
    e.raster_stencil_read_only = d
        .depth_stencil
        .as_ref()
        .is_some_and(|d| d.stencil_read_only);
    e.raster = true;
    e.raster_index_bound = false;
    e.raster_multiview_mask = d.multiview_mask;
    e.secondary_signature = secondary_signature.cloned();
    Ok(())
}

fn store_op(store: api::StoreOp) -> vk::AttachmentStoreOp {
    match store {
        api::StoreOp::Store => vk::AttachmentStoreOp::STORE,
        api::StoreOp::Discard => vk::AttachmentStoreOp::DONT_CARE,
    }
}

fn attachment_view(
    e: &mut VulkanEncoder,
    view: &api::AttachmentView<'_, Vulkan>,
    layers: u32,
    area: api::Rect,
    surfaces: &mut Vec<surface::SurfaceAttachment>,
) -> Result<(vk::ImageView, vk::Format, u32)> {
    match view {
        api::AttachmentView::Texture(view) => {
            // SAFETY: the caller's attachment view has been checked against the device by Core.
            let native = unsafe { view.native_ref() };
            if !native
                .attachment_extent_for_layers(api::TextureAspects::COLOR, layers)
                .is_some_and(|size| contains_area(size, area))
            {
                return Err(Error::InvalidRange);
            }
            let format = super::resource::native_format(&e.device, native.format())
                .ok_or(Error::Unsupported("Vulkan color attachment format"))?;
            e.attachments.push(native.clone());
            Ok((native.native(), format, native.samples()))
        }
        api::AttachmentView::Surface(image) => {
            let surface = surface_attachment(image);
            if layers != 1 || !contains_area([surface.extent.width, surface.extent.height], area) {
                return Err(Error::InvalidRange);
            }
            let result = (surface.view, surface.format, 1);
            surfaces.push(surface);
            Ok(result)
        }
    }
}

fn resolve_mode(format: vk::Format) -> vk::ResolveModeFlags {
    match format {
        vk::Format::R8_UINT
        | vk::Format::R8_SINT
        | vk::Format::R8G8_UINT
        | vk::Format::R8G8_SINT
        | vk::Format::R8G8B8A8_UINT
        | vk::Format::R8G8B8A8_SINT
        | vk::Format::R16_UINT
        | vk::Format::R16_SINT
        | vk::Format::R16G16_UINT
        | vk::Format::R16G16_SINT
        | vk::Format::R16G16B16A16_UINT
        | vk::Format::R16G16B16A16_SINT
        | vk::Format::R32_UINT
        | vk::Format::R32_SINT
        | vk::Format::R32G32_UINT
        | vk::Format::R32G32_SINT
        | vk::Format::R32G32B32A32_UINT
        | vk::Format::R32G32B32A32_SINT
        | vk::Format::A2B10G10R10_UINT_PACK32 => vk::ResolveModeFlags::SAMPLE_ZERO,
        _ => vk::ResolveModeFlags::AVERAGE,
    }
}

fn contains_area(size: [u32; 2], area: api::Rect) -> bool {
    area.width != 0
        && area.height != 0
        && area.x.checked_add(area.width).is_some_and(|v| v <= size[0])
        && area
            .y
            .checked_add(area.height)
            .is_some_and(|v| v <= size[1])
        && area.x <= i32::MAX as u32
        && area.y <= i32::MAX as u32
}
