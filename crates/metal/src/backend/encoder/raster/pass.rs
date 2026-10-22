//! Metal render-pass attachment validation and native descriptor construction.
//!
//! The encoder writes attachments directly to Metal. This module validates sample and resolve compatibility before beginning the native pass.

use fluxel_rhi_core::api;

use super::super::{Metal, MetalCommandEncoder};

pub(super) fn begin_raster(
    encoder: &mut MetalCommandEncoder,
    desc: &api::RasterPassDesc<'_, Metal>,
) -> api::Result<()> {
    #[cfg(target_vendor = "apple")]
    {
        use objc2_metal::{
            MTLCommandBuffer as _, MTLCommandQueue as _, MTLRenderPassDescriptor, MTLResource as _,
            MTLStoreAction, MTLTexture as _,
        };
        if let Some(mask) = desc.multiview_mask {
            if mask == 0 {
                return Err(api::Error::InvalidRange);
            }
            descriptor_multiview_available(encoder, 1)?;
        }
        if encoder.lane != 0
            || encoder.render.is_some()
            || encoder.compute.is_some()
            || desc.colors.len() > 8
        {
            return Err(api::Error::Unsupported("Metal raster pass descriptor"));
        }
        let area = desc.render_area;
        let right = area
            .x
            .checked_add(area.width)
            .ok_or(api::Error::InvalidRange)?;
        let bottom = area
            .y
            .checked_add(area.height)
            .ok_or(api::Error::InvalidRange)?;
        if area.width == 0 || area.height == 0 {
            return Err(api::Error::InvalidRange);
        }
        let descriptor = MTLRenderPassDescriptor::renderPassDescriptor();
        if let Some(mask) = desc.multiview_mask {
            descriptor.setRenderTargetArrayLength((32 - mask.leading_zeros()) as usize);
        }
        let colors = descriptor.colorAttachments();
        let mut color_formats = Vec::with_capacity(desc.colors.len());
        let mut sample_count = None;
        let mut partial_resolves = Vec::new();
        let sparse_views = desc
            .multiview_mask
            .is_some_and(|mask| mask != u32::MAX >> mask.leading_zeros());
        let partial_area = sparse_views
            || desc.colors.iter().flatten().any(|attachment| {
                let texture = match &attachment.view {
                    api::AttachmentView::Texture(view) => &unsafe { view.native_ref() }.raw,
                    api::AttachmentView::Surface(image) => &unsafe { image.native_ref() }.texture,
                };
                area.x != 0
                    || area.y != 0
                    || right as usize != texture.width()
                    || bottom as usize != texture.height()
            })
            || desc
                .depth_stencil
                .as_ref()
                .is_some_and(|depth| match &depth.view {
                    api::AttachmentView::Texture(view) => {
                        let texture = &unsafe { view.native_ref() }.raw;
                        area.x != 0
                            || area.y != 0
                            || right as usize != texture.width()
                            || bottom as usize != texture.height()
                    }
                    api::AttachmentView::Surface(_) => false,
                });
        for (index, attachment) in desc.colors.iter().enumerate() {
            let Some(attachment) = attachment else {
                color_formats.push(None);
                continue;
            };
            let texture: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLTexture> =
                match &attachment.view {
                    // SAFETY: Core validates resource ownership and Surface leases.
                    api::AttachmentView::Texture(view) => &unsafe { view.native_ref() }.raw,
                    api::AttachmentView::Surface(image) => &unsafe { image.native_ref() }.texture,
                };
            if right as usize > texture.width() || bottom as usize > texture.height() {
                return Err(api::Error::InvalidRange);
            }
            if let Some(mask) = desc.multiview_mask {
                if texture.arrayLength() < (32 - mask.leading_zeros()) as usize {
                    return Err(api::Error::InvalidRange);
                }
            }
            let format = texture.pixelFormat();
            let samples = texture.sampleCount();
            if sample_count.is_some_and(|previous| previous != samples) {
                return Err(api::Error::InvalidState);
            }
            sample_count = Some(samples);
            let resolve: Option<&objc2::runtime::ProtocolObject<dyn objc2_metal::MTLTexture>> =
                match &attachment.resolve_target {
                    // SAFETY: Core validates resource ownership and Surface leases.
                    Some(api::AttachmentView::Texture(view)) => {
                        Some(&unsafe { view.native_ref() }.raw)
                    }
                    Some(api::AttachmentView::Surface(image)) => {
                        Some(&unsafe { image.native_ref() }.texture)
                    }
                    None => None,
                };
            if let Some(resolve) = resolve {
                if partial_area {
                    if texture.isFramebufferOnly()
                        || texture.storageMode() == objc2_metal::MTLStorageMode::Memoryless
                        || resolve.storageMode() == objc2_metal::MTLStorageMode::Memoryless
                    {
                        return Err(api::Error::Unsupported(
                            "Metal partial resolve requires persistent attachments",
                        ));
                    }
                    // SAFETY: Core validated attachment ownership and the lease remains live.
                    let source = match &attachment.view {
                        api::AttachmentView::Texture(view) => {
                            unsafe { view.native_ref() }.raw.clone()
                        }
                        api::AttachmentView::Surface(image) => {
                            unsafe { image.native_ref() }.texture.clone()
                        }
                    };
                    let target = match attachment.resolve_target.as_ref().expect("resolve target") {
                        api::AttachmentView::Texture(view) => {
                            unsafe { view.native_ref() }.raw.clone()
                        }
                        api::AttachmentView::Surface(image) => {
                            unsafe { image.native_ref() }.texture.clone()
                        }
                    };
                    partial_resolves.push((source, target, area, desc.multiview_mask));
                }
                let core_format = match &attachment.view {
                    // SAFETY: Core validated ownership of the view.
                    api::AttachmentView::Texture(view) => unsafe { view.native_ref() }.format,
                    api::AttachmentView::Surface(_) => return Err(api::Error::InvalidState),
                };
                if !super::super::super::copy::device_color_resolve_supported(
                    &texture.device(),
                    core_format,
                ) {
                    return Err(api::Error::Unsupported(
                        "Metal device does not support resolve for this color format",
                    ));
                }
                if samples <= 1
                    || resolve.sampleCount() != 1
                    || resolve.pixelFormat() != format
                    || resolve.width() != texture.width()
                    || resolve.height() != texture.height()
                    || desc.multiview_mask.is_some_and(|mask| {
                        resolve.arrayLength() < (32 - mask.leading_zeros()) as usize
                    })
                {
                    return Err(api::Error::InvalidState);
                }
            }
            // SAFETY: the Metal color attachment array has eight entries, checked above.
            let color = unsafe { colors.objectAtIndexedSubscript(index) };
            color.setTexture(Some(texture));
            color.setResolveTexture(if partial_area { None } else { resolve });
            let (load, clear) = super::load_action(attachment.load);
            color.setLoadAction(if partial_area {
                objc2_metal::MTLLoadAction::Load
            } else {
                load
            });
            if let Some(clear) = clear {
                color.setClearColor(clear);
            }
            color.setStoreAction(
                match (
                    resolve.is_some() && !partial_area,
                    if partial_area {
                        api::StoreOp::Store
                    } else {
                        attachment.store
                    },
                ) {
                    (true, api::StoreOp::Store) => MTLStoreAction::StoreAndMultisampleResolve,
                    (true, api::StoreOp::Discard) => MTLStoreAction::MultisampleResolve,
                    (false, api::StoreOp::Store) => MTLStoreAction::Store,
                    (false, api::StoreOp::Discard) => MTLStoreAction::DontCare,
                },
            );
            color_formats.push(Some(format));
        }
        let depth_format = if let Some(depth) = &desc.depth_stencil {
            if depth.depth_read_only
                && depth.depth.is_some_and(|(load, store)| {
                    !matches!((load, store), (api::LoadOp::Load, api::StoreOp::Store))
                })
                || depth.stencil_read_only
                    && depth.stencil.is_some_and(|(load, store)| {
                        !matches!((load, store), (api::LoadOp::Load, api::StoreOp::Store))
                    })
            {
                return Err(api::Error::InvalidState);
            }
            let depth_texture: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLTexture> =
                match &depth.view {
                    api::AttachmentView::Texture(view) => {
                        // SAFETY: Core validated the view's device identity.
                        &unsafe { view.native_ref() }.raw
                    }
                    api::AttachmentView::Surface(_) => {
                        return Err(api::Error::Unsupported("Metal surface depth attachment"));
                    }
                };
            if right as usize > depth_texture.width() || bottom as usize > depth_texture.height() {
                return Err(api::Error::InvalidRange);
            }
            if let Some(mask) = desc.multiview_mask {
                if depth_texture.arrayLength() < (32 - mask.leading_zeros()) as usize {
                    return Err(api::Error::InvalidRange);
                }
            }
            if sample_count.is_some_and(|samples| depth_texture.sampleCount() != samples) {
                return Err(api::Error::InvalidState);
            }
            sample_count = Some(depth_texture.sampleCount());
            if depth_texture.pixelFormat() != objc2_metal::MTLPixelFormat::Stencil8 {
                let depth_attachment = descriptor.depthAttachment();
                depth_attachment.setTexture(Some(depth_texture));
                let (load, clear) = match depth.depth {
                    Some((api::LoadOp::Load, _)) => (objc2_metal::MTLLoadAction::Load, None),
                    Some((api::LoadOp::Discard, _)) => (objc2_metal::MTLLoadAction::DontCare, None),
                    Some((api::LoadOp::Clear(value), _)) => {
                        (objc2_metal::MTLLoadAction::Clear, Some(value))
                    }
                    None if depth.depth_read_only => (objc2_metal::MTLLoadAction::Load, None),
                    None => (objc2_metal::MTLLoadAction::DontCare, None),
                };
                depth_attachment.setLoadAction(if partial_area {
                    objc2_metal::MTLLoadAction::Load
                } else {
                    load
                });
                if let Some(value) = clear {
                    depth_attachment.setClearDepth(value as f64);
                }
                depth_attachment.setStoreAction(if partial_area {
                    MTLStoreAction::Store
                } else {
                    match depth.depth {
                        Some((_, api::StoreOp::Store)) => MTLStoreAction::Store,
                        None if depth.depth_read_only => MTLStoreAction::Store,
                        _ => MTLStoreAction::DontCare,
                    }
                });
            } else if depth.depth.is_some() || depth.depth_read_only {
                return Err(api::Error::InvalidState);
            }
            if matches!(
                depth_texture.pixelFormat(),
                objc2_metal::MTLPixelFormat::Depth32Float_Stencil8
                    | objc2_metal::MTLPixelFormat::Depth24Unorm_Stencil8
                    | objc2_metal::MTLPixelFormat::Stencil8
            ) {
                let stencil_attachment = descriptor.stencilAttachment();
                stencil_attachment.setTexture(Some(depth_texture));
                let (load, clear) = match depth.stencil {
                    Some((api::LoadOp::Load, _)) => (objc2_metal::MTLLoadAction::Load, None),
                    Some((api::LoadOp::Discard, _)) => (objc2_metal::MTLLoadAction::DontCare, None),
                    Some((api::LoadOp::Clear(value), _)) => {
                        (objc2_metal::MTLLoadAction::Clear, Some(value))
                    }
                    None if depth.stencil_read_only => (objc2_metal::MTLLoadAction::Load, None),
                    None => (objc2_metal::MTLLoadAction::DontCare, None),
                };
                stencil_attachment.setLoadAction(if partial_area {
                    objc2_metal::MTLLoadAction::Load
                } else {
                    load
                });
                if let Some(clear) = clear {
                    stencil_attachment.setClearStencil(clear);
                }
                stencil_attachment.setStoreAction(if partial_area {
                    MTLStoreAction::Store
                } else {
                    match depth.stencil {
                        Some((_, api::StoreOp::Store)) => MTLStoreAction::Store,
                        None if depth.stencil_read_only => MTLStoreAction::Store,
                        _ => MTLStoreAction::DontCare,
                    }
                });
            } else if depth.stencil.is_some() {
                return Err(api::Error::InvalidState);
            }
            Some(depth_texture.pixelFormat())
        } else {
            None
        };
        // Reserve the preparation command buffer before the render buffer.
        // They are recorded independently but committed in this order at submit.
        let split = if encoder.indirect_count_enabled {
            let queue = encoder.command.commandQueue();
            let prepass = queue.commandBuffer().ok_or(api::Error::OutOfMemory)?;
            let render_command = queue.commandBuffer().ok_or(api::Error::OutOfMemory)?;
            Some((prepass, render_command))
        } else {
            None
        };
        if let Some(writes) = &desc.timestamp_writes {
            use fluxel_rhi_core::api::QueryBackend as _;
            // SAFETY: Core validated the query set identity, kind and slots.
            let set = unsafe { writes.set.native_ref() };
            if let Some(begin) = writes.begin {
                Metal::write_timestamp(encoder, set, begin)?;
            }
            encoder.raster_end_timestamp = writes.end.map(|end| (set.clone(), end));
        } else {
            encoder.raster_end_timestamp = None;
        }
        let render_command = split
            .as_ref()
            .map_or(&encoder.command, |(_, command)| command)
            .clone();
        if partial_area {
            use objc2_metal::MTLCommandEncoder as _;
            let clear = render_command
                .renderCommandEncoderWithDescriptor(&descriptor)
                .ok_or(api::Error::OutOfMemory)?;
            let result = super::clear_area::clear_area(
                &clear,
                desc,
                sample_count.ok_or(api::Error::InvalidState)?,
            );
            clear.endEncoding();
            result?;
        }
        // Only the user's render pass is sampled; the internal clear draw must
        // not contribute primitives or fragment invocations to pass counters.
        super::super::super::query::configure_raster_statistics(encoder, &descriptor)?;
        super::super::super::query::configure_occlusion(
            encoder,
            &descriptor,
            desc.occlusion_query_set,
        )?;
        let render = render_command
            .renderCommandEncoderWithDescriptor(&descriptor)
            .ok_or(api::Error::OutOfMemory)?;
        if let Some((prepass, render_command)) = split {
            encoder.prior_commands.push(encoder.command.clone());
            encoder.prior_commands.push(prepass.clone());
            encoder.indirect_prepass = Some(prepass);
            encoder.command = render_command;
        }
        use objc2_metal::{MTLRenderCommandEncoder as _, MTLScissorRect, MTLViewport};
        render.setScissorRect(MTLScissorRect {
            x: area.x as usize,
            y: area.y as usize,
            width: area.width as usize,
            height: area.height as usize,
        });
        render.setViewport(MTLViewport {
            originX: area.x as f64,
            originY: area.y as f64,
            width: area.width as f64,
            height: area.height as f64,
            znear: 0.0,
            zfar: 1.0,
        });
        encoder.raster_area = Some(area);
        encoder.raster_view_mask = desc.multiview_mask;
        encoder.raster_resolves = partial_resolves;
        encoder.raster_read_only = desc
            .depth_stencil
            .as_ref()
            .map_or((false, false), |d| (d.depth_read_only, d.stencil_read_only));
        encoder.raster_signature = Some((
            sample_count.ok_or(api::Error::InvalidState)?,
            color_formats,
            depth_format,
        ));
        encoder.raster_pipeline_bound = false;
        encoder.mesh_threads = None;
        encoder.render = Some(render);
        Ok(())
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        let _ = (encoder, desc);
        Err(api::Error::Unsupported("Metal requires an Apple target"))
    }
}

#[cfg(target_vendor = "apple")]
fn descriptor_multiview_available(encoder: &MetalCommandEncoder, count: usize) -> api::Result<()> {
    use objc2::runtime::NSObjectProtocol as _;
    use objc2_metal::{MTLCommandBuffer as _, MTLDevice as _};
    let device = encoder.command.device();
    if !device.respondsToSelector(objc2::sel!(supportsVertexAmplificationCount:))
        || !device.supportsVertexAmplificationCount(count)
    {
        return Err(api::Error::Unsupported(
            "Metal vertex amplification unavailable",
        ));
    }
    Ok(())
}
