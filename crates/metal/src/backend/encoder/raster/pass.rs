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
            MTLCommandBuffer as _, MTLRenderPassDescriptor, MTLStoreAction, MTLTexture as _,
        };
        if encoder.lane != 0
            || encoder.render.is_some()
            || encoder.compute.is_some()
            || desc.colors.len() != 1
            || desc.timestamp_writes.is_some()
            || desc.occlusion_query_set.is_some()
        {
            return Err(api::Error::Unsupported("Metal raster pass descriptor"));
        }
        let attachment = desc.colors[0].as_ref().ok_or(api::Error::InvalidState)?;
        let texture: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLTexture> =
            match &attachment.view {
                api::AttachmentView::Texture(view) => {
                    // SAFETY: Core validated the view's device identity.
                    &unsafe { view.native_ref() }.raw
                }
                api::AttachmentView::Surface(image) => {
                    // SAFETY: Core validated the exclusive, current Surface lease.
                    &unsafe { image.native_ref() }.texture
                }
            };
        let color_format = texture.pixelFormat();
        let sample_count = texture.sampleCount();
        let resolve_texture: Option<&objc2::runtime::ProtocolObject<dyn objc2_metal::MTLTexture>> =
            match &attachment.resolve_target {
                Some(api::AttachmentView::Texture(view)) => {
                    // SAFETY: Core validated the resolve view's device identity.
                    let native = unsafe { view.native_ref() };
                    Some(&native.raw)
                }
                Some(api::AttachmentView::Surface(image)) => {
                    // SAFETY: Core validated the current exclusive drawable lease.
                    let native = unsafe { image.native_ref() };
                    Some(&native.texture)
                }
                None => None,
            };
        if let Some(resolve) = resolve_texture {
            if sample_count <= 1
                || resolve.sampleCount() != 1
                || resolve.pixelFormat() != color_format
                || resolve.width() != texture.width()
                || resolve.height() != texture.height()
            {
                return Err(api::Error::InvalidState);
            }
        }
        let descriptor = MTLRenderPassDescriptor::renderPassDescriptor();
        let colors = descriptor.colorAttachments();
        // SAFETY: this descriptor has a fixed native color attachment array;
        // index zero is valid and Core restricts this pass to one target.
        let color = unsafe { colors.objectAtIndexedSubscript(0) };
        color.setTexture(Some(texture));
        color.setResolveTexture(resolve_texture);
        let (load, clear) = super::load_action(attachment.load);
        color.setLoadAction(load);
        if let Some(clear) = clear {
            color.setClearColor(clear);
        }
        color.setStoreAction(match (resolve_texture.is_some(), attachment.store) {
            (true, api::StoreOp::Store) => MTLStoreAction::StoreAndMultisampleResolve,
            (true, api::StoreOp::Discard) => MTLStoreAction::MultisampleResolve,
            (false, api::StoreOp::Store) => MTLStoreAction::Store,
            (false, api::StoreOp::Discard) => MTLStoreAction::DontCare,
        });
        let depth_format = if let Some(depth) = &desc.depth_stencil {
            if depth.stencil_read_only || depth.depth_read_only {
                return Err(api::Error::Unsupported("Metal read-only depth/stencil"));
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
            if depth_texture.sampleCount() != sample_count {
                return Err(api::Error::InvalidState);
            }
            let depth_attachment = descriptor.depthAttachment();
            depth_attachment.setTexture(Some(depth_texture));
            let (load, clear) = match depth.depth {
                Some((api::LoadOp::Load, _)) => (objc2_metal::MTLLoadAction::Load, None),
                Some((api::LoadOp::Discard, _)) => (objc2_metal::MTLLoadAction::DontCare, None),
                Some((api::LoadOp::Clear(value), _)) => {
                    (objc2_metal::MTLLoadAction::Clear, Some(value))
                }
                None => (objc2_metal::MTLLoadAction::DontCare, None),
            };
            depth_attachment.setLoadAction(load);
            if let Some(value) = clear {
                depth_attachment.setClearDepth(value as f64);
            }
            depth_attachment.setStoreAction(match depth.depth {
                Some((_, api::StoreOp::Store)) => MTLStoreAction::Store,
                _ => MTLStoreAction::DontCare,
            });
            if depth_texture.pixelFormat() == objc2_metal::MTLPixelFormat::Depth32Float_Stencil8 {
                let stencil_attachment = descriptor.stencilAttachment();
                stencil_attachment.setTexture(Some(depth_texture));
                let (load, clear) = match depth.stencil {
                    Some((api::LoadOp::Load, _)) => (objc2_metal::MTLLoadAction::Load, None),
                    Some((api::LoadOp::Discard, _)) => (objc2_metal::MTLLoadAction::DontCare, None),
                    Some((api::LoadOp::Clear(value), _)) => {
                        (objc2_metal::MTLLoadAction::Clear, Some(value))
                    }
                    None => (objc2_metal::MTLLoadAction::DontCare, None),
                };
                stencil_attachment.setLoadAction(load);
                if let Some(clear) = clear {
                    stencil_attachment.setClearStencil(clear);
                }
                stencil_attachment.setStoreAction(match depth.stencil {
                    Some((_, api::StoreOp::Store)) => MTLStoreAction::Store,
                    _ => MTLStoreAction::DontCare,
                });
            } else if depth.stencil.is_some() {
                return Err(api::Error::InvalidState);
            }
            Some(depth_texture.pixelFormat())
        } else {
            None
        };
        let render = encoder
            .command
            .renderCommandEncoderWithDescriptor(&descriptor)
            .ok_or(api::Error::OutOfMemory)?;
        encoder.raster_signature = Some((sample_count, color_format, depth_format));
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
