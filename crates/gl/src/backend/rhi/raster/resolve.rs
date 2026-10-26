//! Native attachment resolves issued at the successful end of an immediate pass.
//!
//! Retained views describe active pass attachments, not recorded draw commands.
//! Each resolve uses single-slot FBOs so MRT holes and source locations cannot
//! change which attachment is read. Aborting a pass never calls this helper.

use crate::backend::api::{
    GlBlitMask, GlBlitRegion, GlError, GlFilterMode, GlFramebufferApi, GlFramebufferDescriptor,
    GlScissorRect, GlTextureView,
};
use crate::backend::native::provider::NativeGlProvider;

pub(super) fn textures(
    provider: &mut NativeGlProvider,
    attachments: &[(GlTextureView, GlTextureView, Option<GlScissorRect>)],
) -> Result<(), GlError> {
    for &(source, destination, area) in attachments {
        let descriptor = |view| GlFramebufferDescriptor {
            color_attachments: vec![Some(view)],
            depth_stencil_attachment: None,
            draw_buffers: vec![0],
        };
        let source_fbo = provider.create_framebuffer(&descriptor(source))?;
        let destination_fbo = match provider.create_framebuffer(&descriptor(destination)) {
            Ok(fbo) => fbo,
            Err(error) => {
                let _ = provider.destroy_framebuffer(source_fbo);
                return Err(error);
            }
        };
        // Texture row zero is Core row zero on both sides of this resolve.
        let (offset, extent) = area.map_or(([0, 0], [source.width, source.height]), |a| {
            ([a.x, a.y], [a.width, a.height])
        });
        let result = provider.blit_framebuffer(
            source_fbo,
            destination_fbo,
            GlBlitRegion {
                src_offset: offset,
                src_extent: extent,
                dst_offset: offset,
                dst_extent: extent,
            },
            GlFilterMode::Nearest,
            GlBlitMask {
                color: true,
                depth: false,
                stencil: false,
            },
        );
        // Retire both temporary FBOs even if the native blit failed.
        let drop_destination = provider.destroy_framebuffer(destination_fbo);
        let drop_source = provider.destroy_framebuffer(source_fbo);
        result.and(drop_destination).and(drop_source)?;
    }
    Ok(())
}

/// Resolves in texture coordinates before the separate window-axis reversal.
/// GL forbids reversing a multisample resolve rectangle directly.
pub(crate) fn surface(
    provider: &mut NativeGlProvider,
    source: GlTextureView,
    target: crate::backend::api::GlDefaultFramebufferTarget,
    area: Option<GlScissorRect>,
) -> Result<(), GlError> {
    use crate::backend::api::{
        GlAttachmentTarget, GlExtent3d, GlResourceApi, GlTextureDesc, GlTextureDimension,
        GlTextureUsage,
    };
    let source_fbo = provider.create_framebuffer(&GlFramebufferDescriptor {
        color_attachments: vec![Some(source)],
        depth_stencil_attachment: None,
        draw_buffers: vec![0],
    })?;
    let texture = match provider.create_texture_resource(GlTextureDesc {
        dimension: GlTextureDimension::D2,
        extent: GlExtent3d {
            width: target.width,
            height: target.height,
            depth_or_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        format: target.color_format,
        usage: GlTextureUsage::RENDER_ATTACHMENT,
    }) {
        Ok(texture) => texture,
        Err(error) => {
            let _ = provider.destroy_framebuffer(source_fbo);
            return Err(error);
        }
    };
    let framebuffer = match provider.create_framebuffer(&GlFramebufferDescriptor {
        color_attachments: vec![Some(GlTextureView {
            aspect: crate::backend::api::GlTextureAspect::All,
            target: GlAttachmentTarget::Texture(texture),
            format: target.color_format,
            mip_level: 0,
            array_layer: 0,
            layer_count: 1,
            width: target.width,
            height: target.height,
            sample_count: 1,
        })],
        depth_stencil_attachment: None,
        draw_buffers: vec![0],
    }) {
        Ok(framebuffer) => framebuffer,
        Err(error) => {
            let _ = provider.destroy_texture_resource(texture);
            let _ = provider.destroy_framebuffer(source_fbo);
            return Err(error);
        }
    };
    let area = area.unwrap_or(GlScissorRect {
        x: 0,
        y: 0,
        width: target.width,
        height: target.height,
    });
    let result = provider
        .blit_framebuffer(
            source_fbo,
            framebuffer,
            GlBlitRegion {
                src_offset: [area.x, area.y],
                src_extent: [area.width, area.height],
                dst_offset: [area.x, area.y],
                dst_extent: [area.width, area.height],
            },
            GlFilterMode::Nearest,
            GlBlitMask {
                color: true,
                depth: false,
                stencil: false,
            },
        )
        .and_then(|()| {
            provider.blit_framebuffer_slot_to_default_region(framebuffer, 0, target, Some(area))
        });
    let drop_framebuffer = provider.destroy_framebuffer(framebuffer);
    let drop_texture = provider.destroy_texture_resource(texture);
    let drop_source = provider.destroy_framebuffer(source_fbo);
    result
        .and(drop_framebuffer)
        .and(drop_texture)
        .and(drop_source)
}
