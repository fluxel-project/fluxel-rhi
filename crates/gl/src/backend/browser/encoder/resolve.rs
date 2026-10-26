//! Immediate WebGL2 multisample renderbuffer resolves.
//!
//! WebGL2 exposes multisampling only through renderbuffers.  This helper
//! attaches that source and a caller-owned single-sample texture to temporary
//! FBOs, issues one nearest framebuffer blit, then restores the RHI's neutral
//! framebuffer/scissor state. No JS object or command is retained.

use fluxel_rhi_core::api::{Error, Result, TextureAspects};
use web_sys::WebGl2RenderingContext as Gl;

use super::super::owner::{BrowserEncoder, BrowserTexture};

impl BrowserEncoder {
    /// Resolves a color region from a multisample renderbuffer into one
    /// single-sample texture.  All ownership and shape rules are checked
    /// before binding either temporary framebuffer.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn resolve_texture_regions(
        &mut self,
        source: &BrowserTexture,
        source_mip: u32,
        source_origin: [u32; 3],
        source_extent: [u32; 3],
        destination: &BrowserTexture,
        destination_mip: u32,
        destination_origin: [u32; 3],
        destination_extent: [u32; 3],
        source_aspect: TextureAspects,
        destination_aspect: TextureAspects,
    ) -> Result<()> {
        use crate::backend::api::{GlFormatResourceKind, GlTextureAspect, GlTextureUsage};

        self.same_owner(&source.owner)?;
        self.same_owner(&destination.owner)?;
        let source_desc = source.desc;
        let destination_desc = destination.desc;
        let source_mip_extent = source_desc
            .mip_extent(source_mip)
            .ok_or(Error::InvalidRange)?;
        let destination_mip_extent = destination_desc
            .mip_extent(destination_mip)
            .ok_or(Error::InvalidRange)?;
        let source_renderbuffer = source.renderbuffer.as_ref();
        let destination_texture = destination.raw.as_ref();
        if source_desc.sample_count <= 1
            || destination_desc.sample_count != 1
            || source_desc.format != destination_desc.format
            || !source_desc
                .format
                .blit_compatible(destination_desc.format, GlTextureAspect::Color)
            || source_aspect != TextureAspects::COLOR
            || destination_aspect != TextureAspects::COLOR
            || source_desc.dimension != crate::backend::api::GlTextureDimension::D2
            || destination_desc.dimension != crate::backend::api::GlTextureDimension::D2
            || source_mip != 0
            || source_origin[2] != 0
            || destination_origin[2] != 0
            || source_extent[2] != 1
            || destination_extent[2] != 1
            || source_extent != destination_extent
            || !source_desc.usage.contains(GlTextureUsage::COPY_SOURCE)
            || !destination_desc
                .usage
                .contains(GlTextureUsage::COPY_DESTINATION)
            || source.raw.is_some()
            || source_renderbuffer.is_none()
            || destination.renderbuffer.is_some()
            || destination_texture.is_none()
            || source_origin[0]
                .checked_add(source_extent[0])
                .is_none_or(|end| end > source_mip_extent.width)
            || source_origin[1]
                .checked_add(source_extent[1])
                .is_none_or(|end| end > source_mip_extent.height)
            || destination_origin[0]
                .checked_add(destination_extent[0])
                .is_none_or(|end| end > destination_mip_extent.width)
            || destination_origin[1]
                .checked_add(destination_extent[1])
                .is_none_or(|end| end > destination_mip_extent.height)
        {
            return Err(Error::InvalidRange);
        }

        let discovery = self.owner.discovery_snapshot();
        if discovery
            .formats()
            .get_for(
                GlFormatResourceKind::Renderbuffer,
                source_desc.format,
                source_desc.sample_count,
            )
            .is_none_or(|facts| !facts.renderable)
            || discovery
                .formats()
                .get_for(GlFormatResourceKind::Texture, destination_desc.format, 1)
                .is_none_or(|facts| !facts.renderable)
        {
            return Err(Error::Unsupported("WebGL2 resolve format/sample count"));
        }

        let to_i32 = |value: u32| i32::try_from(value).map_err(|_| Error::InvalidRange);
        let src_x0 = to_i32(source_origin[0])?;
        let src_y0 = to_i32(source_origin[1])?;
        let src_x1 = to_i32(
            source_origin[0]
                .checked_add(source_extent[0])
                .ok_or(Error::InvalidRange)?,
        )?;
        let src_y1 = to_i32(
            source_origin[1]
                .checked_add(source_extent[1])
                .ok_or(Error::InvalidRange)?,
        )?;
        let dst_x0 = to_i32(destination_origin[0])?;
        let dst_y0 = to_i32(destination_origin[1])?;
        let dst_x1 = to_i32(
            destination_origin[0]
                .checked_add(destination_extent[0])
                .ok_or(Error::InvalidRange)?,
        )?;
        let dst_y1 = to_i32(
            destination_origin[1]
                .checked_add(destination_extent[1])
                .ok_or(Error::InvalidRange)?,
        )?;

        let mut state = self.owner.state.borrow_mut();
        state.ensure_encoder(self.epoch)?;
        state.ensure_active()?;
        let raw = &state.context.raw;
        // WebGL2 inherits ES 3's multisample-blit restriction: source and
        // destination rectangles must have the same coordinates.  Preserve a
        // caller's different destination offset by resolving first into a
        // same-coordinate, single-sample scratch texture, then blitting that
        // texture normally.  The scratch is immediate-call private.
        let offset_workaround = [src_x0, src_y0] != [dst_x0, dst_y0];
        let scratch = if offset_workaround {
            let internal = super::super::format_map::internal_format(source_desc.format)
                .ok_or(Error::Unsupported("WebGL2 resolve scratch format"))?;
            let texture = raw.create_texture().ok_or(Error::OutOfMemory)?;
            raw.bind_texture(Gl::TEXTURE_2D, Some(&texture));
            raw.tex_storage_2d(
                Gl::TEXTURE_2D,
                1,
                internal,
                source_mip_extent.width as i32,
                source_mip_extent.height as i32,
            );
            // Scratch allocation binds the active unit but does not change the
            // active-unit selector. Restore that binding before any later
            // return path so this direct transfer cannot disturb a draw.
            raw.bind_texture(Gl::TEXTURE_2D, None);
            Some(texture)
        } else {
            None
        };
        let Some(source_fbo) = raw.create_framebuffer() else {
            raw.delete_texture(scratch.as_ref());
            return Err(Error::OutOfMemory);
        };
        let Some(destination_fbo) = raw.create_framebuffer() else {
            raw.delete_framebuffer(Some(&source_fbo));
            raw.delete_texture(scratch.as_ref());
            return Err(Error::OutOfMemory);
        };
        let scratch_fbo = if scratch.is_some() {
            match raw.create_framebuffer() {
                Some(framebuffer) => Some(framebuffer),
                None => {
                    raw.delete_framebuffer(Some(&destination_fbo));
                    raw.delete_framebuffer(Some(&source_fbo));
                    raw.delete_texture(scratch.as_ref());
                    return Err(Error::OutOfMemory);
                }
            }
        } else {
            None
        };
        raw.bind_framebuffer(Gl::FRAMEBUFFER, Some(&source_fbo));
        raw.framebuffer_renderbuffer(
            Gl::FRAMEBUFFER,
            Gl::COLOR_ATTACHMENT0,
            Gl::RENDERBUFFER,
            source_renderbuffer,
        );
        raw.bind_framebuffer(Gl::FRAMEBUFFER, Some(&destination_fbo));
        raw.framebuffer_texture_2d(
            Gl::FRAMEBUFFER,
            Gl::COLOR_ATTACHMENT0,
            Gl::TEXTURE_2D,
            destination_texture,
            destination_mip as i32,
        );
        if let (Some(scratch), Some(scratch_fbo)) = (&scratch, &scratch_fbo) {
            raw.bind_framebuffer(Gl::FRAMEBUFFER, Some(scratch_fbo));
            raw.framebuffer_texture_2d(
                Gl::FRAMEBUFFER,
                Gl::COLOR_ATTACHMENT0,
                Gl::TEXTURE_2D,
                Some(scratch),
                0,
            );
        }
        // Source, destination, and optional scratch attachments are created
        // from already validated immutable descriptors. A completeness query
        // per resolve is a synchronous observation, so deferred submission
        // diagnostics own invalid-use reporting instead.
        // An unrelated raster pass may have left this enabled. A transfer is
        // defined by the supplied regions, not by that draw state.
        raw.disable(Gl::SCISSOR_TEST);
        raw.bind_framebuffer(Gl::READ_FRAMEBUFFER, Some(&source_fbo));
        raw.bind_framebuffer(
            Gl::DRAW_FRAMEBUFFER,
            Some(scratch_fbo.as_ref().unwrap_or(&destination_fbo)),
        );
        let (resolve_x0, resolve_y0, resolve_x1, resolve_y1) = if scratch_fbo.is_some() {
            (src_x0, src_y0, src_x1, src_y1)
        } else {
            (dst_x0, dst_y0, dst_x1, dst_y1)
        };
        raw.blit_framebuffer(
            src_x0,
            src_y0,
            src_x1,
            src_y1,
            resolve_x0,
            resolve_y0,
            resolve_x1,
            resolve_y1,
            Gl::COLOR_BUFFER_BIT,
            Gl::NEAREST,
        );
        if let Some(scratch_fbo) = &scratch_fbo {
            raw.bind_framebuffer(Gl::READ_FRAMEBUFFER, Some(scratch_fbo));
            raw.bind_framebuffer(Gl::DRAW_FRAMEBUFFER, Some(&destination_fbo));
            raw.blit_framebuffer(
                src_x0,
                src_y0,
                src_x1,
                src_y1,
                dst_x0,
                dst_y0,
                dst_x1,
                dst_y1,
                Gl::COLOR_BUFFER_BIT,
                Gl::NEAREST,
            );
        }
        self.issued = true;
        raw.bind_framebuffer(Gl::READ_FRAMEBUFFER, None);
        raw.bind_framebuffer(Gl::DRAW_FRAMEBUFFER, None);
        raw.disable(Gl::SCISSOR_TEST);
        raw.delete_framebuffer(Some(&destination_fbo));
        raw.delete_framebuffer(Some(&source_fbo));
        raw.delete_framebuffer(scratch_fbo.as_ref());
        raw.delete_texture(scratch.as_ref());
        if raw.is_context_lost() {
            state.mark_lost();
            return Err(Error::DeviceLost);
        }
        Ok(())
    }
}
