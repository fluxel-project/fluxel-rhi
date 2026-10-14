//! Immediate framebuffer blits between caller-owned 2D texture mip levels.
//!
//! WebGL2 has no `glBlitNamedFramebuffer`; this adapter creates two short-lived
//! FBOs, issues one native blit, and deletes them before returning. No command
//! payload or FBO survives this call.

use fluxel_rhi_core::api::{Error, Result};
use web_sys::WebGl2RenderingContext as Gl;

use super::super::owner::{BrowserEncoder, BrowserTexture};

impl BrowserEncoder {
    pub(crate) fn blit_texture_regions(
        &mut self,
        source: &BrowserTexture,
        source_mip: u32,
        source_origin: [u32; 2],
        source_extent: [u32; 2],
        destination: &BrowserTexture,
        destination_mip: u32,
        destination_origin: [u32; 2],
        destination_extent: [u32; 2],
        filter: crate::backend::gl::api::GlFilterMode,
    ) -> Result<()> {
        self.same_owner(&source.owner)?;
        self.same_owner(&destination.owner)?;
        let src = source.desc;
        let dst = destination.desc;
        let src_extent = src.mip_extent(source_mip).ok_or(Error::InvalidRange)?;
        let dst_extent = dst.mip_extent(destination_mip).ok_or(Error::InvalidRange)?;
        if src.dimension != crate::backend::gl::api::GlTextureDimension::D2
            || dst.dimension != crate::backend::gl::api::GlTextureDimension::D2
            || src.sample_count != 1
            || dst.sample_count != 1
            || !matches!(
                src.format,
                crate::backend::gl::api::GlFormat::Rgba8Unorm
                    | crate::backend::gl::api::GlFormat::Rgba8Srgb
            )
            || dst.format != src.format
            || !src
                .usage
                .contains(crate::backend::gl::api::GlTextureUsage::COPY_SOURCE)
            || !dst
                .usage
                .contains(crate::backend::gl::api::GlTextureUsage::COPY_DESTINATION)
            || source_extent.contains(&0)
            || destination_extent.contains(&0)
            || source_origin[0]
                .checked_add(source_extent[0])
                .is_none_or(|end| end > src_extent.width)
            || source_origin[1]
                .checked_add(source_extent[1])
                .is_none_or(|end| end > src_extent.height)
            || destination_origin[0]
                .checked_add(destination_extent[0])
                .is_none_or(|end| end > dst_extent.width)
            || destination_origin[1]
                .checked_add(destination_extent[1])
                .is_none_or(|end| end > dst_extent.height)
            || (js_sys::Object::is(
                <web_sys::WebGlTexture as AsRef<wasm_bindgen::JsValue>>::as_ref(&source.raw),
                <web_sys::WebGlTexture as AsRef<wasm_bindgen::JsValue>>::as_ref(&destination.raw),
            ) && source_mip == destination_mip)
        {
            return Err(Error::Unsupported("WebGL2 texture blit shape or usage"));
        }
        let src_x0 = i32::try_from(source_origin[0]).map_err(|_| Error::InvalidRange)?;
        let src_x1 =
            i32::try_from(source_origin[0] + source_extent[0]).map_err(|_| Error::InvalidRange)?;
        let src_y0 = i32::try_from(src_extent.height - source_origin[1] - source_extent[1])
            .map_err(|_| Error::InvalidRange)?;
        let src_y1 =
            i32::try_from(src_extent.height - source_origin[1]).map_err(|_| Error::InvalidRange)?;
        let dst_x0 = i32::try_from(destination_origin[0]).map_err(|_| Error::InvalidRange)?;
        let dst_x1 = i32::try_from(destination_origin[0] + destination_extent[0])
            .map_err(|_| Error::InvalidRange)?;
        let dst_y0 =
            i32::try_from(dst_extent.height - destination_origin[1] - destination_extent[1])
                .map_err(|_| Error::InvalidRange)?;
        let dst_y1 = i32::try_from(dst_extent.height - destination_origin[1])
            .map_err(|_| Error::InvalidRange)?;

        let mut state = self.owner.state.borrow_mut();
        state.ensure_encoder(self.epoch)?;
        state.ensure_active()?;
        let raw = &state.context.raw;
        let Some(src_fbo) = raw.create_framebuffer() else {
            return Err(Error::OutOfMemory);
        };
        let Some(dst_fbo) = raw.create_framebuffer() else {
            raw.delete_framebuffer(Some(&src_fbo));
            return Err(Error::OutOfMemory);
        };
        raw.bind_framebuffer(Gl::FRAMEBUFFER, Some(&src_fbo));
        raw.framebuffer_texture_2d(
            Gl::FRAMEBUFFER,
            Gl::COLOR_ATTACHMENT0,
            Gl::TEXTURE_2D,
            Some(&source.raw),
            source_mip as i32,
        );
        let source_complete =
            raw.check_framebuffer_status(Gl::FRAMEBUFFER) == Gl::FRAMEBUFFER_COMPLETE;
        if source_complete {
            raw.bind_framebuffer(Gl::FRAMEBUFFER, Some(&dst_fbo));
            raw.framebuffer_texture_2d(
                Gl::FRAMEBUFFER,
                Gl::COLOR_ATTACHMENT0,
                Gl::TEXTURE_2D,
                Some(&destination.raw),
                destination_mip as i32,
            );
        }
        let destination_complete = source_complete
            && raw.check_framebuffer_status(Gl::FRAMEBUFFER) == Gl::FRAMEBUFFER_COMPLETE;
        if destination_complete {
            // A prior immediate raster pass may leave SCISSOR_TEST enabled.
            // GL applies scissor to framebuffer blits, so the private transfer
            // must not inherit that unrelated draw state.
            raw.disable(Gl::SCISSOR_TEST);
            raw.bind_framebuffer(Gl::READ_FRAMEBUFFER, Some(&src_fbo));
            raw.bind_framebuffer(Gl::DRAW_FRAMEBUFFER, Some(&dst_fbo));
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
                match filter {
                    crate::backend::gl::api::GlFilterMode::Nearest => Gl::NEAREST,
                    crate::backend::gl::api::GlFilterMode::Linear => Gl::LINEAR,
                },
            );
            self.issued = true;
        }
        raw.bind_framebuffer(Gl::FRAMEBUFFER, None);
        raw.delete_framebuffer(Some(&dst_fbo));
        raw.delete_framebuffer(Some(&src_fbo));
        if state.context.raw.is_context_lost() {
            state.mark_lost();
            return Err(Error::DeviceLost);
        }
        let error = raw.get_error();
        if error != Gl::NO_ERROR {
            return Err(Error::Backend(format!(
                "WebGL2 texture blit failed with GL error {error}"
            )));
        }
        if !destination_complete {
            return Err(Error::Backend(
                "WebGL2 texture blit framebuffer is incomplete".into(),
            ));
        }
        Ok(())
    }
}
