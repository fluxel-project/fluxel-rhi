//! Immediate temporary surface framebuffer lifecycle and publication.

use super::super::format_map;
use super::super::owner::{BrowserEncoder, BrowserSurfaceDepthPass};
use fluxel_rhi_core::api::{ClearColor, Error, Result};
use web_sys::WebGl2RenderingContext as Gl;

impl BrowserEncoder {
    /// Starts a surface pass through an owned FBO when the Core pass supplies
    /// a Depth32 texture. Framebuffer zero cannot attach that texture, so the
    /// temporary color image is blitted back during `end_surface_depth_raster`.
    pub(crate) fn begin_surface_depth_raster(
        &mut self,
        extent: [u32; 2],
        surface_format: crate::backend::api::GlFormat,
        color_load: fluxel_rhi_core::api::LoadOp<ClearColor>,
        depth: super::attachments::BrowserDepthAttachment<'_>,
        render_area: fluxel_rhi_core::api::Rect,
    ) -> Result<()> {
        use fluxel_rhi_core::api::LoadOp;
        self.same_owner(&depth.texture.owner)?;
        if extent.contains(&0)
            || depth.texture.desc.sample_count != 1
            || render_area
                .x
                .checked_add(render_area.width)
                .is_none_or(|x| x > extent[0])
            || render_area
                .y
                .checked_add(render_area.height)
                .is_none_or(|y| y > extent[1])
            || render_area.width == 0
            || render_area.height == 0
        {
            return Err(Error::InvalidRange);
        }
        let internal = format_map::internal_format(surface_format)
            .ok_or(Error::Unsupported("WebGL2 intermediate color format"))?;
        let (framebuffer, color) = {
            let mut state = self.owner.state.borrow_mut();
            state.ensure_encoder(self.epoch)?;
            state.ensure_active()?;
            if state.raster_extent.is_some() || self.has_owned_framebuffer_pass() {
                return Err(Error::InvalidState);
            }
            let raw = &state.context.raw;
            let Some(framebuffer) = raw.create_framebuffer() else {
                return Err(Self::allocation_error(&mut state, Error::OutOfMemory));
            };
            let Some(color) = raw.create_texture() else {
                raw.delete_framebuffer(Some(&framebuffer));
                return Err(Self::allocation_error(&mut state, Error::OutOfMemory));
            };
            raw.bind_texture(Gl::TEXTURE_2D, Some(&color));
            raw.tex_storage_2d(
                Gl::TEXTURE_2D,
                1,
                internal,
                extent[0] as i32,
                extent[1] as i32,
            );
            raw.bind_framebuffer(Gl::FRAMEBUFFER, Some(&framebuffer));
            raw.framebuffer_texture_2d(
                Gl::FRAMEBUFFER,
                Gl::COLOR_ATTACHMENT0,
                Gl::TEXTURE_2D,
                Some(&color),
                0,
            );
            super::attachments::attach(
                raw,
                super::attachments::depth_attachment(depth.texture.desc.format),
                depth.texture,
                depth.mip_level,
                depth.layer,
            );
            if raw.check_framebuffer_status(Gl::FRAMEBUFFER) != Gl::FRAMEBUFFER_COMPLETE
                || raw.get_error() != Gl::NO_ERROR
            {
                raw.bind_framebuffer(Gl::FRAMEBUFFER, None);
                raw.delete_texture(Some(&color));
                raw.delete_framebuffer(Some(&framebuffer));
                return Err(Self::allocation_error(
                    &mut state,
                    Error::Backend("WebGL2 surface depth framebuffer is incomplete".into()),
                ));
            }
            (framebuffer, color)
        };
        let srgb = surface_format == crate::backend::api::GlFormat::Rgba8Srgb;
        let mut conversion = Ok(());
        let result = self.with_context("begin_surface_depth_raster", |raw| {
            // Preserve both Load and pixels outside a partial pass.
            raw.disable(Gl::SCISSOR_TEST);
            raw.bind_framebuffer(Gl::READ_FRAMEBUFFER, None);
            raw.bind_framebuffer(Gl::DRAW_FRAMEBUFFER, Some(&framebuffer));
            if srgb {
                conversion = super::surface_color::preload(raw, &framebuffer, 0, extent);
            } else {
                raw.blit_framebuffer(
                    0,
                    extent[1] as i32,
                    extent[0] as i32,
                    0,
                    0,
                    0,
                    extent[0] as i32,
                    extent[1] as i32,
                    Gl::COLOR_BUFFER_BIT,
                    Gl::NEAREST,
                );
            }
            raw.bind_framebuffer(Gl::FRAMEBUFFER, Some(&framebuffer));
            raw.viewport(0, 0, extent[0] as i32, extent[1] as i32);
            raw.enable(Gl::SCISSOR_TEST);
            raw.scissor(
                render_area.x as i32,
                render_area.y as i32,
                render_area.width as i32,
                render_area.height as i32,
            );
            raw.color_mask(true, true, true, true);
            raw.depth_mask(true);
            raw.stencil_mask(u32::MAX);
            match color_load {
                LoadOp::Clear(ClearColor::Float(v)) => {
                    raw.clear_bufferfv_with_f32_array(Gl::COLOR, 0, &v)
                }
                LoadOp::Clear(ClearColor::Uint(v)) => {
                    raw.clear_bufferuiv_with_u32_array(Gl::COLOR, 0, &v)
                }
                LoadOp::Clear(ClearColor::Sint(v)) => {
                    raw.clear_bufferiv_with_i32_array(Gl::COLOR, 0, &v)
                }
                _ => {}
            }
            if let Some((LoadOp::Clear(v), _)) = depth.depth {
                raw.clear_bufferfv_with_f32_array(Gl::DEPTH, 0, &[v]);
            }
            if let Some((LoadOp::Clear(v), _)) = depth.stencil {
                raw.clear_bufferiv_with_i32_array(Gl::STENCIL, 0, &[v as i32]);
            }
            raw.disable(Gl::SCISSOR_TEST);
        });
        if let Err(error) = result.and(conversion) {
            let state = self.owner.state.borrow();
            state.context.raw.bind_framebuffer(Gl::FRAMEBUFFER, None);
            state.context.raw.delete_texture(Some(&color));
            state.context.raw.delete_framebuffer(Some(&framebuffer));
            return Err(error);
        }
        self.owner.state.borrow_mut().raster_extent = Some(extent);
        self.surface_depth_pass = Some(BrowserSurfaceDepthPass {
            framebuffer,
            color,
            depth: depth.texture.clone(),
            extent,
            srgb,
        });
        Ok(())
    }

    /// Resolves the immediate temporary FBO into the acquired default target
    /// and tears down only objects created by `begin_surface_depth_raster`.
    pub(crate) fn end_surface_depth_raster(&mut self) -> Result<()> {
        let pass = self.surface_depth_pass.take().ok_or(Error::InvalidState)?;
        let result = (|| {
            self.with_context("bind_surface_depth_read_framebuffer", |raw| {
                raw.bind_framebuffer(Gl::READ_FRAMEBUFFER, Some(&pass.framebuffer));
            })?;
            self.with_context("bind_surface_depth_draw_framebuffer", |raw| {
                raw.bind_framebuffer(Gl::DRAW_FRAMEBUFFER, None);
            })?;
            let mut conversion = Ok(());
            self.with_context("blit_surface_depth_color", |raw| {
                raw.disable(Gl::SCISSOR_TEST);
                if pass.srgb {
                    let full = fluxel_rhi_core::api::Rect {
                        x: 0,
                        y: 0,
                        width: pass.extent[0],
                        height: pass.extent[1],
                    };
                    conversion = super::surface_color::convert(
                        raw,
                        &pass.color,
                        None,
                        0,
                        pass.extent,
                        full,
                        true,
                    );
                } else {
                    raw.blit_framebuffer(
                        0,
                        pass.extent[1] as i32,
                        pass.extent[0] as i32,
                        0,
                        0,
                        0,
                        pass.extent[0] as i32,
                        pass.extent[1] as i32,
                        Gl::COLOR_BUFFER_BIT,
                        Gl::NEAREST,
                    );
                }
            })?;
            conversion?;
            self.with_context("release_surface_depth_framebuffer", |raw| {
                raw.bind_framebuffer(Gl::FRAMEBUFFER, None);
                raw.delete_texture(Some(&pass.color));
                raw.delete_framebuffer(Some(&pass.framebuffer));
            })
        })();
        // `with_context` can reject before it invokes its closure, for example
        // after context loss. Repeating WebGL deletion after an issued close is
        // harmless; leaving the only owned handles behind is not.
        if result.is_err() {
            self.destroy_surface_depth_pass(pass);
        } else {
            self.owner.state.borrow_mut().raster_extent = None;
        }
        result
    }

    /// Drops a failed surface-depth pass without copying its partial color
    /// image into the acquired default framebuffer.
    pub(crate) fn abort_surface_depth_raster(&mut self) {
        let Some(pass) = self.surface_depth_pass.take() else {
            return;
        };
        self.destroy_surface_depth_pass(pass);
    }

    fn destroy_surface_depth_pass(&mut self, pass: BrowserSurfaceDepthPass) {
        // It is called only after a synchronous owner call returned; no owner
        // borrow can legitimately remain live here. A regular borrow makes a
        // cleanup failure visible instead of silently dropping raw handles.
        let mut state = self.owner.state.borrow_mut();
        state.context.raw.bind_framebuffer(Gl::FRAMEBUFFER, None);
        state.context.raw.delete_texture(Some(&pass.color));
        state
            .context
            .raw
            .delete_framebuffer(Some(&pass.framebuffer));
        state.raster_extent = None;
    }
}
