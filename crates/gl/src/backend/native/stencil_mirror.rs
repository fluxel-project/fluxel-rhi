//! Refresh a private stencil sampling texture from its authoritative source.
use super::provider::NativeGlProvider;
use crate::backend::api::{GlError, GlFamilyApi as _, GlTextureDimension, TextureId};
use crate::backend::state::{ScopedRawAccess, StateEvent};
use glow::HasContext as _;

impl NativeGlProvider {
    pub(crate) fn requires_stencil_extract(&self) -> bool {
        matches!(
            self.discovery.context().profile(),
            crate::backend::api::GlFamilyProfile::Embedded { .. }
        )
    }

    pub(crate) fn sync_stencil_mirror(
        &mut self,
        parent: TextureId,
        mirror: TextureId,
        base_mip: u32,
        base_layer: u32,
    ) -> Result<(), GlError> {
        const OP: &str = "sync stencil mirror";
        self.assert_ready(OP)?;
        let (src, source) = self.texture(OP, parent)?;
        let (dst, view) = self.texture(OP, mirror)?;
        if view.format == crate::backend::api::GlFormat::R8Uint {
            let query = self.suspend_occlusion_for_internal_draw()?;
            let result = self.extract_stencil_array(src, source, dst, view, base_mip, base_layer);
            if result.is_err() {
                self.invalidate_active_occlusion();
            }
            let resume = self.resume_occlusion_after_internal_draw(query);
            resume?;
            return result;
        }
        if !matches!(
            source.dimension,
            GlTextureDimension::D2 | GlTextureDimension::D2Array
        ) || !matches!(
            view.dimension,
            GlTextureDimension::D2 | GlTextureDimension::D2Array
        ) || base_layer
            .checked_add(view.extent.depth_or_layers)
            .is_none_or(|v| v > source.extent.depth_or_layers)
            || source.sample_count != 1
            || view.sample_count != 1
            || source.format != view.format
            || base_mip
                .checked_add(view.mip_level_count)
                .is_none_or(|v| v > source.mip_level_count)
        {
            return Err(GlError::Unsupported {
                operation: OP,
                reason: "stencil mirror texture shape",
            });
        }
        // SAFETY: the textures belong to the current context and the framebuffer
        // bindings and scissor enable are restored before returning.
        unsafe {
            let a = self
                .gl
                .create_framebuffer()
                .map_err(|message| GlError::Driver {
                    operation: OP,
                    message,
                })?;
            let b = match self.gl.create_framebuffer() {
                Ok(b) => b,
                Err(message) => {
                    self.gl.delete_framebuffer(a);
                    return Err(GlError::Driver {
                        operation: OP,
                        message,
                    });
                }
            };
            self.gl.disable(glow::SCISSOR_TEST);
            for mip in 0..view.mip_level_count {
                let extent = view.mip_extent(mip).expect("validated mip");
                for layer in 0..view.extent.depth_or_layers {
                    self.gl.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(a));
                    if source.dimension == GlTextureDimension::D2 {
                        self.gl.framebuffer_texture_2d(
                            glow::READ_FRAMEBUFFER,
                            glow::DEPTH_STENCIL_ATTACHMENT,
                            glow::TEXTURE_2D,
                            Some(src),
                            (base_mip + mip) as i32,
                        );
                    } else {
                        self.gl.framebuffer_texture_layer(
                            glow::READ_FRAMEBUFFER,
                            glow::DEPTH_STENCIL_ATTACHMENT,
                            Some(src),
                            (base_mip + mip) as i32,
                            (base_layer + layer) as i32,
                        );
                    }
                    self.gl.read_buffer(glow::NONE);
                    self.gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, Some(b));
                    if view.dimension == GlTextureDimension::D2 {
                        self.gl.framebuffer_texture_2d(
                            glow::DRAW_FRAMEBUFFER,
                            glow::DEPTH_STENCIL_ATTACHMENT,
                            glow::TEXTURE_2D,
                            Some(dst),
                            mip as i32,
                        );
                    } else {
                        self.gl.framebuffer_texture_layer(
                            glow::DRAW_FRAMEBUFFER,
                            glow::DEPTH_STENCIL_ATTACHMENT,
                            Some(dst),
                            mip as i32,
                            layer as i32,
                        );
                    }
                    self.gl.draw_buffers(&[glow::NONE]);
                    self.gl.blit_framebuffer(
                        0,
                        0,
                        extent.width as i32,
                        extent.height as i32,
                        0,
                        0,
                        extent.width as i32,
                        extent.height as i32,
                        glow::STENCIL_BUFFER_BIT,
                        glow::NEAREST,
                    );
                }
            }
            self.gl.bind_framebuffer(glow::READ_FRAMEBUFFER, None);
            self.gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, None);
            self.context_state
                .event(StateEvent::ScopedRawAccess(ScopedRawAccess::all()));
            self.gl.delete_framebuffer(a);
            self.gl.delete_framebuffer(b);
            // Descriptor/attachment compatibility is established before this
            // per-draw fallback. Re-querying FBO completeness per mip/layer
            // would synchronously observe the stream; report rejection at the
            // deferred publication boundary instead.
            self.defer_driver_error(OP)
        }
    }
}
