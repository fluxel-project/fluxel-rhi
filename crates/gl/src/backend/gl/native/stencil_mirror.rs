//! Refresh a private stencil sampling texture from its authoritative source.
use super::provider::NativeGlProvider;
use crate::backend::gl::api::{GlError, GlFamilyApi as _, GlTextureDimension, TextureId};
use glow::HasContext as _;

impl NativeGlProvider {
    pub(crate) fn requires_stencil_extract(&self) -> bool {
        matches!(
            self.discovery.context().profile(),
            crate::backend::gl::api::GlFamilyProfile::Embedded { .. }
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
        if view.format == crate::backend::gl::api::GlFormat::R8Uint {
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
            let read = self.gl.get_parameter_i32(glow::READ_FRAMEBUFFER_BINDING) as u32;
            let draw = self.gl.get_parameter_i32(glow::DRAW_FRAMEBUFFER_BINDING) as u32;
            let scissor = self.gl.is_enabled(glow::SCISSOR_TEST);
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
            let mut result = Ok(());
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
                    if self.gl.check_framebuffer_status(glow::READ_FRAMEBUFFER)
                        != glow::FRAMEBUFFER_COMPLETE
                    {
                        result = Err(GlError::Unsupported {
                            operation: OP,
                            reason: "stencil source framebuffer incomplete",
                        });
                        break;
                    }
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
                    if self.gl.check_framebuffer_status(glow::DRAW_FRAMEBUFFER)
                        != glow::FRAMEBUFFER_COMPLETE
                    {
                        result = Err(GlError::Unsupported {
                            operation: OP,
                            reason: "stencil mirror framebuffer incomplete",
                        });
                        break;
                    }
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
                    if let Err(error) = self.driver_error(OP) {
                        result = Err(error);
                        break;
                    }
                }
                if result.is_err() {
                    break;
                }
            }
            self.gl.bind_framebuffer(
                glow::READ_FRAMEBUFFER,
                std::num::NonZeroU32::new(read).map(glow::NativeFramebuffer),
            );
            self.gl.bind_framebuffer(
                glow::DRAW_FRAMEBUFFER,
                std::num::NonZeroU32::new(draw).map(glow::NativeFramebuffer),
            );
            if scissor {
                self.gl.enable(glow::SCISSOR_TEST);
            }
            self.gl.delete_framebuffer(a);
            self.gl.delete_framebuffer(b);
            result.and_then(|()| self.driver_error(OP))
        }
    }
}
