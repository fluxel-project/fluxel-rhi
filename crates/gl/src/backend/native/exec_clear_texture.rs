//! Temporary framebuffer clears for standalone texture subresources.
use super::provider::NativeGlProvider;
use crate::backend::api::{
    GlError, GlFamilyApi as _, GlFamilyProfile, GlFormat, GlFormatResourceKind, GlTextureDimension,
    TextureId,
};
use fluxel_rhi_core::api::{ClearColor, TextureClearValue};
use glow::HasContext as _;

impl NativeGlProvider {
    pub(crate) fn clear_texture_subresources(
        &mut self,
        id: TextureId,
        mips: std::ops::Range<u32>,
        layers: std::ops::Range<u32>,
        value: TextureClearValue,
    ) -> Result<(), GlError> {
        const OP: &str = "clear-texture";
        self.assert_ready(OP)?;
        let (texture, desc) = self.texture(OP, id)?;
        if !self
            .discovery
            .formats()
            .get_for(
                GlFormatResourceKind::Texture,
                desc.format,
                desc.sample_count,
            )
            .is_some_and(|f| f.renderable)
        {
            if let TextureClearValue::Color(color) = value {
                if desc.format.compressed_info().is_none() {
                    return if self.clear_texture_entry.is_some() {
                        self.clear_nonrenderable_color(texture, desc, mips, layers, color)
                    } else {
                        self.clear_color_upload(id, desc, mips, layers, color)
                    };
                }
            }
            return Err(GlError::Unsupported {
                operation: OP,
                reason: "format is not proved framebuffer renderable",
            });
        }
        if desc.dimension == GlTextureDimension::D1 && self.framebuffer_texture_1d.is_none() {
            return Err(GlError::Unsupported {
                operation: OP,
                reason: "glFramebufferTexture1D unavailable",
            });
        }
        self.context_state.pipeline_failed();
        // SAFETY: the current context owns the authenticated texture, and the caller validated each mip and layer.
        unsafe {
            let fbo = self
                .gl
                .create_framebuffer()
                .map_err(|_| GlError::OutOfMemory { operation: OP })?;
            self.gl.bind_framebuffer(glow::FRAMEBUFFER, Some(fbo));
            self.gl.disable(glow::SCISSOR_TEST);
            self.gl.color_mask(true, true, true, true);
            self.gl.depth_mask(true);
            if matches!(
                self.discovery.context().profile(),
                GlFamilyProfile::Desktop { .. }
            ) {
                if matches!(desc.format, GlFormat::Rgba8Srgb | GlFormat::Bgra8Srgb) {
                    self.gl.enable(glow::FRAMEBUFFER_SRGB);
                } else {
                    self.gl.disable(glow::FRAMEBUFFER_SRGB);
                }
            }
            let depth = matches!(value, TextureClearValue::Depth(_));
            let attachment = if depth {
                if matches!(
                    desc.format,
                    GlFormat::Depth24PlusStencil8 | GlFormat::Depth32FloatStencil8
                ) {
                    glow::DEPTH_STENCIL_ATTACHMENT
                } else {
                    glow::DEPTH_ATTACHMENT
                }
            } else {
                glow::COLOR_ATTACHMENT0
            };
            if depth {
                self.gl.draw_buffers(&[glow::NONE]);
                self.gl.read_buffer(glow::NONE);
            } else {
                self.gl.draw_buffers(&[glow::COLOR_ATTACHMENT0]);
            }
            let mut result = Ok(());
            'mips: for mip in mips {
                let slices = if desc.dimension == GlTextureDimension::D3 {
                    0..desc.mip_extent(mip).expect("validated mip").depth_or_layers
                } else {
                    layers.clone()
                };
                for layer in slices {
                    if desc.dimension == GlTextureDimension::D1 {
                        self.framebuffer_texture_1d
                            .expect("1D framebuffer entry point was checked before mutation")(
                            glow::FRAMEBUFFER,
                            attachment,
                            glow::TEXTURE_1D,
                            texture.0.get(),
                            mip as i32,
                        );
                    } else if desc.dimension == GlTextureDimension::D2 {
                        self.gl.framebuffer_texture_2d(
                            glow::FRAMEBUFFER,
                            attachment,
                            if desc.sample_count > 1 {
                                glow::TEXTURE_2D_MULTISAMPLE
                            } else {
                                glow::TEXTURE_2D
                            },
                            Some(texture),
                            mip as i32,
                        );
                    } else {
                        self.gl.framebuffer_texture_layer(
                            glow::FRAMEBUFFER,
                            attachment,
                            Some(texture),
                            mip as i32,
                            layer as i32,
                        );
                    }
                    if self.gl.check_framebuffer_status(glow::FRAMEBUFFER)
                        != glow::FRAMEBUFFER_COMPLETE
                    {
                        result = Err(Self::validation(OP, "clear framebuffer incomplete"));
                        break 'mips;
                    }
                    match value {
                        TextureClearValue::Depth(v) => {
                            self.gl.clear_buffer_f32_slice(glow::DEPTH, 0, &[v])
                        }
                        TextureClearValue::Color(ClearColor::Float(v)) => {
                            self.gl.clear_buffer_f32_slice(glow::COLOR, 0, &v)
                        }
                        TextureClearValue::Color(ClearColor::Uint(v)) => {
                            self.gl.clear_buffer_u32_slice(glow::COLOR, 0, &v)
                        }
                        TextureClearValue::Color(ClearColor::Sint(v)) => {
                            self.gl.clear_buffer_i32_slice(glow::COLOR, 0, &v)
                        }
                    }
                    if let Err(e) = self.driver_error(OP) {
                        result = Err(e);
                        break 'mips;
                    }
                }
            }
            self.gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            self.gl.delete_framebuffer(fbo);
            result.and_then(|()| self.driver_error(OP))
        }
    }
}
