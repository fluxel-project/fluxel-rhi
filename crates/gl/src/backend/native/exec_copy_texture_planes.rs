//! Framebuffer copies of selected depth or stencil texture planes.
use super::super::provider::NativeGlProvider;
use crate::backend::api::{
    GlError, GlFormat, GlTextureAspect, GlTextureDesc, GlTextureDimension, GlTextureRegion,
};
use glow::HasContext as _;

impl NativeGlProvider {
    pub(super) fn copy_packed_plane(
        &mut self,
        op: &'static str,
        src: GlTextureRegion,
        dst: GlTextureRegion,
        src_desc: GlTextureDesc,
        dst_desc: GlTextureDesc,
        src_tex: glow::NativeTexture,
        dst_tex: glow::NativeTexture,
    ) -> Result<(), GlError> {
        let mask = match src.subresource.aspect {
            GlTextureAspect::DepthOnly => glow::DEPTH_BUFFER_BIT,
            GlTextureAspect::StencilOnly => glow::STENCIL_BUFFER_BIT,
            _ => {
                return Err(Self::validation(
                    op,
                    "packed copy requires a single depth or stencil aspect",
                ));
            }
        };
        if matches!(src_desc.dimension, GlTextureDimension::D1)
            || matches!(dst_desc.dimension, GlTextureDimension::D1)
        {
            if self.framebuffer_texture_1d.is_none() {
                return Err(GlError::Unsupported {
                    operation: op,
                    reason: "1D framebuffer attachment entry point unavailable",
                });
            }
        }
        let coord = |n: u32| {
            i32::try_from(n).map_err(|_| Self::validation(op, "copy coordinate exceeds GLint"))
        };
        let sx = coord(src.origin[0])?;
        let sy = coord(src.origin[1])?;
        let dx = coord(dst.origin[0])?;
        let dy = coord(dst.origin[1])?;
        let w = coord(src.extent.width)?;
        let h = coord(src.extent.height)?;
        let sl = coord(src.subresource.mip_level)?;
        let dl = coord(dst.subresource.mip_level)?;
        let sz = coord(
            src.subresource
                .base_layer
                .checked_add(src.origin[2])
                .ok_or_else(|| Self::validation(op, "source layer overflow"))?,
        )?;
        let dz = coord(
            dst.subresource
                .base_layer
                .checked_add(dst.origin[2])
                .ok_or_else(|| Self::validation(op, "destination layer overflow"))?,
        )?;
        let layers = coord(src.extent.depth_or_layers)?;
        let sx2 = sx
            .checked_add(w)
            .ok_or_else(|| Self::validation(op, "source width overflow"))?;
        let sy2 = sy
            .checked_add(h)
            .ok_or_else(|| Self::validation(op, "source height overflow"))?;
        let dx2 = dx
            .checked_add(w)
            .ok_or_else(|| Self::validation(op, "destination width overflow"))?;
        let dy2 = dy
            .checked_add(h)
            .ok_or_else(|| Self::validation(op, "destination height overflow"))?;
        sz.checked_add(layers)
            .ok_or_else(|| Self::validation(op, "source layer overflow"))?;
        dz.checked_add(layers)
            .ok_or_else(|| Self::validation(op, "destination layer overflow"))?;
        if !matches!(
            src_desc.dimension,
            GlTextureDimension::D1
                | GlTextureDimension::D2
                | GlTextureDimension::D2Array
                | GlTextureDimension::D3
        ) || !matches!(
            dst_desc.dimension,
            GlTextureDimension::D1
                | GlTextureDimension::D2
                | GlTextureDimension::D2Array
                | GlTextureDimension::D3
        ) {
            return Err(GlError::Unsupported {
                operation: op,
                reason: "packed plane copy texture dimension",
            });
        }
        // SAFETY: validated live GL objects on the current owner context; both
        // scratch FBOs, prior bindings and scissor state are restored on exit.
        unsafe {
            let read = self
                .gl
                .create_framebuffer()
                .map_err(|_| GlError::OutOfMemory { operation: op })?;
            let draw = match self.gl.create_framebuffer() {
                Ok(v) => v,
                Err(_) => {
                    self.gl.delete_framebuffer(read);
                    return Err(GlError::OutOfMemory { operation: op });
                }
            };
            let old_read = self.gl.get_parameter_i32(glow::READ_FRAMEBUFFER_BINDING) as u32;
            let old_draw = self.gl.get_parameter_i32(glow::DRAW_FRAMEBUFFER_BINDING) as u32;
            let scissor = self.gl.is_enabled(glow::SCISSOR_TEST);
            self.gl.disable(glow::SCISSOR_TEST);
            self.gl.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(read));
            self.gl.read_buffer(glow::NONE);
            self.gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, Some(draw));
            self.gl.draw_buffers(&[glow::NONE]);
            let mut result = Ok(());
            for layer in 0..layers {
                for (target, desc, tex, level, z) in [
                    (glow::READ_FRAMEBUFFER, src_desc, src_tex, sl, sz + layer),
                    (glow::DRAW_FRAMEBUFFER, dst_desc, dst_tex, dl, dz + layer),
                ] {
                    let attachment = match desc.format {
                        GlFormat::Depth16Unorm
                        | GlFormat::Depth24Unorm
                        | GlFormat::Depth32Float => glow::DEPTH_ATTACHMENT,
                        GlFormat::Stencil8 => glow::STENCIL_ATTACHMENT,
                        _ => glow::DEPTH_STENCIL_ATTACHMENT,
                    };
                    match desc.dimension {
                        GlTextureDimension::D1 => {
                            self.framebuffer_texture_1d.expect("entry checked")(
                                target,
                                attachment,
                                glow::TEXTURE_1D,
                                tex.0.get(),
                                level,
                            )
                        }
                        GlTextureDimension::D2 => self.gl.framebuffer_texture_2d(
                            target,
                            attachment,
                            glow::TEXTURE_2D,
                            Some(tex),
                            level,
                        ),
                        _ => self.gl.framebuffer_texture_layer(
                            target,
                            attachment,
                            Some(tex),
                            level,
                            z,
                        ),
                    }
                    if self.gl.check_framebuffer_status(target) != glow::FRAMEBUFFER_COMPLETE {
                        result = Err(GlError::Unsupported {
                            operation: op,
                            reason: "packed plane framebuffer incomplete",
                        });
                        break;
                    }
                }
                if result.is_err() {
                    break;
                }
                self.gl
                    .blit_framebuffer(sx, sy, sx2, sy2, dx, dy, dx2, dy2, mask, glow::NEAREST);
                if let Err(e) = self.driver_error(op) {
                    result = Err(e);
                    break;
                }
            }
            self.gl.bind_framebuffer(
                glow::READ_FRAMEBUFFER,
                std::num::NonZeroU32::new(old_read).map(glow::NativeFramebuffer),
            );
            self.gl.bind_framebuffer(
                glow::DRAW_FRAMEBUFFER,
                std::num::NonZeroU32::new(old_draw).map(glow::NativeFramebuffer),
            );
            if scissor {
                self.gl.enable(glow::SCISSOR_TEST);
            }
            self.gl.delete_framebuffer(read);
            self.gl.delete_framebuffer(draw);
            result
        }
    }
}
