//! Immediate texture-to-buffer copies through a GPU pixel-pack buffer.

use super::provider::NativeGlProvider;
use crate::backend::gl::api::{
    GlBufferRange, GlBufferUsage, GlError, GlFamilyApi as _, GlFormat, GlFormatResourceKind,
    GlPixelFormat, GlPixelLayout, GlTextureDimension, GlTextureRegion, GlTextureUsage,
};
use glow::HasContext as _;

impl NativeGlProvider {
    pub(crate) fn copy_texture_to_buffer(
        &mut self,
        source: GlTextureRegion,
        destination: GlBufferRange,
        layout: GlPixelLayout,
    ) -> Result<(), GlError> {
        const OP: &str = "texture-to-buffer";
        self.assert_ready(OP)?;
        let (texture, desc) = self.texture(OP, source.subresource.texture)?;
        let (buffer, buffer_desc) = self.buffer(OP, destination.buffer)?;
        source
            .validate_for(desc)
            .map_err(|_| Self::validation(OP, "invalid texture region"))?;
        destination
            .validate_for(buffer_desc)
            .map_err(|_| Self::validation(OP, "invalid buffer range"))?;
        if !desc.usage.contains(GlTextureUsage::COPY_SOURCE)
            || !buffer_desc.usage.contains(GlBufferUsage::COPY_DESTINATION)
        {
            return Err(Self::validation(OP, "copy usage is missing"));
        }
        if desc.format.compressed_info().is_some() {
            return self.copy_compressed_texture_to_buffer(
                texture,
                desc,
                source,
                buffer,
                destination,
                layout,
            );
        }
        let depth_format = matches!(desc.format, GlFormat::Depth16Unorm | GlFormat::Depth32Float);
        if depth_format
            && !matches!(
                self.discovery.context().profile(),
                crate::backend::gl::api::GlFamilyProfile::Desktop { .. }
            )
        {
            return Err(GlError::Unsupported {
                operation: OP,
                reason: "depth PBO readback requires desktop GL",
            });
        }
        let (read_format, read_type, pixel_format) = match desc.format {
            GlFormat::Depth16Unorm => (
                glow::DEPTH_COMPONENT,
                glow::UNSIGNED_SHORT,
                GlPixelFormat::Depth16,
            ),
            GlFormat::Depth32Float => (
                glow::DEPTH_COMPONENT,
                glow::FLOAT,
                GlPixelFormat::Depth32Float,
            ),
            GlFormat::Rgba8Unorm | GlFormat::Rgba8Srgb => {
                (glow::RGBA, glow::UNSIGNED_BYTE, GlPixelFormat::Rgba8)
            }
            format if super::probes::EXACT_INTEGER_FORMATS.contains(&format) => {
                let (external, kind) =
                    super::exec_copy::upload_encoding(format).ok_or(GlError::Unsupported {
                        operation: OP,
                        reason: "integer readback encoding",
                    })?;
                let pixel =
                    super::exec_copy::upload_pixel_format(format).ok_or(GlError::Unsupported {
                        operation: OP,
                        reason: "integer readback layout",
                    })?;
                (external, kind, pixel)
            }
            _ => {
                return Err(GlError::Unsupported {
                    operation: OP,
                    reason: "texture readback format",
                });
            }
        };
        if desc.sample_count != 1 || layout.format != pixel_format {
            return Err(GlError::Unsupported {
                operation: OP,
                reason: "texture readback format or sample count",
            });
        }
        if (!depth_format && !self.discovery.formats().exact_buffer_readback(desc.format))
            || !self
                .discovery
                .formats()
                .get_for(GlFormatResourceKind::Texture, desc.format, 1)
                .is_some_and(|facts| facts.renderable)
        {
            return Err(GlError::Unsupported {
                operation: OP,
                reason: "format lacks renderable copy source evidence",
            });
        }
        if !matches!(
            desc.dimension,
            GlTextureDimension::D1
                | GlTextureDimension::D2
                | GlTextureDimension::D2Array
                | GlTextureDimension::D3
        ) {
            return Err(GlError::Unsupported {
                operation: OP,
                reason: "texture readback dimension",
            });
        }
        if desc.dimension == GlTextureDimension::D1
            && (source.subresource.base_layer != 0
                || source.origin[1] != 0
                || source.origin[2] != 0
                || source.extent.height != 1
                || source.extent.depth_or_layers != 1)
        {
            return Err(Self::validation(OP, "invalid 1D texture readback region"));
        }
        if desc.dimension == GlTextureDimension::D1 && self.framebuffer_texture_1d.is_none() {
            return Err(GlError::Unsupported {
                operation: OP,
                reason: "glFramebufferTexture1D unavailable",
            });
        }
        let end = layout
            .required_bytes(source)
            .map_err(|_| Self::validation(OP, "invalid pixel layout"))?;
        if end > destination.size {
            return Err(Self::validation(
                OP,
                "destination buffer range is too short",
            ));
        }
        let end_offset = destination
            .offset
            .checked_add(end)
            .ok_or_else(|| Self::validation(OP, "buffer offset overflow"))?;
        if end_offset > u32::MAX as u64 {
            return Err(Self::validation(
                OP,
                "pixel buffer offset exceeds GL offset",
            ));
        }
        let z = source
            .subresource
            .base_layer
            .checked_add(source.origin[2])
            .ok_or_else(|| Self::validation(OP, "layer overflow"))?;
        let level = i32::try_from(source.subresource.mip_level)
            .map_err(|_| Self::validation(OP, "mip exceeds GLint"))?;
        let x =
            i32::try_from(source.origin[0]).map_err(|_| Self::validation(OP, "x exceeds GLint"))?;
        let width = i32::try_from(source.extent.width)
            .map_err(|_| Self::validation(OP, "width exceeds GLint"))?;
        let stride = u64::from(layout.bytes_per_row) * u64::from(layout.rows_per_image);
        let bytes_per_pixel = u64::from(layout.format.bytes_per_pixel());
        let scratch_needed = (destination.offset + layout.offset) % bytes_per_pixel != 0
            || u64::from(layout.bytes_per_row) % bytes_per_pixel != 0;
        let scratch_size = u64::from(source.extent.width)
            .checked_mul(bytes_per_pixel)
            .and_then(|x| i32::try_from(x).ok())
            .ok_or_else(|| Self::validation(OP, "readback row exceeds GL buffer size"))?;
        if scratch_needed && end_offset > i32::MAX as u64 {
            return Err(Self::validation(
                OP,
                "unaligned destination exceeds GL buffer copy range",
            ));
        }
        // SAFETY: all objects belong to the current context; every destination
        // row is within the validated buffer range. A temporary read FBO and
        // PACK state are restored even if completeness or a GL call fails.
        unsafe {
            let prior_fbo = self.gl.get_parameter_i32(glow::READ_FRAMEBUFFER_BINDING) as u32;
            let prior_buffer = self.gl.get_parameter_i32(glow::PIXEL_PACK_BUFFER_BINDING) as u32;
            let scratch = self
                .gl
                .create_framebuffer()
                .map_err(|_| GlError::OutOfMemory { operation: OP })?;
            let aligned = if scratch_needed {
                match self.gl.create_buffer() {
                    Ok(aligned) => {
                        self.gl.bind_buffer(glow::PIXEL_PACK_BUFFER, Some(aligned));
                        self.gl.buffer_data_size(
                            glow::PIXEL_PACK_BUFFER,
                            scratch_size,
                            glow::STREAM_READ,
                        );
                        Some(aligned)
                    }
                    Err(_) => {
                        self.gl.delete_framebuffer(scratch);
                        return Err(GlError::OutOfMemory { operation: OP });
                    }
                }
            } else {
                None
            };
            let saved = self.pixel_store;
            self.gl
                .bind_framebuffer(glow::READ_FRAMEBUFFER, Some(scratch));
            self.gl.read_buffer(if depth_format {
                glow::NONE
            } else {
                glow::COLOR_ATTACHMENT0
            });
            self.gl
                .bind_buffer(glow::PIXEL_PACK_BUFFER, Some(aligned.unwrap_or(buffer)));
            for (key, value) in [
                (glow::PACK_ALIGNMENT, 1),
                (glow::PACK_ROW_LENGTH, 0),
                (glow::PACK_SKIP_PIXELS, 0),
                (glow::PACK_SKIP_ROWS, 0),
            ] {
                self.gl.pixel_store_i32(key, value);
            }
            let mut result = Ok(());
            for layer in 0..source.extent.depth_or_layers {
                let attachment = if depth_format {
                    super::resource::depth_attachment_point(desc.format)
                } else {
                    Some(glow::COLOR_ATTACHMENT0)
                };
                let Some(attachment) = attachment else {
                    result = Err(GlError::Unsupported {
                        operation: OP,
                        reason: "depth attachment format",
                    });
                    break;
                };
                if desc.dimension == GlTextureDimension::D1 {
                    self.framebuffer_texture_1d
                        .expect("1D framebuffer entry point was checked before mutation")(
                        glow::READ_FRAMEBUFFER,
                        attachment,
                        glow::TEXTURE_1D,
                        texture.0.get(),
                        level,
                    );
                } else if desc.dimension == GlTextureDimension::D2 {
                    self.gl.framebuffer_texture_2d(
                        glow::READ_FRAMEBUFFER,
                        attachment,
                        glow::TEXTURE_2D,
                        Some(texture),
                        level,
                    );
                } else {
                    self.gl.framebuffer_texture_layer(
                        glow::READ_FRAMEBUFFER,
                        attachment,
                        Some(texture),
                        level,
                        (z + layer) as i32,
                    );
                }
                if self.gl.check_framebuffer_status(glow::READ_FRAMEBUFFER)
                    != glow::FRAMEBUFFER_COMPLETE
                {
                    result = Err(GlError::Unsupported {
                        operation: OP,
                        reason: "texture slice is not framebuffer complete",
                    });
                    break;
                }
                for row in 0..source.extent.height {
                    let offset = destination.offset
                        + layout.offset
                        + stride * u64::from(layer)
                        + u64::from(layout.bytes_per_row) * u64::from(row);
                    let y = (source.origin[1] + row) as i32;
                    self.gl.read_pixels(
                        x,
                        y,
                        width,
                        1,
                        read_format,
                        read_type,
                        glow::PixelPackData::BufferOffset(if aligned.is_some() {
                            0
                        } else {
                            offset as u32
                        }),
                    );
                    if let Some(aligned) = aligned {
                        self.gl.bind_buffer(glow::COPY_READ_BUFFER, Some(aligned));
                        self.gl.bind_buffer(glow::COPY_WRITE_BUFFER, Some(buffer));
                        self.gl.copy_buffer_sub_data(
                            glow::COPY_READ_BUFFER,
                            glow::COPY_WRITE_BUFFER,
                            0,
                            offset as i32,
                            scratch_size,
                        );
                    }
                }
                if let Err(error) = self.driver_error(OP) {
                    result = Err(error);
                    break;
                }
            }
            self.gl.bind_buffer(
                glow::PIXEL_PACK_BUFFER,
                std::num::NonZeroU32::new(prior_buffer).map(glow::NativeBuffer),
            );
            if let Some(aligned) = aligned {
                self.gl.delete_buffer(aligned);
            }
            for (key, value) in [
                (glow::PACK_ALIGNMENT, i32::from(saved.pack_alignment)),
                (glow::PACK_ROW_LENGTH, saved.pack_row_length as i32),
                (glow::PACK_SKIP_PIXELS, saved.pack_skip_pixels as i32),
                (glow::PACK_SKIP_ROWS, saved.pack_skip_rows as i32),
            ] {
                self.gl.pixel_store_i32(key, value);
            }
            self.gl.bind_framebuffer(
                glow::READ_FRAMEBUFFER,
                std::num::NonZeroU32::new(prior_fbo).map(glow::NativeFramebuffer),
            );
            self.gl.delete_framebuffer(scratch);
            result
        }
    }
}
