//! Immediate pixel-buffer uploads, preserving Core's upper-left row order.

use super::provider::NativeGlProvider;
use crate::backend::gl::api::{
    GlBufferRange, GlBufferUsage, GlError, GlFamilyApi as _, GlFormat, GlFormatResourceKind,
    GlPixelFormat, GlPixelLayout, GlTextureDimension, GlTextureRegion, GlTextureUsage,
};
use glow::HasContext as _;

impl NativeGlProvider {
    pub(super) fn upload_from_pixel_buffer(
        &mut self,
        source: GlBufferRange,
        dest: GlTextureRegion,
        layout: GlPixelLayout,
    ) -> Result<(), GlError> {
        const OP: &str = "buffer-to-texture";
        self.assert_ready(OP)?;
        let (buffer, buffer_desc) = self.buffer(OP, source.buffer)?;
        let (texture, desc) = self.texture(OP, dest.subresource.texture)?;
        source
            .validate_for(buffer_desc)
            .map_err(|_| Self::validation(OP, "invalid source buffer range"))?;
        dest.validate_for(desc)
            .map_err(|_| Self::validation(OP, "invalid destination texture region"))?;
        if !buffer_desc.usage.contains(GlBufferUsage::COPY_SOURCE)
            || !desc.usage.contains(GlTextureUsage::COPY_DESTINATION)
        {
            return Err(Self::validation(OP, "copy usage is missing"));
        }
        if desc.format.compressed_info().is_some() {
            return self.upload_compressed_blocks(
                texture,
                desc,
                dest,
                layout,
                super::exec_compressed_upload::CompressedSource::Buffer(buffer, source),
            );
        }
        if matches!(
            self.discovery.context().profile(),
            crate::backend::gl::api::GlFamilyProfile::Embedded { .. }
        ) && matches!(desc.format, GlFormat::Depth16Unorm | GlFormat::Depth32Float)
        {
            return self.upload_packed_plane(
                OP,
                texture,
                desc,
                dest,
                layout,
                super::exec_copy::packed_upload::PackedSource::Buffer(buffer, source),
            );
        }
        if matches!(
            desc.format,
            GlFormat::Depth24PlusStencil8 | GlFormat::Depth32FloatStencil8
        ) {
            return self.upload_packed_plane(
                OP,
                texture,
                desc,
                dest,
                layout,
                super::exec_copy::packed_upload::PackedSource::Buffer(buffer, source),
            );
        }
        if !matches!(
            layout.format,
            GlPixelFormat::Rgba8
                | GlPixelFormat::Red8
                | GlPixelFormat::Rg8
                | GlPixelFormat::Stencil8
                | GlPixelFormat::Depth16
                | GlPixelFormat::Depth32Float
                | GlPixelFormat::TexelBytes(2 | 4 | 8 | 16)
        ) || desc.sample_count != 1
        {
            return Err(GlError::Unsupported {
                operation: OP,
                reason: "pixel-buffer upload format or sample count",
            });
        }
        let (format, kind) =
            super::exec_copy::upload_encoding(desc.format).ok_or(GlError::Unsupported {
                operation: OP,
                reason: "pixel-buffer upload format",
            })?;
        if super::exec_copy::upload_pixel_format(desc.format) != Some(layout.format) {
            return Err(Self::validation(
                OP,
                "pixel layout does not match texture format",
            ));
        }
        if !self
            .discovery
            .formats()
            .get_for(GlFormatResourceKind::Texture, desc.format, 1)
            .is_some_and(|v| v.copy_destination)
        {
            return Err(GlError::Unsupported {
                operation: OP,
                reason: "format lacks copy destination evidence",
            });
        }
        if !matches!(
            desc.dimension,
            GlTextureDimension::D1
                | GlTextureDimension::D2
                | GlTextureDimension::D2Array
                | GlTextureDimension::D3
        ) || (desc.dimension == GlTextureDimension::D2
            && (dest.subresource.base_layer != 0
                || dest.origin[2] != 0
                || dest.extent.depth_or_layers != 1))
            || (desc.dimension == GlTextureDimension::D1
                && (dest.subresource.base_layer != 0
                    || dest.origin[1] != 0
                    || dest.origin[2] != 0
                    || dest.extent.height != 1
                    || dest.extent.depth_or_layers != 1))
        {
            return Err(GlError::Unsupported {
                operation: OP,
                reason: "pixel-buffer upload dimension",
            });
        }
        let width_bytes = u64::from(dest.extent.width) * u64::from(layout.format.bytes_per_pixel());
        if dest.extent.width == 0
            || dest.extent.height == 0
            || dest.extent.depth_or_layers == 0
            || u64::from(layout.bytes_per_row) < width_bytes
            || layout.rows_per_image < dest.extent.height
        {
            return Err(Self::validation(OP, "invalid pixel layout"));
        }
        let image_stride = u64::from(layout.bytes_per_row) * u64::from(layout.rows_per_image);
        let end = layout
            .offset
            .checked_add(
                image_stride
                    .checked_mul(u64::from(dest.extent.depth_or_layers - 1))
                    .ok_or_else(|| Self::validation(OP, "layout overflow"))?,
            )
            .and_then(|v| {
                v.checked_add(u64::from(layout.bytes_per_row) * u64::from(dest.extent.height - 1))
            })
            .and_then(|v| v.checked_add(width_bytes))
            .ok_or_else(|| Self::validation(OP, "layout overflow"))?;
        if end > source.size {
            return Err(Self::validation(OP, "source buffer range is too short"));
        }
        let last = source
            .offset
            .checked_add(end)
            .ok_or_else(|| Self::validation(OP, "buffer offset overflow"))?;
        if last > u32::MAX as u64 {
            return Err(Self::validation(
                OP,
                "pixel buffer offset exceeds GL offset",
            ));
        }
        if last > i32::MAX as u64
            && ((source.offset + layout.offset) % u64::from(layout.format.bytes_per_pixel()) != 0
                || u64::from(layout.bytes_per_row) % u64::from(layout.format.bytes_per_pixel())
                    != 0)
        {
            return Err(Self::validation(
                OP,
                "unaligned pixel buffer offset exceeds copy range",
            ));
        }
        let z = dest
            .subresource
            .base_layer
            .checked_add(dest.origin[2])
            .ok_or_else(|| Self::validation(OP, "invalid layer"))?;
        let level = i32::try_from(dest.subresource.mip_level)
            .map_err(|_| Self::validation(OP, "mip exceeds GLint"))?;
        let x =
            i32::try_from(dest.origin[0]).map_err(|_| Self::validation(OP, "x exceeds GLint"))?;
        let width = i32::try_from(dest.extent.width)
            .map_err(|_| Self::validation(OP, "width exceeds GLint"))?;
        let target = match desc.dimension {
            GlTextureDimension::D1 => glow::TEXTURE_1D,
            GlTextureDimension::D2 => glow::TEXTURE_2D,
            GlTextureDimension::D3 => glow::TEXTURE_3D,
            _ => glow::TEXTURE_2D_ARRAY,
        };
        if desc.dimension == GlTextureDimension::D1 && self.texture_sub_image_1d.is_none() {
            return Err(GlError::Unsupported {
                operation: OP,
                reason: "glTexSubImage1D unavailable",
            });
        }
        let saved = self.pixel_store;
        // SAFETY: context is current; resources, row bounds and coordinates were checked above.
        unsafe {
            let prior = self.gl.get_parameter_i32(glow::PIXEL_UNPACK_BUFFER_BINDING) as u32;
            // Some GLES implementations require a whole-texel PBO offset even
            // when GL's component alignment would allow a smaller one. GPU
            // copy to offset zero also handles any valid unaligned Core layout.
            let texel_bytes = u64::from(layout.format.bytes_per_pixel());
            let needs_aligned_scratch = (source.offset + layout.offset) % texel_bytes != 0
                || u64::from(layout.bytes_per_row) % texel_bytes != 0;
            let scratch = if needs_aligned_scratch {
                let size = i32::try_from(width_bytes)
                    .map_err(|_| Self::validation(OP, "row exceeds GL buffer size"))?;
                let scratch = self
                    .gl
                    .create_buffer()
                    .map_err(|_| GlError::OutOfMemory { operation: OP })?;
                self.gl.bind_buffer(glow::COPY_WRITE_BUFFER, Some(scratch));
                self.gl
                    .buffer_data_size(glow::COPY_WRITE_BUFFER, size, glow::STREAM_DRAW);
                Some(scratch)
            } else {
                None
            };
            for (key, value) in [
                (glow::UNPACK_ALIGNMENT, 1),
                (glow::UNPACK_ROW_LENGTH, 0),
                (glow::UNPACK_IMAGE_HEIGHT, 0),
                (glow::UNPACK_SKIP_PIXELS, 0),
                (glow::UNPACK_SKIP_ROWS, 0),
                (glow::UNPACK_SKIP_IMAGES, 0),
            ] {
                self.gl.pixel_store_i32(key, value);
            }
            self.gl
                .bind_buffer(glow::PIXEL_UNPACK_BUFFER, Some(scratch.unwrap_or(buffer)));
            self.gl.bind_texture(target, Some(texture));
            for layer in 0..dest.extent.depth_or_layers {
                for row in 0..dest.extent.height {
                    let offset = source.offset
                        + layout.offset
                        + image_stride * u64::from(layer)
                        + u64::from(layout.bytes_per_row) * u64::from(row);
                    let upload_offset = if let Some(scratch) = scratch {
                        self.gl.bind_buffer(glow::COPY_READ_BUFFER, Some(buffer));
                        self.gl.bind_buffer(glow::COPY_WRITE_BUFFER, Some(scratch));
                        if let Some(functions) = self.buffer_functions {
                            (functions.copy_buffer_sub_data)(
                                glow::COPY_READ_BUFFER,
                                glow::COPY_WRITE_BUFFER,
                                isize::try_from(offset).map_err(|_| {
                                    Self::validation(OP, "PBO offset exceeds native GLintptr")
                                })?,
                                0,
                                isize::try_from(width_bytes).map_err(|_| {
                                    Self::validation(OP, "PBO row exceeds native GLsizeiptr")
                                })?,
                            );
                        } else {
                            self.gl.copy_buffer_sub_data(
                                glow::COPY_READ_BUFFER,
                                glow::COPY_WRITE_BUFFER,
                                i32::try_from(offset).map_err(|_| {
                                    Self::validation(OP, "GL buffer ABI entry points unavailable")
                                })?,
                                0,
                                i32::try_from(width_bytes).map_err(|_| {
                                    Self::validation(OP, "GL buffer ABI entry points unavailable")
                                })?,
                            );
                        }
                        0
                    } else {
                        offset
                    };
                    let y = (dest.origin[1] + row) as i32;
                    if target == glow::TEXTURE_1D {
                        let entry = self
                            .buffer_functions
                            .and_then(|f| f.tex_sub_image_1d)
                            .or(self.texture_sub_image_1d)
                            .ok_or(GlError::Unsupported {
                                operation: OP,
                                reason: "glTexSubImage1D unavailable",
                            })?;
                        entry(
                            target,
                            level,
                            x,
                            width,
                            format,
                            kind,
                            usize::try_from(upload_offset).map_err(|_| {
                                Self::validation(OP, "PBO offset exceeds native address range")
                            })? as *const core::ffi::c_void,
                        );
                    } else if target == glow::TEXTURE_2D {
                        if let Some(entry) = self.buffer_functions.and_then(|f| f.tex_sub_image_2d)
                        {
                            entry(
                                target,
                                level,
                                x,
                                y,
                                width,
                                1,
                                format,
                                kind,
                                usize::try_from(upload_offset).map_err(|_| {
                                    Self::validation(OP, "PBO offset exceeds native address range")
                                })? as *const core::ffi::c_void,
                            );
                        } else {
                            self.gl.tex_sub_image_2d(
                                target,
                                level,
                                x,
                                y,
                                width,
                                1,
                                format,
                                kind,
                                glow::PixelUnpackData::BufferOffset(
                                    u32::try_from(upload_offset).map_err(|_| {
                                        Self::validation(
                                            OP,
                                            "wide PBO upload entry point unavailable",
                                        )
                                    })?,
                                ),
                            );
                        }
                    } else {
                        if let Some(entry) = self.buffer_functions.and_then(|f| f.tex_sub_image_3d)
                        {
                            entry(
                                target,
                                level,
                                x,
                                y,
                                (z + layer) as i32,
                                width,
                                1,
                                1,
                                format,
                                kind,
                                usize::try_from(upload_offset).map_err(|_| {
                                    Self::validation(OP, "PBO offset exceeds native address range")
                                })? as *const core::ffi::c_void,
                            );
                        } else {
                            self.gl.tex_sub_image_3d(
                                target,
                                level,
                                x,
                                y,
                                (z + layer) as i32,
                                width,
                                1,
                                1,
                                format,
                                kind,
                                glow::PixelUnpackData::BufferOffset(
                                    u32::try_from(upload_offset).map_err(|_| {
                                        Self::validation(
                                            OP,
                                            "wide PBO upload entry point unavailable",
                                        )
                                    })?,
                                ),
                            );
                        }
                    }
                }
            }
            let result = self.driver_error(OP);
            self.gl.bind_buffer(
                glow::PIXEL_UNPACK_BUFFER,
                std::num::NonZeroU32::new(prior).map(glow::NativeBuffer),
            );
            if let Some(scratch) = scratch {
                self.gl.delete_buffer(scratch);
            }
            for (key, value) in [
                (glow::UNPACK_ALIGNMENT, i32::from(saved.unpack_alignment)),
                (glow::UNPACK_ROW_LENGTH, saved.unpack_row_length as i32),
                (glow::UNPACK_IMAGE_HEIGHT, saved.unpack_image_height as i32),
                (glow::UNPACK_SKIP_PIXELS, saved.unpack_skip_pixels as i32),
                (glow::UNPACK_SKIP_ROWS, saved.unpack_skip_rows as i32),
                (glow::UNPACK_SKIP_IMAGES, saved.unpack_skip_images as i32),
            ] {
                self.gl.pixel_store_i32(key, value);
            }
            result.and_then(|()| self.driver_error(OP))
        }
    }
}
