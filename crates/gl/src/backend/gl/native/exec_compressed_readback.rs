//! Compressed texture block readback using GPU pixel-pack buffers.

use super::provider::NativeGlProvider;
use crate::backend::gl::api::{
    GlBufferRange, GlError, GlFamilyProfile, GlFormatResourceKind, GlPixelFormat, GlPixelLayout,
    GlTextureDesc, GlTextureDimension, GlTextureRegion,
};
use glow::HasContext as _;

impl NativeGlProvider {
    pub(super) fn copy_compressed_texture_to_buffer(
        &mut self,
        texture: glow::NativeTexture,
        desc: GlTextureDesc,
        source: GlTextureRegion,
        destination: glow::NativeBuffer,
        range: GlBufferRange,
        layout: GlPixelLayout,
    ) -> Result<(), GlError> {
        const OP: &str = "compressed-texture-to-buffer";
        if matches!(
            self.discovery.context().profile(),
            GlFamilyProfile::Embedded { .. }
        ) {
            return self.copy_compressed_texture_to_buffer_gles(
                texture,
                desc,
                source,
                destination,
                range,
                layout,
            );
        }
        let get = self.get_compressed_image.ok_or(GlError::Unsupported {
            operation: OP,
            reason: "glGetCompressedTexImage unavailable",
        })?;
        let info = desc
            .format
            .compressed_info()
            .ok_or_else(|| Self::validation(OP, "missing compression format"))?;
        if layout.format
            != (GlPixelFormat::Compressed {
                block_width: info.block_width,
                block_height: info.block_height,
                block_bytes: info.block_bytes,
            })
            || !self
                .discovery
                .formats()
                .get_for(GlFormatResourceKind::Texture, desc.format, 1)
                .is_some_and(|f| f.copy_source)
        {
            return Err(Self::validation(
                OP,
                "compressed layout or copy source unavailable",
            ));
        }
        let target = match desc.dimension {
            GlTextureDimension::D2 => glow::TEXTURE_2D,
            GlTextureDimension::D2Array => glow::TEXTURE_2D_ARRAY,
            _ => {
                return Err(GlError::Unsupported {
                    operation: OP,
                    reason: "compressed target dimension",
                });
            }
        };
        let mip = desc
            .mip_extent(source.subresource.mip_level)
            .ok_or_else(|| Self::validation(OP, "invalid mip"))?;
        let bw = u32::from(info.block_width);
        let bh = u32::from(info.block_height);
        let valid_axis = |origin: u32, extent: u32, size: u32, block: u32| {
            origin % block == 0 && (extent % block == 0 || origin.checked_add(extent) == Some(size))
        };
        if !valid_axis(source.origin[0], source.extent.width, mip.width, bw)
            || !valid_axis(source.origin[1], source.extent.height, mip.height, bh)
        {
            return Err(Self::validation(OP, "compressed copy is not block aligned"));
        }
        let end = layout
            .required_bytes(source)
            .map_err(|_| Self::validation(OP, "invalid block layout"))?;
        if end > range.size {
            return Err(Self::validation(OP, "destination too short"));
        }
        let mip_columns = u64::from(mip.width.div_ceil(bw));
        let mip_rows = u64::from(mip.height.div_ceil(bh));
        let block_bytes = u64::from(info.block_bytes);
        let mip_stride = mip_columns
            .checked_mul(mip_rows)
            .and_then(|x| x.checked_mul(block_bytes))
            .ok_or_else(|| Self::validation(OP, "mip size overflow"))?;
        let scratch_size = mip_stride
            .checked_mul(u64::from(mip.depth_or_layers))
            .and_then(|x| i32::try_from(x).ok())
            .ok_or_else(|| Self::validation(OP, "mip exceeds GL buffer range"))?;
        let max_dst = range
            .offset
            .checked_add(end)
            .and_then(|x| i32::try_from(x).ok())
            .ok_or_else(|| Self::validation(OP, "destination exceeds GL buffer range"))?;
        let _ = max_dst;
        let layers = source
            .subresource
            .base_layer
            .checked_add(source.origin[2])
            .ok_or_else(|| Self::validation(OP, "layer overflow"))?;
        let row_bytes = u64::from(source.extent.width.div_ceil(bw)) * block_bytes;
        let rows = source.extent.height.div_ceil(bh);
        let output_stride = u64::from(layout.bytes_per_row) * u64::from(layout.rows_per_image);
        let level = i32::try_from(source.subresource.mip_level)
            .map_err(|_| Self::validation(OP, "mip exceeds GLint"))?;
        // SAFETY: the context owns every object, the full mip fits the scratch
        // PBO and all row offsets are checked against the validated ranges.
        unsafe {
            let pack = self.gl.get_parameter_i32(glow::PIXEL_PACK_BUFFER_BINDING) as u32;
            let read = self.gl.get_parameter_i32(glow::COPY_READ_BUFFER_BINDING) as u32;
            let write = self.gl.get_parameter_i32(glow::COPY_WRITE_BUFFER_BINDING) as u32;
            let binding = self.gl.get_parameter_i32(if target == glow::TEXTURE_2D {
                glow::TEXTURE_BINDING_2D
            } else {
                glow::TEXTURE_BINDING_2D_ARRAY
            }) as u32;
            let scratch = self
                .gl
                .create_buffer()
                .map_err(|_| GlError::OutOfMemory { operation: OP })?;
            self.gl.bind_buffer(glow::PIXEL_PACK_BUFFER, Some(scratch));
            self.gl
                .buffer_data_size(glow::PIXEL_PACK_BUFFER, scratch_size, glow::STREAM_READ);
            self.gl.bind_texture(target, Some(texture));
            get(target, level, core::ptr::null_mut());
            let mut result = self.driver_error(OP);
            if result.is_ok() {
                self.gl.bind_buffer(glow::COPY_READ_BUFFER, Some(scratch));
                self.gl
                    .bind_buffer(glow::COPY_WRITE_BUFFER, Some(destination));
                for layer in 0..source.extent.depth_or_layers {
                    for row in 0..rows {
                        let src = (u64::from(layers + layer) * mip_stride)
                            + (u64::from(source.origin[1] / bh + row) * mip_columns
                                + u64::from(source.origin[0] / bw))
                                * block_bytes;
                        let dst = range.offset
                            + layout.offset
                            + u64::from(layer) * output_stride
                            + u64::from(row) * u64::from(layout.bytes_per_row);
                        self.gl.copy_buffer_sub_data(
                            glow::COPY_READ_BUFFER,
                            glow::COPY_WRITE_BUFFER,
                            src as i32,
                            dst as i32,
                            row_bytes as i32,
                        );
                    }
                }
                result = self.driver_error(OP);
            }
            self.gl.bind_texture(
                target,
                std::num::NonZeroU32::new(binding).map(glow::NativeTexture),
            );
            self.gl.bind_buffer(
                glow::PIXEL_PACK_BUFFER,
                std::num::NonZeroU32::new(pack).map(glow::NativeBuffer),
            );
            self.gl.bind_buffer(
                glow::COPY_READ_BUFFER,
                std::num::NonZeroU32::new(read).map(glow::NativeBuffer),
            );
            self.gl.bind_buffer(
                glow::COPY_WRITE_BUFFER,
                std::num::NonZeroU32::new(write).map(glow::NativeBuffer),
            );
            self.gl.delete_buffer(scratch);
            result.and_then(|()| self.driver_error(OP))
        }
    }
}

impl NativeGlProvider {
    fn copy_compressed_texture_to_buffer_gles(
        &mut self,
        texture: glow::NativeTexture,
        desc: GlTextureDesc,
        source: GlTextureRegion,
        destination: glow::NativeBuffer,
        range: GlBufferRange,
        layout: GlPixelLayout,
    ) -> Result<(), GlError> {
        use crate::backend::gl::api::GlFormat;
        const OP: &str = "compressed-texture-to-buffer";
        let info = desc
            .format
            .compressed_info()
            .ok_or_else(|| Self::validation(OP, "missing compressed format"))?;
        let format = |f| {
            self.discovery
                .formats()
                .get_for(GlFormatResourceKind::Texture, f, 1)
                .is_some_and(|v| v.renderable && v.copy_source)
        };
        let (internal, external, kind) = match info.block_bytes {
            8 if format(GlFormat::Rg32Uint) => (glow::RG32UI, glow::RG_INTEGER, glow::UNSIGNED_INT),
            8 if format(GlFormat::Rgba16Uint) => {
                (glow::RGBA16UI, glow::RGBA_INTEGER, glow::UNSIGNED_SHORT)
            }
            16 if format(GlFormat::Rgba32Uint) => {
                (glow::RGBA32UI, glow::RGBA_INTEGER, glow::UNSIGNED_INT)
            }
            _ => {
                return Err(GlError::Unsupported {
                    operation: OP,
                    reason: "integer block readback route unavailable",
                });
            }
        };
        if !self
            .discovery
            .formats()
            .get_for(GlFormatResourceKind::Texture, desc.format, 1)
            .is_some_and(|v| v.copy_source)
            || layout.format
                != (GlPixelFormat::Compressed {
                    block_width: info.block_width,
                    block_height: info.block_height,
                    block_bytes: info.block_bytes,
                })
        {
            return Err(Self::validation(
                OP,
                "compressed layout or copy source unavailable",
            ));
        }
        let target = match desc.dimension {
            GlTextureDimension::D2 => glow::TEXTURE_2D,
            GlTextureDimension::D2Array => glow::TEXTURE_2D_ARRAY,
            _ => return Err(Self::validation(OP, "compressed dimension")),
        };
        let mip = desc
            .mip_extent(source.subresource.mip_level)
            .ok_or_else(|| Self::validation(OP, "invalid mip"))?;
        let bw = u32::from(info.block_width);
        let bh = u32::from(info.block_height);
        let valid = |origin: u32, extent: u32, size: u32, block: u32| {
            origin % block == 0 && (extent % block == 0 || origin.checked_add(extent) == Some(size))
        };
        if !valid(source.origin[0], source.extent.width, mip.width, bw)
            || !valid(source.origin[1], source.extent.height, mip.height, bh)
        {
            return Err(Self::validation(OP, "compressed copy alignment"));
        }
        let end = layout
            .required_bytes(source)
            .map_err(|_| Self::validation(OP, "invalid layout"))?;
        if end > range.size
            || range
                .offset
                .checked_add(end)
                .is_none_or(|v| v > i32::MAX as u64)
        {
            return Err(Self::validation(OP, "destination range"));
        }
        let cols = source.extent.width.div_ceil(bw);
        let rows = source.extent.height.div_ceil(bh);
        let row_bytes = cols
            .checked_mul(u32::from(info.block_bytes))
            .and_then(|v| i32::try_from(v).ok())
            .ok_or_else(|| Self::validation(OP, "row exceeds GL range"))?;
        let cv = |v| i32::try_from(v).map_err(|_| Self::validation(OP, "value exceeds GLint"));
        let (cols_i, rows_i, width, height, sx, sy, level) = (
            cv(cols)?,
            cv(rows)?,
            cv(source.extent.width)?,
            cv(source.extent.height)?,
            cv(source.origin[0])?,
            cv(source.origin[1])?,
            cv(source.subresource.mip_level)?,
        );
        let z = source
            .subresource
            .base_layer
            .checked_add(source.origin[2])
            .ok_or_else(|| Self::validation(OP, "layer overflow"))?;
        cv(z.checked_add(source.extent.depth_or_layers - 1)
            .ok_or_else(|| Self::validation(OP, "layer overflow"))?)?;
        let stride = u64::from(layout.bytes_per_row) * u64::from(layout.rows_per_image);
        // SAFETY: all names belong to the current context, source/destination ranges
        // were validated by the caller, and each temporary object's lifetime spans
        // every recorded copy and read. GL retains deleted in-flight objects.
        unsafe {
            let old_texture = self.gl.get_parameter_i32(glow::TEXTURE_BINDING_2D) as u32;
            let old_fbo = self.gl.get_parameter_i32(glow::READ_FRAMEBUFFER_BINDING) as u32;
            let old_pack = self.gl.get_parameter_i32(glow::PIXEL_PACK_BUFFER_BINDING) as u32;
            let old_read = self.gl.get_parameter_i32(glow::COPY_READ_BUFFER_BINDING) as u32;
            let old_write = self.gl.get_parameter_i32(glow::COPY_WRITE_BUFFER_BINDING) as u32;
            let tmp = self
                .gl
                .create_texture()
                .map_err(|_| GlError::OutOfMemory { operation: OP })?;
            let fb = match self.gl.create_framebuffer() {
                Ok(v) => v,
                Err(_) => {
                    self.gl.delete_texture(tmp);
                    return Err(GlError::OutOfMemory { operation: OP });
                }
            };
            let pbo = match self.gl.create_buffer() {
                Ok(v) => v,
                Err(_) => {
                    self.gl.delete_framebuffer(fb);
                    self.gl.delete_texture(tmp);
                    return Err(GlError::OutOfMemory { operation: OP });
                }
            };
            self.gl.bind_texture(glow::TEXTURE_2D, Some(tmp));
            self.gl
                .tex_storage_2d(glow::TEXTURE_2D, 1, internal, cols_i, rows_i);
            self.gl.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(fb));
            self.gl.framebuffer_texture_2d(
                glow::READ_FRAMEBUFFER,
                glow::COLOR_ATTACHMENT0,
                glow::TEXTURE_2D,
                Some(tmp),
                0,
            );
            self.gl.read_buffer(glow::COLOR_ATTACHMENT0);
            self.gl.bind_buffer(glow::PIXEL_PACK_BUFFER, Some(pbo));
            self.gl
                .buffer_data_size(glow::PIXEL_PACK_BUFFER, row_bytes, glow::STREAM_READ);
            let saved = self.pixel_store;
            for (key, value) in [
                (glow::PACK_ALIGNMENT, 1),
                (glow::PACK_ROW_LENGTH, 0),
                (glow::PACK_SKIP_PIXELS, 0),
                (glow::PACK_SKIP_ROWS, 0),
            ] {
                self.gl.pixel_store_i32(key, value);
            }
            let mut result = self.driver_error(OP);
            if result.is_ok()
                && self.gl.check_framebuffer_status(glow::READ_FRAMEBUFFER)
                    != glow::FRAMEBUFFER_COMPLETE
            {
                result = Err(GlError::Unsupported {
                    operation: OP,
                    reason: "integer block framebuffer incomplete",
                });
            }
            for layer in 0..source.extent.depth_or_layers {
                if result.is_err() {
                    break;
                }
                let z = (z + layer) as i32;
                if let Some(copy) = self.copy_image_extension {
                    copy(
                        texture.0.get(),
                        target,
                        level,
                        sx,
                        sy,
                        z,
                        tmp.0.get(),
                        glow::TEXTURE_2D,
                        0,
                        0,
                        0,
                        0,
                        width,
                        height,
                        1,
                    );
                } else {
                    self.gl.copy_image_sub_data(
                        texture,
                        target,
                        level,
                        sx,
                        sy,
                        z,
                        tmp,
                        glow::TEXTURE_2D,
                        0,
                        0,
                        0,
                        0,
                        width,
                        height,
                        1,
                    );
                }
                result = self.driver_error(OP);
                if result.is_err() {
                    break;
                }
                self.gl.bind_buffer(glow::COPY_READ_BUFFER, Some(pbo));
                self.gl
                    .bind_buffer(glow::COPY_WRITE_BUFFER, Some(destination));
                for row in 0..rows {
                    self.gl.read_pixels(
                        0,
                        row as i32,
                        cols_i,
                        1,
                        external,
                        kind,
                        glow::PixelPackData::BufferOffset(0),
                    );
                    let offset = range.offset
                        + layout.offset
                        + u64::from(layer) * stride
                        + u64::from(row) * u64::from(layout.bytes_per_row);
                    self.gl.copy_buffer_sub_data(
                        glow::COPY_READ_BUFFER,
                        glow::COPY_WRITE_BUFFER,
                        0,
                        offset as i32,
                        row_bytes,
                    );
                }
                result = self.driver_error(OP);
            }
            for (key, value) in [
                (glow::PACK_ALIGNMENT, i32::from(saved.pack_alignment)),
                (glow::PACK_ROW_LENGTH, saved.pack_row_length as i32),
                (glow::PACK_SKIP_PIXELS, saved.pack_skip_pixels as i32),
                (glow::PACK_SKIP_ROWS, saved.pack_skip_rows as i32),
            ] {
                self.gl.pixel_store_i32(key, value);
            }
            self.gl.bind_texture(
                glow::TEXTURE_2D,
                std::num::NonZeroU32::new(old_texture).map(glow::NativeTexture),
            );
            self.gl.bind_framebuffer(
                glow::READ_FRAMEBUFFER,
                std::num::NonZeroU32::new(old_fbo).map(glow::NativeFramebuffer),
            );
            self.gl.bind_buffer(
                glow::PIXEL_PACK_BUFFER,
                std::num::NonZeroU32::new(old_pack).map(glow::NativeBuffer),
            );
            self.gl.bind_buffer(
                glow::COPY_READ_BUFFER,
                std::num::NonZeroU32::new(old_read).map(glow::NativeBuffer),
            );
            self.gl.bind_buffer(
                glow::COPY_WRITE_BUFFER,
                std::num::NonZeroU32::new(old_write).map(glow::NativeBuffer),
            );
            self.gl.delete_buffer(pbo);
            self.gl.delete_framebuffer(fb);
            self.gl.delete_texture(tmp);
            result.and_then(|()| self.driver_error(OP))
        }
    }
}
