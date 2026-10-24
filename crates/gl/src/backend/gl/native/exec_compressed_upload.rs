//! Direct compressed block uploads preserving the native block orientation.

use super::provider::NativeGlProvider;
use crate::backend::gl::api::{
    GlBufferRange, GlError, GlFormat, GlFormatResourceKind, GlPixelFormat, GlPixelLayout,
    GlTextureDesc, GlTextureDimension, GlTextureRegion, GlTextureUsage,
};
use glow::HasContext as _;

#[derive(Clone, Copy)]
pub(super) enum CompressedSource<'a> {
    Cpu(&'a [u8]),
    Buffer(glow::NativeBuffer, GlBufferRange),
}

impl NativeGlProvider {
    pub(super) fn upload_compressed_blocks(
        &mut self,
        texture: glow::NativeTexture,
        desc: GlTextureDesc,
        dest: GlTextureRegion,
        layout: GlPixelLayout,
        source: CompressedSource<'_>,
    ) -> Result<(), GlError> {
        const OP: &str = "compressed-upload";
        let info = desc
            .format
            .compressed_info()
            .ok_or_else(|| Self::validation(OP, "missing compressed format"))?;
        let expected = GlPixelFormat::Compressed {
            block_width: info.block_width,
            block_height: info.block_height,
            block_bytes: info.block_bytes,
        };
        if layout.format != expected
            || desc.sample_count != 1
            || !desc.usage.contains(GlTextureUsage::COPY_DESTINATION)
        {
            return Err(Self::validation(OP, "invalid compressed format or usage"));
        }
        if !self
            .discovery
            .formats()
            .get_for(GlFormatResourceKind::Texture, desc.format, 1)
            .is_some_and(|facts| facts.copy_destination)
        {
            return Err(GlError::Unsupported {
                operation: OP,
                reason: "compressed upload format unavailable",
            });
        }
        if !matches!(
            desc.dimension,
            GlTextureDimension::D2 | GlTextureDimension::D2Array
        ) && !(desc.dimension == GlTextureDimension::D3
            && desc.format.supports_compressed_3d(
                self.discovery.context().profile(),
                self.discovery.extensions(),
            ))
        {
            return Err(GlError::Unsupported {
                operation: OP,
                reason: "compressed upload texture dimension",
            });
        }
        dest.validate_for(desc)
            .map_err(|_| Self::validation(OP, "invalid compressed region"))?;
        let mip = desc
            .mip_extent(dest.subresource.mip_level)
            .ok_or_else(|| Self::validation(OP, "invalid mip"))?;
        let aligned = |origin: u32, extent: u32, full: u32, block: u8| {
            origin % u32::from(block) == 0
                && (extent % u32::from(block) == 0 || origin.checked_add(extent) == Some(full))
        };
        if !aligned(
            dest.origin[0],
            dest.extent.width,
            mip.width,
            info.block_width,
        ) || !aligned(
            dest.origin[1],
            dest.extent.height,
            mip.height,
            info.block_height,
        ) {
            return Err(Self::validation(OP, "compressed block alignment"));
        }
        if matches!(
            desc.format,
            GlFormat::Bc1RgbUnorm
                | GlFormat::Bc1RgbaUnorm
                | GlFormat::Bc1RgbSrgb
                | GlFormat::Bc1RgbaSrgb
                | GlFormat::Bc2RgbaUnorm
                | GlFormat::Bc2RgbaSrgb
                | GlFormat::Bc3RgbaUnorm
                | GlFormat::Bc3RgbaSrgb
        ) && ((dest.extent.width % u32::from(info.block_width) != 0
            && dest.extent.width != mip.width)
            || (dest.extent.height % u32::from(info.block_height) != 0
                && dest.extent.height != mip.height))
        {
            return Err(GlError::Unsupported {
                operation: OP,
                reason: "S3TC partial edge block upload",
            });
        }
        let needed = layout
            .required_bytes(dest)
            .map_err(|_| Self::validation(OP, "invalid block layout"))?;
        let base = match source {
            CompressedSource::Cpu(bytes) if needed <= bytes.len() as u64 => 0,
            CompressedSource::Buffer(_, range) if needed <= range.size => range.offset,
            _ => return Err(Self::validation(OP, "compressed source too short")),
        };
        let end = base
            .checked_add(needed)
            .ok_or_else(|| Self::validation(OP, "compressed source overflow"))?;
        if matches!(source, CompressedSource::Buffer(..)) && end > u64::from(u32::MAX) {
            return Err(Self::validation(OP, "pixel buffer offset exceeds GL range"));
        }
        let internal =
            super::resource::native_texture_format(desc.format).ok_or(GlError::Unsupported {
                operation: OP,
                reason: "compressed native format",
            })?;
        let width = i32::try_from(dest.extent.width)
            .map_err(|_| Self::validation(OP, "width exceeds GLint"))?;
        let x =
            i32::try_from(dest.origin[0]).map_err(|_| Self::validation(OP, "x exceeds GLint"))?;
        let level = i32::try_from(dest.subresource.mip_level)
            .map_err(|_| Self::validation(OP, "mip exceeds GLint"))?;
        let z = dest
            .subresource
            .base_layer
            .checked_add(dest.origin[2])
            .ok_or_else(|| Self::validation(OP, "layer overflow"))?;
        let rows = dest.extent.height.div_ceil(u32::from(info.block_height));
        let row_bytes =
            dest.extent.width.div_ceil(u32::from(info.block_width)) * u32::from(info.block_bytes);
        if row_bytes > i32::MAX as u32
            || dest.origin[1] + dest.extent.height > i32::MAX as u32
            || z + dest.extent.depth_or_layers > i32::MAX as u32
        {
            return Err(Self::validation(OP, "compressed geometry exceeds GLint"));
        }
        let tight_size = u64::from(row_bytes) * u64::from(rows);
        if tight_size > i32::MAX as u64
            || (matches!(source, CompressedSource::Buffer(..)) && end > i32::MAX as u64)
        {
            return Err(Self::validation(
                OP,
                "compressed transfer exceeds GL buffer range",
            ));
        }
        let mut packed = Vec::new();
        if matches!(source, CompressedSource::Cpu(_)) {
            packed
                .try_reserve_exact(tight_size as usize)
                .map_err(|_| Self::validation(OP, "compressed staging allocation failed"))?;
            packed.resize(tight_size as usize, 0);
        }
        let target = match desc.dimension {
            GlTextureDimension::D2 => glow::TEXTURE_2D,
            GlTextureDimension::D3 => glow::TEXTURE_3D,
            _ => glow::TEXTURE_2D_ARRAY,
        };
        let saved = self.pixel_store;
        // SAFETY: the context is current and the source, region, and packed
        // buffer bounds were checked above. GPU copies precede each upload in
        // the same context; the scratch buffer is deleted only after recording.
        unsafe {
            let unpack = self.gl.get_parameter_i32(glow::PIXEL_UNPACK_BUFFER_BINDING) as u32;
            let read = self.gl.get_parameter_i32(glow::COPY_READ_BUFFER_BINDING) as u32;
            let write = self.gl.get_parameter_i32(glow::COPY_WRITE_BUFFER_BINDING) as u32;
            let scratch = if let CompressedSource::Buffer(buffer, _) = source {
                let scratch = self.gl.create_buffer().map_err(|_| {
                    Self::validation(OP, "compressed staging buffer allocation failed")
                })?;
                self.gl.bind_buffer(glow::COPY_READ_BUFFER, Some(buffer));
                self.gl.bind_buffer(glow::COPY_WRITE_BUFFER, Some(scratch));
                self.gl.buffer_data_size(
                    glow::COPY_WRITE_BUFFER,
                    tight_size as i32,
                    glow::STREAM_DRAW,
                );
                Some(scratch)
            } else {
                None
            };
            self.gl.bind_buffer(glow::PIXEL_UNPACK_BUFFER, scratch);
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
            self.gl.bind_texture(target, Some(texture));
            for layer in 0..dest.extent.depth_or_layers {
                for row in 0..rows {
                    let offset = base
                        + layout.offset
                        + u64::from(layout.bytes_per_row)
                            * (u64::from(layout.rows_per_image) * u64::from(layer)
                                + u64::from(row));
                    let dst = u64::from(row) * u64::from(row_bytes);
                    match source {
                        CompressedSource::Cpu(bytes) => packed
                            [dst as usize..(dst + u64::from(row_bytes)) as usize]
                            .copy_from_slice(
                                &bytes[offset as usize..(offset + u64::from(row_bytes)) as usize],
                            ),
                        CompressedSource::Buffer(..) => self.copy_buffer_sub_data_wide(
                            glow::COPY_READ_BUFFER,
                            glow::COPY_WRITE_BUFFER,
                            offset,
                            dst,
                            u64::from(row_bytes),
                            OP,
                        )?,
                    }
                }
                let pixels = match source {
                    CompressedSource::Cpu(_) => glow::CompressedPixelUnpackData::Slice(&packed),
                    CompressedSource::Buffer(..) => {
                        glow::CompressedPixelUnpackData::BufferRange(0..tight_size as u32)
                    }
                };
                if target == glow::TEXTURE_2D {
                    self.gl.compressed_tex_sub_image_2d(
                        target,
                        level,
                        x,
                        dest.origin[1] as i32,
                        width,
                        dest.extent.height as i32,
                        internal,
                        pixels,
                    );
                } else {
                    self.gl.compressed_tex_sub_image_3d(
                        target,
                        level,
                        x,
                        dest.origin[1] as i32,
                        (z + layer) as i32,
                        width,
                        dest.extent.height as i32,
                        1,
                        internal,
                        pixels,
                    );
                }
            }
            let result = self.driver_error(OP);
            self.gl.bind_buffer(
                glow::PIXEL_UNPACK_BUFFER,
                std::num::NonZeroU32::new(unpack).map(glow::NativeBuffer),
            );
            self.gl.bind_buffer(
                glow::COPY_READ_BUFFER,
                std::num::NonZeroU32::new(read).map(glow::NativeBuffer),
            );
            self.gl.bind_buffer(
                glow::COPY_WRITE_BUFFER,
                std::num::NonZeroU32::new(write).map(glow::NativeBuffer),
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
