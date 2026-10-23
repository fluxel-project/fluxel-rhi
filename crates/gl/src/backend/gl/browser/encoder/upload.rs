//! Immediate WebGL2 texture upload lowering.
//!
//! Uploads cross the browser boundary synchronously and restore their pixel
//! store state before returning; this module never retains source commands.

use fluxel_rhi_core::api::{Error, Result};
use web_sys::WebGl2RenderingContext as Gl;

use super::super::owner::{BrowserBuffer, BrowserEncoder, BrowserTexture};

impl BrowserEncoder {
    /// Copies a buffer region using native pixel-unpack offsets, preserving Core's
    /// upper-left row order without mapping or reading GPU data on the CPU.
    pub(crate) fn copy_buffer_to_texture(
        &mut self,
        source: &BrowserBuffer,
        offset: u64,
        bytes_per_row: u32,
        rows_per_image: u32,
        destination: &BrowserTexture,
        mip: u32,
        origin: [u32; 3],
        extent: [u32; 3],
    ) -> Result<()> {
        use crate::backend::gl::api::GlTextureDimension;
        self.same_owner(&source.owner)?;
        self.same_owner(&destination.owner)?;
        let texture = destination.texture()?;
        let (format, kind) = super::super::format_map::upload_encoding(destination.desc.format)
            .ok_or(Error::Unsupported("WebGL2 buffer-to-texture format"))?;
        let bytes_per_pixel =
            if destination.desc.format == crate::backend::gl::api::GlFormat::Depth16Unorm {
                2
            } else {
                u64::from(
                    super::super::ordinary_format::encoding(destination.desc.format)
                        .ok_or(Error::Unsupported("WebGL2 buffer-to-texture format"))?
                        .bytes,
                )
            };
        if mip >= destination.desc.mip_level_count || extent.contains(&0) {
            return Err(Error::InvalidRange);
        }
        let width = destination
            .desc
            .extent
            .width
            .checked_shr(mip)
            .unwrap_or(0)
            .max(1);
        let height = destination
            .desc
            .extent
            .height
            .checked_shr(mip)
            .unwrap_or(0)
            .max(1);
        let layers = destination
            .desc
            .mip_extent(mip)
            .ok_or(Error::InvalidRange)?
            .depth_or_layers;
        let row_bytes = u64::from(extent[0])
            .checked_mul(bytes_per_pixel)
            .ok_or(Error::InvalidRange)?;
        if origin[0].checked_add(extent[0]).is_none_or(|x| x > width)
            || origin[1].checked_add(extent[1]).is_none_or(|y| y > height)
            || origin[2].checked_add(extent[2]).is_none_or(|z| z > layers)
            || u64::from(bytes_per_row) < row_bytes
            || rows_per_image < extent[1]
        {
            return Err(Error::InvalidRange);
        }
        let layer_stride = u64::from(bytes_per_row)
            .checked_mul(u64::from(rows_per_image))
            .ok_or(Error::InvalidRange)?;
        let end = offset
            .checked_add(
                layer_stride
                    .checked_mul(u64::from(extent[2] - 1))
                    .ok_or(Error::InvalidRange)?,
            )
            .and_then(|v| v.checked_add(u64::from(bytes_per_row) * u64::from(extent[1] - 1)))
            .and_then(|v| v.checked_add(row_bytes))
            .ok_or(Error::InvalidRange)?;
        if end > source.size || end > (1u64 << 53) {
            return Err(Error::InvalidRange);
        }
        let target = match destination.desc.dimension {
            GlTextureDimension::D2 if origin[2] == 0 && extent[2] == 1 => Gl::TEXTURE_2D,
            GlTextureDimension::D2Array => Gl::TEXTURE_2D_ARRAY,
            GlTextureDimension::D3 => Gl::TEXTURE_3D,
            GlTextureDimension::D2 => return Err(Error::InvalidRange),
            _ => return Err(Error::Unsupported("WebGL2 texture dimension")),
        };
        // WebGL requires the unpack offset to be aligned to the transfer type,
        // while Core permits arbitrary byte offsets. Copy each row on GPU to an
        // aligned scratch buffer when necessary; never map the source on CPU.
        let alignment = super::super::ordinary_format::type_bytes(kind) as u64;
        let unaligned = bytes_per_pixel >= 16
            || offset % alignment != 0
            || u64::from(bytes_per_row) % alignment != 0
            || layer_stride % alignment != 0;
        let mut scratch = None;
        if unaligned {
            let size = i32::try_from(row_bytes).map_err(|_| Error::InvalidRange)?;
            let allocated = self.with_context("allocate_unpack_scratch", |raw| {
                scratch = raw.create_buffer();
                if let Some(buffer) = &scratch {
                    raw.bind_buffer(Gl::COPY_WRITE_BUFFER, Some(buffer));
                    raw.buffer_data_with_i32(Gl::COPY_WRITE_BUFFER, size, Gl::STREAM_DRAW);
                    raw.bind_buffer(Gl::COPY_WRITE_BUFFER, None);
                }
            });
            if let Err(error) = allocated {
                let _ = self.with_context("cleanup_unpack_scratch", |raw| {
                    raw.bind_buffer(Gl::COPY_WRITE_BUFFER, None);
                    if let Some(buffer) = &scratch {
                        raw.delete_buffer(Some(buffer));
                    }
                });
                return Err(error);
            }
            if scratch.is_none() {
                return Err(Error::OutOfMemory);
            }
        }
        let prepared = self.with_context("prepare_buffer_to_texture", |raw| {
            raw.bind_texture(target, Some(texture));
            raw.bind_buffer(
                Gl::PIXEL_UNPACK_BUFFER,
                scratch.as_ref().or(Some(&source.raw)),
            );
            raw.pixel_storei(Gl::UNPACK_ALIGNMENT, 1);
            raw.pixel_storei(Gl::UNPACK_ROW_LENGTH, 0);
            raw.pixel_storei(Gl::UNPACK_IMAGE_HEIGHT, 0);
            raw.pixel_storei(Gl::UNPACK_SKIP_PIXELS, 0);
            raw.pixel_storei(Gl::UNPACK_SKIP_ROWS, 0);
            raw.pixel_storei(Gl::UNPACK_SKIP_IMAGES, 0);
            raw.pixel_storei(Gl::UNPACK_FLIP_Y_WEBGL, 0);
        });
        if let Err(error) = prepared {
            let _ = self.with_context("cleanup_unpack_scratch", |raw| {
                raw.bind_buffer(Gl::PIXEL_UNPACK_BUFFER, None);
                raw.pixel_storei(Gl::UNPACK_ALIGNMENT, 4);
                if let Some(buffer) = &scratch {
                    raw.delete_buffer(Some(buffer));
                }
            });
            return Err(error);
        }
        let mut js_failed = false;
        let upload = self.with_context("copy_buffer_to_texture", |raw| {
            for layer in 0..extent[2] {
                for row in 0..extent[1] {
                    let row_offset = offset
                        + u64::from(layer) * layer_stride
                        + u64::from(row) * u64::from(bytes_per_row);
                    let row_offset = if let Some(buffer) = &scratch {
                        raw.bind_buffer(Gl::COPY_READ_BUFFER, Some(&source.raw));
                        raw.bind_buffer(Gl::COPY_WRITE_BUFFER, Some(buffer));
                        raw.copy_buffer_sub_data_with_f64_and_f64_and_f64(
                            Gl::COPY_READ_BUFFER,
                            Gl::COPY_WRITE_BUFFER,
                            row_offset as f64,
                            0.0,
                            row_bytes as f64,
                        );
                        0
                    } else {
                        row_offset
                    };
                    let y = origin[1] + row;
                    let result = if target == Gl::TEXTURE_2D {
                        raw.tex_sub_image_2d_with_i32_and_i32_and_u32_and_type_and_f64(
                            target,
                            mip as i32,
                            origin[0] as i32,
                            y as i32,
                            extent[0] as i32,
                            1,
                            format,
                            kind,
                            row_offset as f64,
                        )
                    } else {
                        raw.tex_sub_image_3d_with_f64(
                            target,
                            mip as i32,
                            origin[0] as i32,
                            y as i32,
                            (origin[2] + layer) as i32,
                            extent[0] as i32,
                            1,
                            1,
                            format,
                            kind,
                            row_offset as f64,
                        )
                    };
                    if result.is_err() {
                        js_failed = true;
                        return;
                    }
                }
            }
        });
        let restore = self.with_context("restore_buffer_to_texture", |raw| {
            raw.bind_buffer(Gl::PIXEL_UNPACK_BUFFER, None);
            if let Some(buffer) = &scratch {
                raw.bind_buffer(Gl::COPY_READ_BUFFER, None);
                raw.bind_buffer(Gl::COPY_WRITE_BUFFER, None);
                raw.delete_buffer(Some(buffer));
            }
            raw.pixel_storei(Gl::UNPACK_ALIGNMENT, 4);
        });
        upload?;
        restore?;
        if js_failed {
            return Err(Error::Backend(
                "WebGL2 buffer-to-texture upload threw an exception".into(),
            ));
        }
        Ok(())
    }

    /// Uploads one color region into a 2D, 2D-array, or 3D immutable texture.
    ///
    /// Core row zero is stored at native texture row zero.
    pub(crate) fn write_texture_region(
        &mut self,
        destination: &BrowserTexture,
        mip: u32,
        origin: [u32; 3],
        extent: [u32; 3],
        bytes_per_row: u32,
        rows_per_image: u32,
        bytes: &[u8],
    ) -> Result<()> {
        use crate::backend::gl::api::GlTextureDimension;

        self.same_owner(&destination.owner)?;
        let destination_raw = destination.texture()?;
        let encoding = super::super::ordinary_format::encoding(destination.desc.format)
            .ok_or(Error::Unsupported("WebGL2 texture upload format"))?;
        if mip >= destination.desc.mip_level_count || extent.contains(&0) {
            return Err(Error::InvalidRange);
        }
        let row_bytes = (extent[0] as usize)
            .checked_mul(encoding.bytes as usize)
            .ok_or(Error::InvalidRange)?;
        let row_stride = bytes_per_row as usize;
        let layer_stride = row_stride
            .checked_mul(rows_per_image as usize)
            .ok_or(Error::InvalidRange)?;
        let required = layer_stride
            .checked_mul((extent[2] - 1) as usize)
            .and_then(|v| v.checked_add(row_stride.checked_mul((extent[1] - 1) as usize)?))
            .and_then(|v| v.checked_add(row_bytes))
            .ok_or(Error::InvalidRange)?;
        if row_stride < row_bytes || rows_per_image < extent[1] || bytes.len() < required {
            return Err(Error::InvalidRange);
        }
        let width = destination
            .desc
            .extent
            .width
            .checked_shr(mip)
            .unwrap_or(0)
            .max(1);
        let height = destination
            .desc
            .extent
            .height
            .checked_shr(mip)
            .unwrap_or(0)
            .max(1);
        let layers = destination
            .desc
            .mip_extent(mip)
            .ok_or(Error::InvalidRange)?
            .depth_or_layers;
        if origin[0].checked_add(extent[0]).is_none_or(|x| x > width)
            || origin[1].checked_add(extent[1]).is_none_or(|y| y > height)
            || origin[2].checked_add(extent[2]).is_none_or(|z| z > layers)
        {
            return Err(Error::InvalidRange);
        }
        let native_y = origin[1];
        let target = match destination.desc.dimension {
            GlTextureDimension::D2 => {
                if origin[2] != 0 || extent[2] != 1 {
                    return Err(Error::InvalidRange);
                }
                Gl::TEXTURE_2D
            }
            GlTextureDimension::D2Array => Gl::TEXTURE_2D_ARRAY,
            GlTextureDimension::D3 => Gl::TEXTURE_3D,
            _ => return Err(Error::Unsupported("WebGL2 texture dimension")),
        };
        let mut packed = Vec::with_capacity(
            row_bytes
                .checked_mul(extent[1] as usize)
                .and_then(|v| v.checked_mul(extent[2] as usize))
                .ok_or(Error::InvalidRange)?,
        );
        for layer in 0..extent[2] as usize {
            for row in 0..extent[1] as usize {
                let start = layer * layer_stride + row * row_stride;
                packed.extend_from_slice(&bytes[start..start + row_bytes]);
            }
        }
        let (pixel_format, pixel_type) = (encoding.pixel, encoding.ty);
        self.with_context("prepare_texture_region_upload", |raw| {
            raw.bind_texture(target, Some(destination_raw));
            raw.bind_buffer(Gl::PIXEL_UNPACK_BUFFER, None);
            raw.pixel_storei(Gl::UNPACK_ALIGNMENT, 1);
            raw.pixel_storei(Gl::UNPACK_FLIP_Y_WEBGL, 0);
        })?;
        let pixels = super::super::ordinary_format::typed_pixels(&packed, pixel_type)?;
        let mut js_failed = false;
        let upload = self.with_context("write_texture_region", |raw| {
            let result = match destination.desc.dimension {
                GlTextureDimension::D2 => raw
                    .tex_sub_image_2d_with_i32_and_i32_and_u32_and_type_and_opt_array_buffer_view(
                        target,
                        mip as i32,
                        origin[0] as i32,
                        native_y as i32,
                        extent[0] as i32,
                        extent[1] as i32,
                        pixel_format,
                        pixel_type,
                        Some(&pixels),
                    ),
                GlTextureDimension::D2Array | GlTextureDimension::D3 => raw
                    .tex_sub_image_3d_with_opt_array_buffer_view(
                        target,
                        mip as i32,
                        origin[0] as i32,
                        native_y as i32,
                        origin[2] as i32,
                        extent[0] as i32,
                        extent[1] as i32,
                        extent[2] as i32,
                        pixel_format,
                        pixel_type,
                        Some(&pixels),
                    ),
                _ => unreachable!(),
            };
            js_failed = result.is_err();
        });
        let restore = self.with_context("restore_texture_region_unpack_state", |raw| {
            raw.pixel_storei(Gl::UNPACK_FLIP_Y_WEBGL, 0);
            raw.pixel_storei(Gl::UNPACK_ALIGNMENT, 4);
        });
        upload?;
        restore?;
        if js_failed {
            return Err(Error::Backend(
                "WebGL2 texture upload threw an exception".into(),
            ));
        }
        Ok(())
    }
}
