//! Native compressed block uploads, preserving the caller's block orientation.

use super::super::owner::{BrowserBuffer, BrowserEncoder, BrowserTexture};
use crate::backend::api::{GlCompressedTextureFamily, GlFormat, GlTextureDimension};
use fluxel_rhi_core::api::{Error, Result};
use web_sys::WebGl2RenderingContext as Gl;

struct Transfer {
    target: u32,
    format: u32,
    rows: u32,
    row_bytes: u64,
    layer_stride: u64,
    required: u64,
}

fn validate(
    texture: &BrowserTexture,
    mip: u32,
    origin: [u32; 3],
    extent: [u32; 3],
    offset: u64,
    pitch: u32,
    rows_per_image: u32,
) -> Result<Transfer> {
    let info = texture
        .desc
        .format
        .compressed_info()
        .ok_or(Error::InvalidRange)?;
    let size = texture.desc.mip_extent(mip).ok_or(Error::InvalidRange)?;
    if extent.contains(&0)
        || origin[0]
            .checked_add(extent[0])
            .is_none_or(|n| n > size.width)
        || origin[1]
            .checked_add(extent[1])
            .is_none_or(|n| n > size.height)
        || origin[2]
            .checked_add(extent[2])
            .is_none_or(|n| n > size.depth_or_layers)
    {
        return Err(Error::InvalidRange);
    }
    let bw = u32::from(info.block_width);
    let bh = u32::from(info.block_height);
    if origin[0] % bw != 0
        || origin[1] % bh != 0
        || (extent[0] % bw != 0 && origin[0] + extent[0] != size.width)
        || (extent[1] % bh != 0 && origin[1] + extent[1] != size.height)
    {
        return Err(Error::InvalidRange);
    }
    let target = match texture.desc.dimension {
        GlTextureDimension::D2 if origin[2] == 0 && extent[2] == 1 => Gl::TEXTURE_2D,
        GlTextureDimension::D2Array
            if matches!(
                info.family,
                GlCompressedTextureFamily::Etc2 | GlCompressedTextureFamily::Eac
            ) =>
        {
            Gl::TEXTURE_2D_ARRAY
        }
        _ => return Err(Error::Unsupported("WebGL2 compressed texture target")),
    };
    let rows = extent[1].div_ceil(bh);
    // WEBGL_compressed_texture_s3tc only permits a non-block-sized
    // subimage dimension when it equals the entire mip dimension. This
    // block-row lowering cannot represent a partial final row in a larger
    // non-aligned mip; reject before issuing any upload in that case.
    if matches!(
        texture.desc.format,
        GlFormat::Bc1RgbUnorm
            | GlFormat::Bc1RgbaUnorm
            | GlFormat::Bc1RgbSrgb
            | GlFormat::Bc1RgbaSrgb
            | GlFormat::Bc2RgbaUnorm
            | GlFormat::Bc2RgbaSrgb
            | GlFormat::Bc3RgbaUnorm
            | GlFormat::Bc3RgbaSrgb
    ) && ((extent[0] % bw != 0 && extent[0] != size.width)
        || (extent[1] % bh != 0 && extent[1] != size.height))
    {
        return Err(Error::Unsupported("WebGL2 S3TC partial edge block upload"));
    }
    let row_bytes = u64::from(extent[0].div_ceil(bw)) * u64::from(info.block_bytes);
    if u64::from(pitch) < row_bytes
        || rows_per_image < rows
        || pitch % u32::from(info.block_bytes) != 0
    {
        return Err(Error::InvalidRange);
    }
    let layer_stride = u64::from(pitch) * u64::from(rows_per_image);
    let required = offset
        .checked_add(
            layer_stride
                .checked_mul(u64::from(extent[2] - 1))
                .ok_or(Error::InvalidRange)?,
        )
        .and_then(|x| x.checked_add(u64::from(pitch) * u64::from(rows - 1)))
        .and_then(|x| x.checked_add(row_bytes))
        .ok_or(Error::InvalidRange)?;
    if required > (1u64 << 53) || row_bytes * u64::from(rows) > i32::MAX as u64 {
        return Err(Error::InvalidRange);
    }
    Ok(Transfer {
        target,
        format: super::super::format_map::internal_format(texture.desc.format)
            .ok_or(Error::InvalidRange)?,
        rows,
        row_bytes,
        layer_stride,
        required,
    })
}

impl BrowserEncoder {
    pub(crate) fn write_compressed_texture(
        &mut self,
        destination: &BrowserTexture,
        mip: u32,
        origin: [u32; 3],
        extent: [u32; 3],
        offset: u64,
        pitch: u32,
        rows_per_image: u32,
        bytes: &[u8],
    ) -> Result<()> {
        self.same_owner(&destination.owner)?;
        let transfer = validate(
            destination,
            mip,
            origin,
            extent,
            offset,
            pitch,
            rows_per_image,
        )?;
        if transfer.required > bytes.len() as u64 {
            return Err(Error::InvalidRange);
        }
        let texture = destination.texture()?;
        self.with_context("compressed_texture_upload", |raw| {
            raw.bind_buffer(Gl::PIXEL_UNPACK_BUFFER, None);
            raw.bind_texture(transfer.target, Some(texture));
            for layer in 0..extent[2] {
                let mut packed =
                    Vec::with_capacity((transfer.row_bytes * u64::from(transfer.rows)) as usize);
                for row in 0..transfer.rows {
                    let start = (offset
                        + u64::from(layer) * transfer.layer_stride
                        + u64::from(row) * u64::from(pitch))
                        as usize;
                    packed.extend_from_slice(&bytes[start..start + transfer.row_bytes as usize]);
                }
                let data = js_sys::Uint8Array::from(packed.as_slice());
                if transfer.target == Gl::TEXTURE_2D {
                    raw.compressed_tex_sub_image_2d_with_js_u8_array(
                        transfer.target,
                        mip as i32,
                        origin[0] as i32,
                        origin[1] as i32,
                        extent[0] as i32,
                        extent[1] as i32,
                        transfer.format,
                        &data,
                    );
                } else {
                    raw.compressed_tex_sub_image_3d_with_js_u8_array(
                        transfer.target,
                        mip as i32,
                        origin[0] as i32,
                        origin[1] as i32,
                        (origin[2] + layer) as i32,
                        extent[0] as i32,
                        extent[1] as i32,
                        1,
                        transfer.format,
                        &data,
                    );
                }
            }
        })
    }

    pub(crate) fn copy_compressed_buffer_to_texture(
        &mut self,
        source: &BrowserBuffer,
        offset: u64,
        pitch: u32,
        rows_per_image: u32,
        destination: &BrowserTexture,
        mip: u32,
        origin: [u32; 3],
        extent: [u32; 3],
    ) -> Result<()> {
        self.same_owner(&source.owner)?;
        self.same_owner(&destination.owner)?;
        let transfer = validate(
            destination,
            mip,
            origin,
            extent,
            offset,
            pitch,
            rows_per_image,
        )?;
        if transfer.required > source.size {
            return Err(Error::InvalidRange);
        }
        let texture = destination.texture()?;
        let mut scratch = None;
        let prepare = self.with_context("prepare_compressed_pbo", |raw| {
            raw.bind_texture(transfer.target, Some(texture));
            scratch = raw.create_buffer();
            if let Some(buffer) = &scratch {
                raw.bind_buffer(Gl::COPY_WRITE_BUFFER, Some(buffer));
                raw.buffer_data_with_f64(
                    Gl::COPY_WRITE_BUFFER,
                    (transfer.row_bytes * u64::from(transfer.rows)) as f64,
                    Gl::STREAM_DRAW,
                );
            }
        });
        if let Err(error) = prepare {
            if let Some(buffer) = &scratch {
                let _ = self.with_context("release_compressed_pbo", |raw| {
                    raw.delete_buffer(Some(buffer))
                });
            }
            return Err(error);
        }
        let scratch = scratch.ok_or(Error::OutOfMemory)?;
        let result = self.with_context("compressed_pbo_upload", |raw| {
            raw.bind_buffer(Gl::COPY_READ_BUFFER, Some(&source.raw));
            raw.bind_buffer(Gl::COPY_WRITE_BUFFER, Some(&scratch));
            for layer in 0..extent[2] {
                for row in 0..transfer.rows {
                    let pos = offset
                        + u64::from(layer) * transfer.layer_stride
                        + u64::from(row) * u64::from(pitch);
                    raw.copy_buffer_sub_data_with_f64_and_f64_and_f64(
                        Gl::COPY_READ_BUFFER,
                        Gl::COPY_WRITE_BUFFER,
                        pos as f64,
                        (u64::from(row) * transfer.row_bytes) as f64,
                        transfer.row_bytes as f64,
                    );
                }
                raw.bind_buffer(Gl::PIXEL_UNPACK_BUFFER, Some(&scratch));
                if transfer.target == Gl::TEXTURE_2D {
                    raw.compressed_tex_sub_image_2d_with_i32_and_f64(
                        transfer.target,
                        mip as i32,
                        origin[0] as i32,
                        origin[1] as i32,
                        extent[0] as i32,
                        extent[1] as i32,
                        transfer.format,
                        (transfer.row_bytes * u64::from(transfer.rows)) as i32,
                        0.0,
                    );
                } else {
                    raw.compressed_tex_sub_image_3d_with_i32_and_f64(
                        transfer.target,
                        mip as i32,
                        origin[0] as i32,
                        origin[1] as i32,
                        (origin[2] + layer) as i32,
                        extent[0] as i32,
                        extent[1] as i32,
                        1,
                        transfer.format,
                        (transfer.row_bytes * u64::from(transfer.rows)) as i32,
                        0.0,
                    );
                }
            }
        });
        let cleanup = self.with_context("restore_compressed_pbo", |raw| {
            raw.bind_buffer(Gl::PIXEL_UNPACK_BUFFER, None);
            raw.delete_buffer(Some(&scratch));
        });
        result.and(cleanup)
    }
}
