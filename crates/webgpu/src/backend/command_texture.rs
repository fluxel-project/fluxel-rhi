//! Texture copy region and staging validation for the WebGPU command encoder.
use super::{
    Error, Result, WebGpu, WebGpuBuffer, WebGpuEncoder, api, error, field, js, number, object,
    unaligned,
};
use wasm_bindgen::JsValue;

pub(super) fn copy_origin(origin: [u32; 3]) -> Result<JsValue> {
    let value = object();
    field(&value, "x", number(origin[0]))?;
    field(&value, "y", number(origin[1]))?;
    field(&value, "z", number(origin[2]))?;
    Ok(value)
}

pub(super) fn copy_extent(extent: [u32; 3]) -> Result<JsValue> {
    let value = object();
    field(&value, "width", number(extent[0]))?;
    field(&value, "height", number(extent[1]))?;
    field(&value, "depthOrArrayLayers", number(extent[2]))?;
    Ok(value)
}

pub(super) fn validate_texture_region(region: &api::TextureRegion<'_, WebGpu>) -> Result<()> {
    // SAFETY: callers have checked the texture's device against this encoder.
    let desc = &unsafe { region.texture.native_ref() }.desc;
    if region.mip >= desc.mip_levels || region.extent.contains(&0) {
        return Err(Error::InvalidRange);
    }
    let format = super::super::resource::texture_format(desc.format)?;
    let depth = format.starts_with("depth");
    let stencil = format == "stencil8" || format.ends_with("stencil8");
    let valid_aspect = if depth && stencil {
        region.aspect == api::TextureAspects::DEPTH
            || region.aspect == api::TextureAspects::STENCIL
            || region.aspect == (api::TextureAspects::DEPTH | api::TextureAspects::STENCIL)
    } else if depth {
        region.aspect == api::TextureAspects::DEPTH
    } else if stencil {
        region.aspect == api::TextureAspects::STENCIL
    } else {
        region.aspect == api::TextureAspects::COLOR
    };
    if !valid_aspect {
        return Err(Error::InvalidRange);
    }
    let mip = |value: u32| value.checked_shr(region.mip).unwrap_or(0).max(1);
    let size = [
        mip(desc.extent[0]),
        mip(desc.extent[1]),
        if desc.dimension == api::TextureDimension::D3 {
            mip(desc.extent[2])
        } else {
            desc.array_layers
        },
    ];
    let (bw, bh) = if region.aspect == api::TextureAspects::COLOR {
        let (w, h, _) = texel_block(desc.format, region.aspect)?;
        (w, h)
    } else {
        (1, 1)
    };
    let physical = [
        size[0].div_ceil(bw) * bw,
        size[1].div_ceil(bh) * bh,
        size[2],
    ];
    if (0..3).any(|i| {
        region.origin[i]
            .checked_add(region.extent[i])
            .is_none_or(|end| end > physical[i])
    }) || region.origin[0] % bw != 0
        || region.origin[1] % bh != 0
        || region.extent[0] % bw != 0
        || region.extent[1] % bh != 0
    {
        return Err(Error::InvalidRange);
    }
    if (depth || stencil || desc.samples > 1)
        && (region.origin[0] != 0
            || region.origin[1] != 0
            || region.extent[0] != size[0]
            || region.extent[1] != size[1])
    {
        return Err(Error::InvalidRange);
    }
    Ok(())
}

pub(super) fn copy_texture_desc(region: &api::TextureRegion<'_, WebGpu>) -> Result<JsValue> {
    let value = object();
    // SAFETY: every caller is a CommandEncoder entry point that checked the
    // region's DeviceId against the encoder; the JS clone is retained by it.
    let texture = unsafe { region.texture.native_ref() }.raw.clone();
    field(&value, "texture", texture)?;
    field(&value, "mipLevel", number(region.mip))?;
    field(&value, "origin", copy_origin(region.origin)?)?;
    field(
        &value,
        "aspect",
        JsValue::from_str(
            if region.aspect == api::TextureAspects::COLOR
                || region.aspect == (api::TextureAspects::DEPTH | api::TextureAspects::STENCIL)
            {
                "all"
            } else if region.aspect == api::TextureAspects::DEPTH {
                "depth-only"
            } else if region.aspect == api::TextureAspects::STENCIL {
                "stencil-only"
            } else {
                return Err(Error::Unsupported("WebGPU texture copy aspect"));
            },
        ),
    )?;
    Ok(value)
}

pub(crate) fn texel_block(
    format: api::TextureFormat,
    aspect: api::TextureAspects,
) -> Result<(u32, u32, u64)> {
    use api::TextureFormat as F;
    if aspect == api::TextureAspects::STENCIL {
        return if matches!(
            format,
            F::Stencil8 | F::Depth24PlusStencil8 | F::Depth32FloatStencil8
        ) {
            Ok((1, 1, 1))
        } else {
            Err(Error::InvalidRange)
        };
    }
    if aspect == api::TextureAspects::DEPTH {
        return match format {
            F::Depth16Unorm => Ok((1, 1, 2)),
            F::Depth32Float | F::Depth32FloatStencil8 => Ok((1, 1, 4)),
            _ => Err(Error::Unsupported("WebGPU depth buffer copy format")),
        };
    }
    if aspect != api::TextureAspects::COLOR {
        return Err(Error::InvalidRange);
    }
    let bytes = match format {
        F::R8Unorm | F::R8Snorm | F::R8Uint | F::R8Sint => 1,
        F::Rg8Unorm
        | F::Rg8Snorm
        | F::Rg8Uint
        | F::Rg8Sint
        | F::R16Unorm
        | F::R16Snorm
        | F::R16Uint
        | F::R16Sint
        | F::R16Float => 2,
        F::Rgba8Unorm
        | F::Rgba8UnormSrgb
        | F::Rgba8Snorm
        | F::Rgba8Uint
        | F::Rgba8Sint
        | F::Bgra8Unorm
        | F::Bgra8UnormSrgb
        | F::Rg16Unorm
        | F::Rg16Snorm
        | F::Rg16Uint
        | F::Rg16Sint
        | F::Rg16Float
        | F::R32Uint
        | F::R32Sint
        | F::R32Float
        | F::Rgb10a2Unorm
        | F::Rgb10a2Uint
        | F::Rg11b10Ufloat
        | F::Rgb9e5Ufloat => 4,
        F::Rgba16Unorm
        | F::Rgba16Snorm
        | F::Rgba16Uint
        | F::Rgba16Sint
        | F::Rgba16Float
        | F::Rg32Uint
        | F::Rg32Sint
        | F::Rg32Float => 8,
        F::Rgba32Uint | F::Rgba32Sint | F::Rgba32Float => 16,
        F::Bc1RgbaUnorm
        | F::Bc1RgbaUnormSrgb
        | F::Bc4RUnorm
        | F::Bc4RSnorm
        | F::Etc2Rgb8Unorm
        | F::Etc2Rgb8UnormSrgb
        | F::Etc2Rgb8A1Unorm
        | F::Etc2Rgb8A1UnormSrgb
        | F::EacR11Unorm
        | F::EacR11Snorm => return Ok((4, 4, 8)),
        F::Bc2RgbaUnorm
        | F::Bc2RgbaUnormSrgb
        | F::Bc3RgbaUnorm
        | F::Bc3RgbaUnormSrgb
        | F::Bc5RgUnorm
        | F::Bc5RgSnorm
        | F::Bc6hRgbUfloat
        | F::Bc6hRgbFloat
        | F::Bc7RgbaUnorm
        | F::Bc7RgbaUnormSrgb
        | F::Etc2Rgba8Unorm
        | F::Etc2Rgba8UnormSrgb
        | F::EacRg11Unorm
        | F::EacRg11Snorm => return Ok((4, 4, 16)),
        F::Astc {
            block: [w, h],
            hdr: false,
            ..
        } if super::super::resource::texture_format(format).is_ok() => {
            return Ok((u32::from(w), u32::from(h), 16));
        }
        _ => return Err(Error::Unsupported("WebGPU texel copy format")),
    };
    Ok((1, 1, bytes))
}

pub(super) fn validate_texel_buffer(
    buffer: &WebGpuBuffer,
    layout: api::TexelCopyBufferLayout,
    region: &api::TextureRegion<'_, WebGpu>,
) -> Result<()> {
    // SAFETY: callers first check this region's DeviceId against the encoder;
    // this only reads immutable descriptor metadata during that call.
    let format = unsafe { region.texture.native_ref() }.desc.format;
    let (block_w, block_h, bytes_per_block) = texel_block(format, region.aspect)?;
    let [width, height, layers] = region.extent;
    let row_bytes = u64::from(width.div_ceil(block_w))
        .checked_mul(bytes_per_block)
        .ok_or(Error::InvalidRange)?;
    let stride = u64::from(layout.bytes_per_row);
    let rows = u64::from(layout.rows_per_image);
    if width == 0
        || height == 0
        || layers == 0
        || stride < row_bytes
        || rows < u64::from(height.div_ceil(block_h))
    {
        return Err(Error::InvalidRange);
    }
    let layer_offset = u64::from(layers - 1)
        .checked_mul(rows)
        .and_then(|n| n.checked_mul(stride))
        .ok_or(Error::InvalidRange)?;
    let last_row = u64::from(height.div_ceil(block_h) - 1)
        .checked_mul(stride)
        .ok_or(Error::InvalidRange)?;
    let end = layout
        .offset
        .checked_add(layer_offset)
        .and_then(|n| n.checked_add(last_row))
        .and_then(|n| n.checked_add(row_bytes))
        .ok_or(Error::InvalidRange)?;
    if end > buffer.size {
        return Err(Error::InvalidRange);
    }
    Ok(())
}

pub(super) fn copy_unaligned_rows(
    encoder: &mut WebGpuEncoder,
    buffer: &WebGpuBuffer,
    layout: api::TexelCopyBufferLayout,
    region: &api::TextureRegion<'_, WebGpu>,
    buffer_is_source: bool,
) -> Result<()> {
    // SAFETY: caller validated the texture device, format, ranges and buffer bounds.
    let texture = unsafe { region.texture.native_ref() };
    let (bw, bh, block_bytes) = texel_block(texture.desc.format, region.aspect)?;
    let row_bytes = u64::from(region.extent[0].div_ceil(bw)) * block_bytes;
    let rows = u64::from(region.extent[1].div_ceil(bh));
    let aligned_stride = (row_bytes + 255) & !255;
    let is_color = region.aspect == api::TextureAspects::COLOR;
    // Depth/stencil copies must cover a full subresource; color permits a
    // single block row per command, bounding staging allocation size.
    let staging_rows = if is_color { 1 } else { rows };
    let size = aligned_stride
        .checked_mul(staging_rows)
        .ok_or(Error::InvalidRange)?;
    let desc = object();
    field(&desc, "size", number(size as f64))?;
    field(&desc, "usage", number(4u32 | 8))?;
    let raw = js::call1(&encoder.device.0.device, "createBuffer", &desc).map_err(error)?;
    let staging = WebGpuBuffer {
        raw: raw.clone(),
        size,
        usage: api::BufferUsage::COPY_SRC.union(api::BufferUsage::COPY_DST),
        native_usage: 4 | 8,
        mapped: std::rc::Rc::new(std::cell::Cell::new(false)),
    };
    if !buffer_is_source && buffer.usage.contains(api::BufferUsage::MAP_READ) {
        let aligned = layout.offset % 4 == 0
            && u64::from(layout.bytes_per_row) % 4 == 0
            && row_bytes % 4 == 0;
        if !aligned {
            return Err(Error::Unsupported(
                "WebGPU MAP_READ buffer cannot preserve unaligned destination bytes",
            ));
        }
    }
    for layer in 0..region.extent[2] {
        for first_row in (0..rows).step_by(staging_rows as usize) {
            let row_count = staging_rows.min(rows - first_row);
            let move_rows = |encoder: &mut WebGpuEncoder, to_staging: bool| -> Result<()> {
                for row in 0..row_count {
                    let user_offset = layout.offset
                        + (u64::from(layer) * u64::from(layout.rows_per_image) + first_row + row)
                            * u64::from(layout.bytes_per_row);
                    let stage_offset = row * aligned_stride;
                    let (src, so, dst, doff) = if to_staging {
                        (buffer, user_offset, &staging, stage_offset)
                    } else {
                        (&staging, stage_offset, buffer, user_offset)
                    };
                    if so % 4 == 0 && doff % 4 == 0 && row_bytes % 4 == 0 {
                        js::call(
                            &encoder.raw,
                            "copyBufferToBuffer",
                            &[
                                src.raw.clone(),
                                number(so as f64),
                                dst.raw.clone(),
                                number(doff as f64),
                                number(row_bytes as f64),
                            ],
                        )
                        .map_err(error)?;
                    } else {
                        let mut done = 0u64;
                        while done < row_bytes {
                            let count = (1024 * 1024 - ((doff + done) % 4)).min(row_bytes - done);
                            unaligned::copy_word(
                                encoder,
                                src,
                                so + done,
                                dst,
                                doff + done,
                                count as usize,
                            )?;
                            done += count;
                        }
                    }
                }
                Ok(())
            };
            if buffer_is_source {
                move_rows(encoder, true)?;
            }
            let copy_buffer = object();
            field(&copy_buffer, "buffer", raw.clone())?;
            field(&copy_buffer, "offset", number(0u32))?;
            field(&copy_buffer, "bytesPerRow", number(aligned_stride as f64))?;
            field(&copy_buffer, "rowsPerImage", number(row_count as f64))?;
            let copy_texture = copy_texture_desc(region)?;
            field(
                &copy_texture,
                "origin",
                copy_origin([
                    region.origin[0],
                    region.origin[1] + first_row as u32 * bh,
                    region.origin[2] + layer,
                ])?,
            )?;
            let extent = copy_extent([region.extent[0], row_count as u32 * bh, 1])?;
            let (method, args) = if buffer_is_source {
                ("copyBufferToTexture", [copy_buffer, copy_texture, extent])
            } else {
                ("copyTextureToBuffer", [copy_texture, copy_buffer, extent])
            };
            js::call(&encoder.raw, method, &args).map_err(error)?;
            if !buffer_is_source {
                move_rows(encoder, false)?;
            }
        }
    }
    encoder
        .keep_alive
        .extend([raw, buffer.raw.clone(), texture.raw.clone()]);
    Ok(())
}
