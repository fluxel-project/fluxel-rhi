//! Clear texture formats that cannot be directly attached to a render pass.
use super::{Error, Result, WebGpuEncoder, WebGpuTexture, api, error, field, js, number, object};
use js_sys::{Array, Uint8Array};
use wasm_bindgen::JsValue;

pub(super) fn clear(
    encoder: &mut WebGpuEncoder,
    texture: &WebGpuTexture,
    mips: std::ops::Range<u32>,
    layers: std::ops::Range<u32>,
    value: api::TextureClearValue,
) -> Result<()> {
    let d = &texture.desc;
    let format = super::super::resource::texture_format(d.format)?;
    if !d.usage.contains(api::TextureUsage::COPY_DST)
        || mips.start >= mips.end
        || mips.end > d.mip_levels
        || layers.start >= layers.end
        || layers.end > d.array_layers
    {
        return Err(Error::InvalidRange);
    }
    let api::TextureClearValue::Color(color) = value else {
        return Err(Error::InvalidRange);
    };
    if (format.ends_with("uint") && !matches!(color, api::ClearColor::Uint(_)))
        || (format.ends_with("sint") && !matches!(color, api::ClearColor::Sint(_)))
        || (!format.ends_with("uint")
            && !format.ends_with("sint")
            && !matches!(color, api::ClearColor::Float(_)))
    {
        return Err(Error::InvalidRange);
    }
    let components: [f64; 4] = match color {
        api::ClearColor::Float(v) => v.map(f64::from),
        api::ClearColor::Uint(v) => v.map(f64::from),
        api::ClearColor::Sint(v) => v.map(f64::from),
    };
    let renderable = super::super::resource_validation::capabilities(&encoder.device, d.format).3;
    let encoded = if renderable {
        None
    } else {
        let api::ClearColor::Float(c) = color else {
            return Err(Error::InvalidRange);
        };
        let bytes = match format {
            "r8snorm" | "rg8snorm" | "rgba8snorm" => {
                let count = if format == "r8snorm" {
                    1
                } else if format == "rg8snorm" {
                    2
                } else {
                    4
                };
                c[..count]
                    .iter()
                    .map(|v| (v.clamp(-1.0, 1.0) * 127.0).round_ties_even() as i8 as u8)
                    .collect::<Vec<_>>()
            }
            "rgb9e5ufloat" => {
                let vals = [
                    c[0].max(0.0).min(65408.0),
                    c[1].max(0.0).min(65408.0),
                    c[2].max(0.0).min(65408.0),
                ];
                let max = vals[0].max(vals[1]).max(vals[2]);
                let mut exp = if max == 0.0 {
                    0
                } else {
                    (max.log2().floor() as i32 + 16).clamp(0, 31)
                };
                let mut scale = 2f32.powi(exp - 24);
                if (max / scale).round_ties_even() > 511.0 && exp < 31 {
                    exp += 1;
                    scale *= 2.0;
                }
                let m = vals.map(|v| (v / scale).round_ties_even().clamp(0.0, 511.0) as u32);
                (m[0] | (m[1] << 9) | (m[2] << 18) | ((exp as u32) << 27))
                    .to_le_bytes()
                    .to_vec()
            }
            _ => {
                return Err(Error::Unsupported(
                    "WebGPU texture clear requires format encoding",
                ));
            }
        };
        Some(bytes)
    };
    for mip in mips {
        let width = d.extent[0].checked_shr(mip).unwrap_or(0).max(1);
        let height = d.extent[1].checked_shr(mip).unwrap_or(0).max(1);
        let slices = if d.dimension == api::TextureDimension::D3 {
            0..d.extent[2].checked_shr(mip).unwrap_or(0).max(1)
        } else {
            layers.clone()
        };
        for layer in slices.clone() {
            if let Some(pixel) = &encoded {
                let row_size = u64::from(width) * pixel.len() as u64;
                let stride = (row_size + 255) & !255;
                let size = stride * u64::from(height);
                let mut bytes = vec![0u8; usize::try_from(size).map_err(|_| Error::OutOfMemory)?];
                for y in 0..height as usize {
                    for x in 0..width as usize {
                        let start = y * stride as usize + x * pixel.len();
                        bytes[start..start + pixel.len()].copy_from_slice(pixel);
                    }
                }
                let bd = object();
                field(&bd, "size", number(size as f64))?;
                field(&bd, "usage", number(4u32))?;
                field(&bd, "mappedAtCreation", JsValue::TRUE)?;
                let buf =
                    js::call1(&encoder.device.0.device, "createBuffer", &bd).map_err(error)?;
                let map = js::call0(&buf, "getMappedRange").map_err(error)?;
                Uint8Array::new(&map).copy_from(&bytes);
                js::call0(&buf, "unmap").map_err(error)?;
                let source = object();
                field(&source, "buffer", buf.clone())?;
                field(&source, "bytesPerRow", number(stride as f64))?;
                field(&source, "rowsPerImage", number(height))?;
                let dst = object();
                field(&dst, "texture", texture.raw.clone())?;
                field(&dst, "mipLevel", number(mip))?;
                let origin = object();
                field(&origin, "z", number(layer))?;
                field(&dst, "origin", origin)?;
                let extent = object();
                field(&extent, "width", number(width))?;
                field(&extent, "height", number(height))?;
                field(&extent, "depthOrArrayLayers", number(1u32))?;
                js::call(&encoder.raw, "copyBufferToTexture", &[source, dst, extent])
                    .map_err(error)?;
                encoder.keep_alive.push(buf);
            } else {
                let td = object();
                let size = object();
                field(&size, "width", number(width))?;
                field(&size, "height", number(1u32))?;
                field(&size, "depthOrArrayLayers", number(1u32))?;
                field(&td, "size", size)?;
                field(&td, "format", JsValue::from_str(format))?;
                field(&td, "usage", number(1u32 | 16))?;
                let temp =
                    js::call1(&encoder.device.0.device, "createTexture", &td).map_err(error)?;
                let view = js::call0(&temp, "createView").map_err(error)?;
                let att = object();
                field(&att, "view", view)?;
                field(&att, "loadOp", JsValue::from_str("clear"))?;
                field(&att, "storeOp", JsValue::from_str("store"))?;
                let v = object();
                for (key, c) in ["r", "g", "b", "a"].into_iter().zip(components) {
                    field(&v, key, JsValue::from_f64(c))?;
                }
                field(&att, "clearValue", v)?;
                let colors = Array::new();
                colors.push(&att);
                let pd = object();
                field(&pd, "colorAttachments", colors.into())?;
                let pass = js::call1(&encoder.raw, "beginRenderPass", &pd).map_err(error)?;
                js::call0(&pass, "end").map_err(error)?;
                let from = object();
                field(&from, "texture", temp.clone())?;
                let to = object();
                field(&to, "texture", texture.raw.clone())?;
                field(&to, "mipLevel", number(mip))?;
                let extent = object();
                field(&extent, "width", number(width))?;
                field(&extent, "height", number(1u32))?;
                field(&extent, "depthOrArrayLayers", number(1u32))?;
                js::call(&encoder.raw, "copyTextureToTexture", &[from, to, extent])
                    .map_err(error)?;
                encoder.keep_alive.push(temp);
            }
        }
    }
    encoder.keep_alive.push(texture.raw.clone());
    Ok(())
}
