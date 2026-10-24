//! Standalone color resolve recorded as native render passes and exact transfers.
//!
//! WebGPU resolves whole attachments. A private full-mip scratch result allows
//! Core regions without changing pixels outside the destination rectangle.
use super::{
    WebGpuEncoder, WebGpuTexture, api, copy_extent, copy_texture_desc, error, field, js, number,
    object, texture_ops,
};
use fluxel_rhi_core::api::{Error, Result};
use js_sys::Array;
use wasm_bindgen::JsValue;

fn view(texture: &WebGpuTexture, mip: u32, layer: u32) -> Result<JsValue> {
    let desc = object();
    field(&desc, "dimension", JsValue::from_str("2d"))?;
    field(&desc, "baseMipLevel", number(mip))?;
    field(&desc, "mipLevelCount", number(1u32))?;
    field(&desc, "baseArrayLayer", number(layer))?;
    field(&desc, "arrayLayerCount", number(1u32))?;
    js::call1(&texture.raw, "createView", &desc).map_err(error)
}

pub(super) fn record(
    encoder: &mut WebGpuEncoder,
    src: api::TextureRegion<'_, super::WebGpu>,
    dst: api::TextureRegion<'_, super::WebGpu>,
) -> Result<()> {
    if encoder.pass.is_some() {
        return Err(Error::InvalidState);
    }
    // SAFETY: Core checked both texture identities against this encoder. Native
    // textures and their views are retained through submission completion below.
    let (source, target) = unsafe { (src.texture.native_ref(), dst.texture.native_ref()) };
    // The standard table distinguishes multisampling from resolve support.
    // Keep a positive list: integer, 32-bit float and 16-bit normalized formats
    // may admit MSAA attachments without admitting a native resolve target.
    use api::TextureFormat as F;
    if !matches!(
        source.desc.format,
        F::R8Unorm
            | F::R8Snorm
            | F::Rg8Unorm
            | F::Rg8Snorm
            | F::Rgba8Unorm
            | F::Rgba8UnormSrgb
            | F::Rgba8Snorm
            | F::Bgra8Unorm
            | F::Bgra8UnormSrgb
            | F::R16Float
            | F::Rg16Float
            | F::Rgba16Float
            | F::Rgb10a2Unorm
            | F::Rg11b10Ufloat
    ) {
        return Err(Error::Unsupported("WebGPU native color resolve format"));
    }
    if source.desc.format == F::Rgba16Float
        && !js::has_feature(&encoder.device.0.device, "core-features-and-limits")
    {
        return Err(Error::Unsupported("WebGPU core float16 resolve capability"));
    }
    if source.desc.dimension != api::TextureDimension::D2
        || target.desc.dimension != api::TextureDimension::D2
        || source.desc.samples <= 1
        || target.desc.samples != 1
        || source.desc.format != target.desc.format
        || src.aspect != api::TextureAspects::COLOR
        || dst.aspect != api::TextureAspects::COLOR
        || !source
            .desc
            .usage
            .contains(api::TextureUsage::RENDER_ATTACHMENT)
        || !(target
            .desc
            .usage
            .contains(api::TextureUsage::RENDER_ATTACHMENT)
            || target.desc.usage.contains(api::TextureUsage::COPY_DST))
    {
        return Err(Error::Unsupported("WebGPU color resolve format or usage"));
    }
    if src.extent != dst.extent
        || src.extent.contains(&0)
        || src.mip >= source.desc.mip_levels
        || dst.mip >= target.desc.mip_levels
        || js_sys::Object::is(&source.raw, &target.raw)
    {
        return Err(Error::InvalidRange);
    }
    let extent = |t: &WebGpuTexture, mip: u32| {
        [
            t.desc.extent[0].checked_shr(mip).unwrap_or(0).max(1),
            t.desc.extent[1].checked_shr(mip).unwrap_or(0).max(1),
            t.desc.array_layers,
        ]
    };
    let se = extent(source, src.mip);
    let de = extent(target, dst.mip);
    for (region, bounds) in [(&src, se), (&dst, de)] {
        if (0..3).any(|i| {
            region.origin[i]
                .checked_add(region.extent[i])
                .is_none_or(|end| end > bounds[i])
        }) {
            return Err(Error::InvalidRange);
        }
    }
    let direct = src.origin[..2] == [0, 0]
        && dst.origin[..2] == [0, 0]
        && src.extent[..2] == se[..2]
        && dst.extent[..2] == de[..2]
        && target
            .desc
            .usage
            .contains(api::TextureUsage::RENDER_ATTACHMENT);
    for layer in 0..src.extent[2] {
        let source_view = view(source, src.mip, src.origin[2] + layer)?;
        let (resolved_view, scratch) = if direct {
            (view(target, dst.mip, dst.origin[2] + layer)?, None)
        } else {
            let desc = object();
            field(&desc, "size", copy_extent([se[0], se[1], 1])?)?;
            field(
                &desc,
                "format",
                JsValue::from_str(super::super::resource::texture_format(source.desc.format)?),
            )?;
            // COPY_SRC | TEXTURE_BINDING | RENDER_ATTACHMENT.
            field(&desc, "usage", number(21u32))?;
            let texture =
                js::call1(&encoder.device.0.device, "createTexture", &desc).map_err(error)?;
            let resolved_view = js::call1(&texture, "createView", &object()).map_err(error)?;
            (resolved_view, Some(texture))
        };
        let attachment = object();
        field(&attachment, "view", source_view.clone())?;
        field(&attachment, "resolveTarget", resolved_view.clone())?;
        field(&attachment, "loadOp", JsValue::from_str("load"))?;
        // Standalone resolve must preserve the multisampled source.
        field(&attachment, "storeOp", JsValue::from_str("store"))?;
        let colors = Array::new();
        colors.push(&attachment);
        let desc = object();
        field(&desc, "colorAttachments", colors.into())?;
        let pass = js::call1(&encoder.raw, "beginRenderPass", &desc).map_err(error)?;
        js::call0(&pass, "end").map_err(error)?;
        if let Some(scratch) = scratch {
            if target.desc.usage.contains(api::TextureUsage::COPY_DST) {
                let from = object();
                field(&from, "texture", scratch.clone())?;
                field(
                    &from,
                    "origin",
                    texture_ops::copy_origin([src.origin[0], src.origin[1], 0])?,
                )?;
                let to = copy_texture_desc(&dst)?;
                field(
                    &to,
                    "origin",
                    texture_ops::copy_origin([
                        dst.origin[0],
                        dst.origin[1],
                        dst.origin[2] + layer,
                    ])?,
                )?;
                js::call(
                    &encoder.raw,
                    "copyTextureToTexture",
                    &[from, to, copy_extent([src.extent[0], src.extent[1], 1])?],
                )
                .map_err(error)?;
            } else {
                let destination_view = view(target, dst.mip, dst.origin[2] + layer)?;
                super::super::partial::finish_resolve_region(
                    encoder,
                    resolved_view.clone(),
                    destination_view.clone(),
                    source.desc.format,
                    api::Rect {
                        x: dst.origin[0],
                        y: dst.origin[1],
                        width: dst.extent[0],
                        height: dst.extent[1],
                    },
                    [src.origin[0], src.origin[1]],
                )?;
                encoder.keep_alive.push(destination_view);
            }
            encoder.keep_alive.push(scratch);
        }
        encoder.keep_alive.extend([source_view, resolved_view]);
    }
    encoder
        .keep_alive
        .extend([source.raw.clone(), target.raw.clone()]);
    Ok(())
}
