//! Synchronous WebGPU descriptor validation before browser error delivery.

use fluxel_rhi_core::api;

use super::{
    instance::WebGpuDevice,
    js,
    resource::{WebGpuTexture, device_limit, texture_format, texture_usage},
};

pub(super) fn format_available(device: &WebGpuDevice, format: api::TextureFormat) -> bool {
    let Ok(name) = texture_format(format) else {
        return false;
    };
    let feature = if name.starts_with("bc") {
        Some("texture-compression-bc")
    } else if name.starts_with("etc2") || name.starts_with("eac") {
        Some("texture-compression-etc2")
    } else if name.starts_with("astc") {
        Some("texture-compression-astc")
    } else if matches!(format, api::TextureFormat::Depth32FloatStencil8) {
        Some("depth32float-stencil8")
    } else if matches!(
        format,
        api::TextureFormat::R16Unorm
            | api::TextureFormat::R16Snorm
            | api::TextureFormat::Rg16Unorm
            | api::TextureFormat::Rg16Snorm
            | api::TextureFormat::Rgba16Unorm
            | api::TextureFormat::Rgba16Snorm
    ) {
        Some("texture-formats-tier1")
    } else {
        None
    };
    feature.is_none_or(|name| js::has_feature(&device.0.device, name))
}

pub(super) fn compressed(format: api::TextureFormat) -> bool {
    texture_format(format).is_ok_and(|name| {
        name.starts_with("bc")
            || name.starts_with("etc2")
            || name.starts_with("eac")
            || name.starts_with("astc")
    })
}

pub(super) fn capabilities(
    device: &WebGpuDevice,
    format: api::TextureFormat,
) -> (bool, bool, bool, bool, bool) {
    use api::TextureFormat as F;
    let name = texture_format(format).unwrap_or("");
    let tier1 = js::has_feature(&device.0.device, "texture-formats-tier1");
    let depth = matches!(
        format,
        F::Depth16Unorm
            | F::Depth24Plus
            | F::Depth24PlusStencil8
            | F::Depth32Float
            | F::Depth32FloatStencil8
            | F::Stencil8
    );
    let integer = name.ends_with("uint") || name.ends_with("sint");
    let float32 = matches!(format, F::R32Float | F::Rg32Float | F::Rgba32Float);
    let filterable =
        !depth && !integer && (!float32 || js::has_feature(&device.0.device, "float32-filterable"));
    let render = if compressed(format) || matches!(format, F::Rgb9e5Ufloat) {
        false
    } else if matches!(format, F::R8Snorm | F::Rg8Snorm | F::Rgba8Snorm) {
        tier1
    } else if matches!(format, F::Rg11b10Ufloat) {
        js::has_feature(&device.0.device, "rg11b10ufloat-renderable")
    } else {
        true
    };
    let blendable = render
        && !depth
        && !integer
        && (!float32 || js::has_feature(&device.0.device, "float32-blendable"));
    let storage = matches!(
        format,
        F::R32Uint
            | F::R32Sint
            | F::R32Float
            | F::Rg32Uint
            | F::Rg32Sint
            | F::Rg32Float
            | F::Rgba8Unorm
            | F::Rgba8Snorm
            | F::Rgba8Uint
            | F::Rgba8Sint
            | F::Rgba16Uint
            | F::Rgba16Sint
            | F::Rgba16Float
            | F::Rgba32Uint
            | F::Rgba32Sint
            | F::Rgba32Float
    ) || (matches!(format, F::Bgra8Unorm)
        && js::has_feature(&device.0.device, "bgra8unorm-storage"))
        || (tier1
            && matches!(
                format,
                F::R8Unorm
                    | F::R8Snorm
                    | F::R8Uint
                    | F::R8Sint
                    | F::Rg8Unorm
                    | F::Rg8Snorm
                    | F::Rg8Uint
                    | F::Rg8Sint
                    | F::R16Unorm
                    | F::R16Snorm
                    | F::R16Uint
                    | F::R16Sint
                    | F::R16Float
                    | F::Rg16Unorm
                    | F::Rg16Snorm
                    | F::Rg16Uint
                    | F::Rg16Sint
                    | F::Rg16Float
                    | F::Rgba16Unorm
                    | F::Rgba16Snorm
                    | F::Rgb10a2Unorm
                    | F::Rgb10a2Uint
                    | F::Rg11b10Ufloat
            ));
    let multisample = render
        // Compatibility-mode devices do not guarantee multisampled integer,
        // R32Float or Rgba16Float attachments. Use enabled device facts, not
        // format renderability alone, for these standard core-only formats.
        && (!(integer || matches!(format, F::R32Float | F::Rgba16Float))
            || js::has_feature(&device.0.device, "core-features-and-limits"))
        && !matches!(
            format,
            F::R32Uint
                | F::R32Sint
                | F::Rg32Uint
                | F::Rg32Sint
                | F::Rg32Float
                | F::Rgba32Uint
                | F::Rgba32Sint
                | F::Rgba32Float
        );
    (filterable, blendable, storage, render, multisample)
}

fn max_mips(extent: [u32; 3]) -> u32 {
    let largest = extent.into_iter().max().unwrap_or(1);
    u32::BITS - largest.leading_zeros()
}

pub(super) fn compatible_view_format(base: api::TextureFormat, view: api::TextureFormat) -> bool {
    if base == view {
        return true;
    }
    let (Ok(base), Ok(view)) = (texture_format(base), texture_format(view)) else {
        return false;
    };
    base.strip_suffix("-srgb") == Some(view) || view.strip_suffix("-srgb") == Some(base)
}

/// Rejects descriptions which browsers would otherwise report only through an
/// asynchronous validation error.
pub(super) fn valid_texture_desc(device: &WebGpuDevice, desc: &api::TextureDesc) -> bool {
    if desc.extent.contains(&0)
        || desc.array_layers == 0
        || desc.mip_levels == 0
        || !format_available(device, desc.format)
        || texture_usage(desc.usage).is_err()
        || desc.view_formats.iter().any(|&format| {
            !format_available(device, format) || !compatible_view_format(desc.format, format)
        })
        || desc
            .view_formats
            .iter()
            .enumerate()
            .any(|(index, format)| desc.view_formats[..index].contains(format))
    {
        return false;
    }
    let (max_extent, max_layers, mip_extent) = match desc.dimension {
        api::TextureDimension::D1 => {
            if desc.extent[1] != 1 || desc.extent[2] != 1 || desc.array_layers != 1 {
                return false;
            }
            (
                [
                    device_limit(&device.0.device, "maxTextureDimension1D") as u32,
                    1,
                    1,
                ],
                1,
                [desc.extent[0], 1, 1],
            )
        }
        api::TextureDimension::D2 => {
            if desc.extent[2] != 1 {
                return false;
            }
            let limit = device_limit(&device.0.device, "maxTextureDimension2D") as u32;
            (
                [limit, limit, 1],
                device_limit(&device.0.device, "maxTextureArrayLayers") as u32,
                [desc.extent[0], desc.extent[1], 1],
            )
        }
        api::TextureDimension::D3 => {
            if desc.array_layers != 1 {
                return false;
            }
            let limit = device_limit(&device.0.device, "maxTextureDimension3D") as u32;
            ([limit, limit, limit], 1, desc.extent)
        }
    };
    if desc.extent[0] > max_extent[0]
        || desc.extent[1] > max_extent[1]
        || desc.extent[2] > max_extent[2]
        || desc.array_layers > max_layers
        || desc.mip_levels > max_mips(mip_extent)
    {
        return false;
    }
    let depth = matches!(
        desc.format,
        api::TextureFormat::Depth16Unorm
            | api::TextureFormat::Depth24Plus
            | api::TextureFormat::Depth24PlusStencil8
            | api::TextureFormat::Depth32Float
            | api::TextureFormat::Depth32FloatStencil8
            | api::TextureFormat::Stencil8
    );
    if (desc.dimension == api::TextureDimension::D1
        && (desc.mip_levels != 1
            || depth
            || desc.usage.contains(api::TextureUsage::RENDER_ATTACHMENT)))
        || (desc.dimension == api::TextureDimension::D3 && depth)
    {
        return false;
    }
    if !matches!(desc.samples, 1 | 4) {
        return false;
    }
    let (_, _, storage, render, multisample) = capabilities(device, desc.format);
    if (desc.usage.contains(api::TextureUsage::STORAGE_BINDING) && !storage)
        || (desc.usage.contains(api::TextureUsage::RENDER_ATTACHMENT) && !render)
        || (desc.samples > 1 && !multisample)
    {
        return false;
    }
    if compressed(desc.format) {
        let Ok((block_width, block_height, _)) =
            super::command::texel_block(desc.format, api::TextureAspects::COLOR)
        else {
            return false;
        };
        if !js::has_feature(&device.0.device, "texture-compression-unaligned")
            && (desc.extent[0] % block_width != 0 || desc.extent[1] % block_height != 0)
        {
            return false;
        }
        if desc.samples != 1
            || desc.dimension == api::TextureDimension::D1
            || desc.usage.contains(api::TextureUsage::RENDER_ATTACHMENT)
            || desc.usage.contains(api::TextureUsage::STORAGE_BINDING)
        {
            return false;
        }
        if desc.dimension == api::TextureDimension::D3 {
            let name = texture_format(desc.format).unwrap();
            let feature = if name.starts_with("bc") {
                "texture-compression-bc-sliced-3d"
            } else if name.starts_with("astc") {
                "texture-compression-astc-sliced-3d"
            } else {
                return false;
            };
            if !js::has_feature(&device.0.device, feature) {
                return false;
            }
        }
    }
    if desc.samples != 1
        && (desc.dimension != api::TextureDimension::D2
            || desc.mip_levels != 1
            || !desc.usage.contains(api::TextureUsage::RENDER_ATTACHMENT)
            || desc.usage.contains(api::TextureUsage::STORAGE_BINDING))
    {
        return false;
    }
    true
}

pub(super) fn supported_sample_counts(desc: &api::TextureDesc) -> Vec<u32> {
    if desc.dimension == api::TextureDimension::D2
        && desc.mip_levels == 1
        && desc.usage.contains(api::TextureUsage::RENDER_ATTACHMENT)
        && !desc.usage.contains(api::TextureUsage::STORAGE_BINDING)
    {
        vec![1, 4]
    } else {
        vec![1]
    }
}

pub(super) fn max_mips_for(desc: &api::TextureDesc) -> u32 {
    max_mips(match desc.dimension {
        api::TextureDimension::D1 => [desc.extent[0], 1, 1],
        api::TextureDimension::D2 => [desc.extent[0], desc.extent[1], 1],
        api::TextureDimension::D3 => desc.extent,
    })
}

pub(super) fn valid_view_desc(texture: &WebGpuTexture, desc: &api::TextureViewDesc) -> bool {
    let texture_desc = &texture.desc;
    if texture_format(desc.format).is_err()
        || (desc.format != texture_desc.format
            && (!texture_desc.view_formats.contains(&desc.format)
                || !compatible_view_format(texture_desc.format, desc.format)))
        || desc.mips.start >= desc.mips.end
        || desc.mips.end > texture_desc.mip_levels
        || desc.layers.start >= desc.layers.end
    {
        return false;
    }
    let is_depth = matches!(
        texture_desc.format,
        api::TextureFormat::Depth16Unorm
            | api::TextureFormat::Depth24Plus
            | api::TextureFormat::Depth24PlusStencil8
            | api::TextureFormat::Depth32Float
            | api::TextureFormat::Depth32FloatStencil8
    );
    let has_stencil = matches!(
        texture_desc.format,
        api::TextureFormat::Stencil8
            | api::TextureFormat::Depth24PlusStencil8
            | api::TextureFormat::Depth32FloatStencil8
    );
    let aspects_ok = if is_depth || has_stencil {
        (is_depth && desc.aspects == api::TextureAspects::DEPTH)
            || (has_stencil && desc.aspects == api::TextureAspects::STENCIL)
            || (is_depth
                && has_stencil
                && desc.aspects == (api::TextureAspects::DEPTH | api::TextureAspects::STENCIL))
    } else {
        desc.aspects == api::TextureAspects::COLOR
    };
    if !aspects_ok {
        return false;
    }
    if texture_desc.samples > 1 && desc.dimension != api::TextureViewDimension::D2 {
        return false;
    }
    if matches!(
        desc.dimension,
        api::TextureViewDimension::Cube | api::TextureViewDimension::CubeArray
    ) && texture_desc.extent[0] != texture_desc.extent[1]
    {
        return false;
    }
    match texture_desc.dimension {
        api::TextureDimension::D1 => {
            desc.dimension == api::TextureViewDimension::D1 && desc.layers == (0..1)
        }
        api::TextureDimension::D2 => {
            if desc.layers.end > texture_desc.array_layers {
                return false;
            }
            match desc.dimension {
                api::TextureViewDimension::D2 => desc.layers.end - desc.layers.start == 1,
                api::TextureViewDimension::D2Array => true,
                api::TextureViewDimension::Cube => desc.layers.end - desc.layers.start == 6,
                api::TextureViewDimension::CubeArray => {
                    (desc.layers.end - desc.layers.start) % 6 == 0
                }
                _ => false,
            }
        }
        api::TextureDimension::D3 => {
            desc.dimension == api::TextureViewDimension::D3 && desc.layers == (0..1)
        }
    }
}
