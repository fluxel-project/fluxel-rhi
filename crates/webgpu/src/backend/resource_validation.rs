//! Synchronous WebGPU descriptor validation before browser error delivery.

use fluxel_rhi_core::api;

use super::{
    instance::WebGpuDevice,
    resource::{WebGpuTexture, device_limit, texture_format, texture_usage},
};

fn max_mips(extent: [u32; 3]) -> u32 {
    let largest = extent.into_iter().max().unwrap_or(1);
    u32::BITS - largest.leading_zeros()
}

fn compatible_view_format(base: api::TextureFormat, view: api::TextureFormat) -> bool {
    base == view
        || matches!(
            (base, view),
            (
                api::TextureFormat::Rgba8Unorm,
                api::TextureFormat::Rgba8UnormSrgb
            ) | (
                api::TextureFormat::Rgba8UnormSrgb,
                api::TextureFormat::Rgba8Unorm
            ) | (
                api::TextureFormat::Bgra8Unorm,
                api::TextureFormat::Bgra8UnormSrgb
            ) | (
                api::TextureFormat::Bgra8UnormSrgb,
                api::TextureFormat::Bgra8Unorm
            )
        )
}

/// Rejects descriptions which browsers would otherwise report only through an
/// asynchronous validation error.
pub(super) fn valid_texture_desc(device: &WebGpuDevice, desc: &api::TextureDesc) -> bool {
    if desc.extent.contains(&0)
        || desc.array_layers == 0
        || desc.mip_levels == 0
        || texture_format(desc.format).is_err()
        || texture_usage(desc.usage).is_err()
        || desc.view_formats.iter().any(|&format| {
            texture_format(format).is_err() || !compatible_view_format(desc.format, format)
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
    if !matches!(desc.samples, 1 | 4) {
        return false;
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
    );
    let aspects_ok = if is_depth {
        matches!(desc.aspects, api::TextureAspects::DEPTH)
            || (texture_desc.format == api::TextureFormat::Depth24PlusStencil8
                && desc.aspects == api::TextureAspects::STENCIL)
    } else {
        desc.aspects == api::TextureAspects::COLOR
    };
    if !aspects_ok {
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
                api::TextureViewDimension::Cube | api::TextureViewDimension::CubeArray => {
                    desc.layers.start % 6 == 0 && (desc.layers.end - desc.layers.start) % 6 == 0
                }
                _ => false,
            }
        }
        api::TextureDimension::D3 => {
            desc.dimension == api::TextureViewDimension::D3 && desc.layers == (0..1)
        }
    }
}
