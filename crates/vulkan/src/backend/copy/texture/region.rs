//! Vulkan image subresource and buffer footprint validation.
use crate::backend::{Vulkan, VulkanTexture};
use ash::vk;
use fluxel_rhi_core::api::{self, Error, Result};

pub(super) fn block(
    format: api::TextureFormat,
    aspect: api::TextureAspects,
) -> Result<(u32, u32, u32)> {
    use api::TextureFormat::*;
    let bytes = match (format, aspect) {
        (Stencil8 | Depth24PlusStencil8 | Depth32FloatStencil8, api::TextureAspects::STENCIL) => 1,
        (Depth16Unorm, api::TextureAspects::DEPTH) => 2,
        (Depth32Float | Depth32FloatStencil8, api::TextureAspects::DEPTH) => 4,
        // The Vulkan formats chosen for Depth24Plus use a 32-bit packed depth
        // plane. Other backends may choose a different native representation.
        (Depth24Plus | Depth24PlusStencil8, api::TextureAspects::DEPTH) => 4,
        (R8Unorm | R8Snorm | R8Uint | R8Sint, api::TextureAspects::COLOR) => 1,
        (
            Rg8Unorm | Rg8Snorm | Rg8Uint | Rg8Sint | R16Unorm | R16Snorm | R16Uint | R16Sint
            | R16Float,
            api::TextureAspects::COLOR,
        ) => 2,
        (
            Rgba8Unorm | Rgba8UnormSrgb | Rgba8Snorm | Rgba8Uint | Rgba8Sint | Bgra8Unorm
            | Bgra8UnormSrgb | Rg16Unorm | Rg16Snorm | Rg16Uint | Rg16Sint | Rg16Float | R32Uint
            | R32Sint | R32Float | Rgb10a2Unorm | Rgb10a2Uint | Rg11b10Ufloat | Rgb9e5Ufloat,
            api::TextureAspects::COLOR,
        ) => 4,
        (
            Rgba16Unorm | Rgba16Snorm | Rgba16Uint | Rgba16Sint | Rgba16Float | Rg32Uint | Rg32Sint
            | Rg32Float | R64Uint,
            api::TextureAspects::COLOR,
        ) => 8,
        (Rgba32Uint | Rgba32Sint | Rgba32Float | Rg64Uint, api::TextureAspects::COLOR) => 16,
        (
            Bc1RgbaUnorm | Bc1RgbaUnormSrgb | Bc4RUnorm | Bc4RSnorm | Etc2Rgb8Unorm
            | Etc2Rgb8UnormSrgb | Etc2Rgb8A1Unorm | Etc2Rgb8A1UnormSrgb | EacR11Unorm | EacR11Snorm,
            api::TextureAspects::COLOR,
        ) => return Ok((4, 4, 8)),
        (
            Bc2RgbaUnorm | Bc2RgbaUnormSrgb | Bc3RgbaUnorm | Bc3RgbaUnormSrgb | Bc5RgUnorm
            | Bc5RgSnorm | Bc6hRgbUfloat | Bc6hRgbFloat | Bc7RgbaUnorm | Bc7RgbaUnormSrgb
            | Etc2Rgba8Unorm | Etc2Rgba8UnormSrgb | EacRg11Unorm | EacRg11Snorm,
            api::TextureAspects::COLOR,
        ) => return Ok((4, 4, 16)),
        (Astc { block, .. }, api::TextureAspects::COLOR) if block[0] != 0 && block[1] != 0 => {
            return Ok((block[0] as u32, block[1] as u32, 16));
        }
        _ => return Err(Error::InvalidRange),
    };
    Ok((1, 1, bytes))
}

pub(super) fn image_region(t: &VulkanTexture, r: &api::TextureRegion<'_, Vulkan>) -> Result<()> {
    let d = t.desc();
    if r.mip >= d.mip_levels || r.extent.contains(&0) {
        return Err(Error::InvalidRange);
    }
    let aspect_ok = match d.format {
        api::TextureFormat::Depth16Unorm
        | api::TextureFormat::Depth24Plus
        | api::TextureFormat::Depth32Float => r.aspect == api::TextureAspects::DEPTH,
        api::TextureFormat::Stencil8 => r.aspect == api::TextureAspects::STENCIL,
        api::TextureFormat::Depth24PlusStencil8 | api::TextureFormat::Depth32FloatStencil8 => {
            r.aspect == api::TextureAspects::DEPTH || r.aspect == api::TextureAspects::STENCIL
        }
        _ => r.aspect == api::TextureAspects::COLOR,
    };
    let bounds = [
        (d.extent[0] >> r.mip).max(1),
        (d.extent[1] >> r.mip).max(1),
        if d.dimension == api::TextureDimension::D3 {
            (d.extent[2] >> r.mip).max(1)
        } else {
            d.array_layers
        },
    ];
    if !aspect_ok
        || (0..3).any(|i| {
            r.origin[i]
                .checked_add(r.extent[i])
                .is_none_or(|v| v > bounds[i])
        })
    {
        return Err(Error::InvalidRange);
    }
    if let Ok((bw, bh, _)) = block(d.format, r.aspect) {
        for (i, b) in [(0, bw), (1, bh)] {
            if r.origin[i] % b != 0
                || (r.extent[i] % b != 0 && r.origin[i] + r.extent[i] != bounds[i])
            {
                return Err(Error::InvalidRange);
            }
        }
    }
    Ok(())
}

pub(super) fn layers(
    t: &VulkanTexture,
    r: &api::TextureRegion<'_, Vulkan>,
) -> vk::ImageSubresourceLayers {
    vk::ImageSubresourceLayers::default()
        .aspect_mask(if r.aspect == api::TextureAspects::COLOR {
            vk::ImageAspectFlags::COLOR
        } else if r.aspect == api::TextureAspects::DEPTH {
            vk::ImageAspectFlags::DEPTH
        } else {
            vk::ImageAspectFlags::STENCIL
        })
        .mip_level(r.mip)
        .base_array_layer(if t.desc().dimension == api::TextureDimension::D3 {
            0
        } else {
            r.origin[2]
        })
        .layer_count(if t.desc().dimension == api::TextureDimension::D3 {
            1
        } else {
            r.extent[2]
        })
}
pub(super) fn offset(t: &VulkanTexture, r: &api::TextureRegion<'_, Vulkan>) -> vk::Offset3D {
    vk::Offset3D {
        x: r.origin[0] as i32,
        y: r.origin[1] as i32,
        z: if t.desc().dimension == api::TextureDimension::D3 {
            r.origin[2] as i32
        } else {
            0
        },
    }
}
pub(super) fn extent(t: &VulkanTexture, r: &api::TextureRegion<'_, Vulkan>) -> vk::Extent3D {
    vk::Extent3D {
        width: r.extent[0],
        height: r.extent[1],
        depth: if t.desc().dimension == api::TextureDimension::D3 {
            r.extent[2]
        } else {
            1
        },
    }
}

pub(super) fn buffer_region(
    t: &VulkanTexture,
    r: &api::TextureRegion<'_, Vulkan>,
    layout: api::TexelCopyBufferLayout,
    size: u64,
) -> Result<vk::BufferImageCopy> {
    image_region(t, r)?;
    let (bw, bh, bytes) = block(t.desc().format, r.aspect)?;
    let rows = r.extent[1].div_ceil(bh);
    let row_bytes = r.extent[0]
        .div_ceil(bw)
        .checked_mul(bytes)
        .ok_or(Error::InvalidRange)?;
    let alignment = if r.aspect == api::TextureAspects::COLOR {
        bytes
    } else {
        4
    };
    if layout.offset % u64::from(alignment) != 0
        || layout.bytes_per_row < row_bytes
        || layout.bytes_per_row % bytes != 0
        || layout.rows_per_image < rows
    {
        return Err(Error::InvalidRange);
    }
    let stride = u64::from(layout.bytes_per_row)
        .checked_mul(u64::from(layout.rows_per_image))
        .ok_or(Error::InvalidRange)?;
    let end = layout
        .offset
        .checked_add(
            stride
                .checked_mul(u64::from(r.extent[2] - 1))
                .ok_or(Error::InvalidRange)?,
        )
        .and_then(|v| v.checked_add(u64::from(layout.bytes_per_row) * u64::from(rows - 1)))
        .and_then(|v| v.checked_add(u64::from(row_bytes)))
        .ok_or(Error::InvalidRange)?;
    if end > size {
        return Err(Error::InvalidRange);
    }
    let row_length = (layout.bytes_per_row / bytes)
        .checked_mul(bw)
        .ok_or(Error::InvalidRange)?;
    let image_height = layout
        .rows_per_image
        .checked_mul(bh)
        .ok_or(Error::InvalidRange)?;
    Ok(vk::BufferImageCopy::default()
        .buffer_offset(layout.offset)
        .buffer_row_length(row_length)
        .buffer_image_height(image_height)
        .image_subresource(layers(t, r))
        .image_offset(offset(t, r))
        .image_extent(extent(t, r)))
}
