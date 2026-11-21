//! Direct lowering of explicit buffer and texture copy commands.
//!
//! These operations do not infer or restore resource state. Callers transition
//! device-memory resources to `CopySrc` or `CopyDst` beforehand. A readback
//! buffer is the D3D12 exception: its committed heap has the fixed `COPY_DEST`
//! state and therefore needs no transition for a texture-to-buffer copy.

use std::mem::ManuallyDrop;

use fluxel_rhi_core::api::{self, Error, Result};
use windows::{
    Win32::Graphics::Direct3D12::{
        D3D12_BOX, D3D12_PLACED_SUBRESOURCE_FOOTPRINT, D3D12_SUBRESOURCE_FOOTPRINT,
    },
    core::Interface,
};

use super::{Dx12Encoder, copy_location};
use crate::backend::{Dx12Buffer, Dx12Texture};

fn same_resource(
    left: &windows::Win32::Graphics::Direct3D12::ID3D12Resource,
    right: &windows::Win32::Graphics::Direct3D12::ID3D12Resource,
) -> bool {
    left.as_raw() == right.as_raw()
}

pub(super) fn validate_region(
    texture: &Dx12Texture,
    region: &api::TextureRegion<'_, crate::backend::Dx12>,
) -> Result<u32> {
    let desc = &texture.0.desc;
    if region.mip >= desc.mip_levels {
        return Err(Error::InvalidRange);
    }
    if super::super::texture::bc_block_bytes(desc.format).is_some()
        && (region.origin[0] % 4 != 0
            || region.origin[1] % 4 != 0
            || (region.extent[0] % 4 != 0
                && region.origin[0].checked_add(region.extent[0])
                    != Some((desc.extent[0] >> region.mip).max(1)))
            || (region.extent[1] % 4 != 0
                && region.origin[1].checked_add(region.extent[1])
                    != Some((desc.extent[1] >> region.mip).max(1))))
    {
        return Err(Error::InvalidRange);
    }
    let depth = super::super::texture::is_depth(desc.format);
    let aspect_valid = if depth {
        matches!(
            region.aspect,
            api::TextureAspects::DEPTH | api::TextureAspects::STENCIL
        ) && super::super::texture::depth_aspects(desc.format).0 & region.aspect.0 != 0
    } else {
        region.aspect == api::TextureAspects::COLOR
    };
    if !aspect_valid
        || region.extent[0] == 0
        || region.extent[1] == 0
        || region.extent[2] == 0
        || region.mip >= desc.mip_levels
        || region.origin[0]
            .checked_add(region.extent[0])
            .is_none_or(|end| end > (desc.extent[0] >> region.mip).max(1))
        || region.origin[1]
            .checked_add(region.extent[1])
            .is_none_or(|end| end > (desc.extent[1] >> region.mip).max(1))
        || if desc.dimension == api::TextureDimension::D3 {
            region.origin[2]
                .checked_add(region.extent[2])
                .is_none_or(|end| end > (desc.extent[2] >> region.mip).max(1))
        } else {
            region.origin[2]
                .checked_add(region.extent[2])
                .is_none_or(|end| end > desc.array_layers)
        }
    {
        return Err(Error::InvalidRange);
    }
    if desc.dimension == api::TextureDimension::D3 {
        return Ok(region.mip);
    }
    region.origin[2]
        .checked_mul(desc.mip_levels)
        .and_then(|layer| layer.checked_add(region.mip))
        .and_then(|subresource| {
            if region.aspect == api::TextureAspects::STENCIL {
                subresource.checked_add(desc.mip_levels.checked_mul(desc.array_layers)?)
            } else {
                Some(subresource)
            }
        })
        .ok_or(Error::InvalidRange)
}

/// A native buffer footprint covering exactly the copied color rows plus the
/// caller's inter-image padding. Array images occupy consecutive row bands.
pub(super) struct ColorCopyFootprint {
    pub(super) placed: D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
    pub(super) byte_end: u64,
    pub(super) copy_height: u32,
    pub(super) array_layer_height: u32,
}

pub(super) fn color_copy_footprint(
    format: api::TextureFormat,
    dimension: api::TextureDimension,
    layout: api::TexelCopyBufferLayout,
    extent: [u32; 3],
    texel_aligned: bool,
) -> Result<ColorCopyFootprint> {
    use crate::backend::texture::{bc_block_bytes, color_texel_bytes, dxgi_format};
    use windows::Win32::Graphics::Direct3D12::{
        D3D12_TEXTURE_DATA_PITCH_ALIGNMENT, D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT,
    };
    let block = bc_block_bytes(format);
    let block_extent = if block.is_some() { 4 } else { 1 };
    let unit_bytes = block
        .or_else(|| color_texel_bytes(format))
        .ok_or(Error::Unsupported("DX12 color texture copy format"))?;
    let columns = extent[0].div_ceil(block_extent);
    let rows = extent[1].div_ceil(block_extent);
    let tight = columns.checked_mul(unit_bytes).ok_or(Error::InvalidRange)?;
    let (offset_alignment, pitch_alignment) = if texel_aligned {
        (unit_bytes, unit_bytes)
    } else {
        (
            D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT,
            D3D12_TEXTURE_DATA_PITCH_ALIGNMENT,
        )
    };
    if layout.offset % u64::from(offset_alignment) != 0
        || layout.bytes_per_row < tight
        || layout.bytes_per_row % pitch_alignment != 0
        || layout.rows_per_image < rows
    {
        return Err(Error::InvalidRange);
    }
    let preceding_rows = u64::from(layout.rows_per_image) * u64::from(extent[2] - 1);
    let byte_end = preceding_rows
        .checked_add(u64::from(rows - 1))
        .and_then(|n| n.checked_mul(u64::from(layout.bytes_per_row)))
        .and_then(|n| n.checked_add(u64::from(tight)))
        .and_then(|n| n.checked_add(layout.offset))
        .ok_or(Error::InvalidRange)?;
    let is_volume = dimension == api::TextureDimension::D3;
    let footprint_rows = if !is_volume {
        u32::try_from(preceding_rows + u64::from(rows)).map_err(|_| Error::InvalidRange)?
    } else if extent[2] > 1 {
        layout.rows_per_image
    } else {
        rows
    };
    let height = footprint_rows
        .checked_mul(block_extent)
        .ok_or(Error::InvalidRange)?;
    let array_layer_height = if !is_volume && extent[2] > 1 {
        layout
            .rows_per_image
            .checked_mul(block_extent)
            .ok_or(Error::InvalidRange)?
    } else {
        0
    };
    Ok(ColorCopyFootprint {
        placed: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
            Offset: layout.offset,
            Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                Format: dxgi_format(format),
                Width: columns * block_extent,
                Height: height,
                Depth: if is_volume { extent[2] } else { 1 },
                RowPitch: layout.bytes_per_row,
            },
        },
        byte_end,
        copy_height: rows * block_extent,
        array_layer_height,
    })
}

pub(crate) fn buffer(
    encoder: &mut Dx12Encoder,
    src: &Dx12Buffer,
    src_offset: u64,
    dst: &Dx12Buffer,
    dst_offset: u64,
    bytes: u64,
) -> Result<()> {
    if bytes == 0
        || src.memory == api::MemoryClass::Readback
        || src_offset
            .checked_add(bytes)
            .is_none_or(|end| end > src.size)
        || dst_offset
            .checked_add(bytes)
            .is_none_or(|end| end > dst.size)
    {
        return Err(Error::InvalidRange);
    }
    if !src.usage.contains(api::BufferUsage::COPY_SRC)
        || !dst.usage.contains(api::BufferUsage::COPY_DST)
    {
        return Err(Error::Unsupported(
            "DX12 buffer copy requires COPY_SRC and COPY_DST usage",
        ));
    }
    if same_resource(&src.resource, &dst.resource) {
        return Err(Error::Unsupported("DX12 same-resource buffer copy"));
    }
    // SAFETY: ranges/usages were validated above, both buffers are retained
    // below, and this encoder exclusively owns the open native list.
    unsafe {
        encoder
            .list
            .CopyBufferRegion(&dst.resource, dst_offset, &src.resource, src_offset, bytes);
    }
    encoder.buffers.push(src.clone());
    encoder.buffers.push(dst.clone());
    Ok(())
}

struct TextureCopy<'a> {
    source: &'a Dx12Texture,
    target: &'a Dx12Texture,
    source_subresource: u32,
    target_subresource: u32,
    full_copy: bool,
    layers: u32,
}

fn validate_texture_copy<'a>(
    encoder: &Dx12Encoder,
    src: &api::TextureRegion<'a, crate::backend::Dx12>,
    dst: &api::TextureRegion<'a, crate::backend::Dx12>,
) -> Result<TextureCopy<'a>> {
    encoder.device.require_active()?;
    // SAFETY: Core validated both regions' device identity.
    let source_texture = unsafe { src.texture.native_ref() };
    let target_texture = unsafe { dst.texture.native_ref() };
    let source = &source_texture.0;
    let target = &target_texture.0;
    if !source.desc.usage.contains(api::TextureUsage::COPY_SRC)
        || !target.desc.usage.contains(api::TextureUsage::COPY_DST)
    {
        return Err(Error::Unsupported(
            "DX12 texture copy requires COPY_SRC and COPY_DST usage",
        ));
    }
    if source.desc.dimension != target.desc.dimension
        || source.desc.format != target.desc.format
        || source.desc.samples != target.desc.samples
        || src.extent != dst.extent
        || src.aspect != dst.aspect
    {
        return Err(Error::Unsupported("DX12 texture copy format or extent"));
    }
    let source_subresource = validate_region(source_texture, src)?;
    let target_subresource = validate_region(target_texture, dst)?;
    let full_copy = source.desc.samples > 1 || super::super::texture::is_depth(source.desc.format);
    if full_copy {
        let full = |texture: &Dx12Texture,
                    region: &api::TextureRegion<'_, crate::backend::Dx12>| {
            region.origin[0] == 0
                && region.origin[1] == 0
                && region.extent[0] == (texture.0.desc.extent[0] >> region.mip).max(1)
                && region.extent[1] == (texture.0.desc.extent[1] >> region.mip).max(1)
        };
        if !full(source_texture, src) || !full(target_texture, dst) {
            return Err(Error::Unsupported(
                "DX12 depth and MSAA texture copies require full subresources",
            ));
        }
    }
    let layers = if source.desc.dimension == api::TextureDimension::D3 {
        1
    } else {
        src.extent[2]
    };
    if same_resource(&source.resource, &target.resource)
        && src.mip == dst.mip
        && (source.desc.dimension == api::TextureDimension::D3
            || (src.origin[2] < dst.origin[2] + layers && dst.origin[2] < src.origin[2] + layers))
    {
        return Err(Error::Unsupported(
            "DX12 overlapping subresource texture copy",
        ));
    }
    Ok(TextureCopy {
        source: source_texture,
        target: target_texture,
        source_subresource,
        target_subresource,
        full_copy,
        layers,
    })
}

pub(super) fn texture_support(
    encoder: &Dx12Encoder,
    src: &api::TextureRegion<'_, crate::backend::Dx12>,
    dst: &api::TextureRegion<'_, crate::backend::Dx12>,
) -> Result<()> {
    validate_texture_copy(encoder, src, dst).map(drop)
}

pub(crate) fn texture(
    encoder: &mut Dx12Encoder,
    src: api::TextureRegion<'_, crate::backend::Dx12>,
    dst: api::TextureRegion<'_, crate::backend::Dx12>,
) -> Result<()> {
    let copy = validate_texture_copy(encoder, &src, &dst)?;
    let source = &copy.source.0;
    let target = &copy.target.0;
    for layer in 0..copy.layers {
        let mut from = copy_location(
            &source.resource,
            D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(),
            false,
            copy.source_subresource + layer * source.desc.mip_levels,
        );
        let mut to = copy_location(
            &target.resource,
            D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(),
            false,
            copy.target_subresource + layer * target.desc.mip_levels,
        );
        // SAFETY: source/target subresources and box were validated above; copy
        // locations own cloned COM references which are released after this call.
        unsafe {
            encoder.list.CopyTextureRegion(
                &to,
                dst.origin[0],
                dst.origin[1],
                if target.desc.dimension == api::TextureDimension::D3 {
                    dst.origin[2]
                } else {
                    0
                },
                &from,
                if copy.full_copy {
                    None
                } else {
                    Some(&D3D12_BOX {
                        left: src.origin[0],
                        top: src.origin[1],
                        front: if source.desc.dimension == api::TextureDimension::D3 {
                            src.origin[2]
                        } else {
                            0
                        },
                        right: src.origin[0]
                            + if super::super::texture::bc_block_bytes(source.desc.format).is_some()
                            {
                                src.extent[0].div_ceil(4) * 4
                            } else {
                                src.extent[0]
                            },
                        bottom: src.origin[1]
                            + if super::super::texture::bc_block_bytes(source.desc.format).is_some()
                            {
                                src.extent[1].div_ceil(4) * 4
                            } else {
                                src.extent[1]
                            },
                        back: if source.desc.dimension == api::TextureDimension::D3 {
                            src.origin[2] + src.extent[2]
                        } else {
                            1
                        },
                    })
                },
            );
            ManuallyDrop::drop(&mut from.pResource);
            ManuallyDrop::drop(&mut to.pResource);
        }
    }
    encoder.textures.push(copy.source.clone());
    encoder.textures.push(copy.target.clone());
    Ok(())
}

fn validate_color_upload(
    encoder: &Dx12Encoder,
    src: &Dx12Buffer,
    layout: api::TexelCopyBufferLayout,
    dst: &api::TextureRegion<'_, crate::backend::Dx12>,
    target: &Dx12Texture,
) -> Result<(u32, ColorCopyFootprint)> {
    encoder.device.require_active()?;
    let desc = &target.0.desc;
    if src.memory == api::MemoryClass::Readback
        || !src.usage.contains(api::BufferUsage::COPY_SRC)
        || !desc.usage.contains(api::TextureUsage::COPY_DST)
    {
        return Err(Error::Unsupported(
            "DX12 COPY_SRC buffer to COPY_DST texture",
        ));
    }
    if desc.samples != 1 {
        return Err(Error::Unsupported(
            "DX12 buffer to multisample texture copy",
        ));
    }
    let subresource = validate_region(target, dst)?;
    let footprint = color_copy_footprint(
        desc.format,
        desc.dimension,
        layout,
        dst.extent,
        encoder.device.0.texel_aligned_color_copies,
    )?;
    if footprint.byte_end > src.size {
        return Err(Error::InvalidRange);
    }
    Ok((subresource, footprint))
}

pub(super) fn buffer_to_texture_support(
    encoder: &Dx12Encoder,
    src: &Dx12Buffer,
    layout: api::TexelCopyBufferLayout,
    dst: &api::TextureRegion<'_, crate::backend::Dx12>,
) -> Result<()> {
    // SAFETY: Core validated the destination region's device.
    let target = unsafe { dst.texture.native_ref() };
    if super::super::texture::is_depth(target.0.desc.format) {
        return super::depth_readback::validate_upload(encoder, &target.0, src, layout, dst)
            .map(drop);
    }
    validate_color_upload(encoder, src, layout, dst, target).map(drop)
}

pub(crate) fn buffer_to_texture(
    encoder: &mut Dx12Encoder,
    src: &Dx12Buffer,
    layout: api::TexelCopyBufferLayout,
    dst: api::TextureRegion<'_, crate::backend::Dx12>,
) -> Result<()> {
    // SAFETY: Core validated the destination region's device; retention covers execution.
    let target = unsafe { dst.texture.native_ref() };
    if super::super::texture::is_depth(target.0.desc.format) {
        return super::depth_readback::buffer_to_texture(encoder, src, layout, dst);
    }
    let (subresource, footprint) = validate_color_upload(encoder, src, layout, &dst, target)?;
    let is_array = target.0.desc.dimension != api::TextureDimension::D3;
    let layers = if is_array { dst.extent[2] } else { 1 };
    for layer in 0..layers {
        let mut from = copy_location(&src.resource, footprint.placed, true, 0);
        let mut to = copy_location(
            &target.0.resource,
            D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(),
            false,
            subresource + layer * target.0.desc.mip_levels,
        );
        // SAFETY: layout and region validation established copy bounds; both FFI
        // locations own COM clones released immediately after synchronous encoding.
        unsafe {
            let box_ = D3D12_BOX {
                left: 0,
                top: layer * footprint.array_layer_height,
                front: 0,
                right: footprint.placed.Footprint.Width,
                bottom: layer * footprint.array_layer_height + footprint.copy_height,
                back: if is_array { 1 } else { dst.extent[2] },
            };
            encoder.list.CopyTextureRegion(
                &to,
                dst.origin[0],
                dst.origin[1],
                if target.0.desc.dimension == api::TextureDimension::D3 {
                    dst.origin[2]
                } else {
                    0
                },
                &from,
                Some(&box_),
            );
            ManuallyDrop::drop(&mut from.pResource);
            ManuallyDrop::drop(&mut to.pResource);
        }
    }
    encoder.buffers.push(src.clone());
    encoder.textures.push(target.clone());
    Ok(())
}

pub(crate) fn texture_to_buffer(
    encoder: &mut Dx12Encoder,
    src: api::TextureRegion<'_, crate::backend::Dx12>,
    dst: &Dx12Buffer,
    layout: api::TexelCopyBufferLayout,
) -> Result<()> {
    // A READBACK destination is permanently COPY_DEST in D3D12; device-memory
    // destinations use the caller's explicit barriers.
    super::readback::texture_to_buffer(encoder, src, dst, layout)
}

#[cfg(all(test, feature = "hardware-tests"))]
#[path = "../../../tests/backend/encoder/copy.rs"]
mod hardware_tests;
