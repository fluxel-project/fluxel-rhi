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

fn validate_region(
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
        region.aspect == api::TextureAspects::DEPTH
            || (region.aspect == api::TextureAspects::STENCIL
                && super::super::texture::depth_aspects(desc.format) != api::TextureAspects::DEPTH)
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

fn copy_footprint(
    texture: &Dx12Texture,
    subresource: u32,
    layout: api::TexelCopyBufferLayout,
    extent: [u32; 3],
) -> Result<D3D12_PLACED_SUBRESOURCE_FOOTPRINT> {
    let block = super::super::texture::bc_block_bytes(texture.0.desc.format);
    let rows = if block.is_some() {
        extent[1].div_ceil(4)
    } else {
        extent[1]
    };
    let tight = if let Some(block) = block {
        extent[0].div_ceil(4).checked_mul(block)
    } else {
        extent[0].checked_mul(
            super::super::texture::color_texel_bytes(texture.0.desc.format)
                .ok_or(Error::Unsupported("DX12 texture copy format"))?,
        )
    }
    .ok_or(Error::InvalidRange)?;
    if layout.offset % 512 != 0
        || layout.bytes_per_row < tight
        || layout.bytes_per_row % 256 != 0
        || layout.rows_per_image < rows
    {
        return Err(Error::InvalidRange);
    }
    let format = super::super::texture::dxgi_format(texture.0.desc.format);
    if subresource >= texture.0.desc.mip_levels * texture.0.desc.array_layers {
        return Err(Error::Unsupported(
            "DX12 buffer-texture copies support RGBA8 2D textures",
        ));
    }
    Ok(D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
        Offset: layout.offset,
        Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
            Format: format,
            Width: if block.is_some() {
                extent[0].div_ceil(4) * 4
            } else {
                extent[0]
            },
            Height: if extent[2] > 1 {
                layout.rows_per_image
            } else if block.is_some() {
                extent[1].div_ceil(4) * 4
            } else {
                extent[1]
            },
            Depth: extent[2],
            RowPitch: layout.bytes_per_row,
        },
    })
}

fn layout_end(
    layout: api::TexelCopyBufferLayout,
    extent: [u32; 3],
    compressed: bool,
) -> Result<u64> {
    u64::from(layout.rows_per_image)
        .checked_mul(u64::from(extent[2] - 1))
        .and_then(|n| {
            n.checked_add(u64::from(if compressed {
                extent[1].div_ceil(4)
            } else {
                extent[1]
            }))
        })
        .and_then(|n| n.checked_mul(u64::from(layout.bytes_per_row)))
        .and_then(|n| n.checked_add(layout.offset))
        .ok_or(Error::InvalidRange)
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

pub(crate) fn texture(
    encoder: &mut Dx12Encoder,
    src: api::TextureRegion<'_, crate::backend::Dx12>,
    dst: api::TextureRegion<'_, crate::backend::Dx12>,
) -> Result<()> {
    // SAFETY: Core tied both regions to this device; cloning extends native
    // texture lifetime through the recorded copy and submission fence.
    let source = unsafe { src.texture.native_ref() }.0.clone();
    let target = unsafe { dst.texture.native_ref() }.0.clone();
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
    let source_subresource = validate_region(&Dx12Texture(source.clone()), &src)?;
    let target_subresource = validate_region(&Dx12Texture(target.clone()), &dst)?;
    let full_copy = source.desc.samples > 1 || super::super::texture::is_depth(source.desc.format);
    if full_copy {
        let full = |texture: &Dx12Texture,
                    region: &api::TextureRegion<'_, crate::backend::Dx12>| {
            region.origin[0] == 0
                && region.origin[1] == 0
                && region.extent[0] == (texture.0.desc.extent[0] >> region.mip).max(1)
                && region.extent[1] == (texture.0.desc.extent[1] >> region.mip).max(1)
        };
        if !full(&Dx12Texture(source.clone()), &src) || !full(&Dx12Texture(target.clone()), &dst) {
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
    for layer in 0..layers {
        let mut from = copy_location(
            &source.resource,
            D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(),
            false,
            source_subresource + layer * source.desc.mip_levels,
        );
        let mut to = copy_location(
            &target.resource,
            D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(),
            false,
            target_subresource + layer * target.desc.mip_levels,
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
                if full_copy {
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
    encoder.textures.push(Dx12Texture(source));
    encoder.textures.push(Dx12Texture(target));
    Ok(())
}

pub(crate) fn buffer_to_texture(
    encoder: &mut Dx12Encoder,
    src: &Dx12Buffer,
    layout: api::TexelCopyBufferLayout,
    dst: api::TextureRegion<'_, crate::backend::Dx12>,
) -> Result<()> {
    // SAFETY: Core validated the destination region's device; clone retains
    // its native resource until this command list retires.
    let target = unsafe { dst.texture.native_ref() }.0.clone();
    if super::super::texture::is_depth(target.desc.format) {
        return super::depth_readback::buffer_to_texture(encoder, src, layout, dst);
    }
    if src.memory == api::MemoryClass::Readback
        || !src.usage.contains(api::BufferUsage::COPY_SRC)
        || !target.desc.usage.contains(api::TextureUsage::COPY_DST)
    {
        return Err(Error::Unsupported(
            "DX12 COPY_SRC buffer to COPY_DST texture",
        ));
    }
    if !(super::super::texture::bc_block_bytes(target.desc.format).is_some()
        || super::super::texture::color_texel_bytes(target.desc.format).is_some())
        || target.desc.samples != 1
    {
        return Err(Error::Unsupported(
            "DX12 buffer-texture copies support single-sample RGBA8/BGRA8 2D textures",
        ));
    }
    let target = Dx12Texture(target);
    let subresource = validate_region(&target, &dst)?;
    let is_array = target.0.desc.dimension != api::TextureDimension::D3;
    let mut footprint = copy_footprint(
        &target,
        subresource,
        layout,
        if is_array {
            [dst.extent[0], dst.extent[1], 1]
        } else {
            dst.extent
        },
    )?;
    if is_array {
        footprint.Footprint.Height = layout
            .rows_per_image
            .checked_mul(dst.extent[2])
            .and_then(|n| {
                n.checked_mul(
                    if super::super::texture::bc_block_bytes(target.0.desc.format).is_some() {
                        4
                    } else {
                        1
                    },
                )
            })
            .ok_or(Error::InvalidRange)?;
    }
    if layout_end(
        layout,
        dst.extent,
        super::super::texture::bc_block_bytes(target.0.desc.format).is_some(),
    )? > src.size
    {
        return Err(Error::InvalidRange);
    }
    let layers = if is_array { dst.extent[2] } else { 1 };
    for layer in 0..layers {
        let mut from = copy_location(&src.resource, footprint, true, 0);
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
                top: if is_array {
                    layer
                        * layout.rows_per_image
                        * if super::super::texture::bc_block_bytes(target.0.desc.format).is_some() {
                            4
                        } else {
                            1
                        }
                } else {
                    0
                },
                front: 0,
                right: footprint.Footprint.Width,
                bottom: (if is_array {
                    layer
                        * layout.rows_per_image
                        * if super::super::texture::bc_block_bytes(target.0.desc.format).is_some() {
                            4
                        } else {
                            1
                        }
                } else {
                    0
                }) + if super::super::texture::bc_block_bytes(target.0.desc.format)
                    .is_some()
                {
                    dst.extent[1].div_ceil(4) * 4
                } else {
                    dst.extent[1]
                },
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
    encoder.textures.push(target);
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
#[path = "copy/tests.rs"]
mod hardware_tests;
