//! Direct D3D12 texture-to-buffer lowering for explicit copies.

use std::mem::ManuallyDrop;

use fluxel_rhi_core::api::{self, Error, Result};
use windows::Win32::Graphics::Direct3D12::{
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT, D3D12_SUBRESOURCE_FOOTPRINT,
};

use super::{Dx12Encoder, copy_location};
use crate::backend::{Dx12Buffer, Dx12Texture};

/// Copies a color region into a device or readback buffer without repacking.
pub(crate) fn texture_to_buffer(
    encoder: &mut Dx12Encoder,
    src: api::TextureRegion<'_, crate::backend::Dx12>,
    dst: &Dx12Buffer,
    layout: api::TexelCopyBufferLayout,
) -> Result<()> {
    // SAFETY: Core validated the source region's device identity; clone retains
    // native texture storage through the command list and completion boundary.
    let texture = unsafe { src.texture.native_ref() }.0.clone();
    let desc = &texture.desc;
    if desc.samples > 1 {
        return Err(Error::Unsupported(
            "DX12 multisample texture to buffer copy",
        ));
    }
    if super::super::texture::is_depth(desc.format) {
        return super::depth_readback::texture_to_buffer(encoder, src, dst, layout);
    }
    let block = super::super::texture::bc_block_bytes(desc.format);
    if block.is_none() && super::super::texture::color_texel_bytes(desc.format).is_none() {
        return Err(Error::Unsupported("DX12 color texture readback format"));
    }
    let native_format = super::super::texture::dxgi_format(desc.format);
    if src.aspect != api::TextureAspects::COLOR || !desc.usage.contains(api::TextureUsage::COPY_SRC)
    {
        return Err(Error::Unsupported("DX12 RGBA8 COPY_SRC texture readback"));
    }
    if !dst.usage.contains(api::BufferUsage::COPY_DST) || dst.memory == api::MemoryClass::Upload {
        return Err(Error::Unsupported("DX12 texture copy destination buffer"));
    }
    if src.mip >= desc.mip_levels
        || src.extent[0] == 0
        || src.extent[1] == 0
        || src.extent[2] == 0
        || src.origin[0]
            .checked_add(src.extent[0])
            .is_none_or(|end| end > (desc.extent[0] >> src.mip).max(1))
        || src.origin[1]
            .checked_add(src.extent[1])
            .is_none_or(|end| end > (desc.extent[1] >> src.mip).max(1))
        || src.origin[2].checked_add(src.extent[2]).is_none_or(|end| {
            end > if desc.dimension == api::TextureDimension::D3 {
                (desc.extent[2] >> src.mip).max(1)
            } else {
                desc.array_layers
            }
        })
    {
        return Err(Error::InvalidRange);
    }
    if block.is_some()
        && (src.origin[0] % 4 != 0
            || src.origin[1] % 4 != 0
            || (src.extent[0] % 4 != 0
                && src.origin[0].checked_add(src.extent[0])
                    != Some((desc.extent[0] >> src.mip).max(1)))
            || (src.extent[1] % 4 != 0
                && src.origin[1].checked_add(src.extent[1])
                    != Some((desc.extent[1] >> src.mip).max(1))))
    {
        return Err(Error::InvalidRange);
    }
    let rows = if block.is_some() {
        src.extent[1].div_ceil(4)
    } else {
        src.extent[1]
    };
    let tight = if let Some(block) = block {
        src.extent[0].div_ceil(4).checked_mul(block)
    } else {
        src.extent[0].checked_mul(
            super::super::texture::color_texel_bytes(desc.format).ok_or(Error::InvalidRange)?,
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
    let copied = u64::from(layout.rows_per_image)
        .checked_mul(u64::from(src.extent[2] - 1))
        .and_then(|n| n.checked_add(u64::from(rows - 1)))
        .and_then(|n| n.checked_mul(u64::from(layout.bytes_per_row)))
        .and_then(|prefix| prefix.checked_add(u64::from(tight)))
        .ok_or(Error::InvalidRange)?;
    let end = u64::from(layout.offset)
        .checked_add(copied)
        .ok_or(Error::InvalidRange)?;
    if end > dst.size {
        return Err(Error::InvalidRange);
    }
    let subresource = if desc.dimension == api::TextureDimension::D3 {
        src.mip
    } else {
        src.origin[2]
            .checked_mul(desc.mip_levels)
            .and_then(|layer| layer.checked_add(src.mip))
            .ok_or(Error::InvalidRange)?
    };
    let is_array = desc.dimension != api::TextureDimension::D3;
    let array_height = if is_array {
        Some(
            layout
                .rows_per_image
                .checked_mul(src.extent[2])
                .and_then(|n| n.checked_mul(if block.is_some() { 4 } else { 1 }))
                .ok_or(Error::InvalidRange)?,
        )
    } else {
        None
    };
    let footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
        Offset: layout.offset,
        Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
            Format: native_format,
            Width: if block.is_some() {
                src.extent[0].div_ceil(4) * 4
            } else {
                src.extent[0]
            },
            Height: if let Some(height) = array_height {
                height
            } else if src.extent[2] > 1 {
                layout.rows_per_image
            } else if block.is_some() {
                src.extent[1].div_ceil(4) * 4
            } else {
                src.extent[1]
            },
            Depth: if is_array { 1 } else { src.extent[2] },
            RowPitch: layout.bytes_per_row,
        },
    };
    let layers = if is_array { src.extent[2] } else { 1 };
    for layer in 0..layers {
        let mut target = copy_location(&dst.resource, footprint, true, 0);
        let mut source = copy_location(
            &texture.resource,
            D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(),
            false,
            subresource + layer * desc.mip_levels,
        );
        // SAFETY: source/target locations own COM clones, all ranges were checked,
        // and the caller supplies COPY_DEST for device buffers; readback storage
        // has a fixed COPY_DEST state.
        unsafe {
            encoder.list.CopyTextureRegion(
                &target,
                0,
                if is_array {
                    layer * layout.rows_per_image * if block.is_some() { 4 } else { 1 }
                } else {
                    0
                },
                0,
                &source,
                Some(&windows::Win32::Graphics::Direct3D12::D3D12_BOX {
                    left: src.origin[0],
                    top: src.origin[1],
                    front: if desc.dimension == api::TextureDimension::D3 {
                        src.origin[2]
                    } else {
                        0
                    },
                    right: src.origin[0]
                        + if block.is_some() {
                            src.extent[0].div_ceil(4) * 4
                        } else {
                            src.extent[0]
                        },
                    bottom: src.origin[1]
                        + if block.is_some() {
                            src.extent[1].div_ceil(4) * 4
                        } else {
                            src.extent[1]
                        },
                    back: if desc.dimension == api::TextureDimension::D3 {
                        src.origin[2] + src.extent[2]
                    } else {
                        1
                    },
                }),
            );
            ManuallyDrop::drop(&mut target.pResource);
            ManuallyDrop::drop(&mut source.pResource);
        }
    }
    encoder.textures.push(Dx12Texture(texture));
    encoder.buffers.push(dst.clone());
    Ok(())
}
