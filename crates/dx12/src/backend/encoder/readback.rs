//! Direct D3D12 texture-to-buffer lowering for explicit readback copies.

use std::mem::ManuallyDrop;

use fluxel_rhi_core::api::{self, Error, Result};
use windows::Win32::Graphics::{
    Direct3D12::{D3D12_PLACED_SUBRESOURCE_FOOTPRINT, D3D12_SUBRESOURCE_FOOTPRINT},
    Dxgi::Common::{DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_R8G8B8A8_UNORM},
};

use super::{Dx12Encoder, copy_location};
use crate::backend::{Dx12Buffer, Dx12Texture};

/// Lowers one RGBA8 2D subresource region into a caller-provided READBACK
/// buffer. The public layout is the contract: no hidden repacking occurs.
pub(crate) fn texture_to_buffer(
    encoder: &mut Dx12Encoder,
    src: api::TextureRegion<'_, crate::backend::Dx12>,
    dst: &Dx12Buffer,
    layout: api::TexelCopyBufferLayout,
) -> Result<()> {
    let texture = unsafe { src.texture.native_ref() }.0.clone();
    let desc = &texture.desc;
    let native_format = match desc.format {
        api::TextureFormat::Rgba8Unorm => DXGI_FORMAT_R8G8B8A8_UNORM,
        api::TextureFormat::Bgra8Unorm => DXGI_FORMAT_B8G8R8A8_UNORM,
        _ => return Err(Error::Unsupported("DX12 RGBA8/BGRA8 texture readback")),
    };
    if src.aspect != api::TextureAspects::COLOR || !desc.usage.contains(api::TextureUsage::COPY_SRC)
    {
        return Err(Error::Unsupported("DX12 RGBA8 COPY_SRC texture readback"));
    }
    if !dst.usage.contains(api::BufferUsage::COPY_DST)
        || !dst.usage.contains(api::BufferUsage::MAP_READ)
    {
        return Err(Error::Unsupported("DX12 READBACK COPY_DST|MAP_READ buffer"));
    }
    if src.mip >= desc.mip_levels
        || src.extent[0] == 0
        || src.extent[1] == 0
        || src.extent[2] != 1
        || src.origin[0]
            .checked_add(src.extent[0])
            .is_none_or(|end| end > (desc.extent[0] >> src.mip).max(1))
        || src.origin[1]
            .checked_add(src.extent[1])
            .is_none_or(|end| end > (desc.extent[1] >> src.mip).max(1))
        || src.origin[2]
            .checked_add(src.extent[2])
            .is_none_or(|end| end > desc.array_layers)
    {
        return Err(Error::InvalidRange);
    }
    let tight = src.extent[0].checked_mul(4).ok_or(Error::InvalidRange)?;
    if layout.offset % 512 != 0
        || layout.bytes_per_row < tight
        || layout.bytes_per_row % 256 != 0
        || layout.rows_per_image < src.extent[1]
    {
        return Err(Error::InvalidRange);
    }
    let copied = u64::from(layout.bytes_per_row)
        .checked_mul(u64::from(src.extent[1] - 1))
        .and_then(|prefix| prefix.checked_add(u64::from(tight)))
        .ok_or(Error::InvalidRange)?;
    let end = u64::from(layout.offset)
        .checked_add(copied)
        .ok_or(Error::InvalidRange)?;
    if end > dst.size {
        return Err(Error::InvalidRange);
    }
    let subresource = src.origin[2]
        .checked_mul(desc.mip_levels)
        .and_then(|layer| layer.checked_add(src.mip))
        .ok_or(Error::InvalidRange)?;
    let footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
        Offset: layout.offset,
        Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
            Format: native_format,
            Width: src.extent[0],
            Height: src.extent[1],
            Depth: 1,
            RowPitch: layout.bytes_per_row,
        },
    };
    let mut target = copy_location(&dst.resource, footprint, true, 0);
    let mut source = copy_location(
        &texture.resource,
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(),
        false,
        subresource,
    );
    unsafe {
        encoder.list.CopyTextureRegion(
            &target,
            0,
            0,
            0,
            &source,
            Some(&windows::Win32::Graphics::Direct3D12::D3D12_BOX {
                left: src.origin[0],
                top: src.origin[1],
                front: 0,
                right: src.origin[0] + src.extent[0],
                bottom: src.origin[1] + src.extent[1],
                back: 1,
            }),
        );
        ManuallyDrop::drop(&mut target.pResource);
        ManuallyDrop::drop(&mut source.pResource);
    }
    encoder.textures.push(Dx12Texture(texture));
    encoder.buffers.push(dst.clone());
    Ok(())
}
