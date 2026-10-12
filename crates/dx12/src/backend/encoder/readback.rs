//! Direct D3D12 texture-to-buffer lowering for explicit readback copies.

use std::mem::ManuallyDrop;

use fluxel_rhi_core::api::{self, Error, Result};
use windows::Win32::Graphics::Direct3D12::{D3D12_PLACED_SUBRESOURCE_FOOTPRINT, ID3D12Device};

use super::{Dx12Encoder, copy_location, native_error};
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
    if desc.format != api::TextureFormat::Rgba8Unorm
        || src.aspect != api::TextureAspects::COLOR
        || !desc.usage.contains(api::TextureUsage::COPY_SRC)
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
        || layout.rows_per_image != src.extent[1]
    {
        return Err(Error::InvalidRange);
    }
    let copied = u64::from(layout.bytes_per_row)
        .checked_mul(u64::from(src.extent[1]))
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
    let native_desc = unsafe { texture.resource.GetDesc() };
    let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
    let mut rows = 0;
    let mut row_size = 0;
    let mut total = 0;
    let mut device: Option<ID3D12Device> = None;
    unsafe { texture.resource.GetDevice(&mut device) }.map_err(native_error)?;
    let device = device.ok_or_else(|| Error::Backend("texture has no D3D12 device".into()))?;
    unsafe {
        device.GetCopyableFootprints(
            &native_desc,
            subresource,
            1,
            u64::from(layout.offset),
            Some(&mut footprint),
            Some(&mut rows),
            Some(&mut row_size),
            Some(&mut total),
        );
    }
    if rows < src.extent[1]
        || row_size < u64::from(tight)
        || footprint.Footprint.RowPitch != layout.bytes_per_row
    {
        return Err(Error::InvalidRange);
    }
    footprint.Footprint.Width = src.extent[0];
    footprint.Footprint.Height = src.extent[1];
    footprint.Footprint.Depth = 1;
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
