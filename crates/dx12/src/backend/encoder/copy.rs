//! Direct lowering of explicit buffer and texture copy commands.
//!
//! These operations do not infer or restore resource state. Callers transition
//! device-memory resources to `CopySrc` or `CopyDst` beforehand. A readback
//! buffer is the D3D12 exception: its committed heap has the fixed `COPY_DEST`
//! state and therefore needs no transition for a texture-to-buffer copy.

use std::mem::ManuallyDrop;

use fluxel_rhi_core::api::{self, Error, Result};
use windows::{
    Win32::Graphics::{
        Direct3D12::{D3D12_BOX, D3D12_PLACED_SUBRESOURCE_FOOTPRINT, D3D12_SUBRESOURCE_FOOTPRINT},
        Dxgi::Common::{DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_R8G8B8A8_UNORM},
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
    let expected_aspect = if desc.format == api::TextureFormat::Depth32Float {
        api::TextureAspects::DEPTH
    } else {
        api::TextureAspects::COLOR
    };
    if region.aspect != expected_aspect
        || region.extent[0] == 0
        || region.extent[1] == 0
        || region.extent[2] != 1
        || region.mip >= desc.mip_levels
        || region.origin[0]
            .checked_add(region.extent[0])
            .is_none_or(|end| end > (desc.extent[0] >> region.mip).max(1))
        || region.origin[1]
            .checked_add(region.extent[1])
            .is_none_or(|end| end > (desc.extent[1] >> region.mip).max(1))
        || region.origin[2] >= desc.array_layers
    {
        return Err(Error::InvalidRange);
    }
    region.origin[2]
        .checked_mul(desc.mip_levels)
        .and_then(|layer| layer.checked_add(region.mip))
        .ok_or(Error::InvalidRange)
}

fn copy_footprint(
    texture: &Dx12Texture,
    subresource: u32,
    layout: api::TexelCopyBufferLayout,
    extent: [u32; 3],
) -> Result<D3D12_PLACED_SUBRESOURCE_FOOTPRINT> {
    let tight = extent[0].checked_mul(4).ok_or(Error::InvalidRange)?;
    if layout.offset % 512 != 0
        || layout.bytes_per_row < tight
        || layout.bytes_per_row % 256 != 0
        || layout.rows_per_image < extent[1]
    {
        return Err(Error::InvalidRange);
    }
    let format = match texture.0.desc.format {
        api::TextureFormat::Rgba8Unorm => DXGI_FORMAT_R8G8B8A8_UNORM,
        api::TextureFormat::Bgra8Unorm => DXGI_FORMAT_B8G8R8A8_UNORM,
        _ => {
            return Err(Error::Unsupported(
                "DX12 buffer-texture copies support RGBA8/BGRA8 2D textures",
            ));
        }
    };
    if subresource >= texture.0.desc.mip_levels * texture.0.desc.array_layers {
        return Err(Error::Unsupported(
            "DX12 buffer-texture copies support RGBA8 2D textures",
        ));
    }
    Ok(D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
        Offset: layout.offset,
        Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
            Format: format,
            Width: extent[0],
            Height: extent[1],
            Depth: 1,
            RowPitch: layout.bytes_per_row,
        },
    })
}

fn layout_end(layout: api::TexelCopyBufferLayout, extent: [u32; 3]) -> Result<u64> {
    layout
        .offset
        .checked_add(u64::from(layout.bytes_per_row) * u64::from(extent[1]))
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
    if source.desc.format != target.desc.format
        || source.desc.samples != target.desc.samples
        || src.extent != dst.extent
    {
        return Err(Error::Unsupported("DX12 texture copy format or extent"));
    }
    let source_subresource = validate_region(&Dx12Texture(source.clone()), &src)?;
    let target_subresource = validate_region(&Dx12Texture(target.clone()), &dst)?;
    if source.desc.samples > 1 {
        let full = |texture: &Dx12Texture,
                    region: &api::TextureRegion<'_, crate::backend::Dx12>| {
            region.origin == [0, 0, 0]
                && region.extent
                    == [
                        texture.0.desc.extent[0] >> region.mip,
                        texture.0.desc.extent[1] >> region.mip,
                        1,
                    ]
        };
        if !full(&Dx12Texture(source.clone()), &src) || !full(&Dx12Texture(target.clone()), &dst) {
            return Err(Error::Unsupported(
                "DX12 MSAA texture copies require full subresources",
            ));
        }
    }
    if same_resource(&source.resource, &target.resource) {
        return Err(Error::Unsupported("DX12 same-resource texture copy"));
    }
    let mut from = copy_location(
        &source.resource,
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(),
        false,
        source_subresource,
    );
    let mut to = copy_location(
        &target.resource,
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(),
        false,
        target_subresource,
    );
    // SAFETY: source/target subresources and box were validated above; copy
    // locations own cloned COM references which are released after this call.
    unsafe {
        encoder.list.CopyTextureRegion(
            &to,
            dst.origin[0],
            dst.origin[1],
            0,
            &from,
            Some(&D3D12_BOX {
                left: src.origin[0],
                top: src.origin[1],
                front: 0,
                right: src.origin[0] + src.extent[0],
                bottom: src.origin[1] + src.extent[1],
                back: 1,
            }),
        );
        ManuallyDrop::drop(&mut from.pResource);
        ManuallyDrop::drop(&mut to.pResource);
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
    if src.memory == api::MemoryClass::Readback
        || !src.usage.contains(api::BufferUsage::COPY_SRC)
        || !target.desc.usage.contains(api::TextureUsage::COPY_DST)
    {
        return Err(Error::Unsupported(
            "DX12 COPY_SRC buffer to COPY_DST texture",
        ));
    }
    if !matches!(
        target.desc.format,
        api::TextureFormat::Rgba8Unorm | api::TextureFormat::Bgra8Unorm
    ) || target.desc.samples != 1
    {
        return Err(Error::Unsupported(
            "DX12 buffer-texture copies support single-sample RGBA8/BGRA8 2D textures",
        ));
    }
    let target = Dx12Texture(target);
    let subresource = validate_region(&target, &dst)?;
    let footprint = copy_footprint(&target, subresource, layout, dst.extent)?;
    if layout_end(layout, dst.extent)? > src.size {
        return Err(Error::InvalidRange);
    }
    let mut from = copy_location(&src.resource, footprint, true, 0);
    let mut to = copy_location(
        &target.0.resource,
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(),
        false,
        subresource,
    );
    // SAFETY: layout and region validation established copy bounds; both FFI
    // locations own COM clones released immediately after synchronous encoding.
    unsafe {
        encoder
            .list
            .CopyTextureRegion(&to, dst.origin[0], dst.origin[1], 0, &from, None);
        ManuallyDrop::drop(&mut from.pResource);
        ManuallyDrop::drop(&mut to.pResource);
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
    // A READBACK destination is permanently COPY_DEST in D3D12.  This is the
    // sole fixed-state path; all device-memory resources use explicit barriers.
    super::readback::texture_to_buffer(encoder, src, dst, layout)
}

#[cfg(all(test, feature = "hardware-tests"))]
#[path = "copy/tests.rs"]
mod hardware_tests;
