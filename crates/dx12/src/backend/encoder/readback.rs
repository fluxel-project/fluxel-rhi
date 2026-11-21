//! Direct D3D12 texture-to-buffer lowering for explicit copies.

use std::mem::ManuallyDrop;

use fluxel_rhi_core::api::{self, Error, Result};
use windows::Win32::Graphics::Direct3D12::{D3D12_BOX, D3D12_PLACED_SUBRESOURCE_FOOTPRINT};

use super::{
    Dx12Encoder,
    copy::{ColorCopyFootprint, color_copy_footprint, validate_region},
    copy_location,
};
use crate::backend::{Dx12Buffer, Dx12Texture};

fn validate_color_readback(
    encoder: &Dx12Encoder,
    source: &Dx12Texture,
    src: &api::TextureRegion<'_, crate::backend::Dx12>,
    dst: &Dx12Buffer,
    layout: api::TexelCopyBufferLayout,
) -> Result<(u32, ColorCopyFootprint)> {
    encoder.device.require_active()?;
    let desc = &source.0.desc;
    if desc.samples > 1 {
        return Err(Error::Unsupported(
            "DX12 multisample texture to buffer copy",
        ));
    }
    if !desc.usage.contains(api::TextureUsage::COPY_SRC) {
        return Err(Error::Unsupported("DX12 COPY_SRC texture readback"));
    }
    if !dst.usage.contains(api::BufferUsage::COPY_DST) || dst.memory == api::MemoryClass::Upload {
        return Err(Error::Unsupported("DX12 texture copy destination buffer"));
    }
    let subresource = validate_region(source, src)?;
    let footprint = color_copy_footprint(
        desc.format
            .planar_view_format(src.aspect)
            .unwrap_or(desc.format),
        desc.dimension,
        layout,
        src.extent,
        encoder.device.0.texel_aligned_color_copies,
    )?;
    if footprint.byte_end > dst.size {
        return Err(Error::InvalidRange);
    }
    Ok((subresource, footprint))
}

pub(super) fn texture_to_buffer_support(
    encoder: &Dx12Encoder,
    src: &api::TextureRegion<'_, crate::backend::Dx12>,
    dst: &Dx12Buffer,
    layout: api::TexelCopyBufferLayout,
) -> Result<()> {
    // SAFETY: Core validated the source region's device identity.
    let texture = unsafe { src.texture.native_ref() };
    if super::super::texture::is_depth(texture.0.desc.format) {
        return super::depth_readback::validate_readback(encoder, &texture.0, src, dst, layout)
            .map(drop);
    }
    validate_color_readback(encoder, texture, src, dst, layout).map(drop)
}

/// Copies a region into a device or readback buffer.
pub(crate) fn texture_to_buffer(
    encoder: &mut Dx12Encoder,
    src: api::TextureRegion<'_, crate::backend::Dx12>,
    dst: &Dx12Buffer,
    layout: api::TexelCopyBufferLayout,
) -> Result<()> {
    // SAFETY: Core validated the source region's device identity; retention covers execution.
    let texture = unsafe { src.texture.native_ref() };
    let desc = &texture.0.desc;
    if super::super::texture::is_depth(desc.format) {
        return super::depth_readback::texture_to_buffer(encoder, src, dst, layout);
    }
    let (subresource, footprint) = validate_color_readback(encoder, texture, &src, dst, layout)?;
    let is_array = desc.dimension != api::TextureDimension::D3;
    let layers = if is_array { src.extent[2] } else { 1 };
    for layer in 0..layers {
        let mut target = copy_location(&dst.resource, footprint.placed, true, 0);
        let mut source = copy_location(
            &texture.0.resource,
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
                layer * footprint.array_layer_height,
                0,
                &source,
                Some(&D3D12_BOX {
                    left: src.origin[0],
                    top: src.origin[1],
                    front: if is_array { 0 } else { src.origin[2] },
                    right: src.origin[0] + footprint.placed.Footprint.Width,
                    bottom: src.origin[1] + footprint.copy_height,
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
    encoder.textures.push(texture.clone());
    encoder.buffers.push(dst.clone());
    Ok(())
}
