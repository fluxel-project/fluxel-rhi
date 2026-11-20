//! Direct D3D12 texture-to-buffer lowering for explicit copies.

use std::mem::ManuallyDrop;

use fluxel_rhi_core::api::{self, Error, Result};
use windows::Win32::Graphics::Direct3D12::{D3D12_BOX, D3D12_PLACED_SUBRESOURCE_FOOTPRINT};

use super::{
    Dx12Encoder,
    copy::{color_copy_footprint, validate_region},
    copy_location,
};
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
    if src.aspect != api::TextureAspects::COLOR || !desc.usage.contains(api::TextureUsage::COPY_SRC)
    {
        return Err(Error::Unsupported("DX12 COPY_SRC color texture readback"));
    }
    if !dst.usage.contains(api::BufferUsage::COPY_DST) || dst.memory == api::MemoryClass::Upload {
        return Err(Error::Unsupported("DX12 texture copy destination buffer"));
    }
    let texture = Dx12Texture(texture);
    let desc = &texture.0.desc;
    let subresource = validate_region(&texture, &src)?;
    let is_array = desc.dimension != api::TextureDimension::D3;
    let footprint = color_copy_footprint(
        desc.format,
        desc.dimension,
        layout,
        src.extent,
        encoder.device.0.texel_aligned_color_copies,
    )?;
    if footprint.byte_end > dst.size {
        return Err(Error::InvalidRange);
    }
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
    encoder.textures.push(texture);
    encoder.buffers.push(dst.clone());
    Ok(())
}
