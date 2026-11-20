//! CPU texture writes using queue-retired upload pages.
//!
//! This module validates the upload region and repacks source rows into the
//! native footprint before recording any texture state transitions.

use super::*;

pub(super) fn write(
    encoder: &mut Dx12Encoder,
    dst: api::TextureRegion<'_, Dx12>,
    layout: api::TexelCopyBufferLayout,
    bytes: &[u8],
) -> Result<(), Error> {
    // SAFETY: Core tied the region to this device; the clone stored below
    // keeps the texture and its COM resource alive through submission.
    let texture = unsafe { dst.texture.native_ref() };
    if texture.0.desc.samples != 1 {
        return Err(Error::Unsupported(
            "DX12 CPU upload to multisampled texture",
        ));
    }
    let block = super::super::texture::bc_block_bytes(texture.0.desc.format);
    let texel = super::super::texture::color_texel_bytes(texture.0.desc.format);
    let depth = super::super::texture::is_depth(texture.0.desc.format);
    let plane_bytes = if depth {
        use api::TextureFormat as F;
        match (texture.0.desc.format, dst.aspect) {
            (F::Depth16Unorm, api::TextureAspects::DEPTH) => Some(2),
            (F::Depth32Float | F::Depth32FloatStencil8, api::TextureAspects::DEPTH) => Some(4),
            (F::Depth24PlusStencil8 | F::Depth32FloatStencil8, api::TextureAspects::STENCIL) => {
                Some(1)
            }
            (F::Depth24Plus | F::Depth24PlusStencil8, api::TextureAspects::DEPTH) => {
                return Err(Error::Unsupported(
                    "DX12 opaque Depth24Plus buffer representation",
                ));
            }
            _ => return Err(Error::Unsupported("DX12 texture write format or aspect")),
        }
    } else {
        if !(block.is_some() || texel.is_some()) || dst.aspect != api::TextureAspects::COLOR {
            return Err(Error::Unsupported("DX12 texture write format or aspect"));
        }
        None
    };
    if !texture.0.desc.usage.contains(api::TextureUsage::COPY_DST) {
        return Err(Error::Unsupported(
            "DX12 texture write requires COPY_DST usage",
        ));
    }
    if dst.mip >= texture.0.desc.mip_levels
        || dst.extent[0] == 0
        || dst.extent[1] == 0
        || dst.extent[2] == 0
        || dst.origin[0]
            .checked_add(dst.extent[0])
            .is_none_or(|end| end > (texture.0.desc.extent[0] >> dst.mip).max(1))
        || dst.origin[1]
            .checked_add(dst.extent[1])
            .is_none_or(|end| end > (texture.0.desc.extent[1] >> dst.mip).max(1))
        || dst.origin[2].checked_add(dst.extent[2]).is_none_or(|end| {
            end > if texture.0.desc.dimension == api::TextureDimension::D3 {
                (texture.0.desc.extent[2] >> dst.mip).max(1)
            } else {
                texture.0.desc.array_layers
            }
        })
    {
        return Err(Error::InvalidRange);
    }
    if depth
        && (dst.origin[0] != 0
            || dst.origin[1] != 0
            || dst.extent[0] != (texture.0.desc.extent[0] >> dst.mip).max(1)
            || dst.extent[1] != (texture.0.desc.extent[1] >> dst.mip).max(1)
            || texture.0.desc.samples != 1)
    {
        return Err(Error::InvalidRange);
    }
    if block.is_some()
        && (dst.origin[0] % 4 != 0
            || dst.origin[1] % 4 != 0
            || (dst.extent[0] % 4 != 0
                && dst.origin[0].checked_add(dst.extent[0])
                    != Some((texture.0.desc.extent[0] >> dst.mip).max(1)))
            || (dst.extent[1] % 4 != 0
                && dst.origin[1].checked_add(dst.extent[1])
                    != Some((texture.0.desc.extent[1] >> dst.mip).max(1))))
    {
        return Err(Error::InvalidRange);
    }
    let rows_needed = if block.is_some() {
        dst.extent[1].div_ceil(4)
    } else {
        dst.extent[1]
    };
    let tight = if let Some(block) = block {
        dst.extent[0].div_ceil(4).checked_mul(block)
    } else {
        dst.extent[0].checked_mul(plane_bytes.or(texel).ok_or(Error::InvalidRange)?)
    }
    .ok_or(Error::InvalidRange)?;
    if layout.bytes_per_row < tight || layout.rows_per_image < rows_needed {
        return Err(Error::InvalidRange);
    }
    let input_end = u64::from(layout.rows_per_image)
        .checked_mul(u64::from(dst.extent[2] - 1))
        .and_then(|n| n.checked_add(u64::from(rows_needed - 1)))
        .and_then(|n| n.checked_mul(u64::from(layout.bytes_per_row)))
        .and_then(|n| n.checked_add(u64::from(tight)))
        .and_then(|n| n.checked_add(layout.offset))
        .ok_or(Error::InvalidRange)?;
    if input_end > bytes.len() as u64 {
        return Err(Error::InvalidRange);
    }
    // D3D12 enumerates mips consecutively inside each array layer.
    let subresource = if texture.0.desc.dimension == api::TextureDimension::D3 {
        dst.mip
    } else {
        dst.origin[2]
            .checked_mul(texture.0.desc.mip_levels)
            .and_then(|layer| layer.checked_add(dst.mip))
            .ok_or(Error::InvalidRange)?
    };
    let subresource = if depth && dst.aspect == api::TextureAspects::STENCIL {
        subresource
            .checked_add(
                texture
                    .0
                    .desc
                    .mip_levels
                    .checked_mul(texture.0.desc.array_layers)
                    .ok_or(Error::InvalidRange)?,
            )
            .ok_or(Error::InvalidRange)?
    } else {
        subresource
    };
    // SAFETY: the retained resource exposes an immutable descriptor only.
    let desc = unsafe { texture.0.resource.GetDesc() };
    let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
    let mut rows = 0;
    let mut row_size = 0;
    let mut total = 0;
    let device = &encoder.device.0.native;
    // SAFETY: all footprint out-pointers are valid locals and `desc` is
    // an immutable descriptor for the retained texture resource.
    unsafe {
        device.GetCopyableFootprints(
            &desc,
            subresource,
            1,
            0,
            Some(&mut footprint),
            Some(&mut rows),
            Some(&mut row_size),
            Some(&mut total),
        );
    }
    if rows < rows_needed || footprint.Footprint.RowPitch < tight {
        return Err(Error::Backend(
            "DX12 returned an invalid texture copy footprint".into(),
        ));
    }
    footprint.Footprint.Width = if block.is_some() {
        dst.extent[0].div_ceil(4) * 4
    } else {
        dst.extent[0]
    };
    footprint.Footprint.Height = if block.is_some() {
        dst.extent[1].div_ceil(4) * 4
    } else {
        dst.extent[1]
    };
    let is_array = texture.0.desc.dimension != api::TextureDimension::D3;
    footprint.Footprint.Depth = if is_array { 1 } else { dst.extent[2] };
    let image_bytes = u64::from(footprint.Footprint.RowPitch)
        .checked_mul(u64::from(rows_needed))
        .ok_or(Error::InvalidRange)?;
    let alignment = u64::from(D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT);
    let image_stride = if is_array {
        image_bytes
            .checked_add(alignment - 1)
            .ok_or(Error::InvalidRange)?
            & !(alignment - 1)
    } else {
        image_bytes
    };
    let upload_bytes = image_stride
        .checked_mul(u64::from(dst.extent[2] - 1))
        .and_then(|n| n.checked_add(image_bytes))
        .ok_or(Error::InvalidRange)?;
    let upload_bytes = usize::try_from(upload_bytes).map_err(|_| Error::InvalidRange)?;
    let source_base = usize::try_from(layout.offset).map_err(|_| Error::InvalidRange)?;
    let source_pitch = layout.bytes_per_row as usize;
    let row_bytes = tight as usize;
    let target_pitch = footprint.Footprint.RowPitch as usize;
    let image_stride_bytes = usize::try_from(image_stride).map_err(|_| Error::InvalidRange)?;
    let (upload, base_offset) = encoder
        .upload_arena
        .write_with(device, upload_bytes, alignment, |staging| {
            // The source end and staging length above bound every row of
            // every image, including the final row without caller padding.
            for slice in 0..dst.extent[2] as usize {
                for row in 0..rows_needed as usize {
                    let source =
                        source_base + (slice * layout.rows_per_image as usize + row) * source_pitch;
                    let target = slice * image_stride_bytes + row * target_pitch;
                    staging[target..target + row_bytes]
                        .copy_from_slice(&bytes[source..source + row_bytes]);
                }
            }
            true
        })?
        .expect("the texture staging copy commits its reservation");
    let stable_state = if depth {
        D3D12_RESOURCE_STATE_DEPTH_WRITE
    } else {
        D3D12_RESOURCE_STATE_COMMON
    };
    let layers = if is_array { dst.extent[2] } else { 1 };
    for layer in 0..layers {
        let subresource = subresource + layer * texture.0.desc.mip_levels;
        footprint.Offset = base_offset + u64::from(layer) * image_stride;
        barrier_sub(
            &encoder.list,
            &texture.0.resource,
            subresource,
            stable_state,
            D3D12_RESOURCE_STATE_COPY_DEST,
        );
        let mut source = copy_location(&upload, footprint, true, 0);
        let mut target = copy_location(
            &texture.0.resource,
            D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(),
            false,
            subresource,
        );
        // SAFETY: copy locations own cloned COM references; both resources are
        // retained and are in the explicit COPY states around this call.
        unsafe {
            encoder.list.CopyTextureRegion(
                &target,
                dst.origin[0],
                dst.origin[1],
                if texture.0.desc.dimension == api::TextureDimension::D3 {
                    dst.origin[2]
                } else {
                    0
                },
                &source,
                None,
            );
        }
        // SAFETY: destroy exactly the cloned COM references installed in the
        // FFI unions after D3D12 synchronously consumed the locations.
        unsafe {
            ManuallyDrop::drop(&mut source.pResource);
            ManuallyDrop::drop(&mut target.pResource);
        }
        barrier_sub(
            &encoder.list,
            &texture.0.resource,
            subresource,
            D3D12_RESOURCE_STATE_COPY_DEST,
            stable_state,
        );
    }
    encoder.textures.push(texture.clone());
    Ok(())
}
