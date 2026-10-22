//! Full depth/stencil plane readback using the native copyable footprint.
use super::*;
use std::mem::ManuallyDrop;

fn scratch_buffer(encoder: &Dx12Encoder, size: u64) -> Result<ID3D12Resource, Error> {
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
        MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
        CreationNodeMask: 1,
        VisibleNodeMask: 1,
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Alignment: 0,
        Width: size,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_UNKNOWN,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_NONE,
    };
    let mut resource = None;
    // SAFETY: initialized descriptors and output slot are valid for the call.
    unsafe {
        encoder.device.0.native.CreateCommittedResource(
            &heap,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_COMMON,
            None,
            &mut resource,
        )
    }
    .map_err(native_error)?;
    let resource = resource.ok_or_else(|| {
        Error::Backend("DX12 depth copy scratch allocation returned no resource".into())
    })?;
    barrier(
        &encoder.list,
        &resource,
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_COPY_DEST,
    );
    Ok(resource)
}

pub(super) fn texture_to_buffer(
    encoder: &mut Dx12Encoder,
    src: api::TextureRegion<'_, Dx12>,
    dst: &Dx12Buffer,
    layout: api::TexelCopyBufferLayout,
) -> Result<(), Error> {
    // SAFETY: Core checked device identity and the clone retains the resource.
    let texture = unsafe { src.texture.native_ref() }.0.clone();
    let desc = &texture.desc;
    if !desc.usage.contains(api::TextureUsage::COPY_SRC)
        || !dst.usage.contains(api::BufferUsage::COPY_DST)
        || dst.memory == api::MemoryClass::Upload
    {
        return Err(Error::Unsupported("DX12 depth plane readback usage"));
    }
    let copies = plane_footprints(encoder, &texture, &src, layout, dst.size)?;
    let scratch = if copies.iter().any(|(_, fp, _, _)| fp.Offset % 512 != 0) {
        Some(scratch_buffer(
            encoder,
            u64::from(layout.bytes_per_row) * u64::from(copies[0].2),
        )?)
    } else {
        None
    };
    for (subresource, mut footprint, rows, row_size) in copies {
        let offset = footprint.Offset;
        let use_scratch = offset % 512 != 0;
        let target_resource = if use_scratch {
            footprint.Offset = 0;
            scratch.as_ref().unwrap()
        } else {
            &dst.resource
        };
        let mut target = copy_location(target_resource, footprint, true, 0);
        let mut source = copy_location(
            &texture.resource,
            D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(),
            false,
            subresource,
        );
        // SAFETY: full subresource and the device-reported footprint are validated;
        // callers supply COPY_SOURCE and destination COPY_DEST states; readback
        // heaps have a fixed COPY_DEST state.
        unsafe {
            encoder
                .list
                .CopyTextureRegion(&target, 0, 0, 0, &source, None);
            ManuallyDrop::drop(&mut source.pResource);
            ManuallyDrop::drop(&mut target.pResource);
        }
        if use_scratch {
            let scratch = scratch.as_ref().unwrap();
            barrier(
                &encoder.list,
                scratch,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            );
            // SAFETY: validated rows fit both buffers and resources remain retained.
            unsafe {
                for row in 0..rows {
                    let position = u64::from(row) * u64::from(layout.bytes_per_row);
                    encoder.list.CopyBufferRegion(
                        &dst.resource,
                        offset + position,
                        scratch,
                        position,
                        row_size,
                    );
                }
            }
            barrier(
                &encoder.list,
                scratch,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_COPY_DEST,
            );
        }
    }
    if let Some(scratch) = scratch {
        encoder.uploads.push(scratch);
    }
    encoder.textures.push(Dx12Texture(texture));
    encoder.buffers.push(dst.clone());
    Ok(())
}

fn plane_footprints(
    encoder: &Dx12Encoder,
    texture: &std::sync::Arc<super::super::texture::TextureState>,
    region: &api::TextureRegion<'_, Dx12>,
    layout: api::TexelCopyBufferLayout,
    buffer_size: u64,
) -> Result<Vec<(u32, D3D12_PLACED_SUBRESOURCE_FOOTPRINT, u32, u64)>, Error> {
    let desc = &texture.desc;
    let stencil = region.aspect == api::TextureAspects::STENCIL;
    if !(region.aspect == api::TextureAspects::DEPTH
        || (stencil
            && super::super::texture::depth_aspects(desc.format) != api::TextureAspects::DEPTH))
    {
        return Err(Error::InvalidRange);
    }
    if !stencil
        && matches!(
            desc.format,
            api::TextureFormat::Depth24Plus | api::TextureFormat::Depth24PlusStencil8
        )
    {
        return Err(Error::Unsupported(
            "DX12 opaque Depth24Plus buffer representation",
        ));
    }
    if region.mip >= desc.mip_levels
        || region.origin[0] != 0
        || region.origin[1] != 0
        || region.extent[2] == 0
        || region.origin[2]
            .checked_add(region.extent[2])
            .is_none_or(|end| end > desc.array_layers)
        || region.extent[0] != (desc.extent[0] >> region.mip).max(1)
        || region.extent[1] != (desc.extent[1] >> region.mip).max(1)
    {
        return Err(Error::InvalidRange);
    }
    let subresource = region.mip
        + region.origin[2] * desc.mip_levels
        + if stencil {
            desc.mip_levels * desc.array_layers
        } else {
            0
        };
    let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
    let mut rows = 0u32;
    let mut row_size = 0u64;
    // SAFETY: the retained native resource and matching output slots are valid.
    let rd = unsafe { texture.resource.GetDesc() };
    unsafe {
        encoder.device.0.native.GetCopyableFootprints(
            &rd,
            subresource,
            1,
            0,
            Some(&mut footprint),
            Some(&mut rows),
            Some(&mut row_size),
            None,
        )
    };
    if rows == 0 || row_size == 0 {
        return Err(Error::Backend(
            "DX12 returned an invalid depth plane footprint".into(),
        ));
    }
    if layout.offset % 512 != 0
        || layout.bytes_per_row % 256 != 0
        || layout.bytes_per_row < footprint.Footprint.RowPitch
        || layout.rows_per_image < rows
    {
        return Err(Error::InvalidRange);
    }
    let stride = u64::from(layout.rows_per_image) * u64::from(layout.bytes_per_row);
    let end = layout
        .offset
        .checked_add(
            stride
                .checked_mul(u64::from(region.extent[2] - 1))
                .ok_or(Error::InvalidRange)?,
        )
        .and_then(|n| n.checked_add(u64::from(rows - 1) * u64::from(layout.bytes_per_row)))
        .and_then(|n| n.checked_add(row_size))
        .ok_or(Error::InvalidRange)?;
    if end > buffer_size {
        return Err(Error::InvalidRange);
    }
    footprint.Footprint.RowPitch = layout.bytes_per_row;
    Ok((0..region.extent[2])
        .map(|layer| {
            let mut placed = footprint;
            placed.Offset = layout.offset + u64::from(layer) * stride;
            (
                subresource + layer * desc.mip_levels,
                placed,
                rows,
                row_size,
            )
        })
        .collect())
}

pub(super) fn buffer_to_texture(
    encoder: &mut Dx12Encoder,
    src: &Dx12Buffer,
    layout: api::TexelCopyBufferLayout,
    dst: api::TextureRegion<'_, Dx12>,
) -> Result<(), Error> {
    // SAFETY: Core validates the destination identity; retaining its native handle covers GPU use.
    let texture = unsafe { dst.texture.native_ref() }.0.clone();
    if src.memory == api::MemoryClass::Readback
        || !src.usage.contains(api::BufferUsage::COPY_SRC)
        || !texture.desc.usage.contains(api::TextureUsage::COPY_DST)
    {
        return Err(Error::Unsupported("DX12 depth plane upload usage"));
    }
    let copies = plane_footprints(encoder, &texture, &dst, layout, src.size)?;
    let scratch = if copies.iter().any(|(_, fp, _, _)| fp.Offset % 512 != 0) {
        Some(scratch_buffer(
            encoder,
            u64::from(layout.bytes_per_row) * u64::from(copies[0].2),
        )?)
    } else {
        None
    };
    for (subresource, mut footprint, rows, row_size) in copies {
        let offset = footprint.Offset;
        let use_scratch = offset % 512 != 0;
        let source_resource = if use_scratch {
            let scratch = scratch.as_ref().unwrap();
            // SAFETY: validated rows fit both resources, which are retained through completion.
            unsafe {
                for row in 0..rows {
                    let position = u64::from(row) * u64::from(layout.bytes_per_row);
                    encoder.list.CopyBufferRegion(
                        scratch,
                        position,
                        &src.resource,
                        offset + position,
                        row_size,
                    );
                }
            }
            barrier(
                &encoder.list,
                scratch,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            );
            footprint.Offset = 0;
            scratch
        } else {
            &src.resource
        };
        let mut source = copy_location(source_resource, footprint, true, 0);
        let mut target = copy_location(
            &texture.resource,
            D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(),
            false,
            subresource,
        );
        // SAFETY: the whole plane and native footprint are checked; caller supplies COPY_DEST and source state.
        unsafe {
            encoder
                .list
                .CopyTextureRegion(&target, 0, 0, 0, &source, None);
            ManuallyDrop::drop(&mut source.pResource);
            ManuallyDrop::drop(&mut target.pResource);
        }
        if use_scratch {
            barrier(
                &encoder.list,
                scratch.as_ref().unwrap(),
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_COPY_DEST,
            );
        }
    }
    if let Some(scratch) = scratch {
        encoder.uploads.push(scratch);
    }
    encoder.textures.push(Dx12Texture(texture));
    encoder.buffers.push(src.clone());
    Ok(())
}
