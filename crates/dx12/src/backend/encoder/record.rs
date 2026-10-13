//! Direct command encoder construction and generic command recording.
//!
//! The methods here translate validated Core operations straight to the native
//! command list.  Resource transitions are supplied explicitly by the graph;
//! this module does not keep a global state tracker or derive future uses from
//! creation flags.  References retained by `Dx12Encoder` are GPU-lifetime
//! keep-alives, released only after the submission fence retires the work.

use super::*;
impl api::CommandBackend for Dx12 {
    fn begin_encoder(device: &Dx12Device, lane: LaneId) -> Result<Dx12Encoder, Error> {
        if lane.ordinal != 0 {
            return Err(Error::WrongLane);
        }
        Dx12Encoder::new(device)
    }

    fn transition(
        encoder: &mut Dx12Encoder,
        ops: &[api::ResourceTransition<'_, Self>],
    ) -> Result<(), Error> {
        for op in ops {
            match op {
                api::ResourceTransition::Buffer(barrier_op) => {
                    if barrier_op.ownership.is_some() {
                        return Err(Error::Unsupported("DX12 buffer ownership transfer"));
                    }
                    // SAFETY: Core checked the target's device identity; this
                    // borrows its native state only while a clone is retained below.
                    let buffer = unsafe { barrier_op.target.buffer.native_ref() };
                    if barrier_op.target.range != (0..buffer.size) {
                        return Err(Error::Unsupported("DX12 partial buffer transition"));
                    }
                    if buffer.memory == api::MemoryClass::Readback {
                        return Err(Error::Unsupported(
                            "DX12 readback buffers have fixed COPY_DEST state",
                        ));
                    }
                    let before = buffer_state(barrier_op.usage.from)?;
                    let after = buffer_state(barrier_op.usage.to)?;
                    barrier(&encoder.list, &buffer.resource, before, after);
                    encoder.buffers.push(buffer.clone());
                }
                api::ResourceTransition::Texture(barrier_op) => {
                    if barrier_op.ownership.is_some() {
                        return Err(Error::Unsupported("DX12 texture ownership transfer"));
                    }
                    // SAFETY: as for buffers, the typed texture is same-device
                    // and its clone enters encoder lifetime retention below.
                    let texture = unsafe { barrier_op.target.texture.native_ref() };
                    let desc = &texture.0.desc;
                    if barrier_op.target.aspects != api::TextureAspects::COLOR {
                        return Err(Error::Unsupported("DX12 texture transition aspect"));
                    }
                    if barrier_op.target.mips.start >= barrier_op.target.mips.end
                        || barrier_op.target.mips.end > desc.mip_levels
                        || barrier_op.target.layers.start >= barrier_op.target.layers.end
                        || barrier_op.target.layers.end > desc.array_layers
                    {
                        return Err(Error::InvalidRange);
                    }
                    let (from_layout, to_layout) = barrier_op.layout.ok_or(Error::Unsupported(
                        "DX12 texture transition requires explicit layouts",
                    ))?;
                    let before = texture_state(from_layout, barrier_op.usage.from)?;
                    let after = texture_state(to_layout, barrier_op.usage.to)?;
                    for layer in barrier_op.target.layers.clone() {
                        for mip in barrier_op.target.mips.clone() {
                            // D3D12 subresource order is mip-major within each
                            // array layer for the single color plane exposed by
                            // this RHI path. Core supplied this exact range.
                            let subresource = mip + layer * desc.mip_levels;
                            barrier_sub(
                                &encoder.list,
                                &texture.0.resource,
                                subresource,
                                before,
                                after,
                            );
                        }
                    }
                    encoder.textures.push(texture.clone());
                }
                _ => return Err(Error::Unsupported("DX12 resource transition")),
            }
        }
        Ok(())
    }

    fn copy_buffer(
        encoder: &mut Dx12Encoder,
        src: &super::Dx12Buffer,
        src_offset: u64,
        dst: &super::Dx12Buffer,
        dst_offset: u64,
        bytes: u64,
    ) -> Result<(), Error> {
        super::copy::buffer(encoder, src, src_offset, dst, dst_offset, bytes)
    }
    fn copy_texture(
        encoder: &mut Dx12Encoder,
        src: api::TextureRegion<'_, Self>,
        dst: api::TextureRegion<'_, Self>,
    ) -> Result<(), Error> {
        super::copy::texture(encoder, src, dst)
    }
    fn buffer_to_texture(
        encoder: &mut Dx12Encoder,
        src: &super::Dx12Buffer,
        layout: api::TexelCopyBufferLayout,
        dst: api::TextureRegion<'_, Self>,
    ) -> Result<(), Error> {
        super::copy::buffer_to_texture(encoder, src, layout, dst)
    }
    fn texture_to_buffer(
        encoder: &mut Dx12Encoder,
        src: api::TextureRegion<'_, Self>,
        dst: &super::Dx12Buffer,
        layout: api::TexelCopyBufferLayout,
    ) -> Result<(), Error> {
        super::copy::texture_to_buffer(encoder, src, dst, layout)
    }
    fn write_buffer(
        encoder: &mut Dx12Encoder,
        dst: &super::Dx12Buffer,
        offset: u64,
        bytes: &[u8],
    ) -> Result<(), Error> {
        if bytes.is_empty() {
            return Ok(());
        }
        let end = offset
            .checked_add(bytes.len() as u64)
            .ok_or(Error::InvalidRange)?;
        if end > dst.size {
            return Err(Error::InvalidRange);
        }
        let upload = upload_buffer(&dst.resource, bytes)?;
        // Default-heap buffers begin and end every upload in COMMON.  This is
        // an explicit stable boundary, so no hidden cross-frame state tracker
        // is needed for the minimal path.
        barrier(
            &encoder.list,
            &dst.resource,
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_COPY_DEST,
        );
        // SAFETY: both resources are retained by the encoder and range checks
        // above ensure the native copy does not address outside either buffer.
        unsafe {
            encoder
                .list
                .CopyBufferRegion(&dst.resource, offset, &upload, 0, bytes.len() as u64);
        }
        barrier(
            &encoder.list,
            &dst.resource,
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_COMMON,
        );
        encoder.uploads.push(upload);
        encoder.buffers.push(dst.clone());
        Ok(())
    }
    fn write_texture(
        encoder: &mut Dx12Encoder,
        dst: api::TextureRegion<'_, Self>,
        layout: api::TexelCopyBufferLayout,
        bytes: &[u8],
    ) -> Result<(), Error> {
        // SAFETY: Core tied the region to this device; the clone stored below
        // keeps the texture and its COM resource alive through submission.
        let texture = unsafe { dst.texture.native_ref() };
        if texture.0.desc.format != api::TextureFormat::Rgba8Unorm
            || dst.aspect != api::TextureAspects::COLOR
        {
            return Err(Error::Unsupported("DX12 texture write format or aspect"));
        }
        if !texture.0.desc.usage.contains(api::TextureUsage::COPY_DST) {
            return Err(Error::Unsupported(
                "DX12 texture write requires COPY_DST usage",
            ));
        }
        if dst.mip >= texture.0.desc.mip_levels
            || dst.extent[0] == 0
            || dst.extent[1] == 0
            || dst.extent[2] != 1
            || dst.origin[0]
                .checked_add(dst.extent[0])
                .is_none_or(|end| end > (texture.0.desc.extent[0] >> dst.mip).max(1))
            || dst.origin[1]
                .checked_add(dst.extent[1])
                .is_none_or(|end| end > (texture.0.desc.extent[1] >> dst.mip).max(1))
            || dst.origin[2]
                .checked_add(dst.extent[2])
                .is_none_or(|end| end > texture.0.desc.array_layers)
        {
            return Err(Error::InvalidRange);
        }
        let tight = dst.extent[0].checked_mul(4).ok_or(Error::InvalidRange)?;
        if layout.bytes_per_row < tight || layout.rows_per_image < dst.extent[1] {
            return Err(Error::InvalidRange);
        }
        let input_end = u64::from(layout.offset)
            .checked_add(u64::from(layout.bytes_per_row) * u64::from(dst.extent[1]))
            .ok_or(Error::InvalidRange)?;
        if input_end > bytes.len() as u64 {
            return Err(Error::InvalidRange);
        }
        // D3D12 enumerates mips consecutively inside each array layer.
        let subresource = dst.origin[2]
            .checked_mul(texture.0.desc.mip_levels)
            .and_then(|layer| layer.checked_add(dst.mip))
            .ok_or(Error::InvalidRange)?;
        // SAFETY: the retained resource exposes an immutable descriptor only.
        let desc = unsafe { texture.0.resource.GetDesc() };
        let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        let mut rows = 0;
        let mut row_size = 0;
        let mut total = 0;
        let mut device = None;
        // SAFETY: the resource is retained and D3D12 writes the device COM
        // reference only into the valid local Option output slot.
        unsafe {
            texture
                .0
                .resource
                .GetDevice::<windows::Win32::Graphics::Direct3D12::ID3D12Device>(&mut device)
        }
        .map_err(native_error)?;
        let device = device.ok_or_else(|| Error::Backend("texture has no D3D12 device".into()))?;
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
        if rows < dst.extent[1] || footprint.Footprint.RowPitch < tight {
            return Err(Error::Backend(
                "DX12 returned an invalid RGBA8 copy footprint".into(),
            ));
        }
        footprint.Footprint.Width = dst.extent[0];
        footprint.Footprint.Height = dst.extent[1];
        footprint.Footprint.Depth = 1;
        let upload_bytes = u64::from(footprint.Footprint.RowPitch)
            .checked_mul(u64::from(dst.extent[1]))
            .ok_or(Error::InvalidRange)?;
        let upload = texture_upload_buffer(&device, upload_bytes)?;
        let mut mapped = std::ptr::null_mut();
        // SAFETY: upload heap storage is CPU writable and Map initializes the
        // local pointer for the no-read range before row copies below.
        unsafe {
            upload.Map(
                0,
                Some(&D3D12_RANGE { Begin: 0, End: 0 }),
                Some(&mut mapped),
            )
        }
        .map_err(native_error)?;
        let source_base = usize::try_from(layout.offset).map_err(|_| Error::InvalidRange)?;
        let source_pitch =
            usize::try_from(layout.bytes_per_row).map_err(|_| Error::InvalidRange)?;
        let row_bytes = usize::try_from(tight).map_err(|_| Error::InvalidRange)?;
        for row in 0..dst.extent[1] as usize {
            let source = source_base
                .checked_add(row.checked_mul(source_pitch).ok_or(Error::InvalidRange)?)
                .ok_or(Error::InvalidRange)?;
            // SAFETY: `row` is bounded by the checked footprint height and
            // RowPitch, so the calculated address stays within the mapped upload range.
            let target = unsafe {
                mapped
                    .cast::<u8>()
                    .add(row * footprint.Footprint.RowPitch as usize)
            };
            let source_end = source.checked_add(row_bytes).ok_or(Error::InvalidRange)?;
            // SAFETY: source slice bounds and destination row bounds were
            // checked above; the two allocations cannot overlap.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    bytes[source..source_end].as_ptr(),
                    target,
                    row_bytes,
                );
            }
        }
        // SAFETY: the mapped upload allocation is no longer accessed after
        // this one matching Unmap call.
        unsafe {
            upload.Unmap(0, None);
        }
        barrier(
            &encoder.list,
            &texture.0.resource,
            D3D12_RESOURCE_STATE_COMMON,
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
            encoder
                .list
                .CopyTextureRegion(&target, dst.origin[0], dst.origin[1], 0, &source, None);
        }
        // SAFETY: destroy exactly the cloned COM references installed in the
        // FFI unions after D3D12 synchronously consumed the locations.
        unsafe {
            ManuallyDrop::drop(&mut source.pResource);
            ManuallyDrop::drop(&mut target.pResource);
        }
        barrier(
            &encoder.list,
            &texture.0.resource,
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_COMMON,
        );
        encoder.uploads.push(upload);
        encoder.textures.push(texture.clone());
        Ok(())
    }
    fn blit(
        encoder: &mut Dx12Encoder,
        src: api::TextureRegion<'_, Self>,
        dst: api::TextureRegion<'_, Self>,
        filter: api::FilterMode,
    ) -> Result<(), Error> {
        if filter != api::FilterMode::Linear {
            return Err(Error::Unsupported("DX12 mip blit filter"));
        }
        // SAFETY: Core validated both regions against this encoder's device;
        // cloning state extends source and target lifetime through the blit.
        let source = unsafe { src.texture.native_ref() }.0.clone();
        let target = unsafe { dst.texture.native_ref() }.0.clone();
        validate_mip_blit(&source, &src, true)?;
        validate_mip_blit(&target, &dst, false)?;
        if !source.desc.usage.contains(api::TextureUsage::COPY_SRC)
            || !target.desc.usage.contains(api::TextureUsage::COPY_DST)
        {
            return Err(Error::Unsupported(
                "DX12 mip blit requires COPY_SRC and COPY_DST",
            ));
        }
        linear_mip_blit(
            encoder,
            &source.resource,
            src.mip,
            &target.resource,
            dst.mip,
            dst.extent[0],
            dst.extent[1],
        )?;
        encoder.textures.push(Dx12Texture(source));
        encoder.textures.push(Dx12Texture(target));
        Ok(())
    }
}
