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
                    let texture = unsafe { barrier_op.target.texture.native_ref() };
                    let desc = &texture.0.desc;
                    if barrier_op.target.aspects != api::TextureAspects::COLOR
                        || barrier_op.target.mips != (0..desc.mip_levels)
                        || barrier_op.target.layers != (0..desc.array_layers)
                    {
                        return Err(Error::Unsupported("DX12 partial texture transition"));
                    }
                    let (from_layout, to_layout) = barrier_op.layout.ok_or(Error::Unsupported(
                        "DX12 texture transition requires explicit layouts",
                    ))?;
                    let before = texture_state(from_layout, barrier_op.usage.from)?;
                    let after = texture_state(to_layout, barrier_op.usage.to)?;
                    barrier(&encoder.list, &texture.0.resource, before, after);
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
        let desc = unsafe { texture.0.resource.GetDesc() };
        let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        let mut rows = 0;
        let mut row_size = 0;
        let mut total = 0;
        let mut device = None;
        unsafe {
            texture
                .0
                .resource
                .GetDevice::<windows::Win32::Graphics::Direct3D12::ID3D12Device>(&mut device)
        }
        .map_err(native_error)?;
        let device = device.ok_or_else(|| Error::Backend("texture has no D3D12 device".into()))?;
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
            let target = unsafe {
                mapped
                    .cast::<u8>()
                    .add(row * footprint.Footprint.RowPitch as usize)
            };
            let source_end = source.checked_add(row_bytes).ok_or(Error::InvalidRange)?;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    bytes[source..source_end].as_ptr(),
                    target,
                    row_bytes,
                );
            }
        }
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
        unsafe {
            encoder
                .list
                .CopyTextureRegion(&target, dst.origin[0], dst.origin[1], 0, &source, None);
        }
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
