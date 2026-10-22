//! Direct command encoder construction and generic command recording.
//!
//! The methods here translate validated Core operations straight to the native
//! command list.  Resource transitions are supplied explicitly by the graph;
//! this module does not keep a global state tracker or derive future uses from
//! creation flags.  References retained by `Dx12Encoder` are GPU-lifetime
//! keep-alives, released only after the submission fence retires the work.

use super::*;
impl api::CommandBackend for Dx12 {
    fn clear_texture(
        encoder: &mut Dx12Encoder,
        texture: &super::Dx12Texture,
        mips: std::ops::Range<u32>,
        layers: std::ops::Range<u32>,
        value: api::TextureClearValue,
    ) -> Result<(), Error> {
        if encoder.is_compute {
            return Err(Error::Unsupported("DX12 compute queue texture clear"));
        }
        super::clear::texture(encoder, texture, mips, layers, value)
    }
    fn begin_encoder(device: &Dx12Device, lane: LaneId) -> Result<Dx12Encoder, Error> {
        if lane.ordinal > 1 {
            return Err(Error::WrongLane);
        }
        Dx12Encoder::new(device, lane)
    }

    fn transition(
        encoder: &mut Dx12Encoder,
        ops: &[api::ResourceTransition<'_, Self>],
    ) -> Result<(), Error> {
        for op in ops {
            match op {
                api::ResourceTransition::Buffer(barrier_op) => {
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
                    if buffer.memory == api::MemoryClass::Upload {
                        if barrier_op.ownership.is_some() {
                            return Err(Error::Unsupported(
                                "DX12 upload buffer ownership transfer",
                            ));
                        }
                        let fixed =
                            windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATE_GENERIC_READ;
                        if before.0 & !fixed.0 != 0 || after.0 & !fixed.0 != 0 {
                            return Err(Error::Unsupported(
                                "DX12 upload buffers have fixed GENERIC_READ state",
                            ));
                        }
                        encoder.buffers.push(buffer.clone());
                        continue;
                    }
                    let (before, after) = if let Some(transfer) = barrier_op.ownership {
                        super::ownership::states(
                            encoder,
                            transfer,
                            before,
                            after,
                            barrier_op.usage.from.stages,
                            barrier_op.usage.to.stages,
                        )?
                    } else {
                        validate_compute_states(encoder, before, after)?;
                        (before, after)
                    };
                    barrier(&encoder.list, &buffer.resource, before, after);
                    encoder.buffers.push(buffer.clone());
                }
                api::ResourceTransition::Texture(barrier_op) => {
                    // SAFETY: as for buffers, the typed texture is same-device
                    // and its clone enters encoder lifetime retention below.
                    let texture = unsafe { barrier_op.target.texture.native_ref() };
                    let desc = &texture.0.desc;
                    let expected_aspect = if super::super::texture::is_depth(desc.format) {
                        super::super::texture::depth_aspects(desc.format)
                    } else {
                        api::TextureAspects::COLOR
                    };
                    let aspects = barrier_op.target.aspects.0;
                    if aspects == 0 || aspects & !expected_aspect.0 != 0 {
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
                    let depth = super::super::texture::is_depth(desc.format);
                    if [from_layout, to_layout].into_iter().any(|layout| {
                        (depth && layout == api::ImageLayout::ColorAttachment)
                            || (!depth && layout == api::ImageLayout::DepthStencilAttachment)
                    }) {
                        return Err(Error::Unsupported("DX12 texture layout/format mismatch"));
                    }
                    let before = if depth && from_layout == api::ImageLayout::Undefined {
                        // Depth resources are created in DEPTH_WRITE. Undefined
                        // names the initial contents, not a native COMMON state.
                        if barrier_op.usage.from.accesses != api::AccessMask::EMPTY {
                            return Err(Error::Unsupported("DX12 Undefined layout access"));
                        }
                        D3D12_RESOURCE_STATE_DEPTH_WRITE
                    } else {
                        texture_state(from_layout, barrier_op.usage.from)?
                    };
                    let after = texture_state(to_layout, barrier_op.usage.to)?;
                    let (before, after) = if let Some(transfer) = barrier_op.ownership {
                        super::ownership::states(
                            encoder,
                            transfer,
                            before,
                            after,
                            barrier_op.usage.from.stages,
                            barrier_op.usage.to.stages,
                        )?
                    } else {
                        validate_compute_states(encoder, before, after)?;
                        (before, after)
                    };
                    let discard = texture.0.sparse.is_some()
                        && from_layout == api::ImageLayout::Undefined
                        && barrier_op.ownership.is_none()
                        && ((!depth && after == D3D12_RESOURCE_STATE_RENDER_TARGET)
                            || (depth && after == D3D12_RESOURCE_STATE_DEPTH_WRITE));
                    if discard && encoder.is_compute {
                        return Err(Error::Unsupported(
                            "DX12 sparse attachment discard requires graphics queue",
                        ));
                    }
                    for layer in barrier_op.target.layers.clone() {
                        for mip in barrier_op.target.mips.clone() {
                            for plane in 0..2 {
                                let bit = if plane == 0 {
                                    if depth {
                                        api::TextureAspects::DEPTH.0
                                    } else {
                                        api::TextureAspects::COLOR.0
                                    }
                                } else {
                                    api::TextureAspects::STENCIL.0
                                };
                                if aspects & bit == 0 {
                                    continue;
                                }
                                let subresource = mip
                                    + layer * desc.mip_levels
                                    + plane * desc.mip_levels * desc.array_layers;
                                barrier_sub(
                                    &encoder.list,
                                    &texture.0.resource,
                                    subresource,
                                    before,
                                    after,
                                );
                                if discard {
                                    let region = D3D12_DISCARD_REGION {
                                        NumRects: 0,
                                        pRects: std::ptr::null(),
                                        FirstSubresource: subresource,
                                        NumSubresources: 1,
                                    };
                                    // SAFETY: Undefined explicitly relinquishes these contents,
                                    // the transition above establishes the required attachment
                                    // state, and the region selects this complete subresource.
                                    unsafe {
                                        encoder
                                            .list
                                            .DiscardResource(&texture.0.resource, Some(&region));
                                    }
                                }
                            }
                        }
                    }
                    encoder.textures.push(texture.clone());
                }
                api::ResourceTransition::Acceleration {
                    target,
                    before,
                    after,
                } => {
                    if !before.accesses.contains(api::AccessMask::ACCEL_WRITE)
                        || !(after.accesses.contains(api::AccessMask::ACCEL_READ)
                            || after.accesses.contains(api::AccessMask::ACCEL_WRITE))
                    {
                        return Err(Error::Unsupported("DX12 acceleration barrier scope"));
                    }
                    // SAFETY: Core checked the acceleration structure's device;
                    // the encoder retains its native resource through completion.
                    let acceleration = unsafe { target.native_ref() };
                    let mut native = D3D12_RESOURCE_BARRIER {
                        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
                        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                        Anonymous: D3D12_RESOURCE_BARRIER_0 {
                            UAV: ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                                pResource: ManuallyDrop::new(Some(acceleration.0.resource.clone())),
                            }),
                        },
                    };
                    // SAFETY: the barrier owns a COM clone for the duration of
                    // this synchronous recording call, then releases it once.
                    unsafe {
                        encoder.list.ResourceBarrier(std::slice::from_ref(&native));
                        ManuallyDrop::drop(&mut (*native.Anonymous.UAV).pResource);
                    }
                    encoder.accelerations.push(acceleration.clone());
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
        if dst.memory != api::MemoryClass::Device || !dst.usage.contains(api::BufferUsage::COPY_DST)
        {
            return Err(Error::Unsupported(
                "DX12 write_buffer requires a device COPY_DST buffer",
            ));
        }
        if bytes.is_empty() {
            return Ok(());
        }
        let end = offset
            .checked_add(bytes.len() as u64)
            .ok_or(Error::InvalidRange)?;
        if end > dst.size {
            return Err(Error::InvalidRange);
        }
        let (upload, upload_offset) = encoder
            .upload_arena
            .write(&encoder.device.0.native, bytes)?;
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
            encoder.list.CopyBufferRegion(
                &dst.resource,
                offset,
                &upload,
                upload_offset,
                bytes.len() as u64,
            );
        }
        barrier(
            &encoder.list,
            &dst.resource,
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_COMMON,
        );
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
                (
                    F::Depth24PlusStencil8 | F::Depth32FloatStencil8,
                    api::TextureAspects::STENCIL,
                ) => Some(1),
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
        if rows < rows_needed || footprint.Footprint.RowPitch < tight {
            return Err(Error::Backend(
                "DX12 returned an invalid RGBA8 copy footprint".into(),
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
        let image_stride = if is_array {
            image_bytes.checked_add(511).ok_or(Error::InvalidRange)? & !511
        } else {
            image_bytes
        };
        let upload_bytes = image_stride
            .checked_mul(u64::from(dst.extent[2] - 1))
            .and_then(|n| n.checked_add(image_bytes))
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
        for slice in 0..dst.extent[2] as usize {
            for row in 0..rows_needed as usize {
                let source = source_base
                    .checked_add(
                        (slice * layout.rows_per_image as usize + row)
                            .checked_mul(source_pitch)
                            .ok_or(Error::InvalidRange)?,
                    )
                    .ok_or(Error::InvalidRange)?;
                // SAFETY: `row` is bounded by the checked footprint height and
                // RowPitch, so the calculated address stays within the mapped upload range.
                let target = unsafe {
                    mapped.cast::<u8>().add(
                        slice * image_stride as usize + row * footprint.Footprint.RowPitch as usize,
                    )
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
        }
        // SAFETY: the mapped upload allocation is no longer accessed after
        // this one matching Unmap call.
        unsafe {
            upload.Unmap(0, None);
        }
        let stable_state = if depth {
            D3D12_RESOURCE_STATE_DEPTH_WRITE
        } else {
            D3D12_RESOURCE_STATE_COMMON
        };
        let layers = if is_array { dst.extent[2] } else { 1 };
        for layer in 0..layers {
            let subresource = subresource + layer * texture.0.desc.mip_levels;
            footprint.Offset = u64::from(layer) * image_stride;
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
        // SAFETY: Core validated both regions against this encoder's device;
        // cloning state extends source and target lifetime through the blit.
        let source = unsafe { src.texture.native_ref() }.0.clone();
        let target = unsafe { dst.texture.native_ref() }.0.clone();
        validate_mip_blit(&source, &src, true)?;
        validate_mip_blit(&target, &dst, false)?;
        let source_3d = source.desc.dimension == api::TextureDimension::D3;
        let target_3d = target.desc.dimension == api::TextureDimension::D3;
        if source_3d != target_3d || (!source_3d && src.extent[2] != dst.extent[2]) {
            return Err(Error::Unsupported("DX12 blit dimensionality"));
        }
        if source.resource == target.resource
            && src.mip == dst.mip
            && (source_3d
                || (src.origin[2] < dst.origin[2] + dst.extent[2]
                    && dst.origin[2] < src.origin[2] + src.extent[2]))
        {
            return Err(Error::Unsupported("DX12 same-subresource blit"));
        }
        // SAFETY: these immutable native descriptors belong to retained resources.
        let source_format = super::super::texture::dxgi_format(source.desc.format);
        let target_desc = unsafe { target.resource.GetDesc() };
        let target_format = super::super::texture::dxgi_format(target.desc.format);
        let kind = super::mipmap::blit_kind(source.desc.format);
        if kind != super::mipmap::blit_kind(target.desc.format)
            || (kind != 0 && filter != api::FilterMode::Nearest)
        {
            return Err(Error::Unsupported("DX12 blit numeric type or filter"));
        }
        let source_support = if kind == 0 {
            D3D12_FORMAT_SUPPORT1_SHADER_SAMPLE
        } else {
            D3D12_FORMAT_SUPPORT1_SHADER_LOAD
        };
        if !super::super::texture::format_has_support(
            &encoder.device,
            source.desc.format,
            source_support,
        ) {
            return Err(Error::Unsupported("DX12 blit source format support"));
        }
        let render_target_supported = super::super::texture::format_has_support(
            &encoder.device,
            target.desc.format,
            D3D12_FORMAT_SUPPORT1_RENDER_TARGET,
        );
        if !source.desc.usage.contains(api::TextureUsage::COPY_SRC)
            || !target.desc.usage.contains(api::TextureUsage::COPY_DST)
        {
            return Err(Error::Unsupported(
                "DX12 mip blit requires COPY_SRC and COPY_DST",
            ));
        }
        // Keep both resources alive even if a later per-layer allocation fails
        // after earlier layers have already recorded commands.
        encoder.textures.push(Dx12Texture(source.clone()));
        encoder.textures.push(Dx12Texture(target.clone()));
        if encoder.is_compute || !render_target_supported {
            super::blit_compute::blit_compute(encoder, &source, &target, &src, &dst, filter)?;
            if !encoder.is_compute {
                super::restore_after_internal_draw(encoder)?;
            }
            return Ok(());
        }
        for layer in 0..dst.extent[2] {
            let temporary = if target_desc
                .Flags
                .contains(D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET)
            {
                None
            } else {
                Some(super::mipmap::temporary_blit_target(
                    encoder,
                    [dst.extent[0], dst.extent[1]],
                    target_format,
                    target.desc.dimension,
                )?)
            };
            if let Some(resource) = &temporary {
                encoder.uploads.push(resource.clone());
            }
            let target_resource = temporary.as_ref().unwrap_or(&target.resource);
            let target_mip = if temporary.is_some() { 0 } else { dst.mip };
            let target_layer = if temporary.is_some() {
                0
            } else {
                dst.origin[2] + layer
            };
            let target_rect = if temporary.is_some() {
                [0, 0, dst.extent[0], dst.extent[1]]
            } else {
                [dst.origin[0], dst.origin[1], dst.extent[0], dst.extent[1]]
            };
            linear_mip_blit(
                encoder,
                &source.resource,
                src.mip,
                if source_3d { 0 } else { src.origin[2] + layer },
                source.desc.mip_levels,
                match source.desc.dimension {
                    api::TextureDimension::D1 => 2,
                    api::TextureDimension::D3 => 3,
                    api::TextureDimension::D2 => u8::from(source.desc.array_layers > 1),
                },
                if source_3d {
                    (src.origin[2] as f32
                        + (layer as f32 + 0.5) * src.extent[2] as f32 / dst.extent[2] as f32)
                        / (source.desc.extent[2] >> src.mip).max(1) as f32
                } else {
                    0.0
                },
                kind,
                [src.origin[0], src.origin[1], src.extent[0], src.extent[1]],
                [
                    (source.desc.extent[0] >> src.mip).max(1),
                    (source.desc.extent[1] >> src.mip).max(1),
                ],
                source_format,
                target_resource,
                target_mip,
                target_layer,
                if temporary.is_some() {
                    1
                } else {
                    target.desc.mip_levels
                },
                target.desc.dimension,
                temporary.is_none() && target.desc.array_layers > 1,
                target_rect,
                target_format,
                filter,
            )?;
            if let Some(temporary) = temporary {
                super::mipmap::copy_temporary_blit(
                    encoder,
                    temporary,
                    &target.resource,
                    dst.mip
                        + if target_3d {
                            0
                        } else {
                            (dst.origin[2] + layer) * target.desc.mip_levels
                        },
                    [
                        dst.origin[0],
                        dst.origin[1],
                        if target_3d { dst.origin[2] + layer } else { 0 },
                    ],
                );
            }
        }
        Ok(())
    }
}

fn validate_compute_states(
    encoder: &Dx12Encoder,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) -> Result<(), Error> {
    if encoder.is_compute {
        let allowed = D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER.0
            | D3D12_RESOURCE_STATE_UNORDERED_ACCESS.0
            | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE.0
            | D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT.0
            | D3D12_RESOURCE_STATE_COPY_SOURCE.0
            | D3D12_RESOURCE_STATE_COPY_DEST.0;
        if (before.0 | after.0) & !allowed != 0 {
            return Err(Error::Unsupported("DX12 compute queue resource state"));
        }
    }
    Ok(())
}
