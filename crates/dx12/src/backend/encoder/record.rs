//! Direct command encoder construction and generic command recording.
//!
//! The methods here translate validated Core operations straight to the native
//! command list.  Resource transitions are supplied explicitly by the graph;
//! this module does not keep a global state tracker or derive future uses from
//! creation flags.  References retained by `Dx12Encoder` are GPU-lifetime
//! keep-alives, released only after the submission fence retires the work.

use super::*;
impl fluxel_rhi_core::backend::CommandBackend for Dx12 {
    unsafe fn wait_for_residency(
        encoder: &mut Dx12Encoder,
        point: &Dx12Completion,
    ) -> Result<(), Error> {
        if encoder.device.0.residency_device.is_none() {
            return Err(Error::Unsupported("DX12 residency queue waits"));
        }
        encoder.device.require_active()?;
        encoder.split_queue_boundary(super::external_segment::QueueBoundary::Wait {
            fence: point.fence.clone(),
            value: point.value,
        })
    }
    fn debug_marker(
        encoder: &mut Dx12Encoder,
        _: fluxel_rhi_core::backend::DebugMarkerScope,
        op: fluxel_rhi_core::backend::DebugMarkerOp<'_>,
    ) -> Result<(), Error> {
        use std::ffi::c_void;

        // PIX recognizes UTF-16 event data on a command list. The vector owns
        // the conversion until the synchronous native call returns.
        let label = |text: &str| text.encode_utf16().chain(Some(0)).collect::<Vec<_>>();
        match op {
            fluxel_rhi_core::backend::DebugMarkerOp::Insert(text) => {
                let label = label(text);
                // SAFETY: the command list is open and the temporary UTF-16
                // buffer remains live for SetMarker's synchronous copy.
                unsafe {
                    encoder.list.SetMarker(
                        0,
                        Some(label.as_ptr().cast::<c_void>()),
                        (label.len() * std::mem::size_of::<u16>()) as u32,
                    );
                }
            }
            fluxel_rhi_core::backend::DebugMarkerOp::Push(text) => {
                let label = label(text);
                // SAFETY: as above for SetMarker.
                unsafe {
                    encoder.list.BeginEvent(
                        0,
                        Some(label.as_ptr().cast::<c_void>()),
                        (label.len() * std::mem::size_of::<u16>()) as u32,
                    );
                }
                encoder.debug_group_depth = encoder
                    .debug_group_depth
                    .checked_add(1)
                    .ok_or(Error::InvalidState)?;
            }
            fluxel_rhi_core::backend::DebugMarkerOp::Pop => {
                if encoder.debug_group_depth == 0 {
                    return Err(Error::InvalidState);
                }
                // SAFETY: the list is open and the native depth was checked.
                unsafe { encoder.list.EndEvent() };
                encoder.debug_group_depth -= 1;
            }
        }
        Ok(())
    }

    fn resolve_support(
        encoder: &Dx12Encoder,
        src: &api::TextureRegion<'_, Self>,
        dst: &api::TextureRegion<'_, Self>,
    ) -> Result<(), Error> {
        super::resolve::support(encoder, src, dst)
    }

    fn resolve(
        encoder: &mut Dx12Encoder,
        src: api::TextureRegion<'_, Self>,
        dst: api::TextureRegion<'_, Self>,
    ) -> Result<(), Error> {
        super::resolve::texture(encoder, src, dst)
    }
    fn clear_buffer(
        encoder: &mut Dx12Encoder,
        dst: &super::Dx12Buffer,
        range: std::ops::Range<u64>,
        value: u32,
    ) -> Result<(), Error> {
        super::buffer_clear::record(encoder, dst, range, value)
    }
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
                    if transition::needs_uav_barrier(
                        before,
                        after,
                        barrier_op.usage.from.accesses,
                        barrier_op.usage.to.accesses,
                    ) {
                        transition::uav_barrier(&encoder.list, &buffer.resource);
                    }
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
                    let discard = (texture.0.sparse.is_some() || texture.0.backing_heap.is_some())
                        && from_layout == api::ImageLayout::Undefined
                        && barrier_op.ownership.is_none()
                        && ((!depth && after == D3D12_RESOURCE_STATE_RENDER_TARGET)
                            || (depth && after == D3D12_RESOURCE_STATE_DEPTH_WRITE));
                    if discard && encoder.is_compute {
                        return Err(Error::Unsupported(
                            "DX12 attachment discard requires graphics queue",
                        ));
                    }
                    // Legacy UAV barriers cover the entire resource. Emit at
                    // most one, outside the mip/layer/plane transition loop.
                    if transition::needs_uav_barrier(
                        before,
                        after,
                        barrier_op.usage.from.accesses,
                        barrier_op.usage.to.accesses,
                    ) {
                        transition::uav_barrier(&encoder.list, &texture.0.resource);
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
                                    // SAFETY: Undefined explicitly relinquishes these contents.
                                    // This is needed for sparse and placed identities: an alias
                                    // barrier does not initialize a newly activated RT/DS
                                    // resource. The transition establishes attachment state and
                                    // the region selects this complete subresource.
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
                api::ResourceTransition::Aliasing { before, after } => {
                    super::aliasing::record(encoder, before, after);
                }
                api::ResourceTransition::Global { before, after } => {
                    super::global_barrier::record(encoder, *before, *after)?;
                }
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
        super::upload::write_buffer(encoder, dst, offset, bytes)
    }
    fn write_buffer_with(
        encoder: &mut Dx12Encoder,
        dst: &super::Dx12Buffer,
        offset: u64,
        size: usize,
        writer: impl FnOnce(&mut [u8]) -> bool,
    ) -> Result<(), Error> {
        super::upload::write_buffer_with(encoder, dst, offset, size, writer)
    }
    fn write_texture(
        encoder: &mut Dx12Encoder,
        dst: api::TextureRegion<'_, Self>,
        layout: api::TexelCopyBufferLayout,
        bytes: &[u8],
    ) -> Result<(), Error> {
        super::texture_upload::write(encoder, dst, layout, bytes)
    }
    fn write_texture_with(
        encoder: &mut Dx12Encoder,
        dst: api::TextureRegion<'_, Self>,
        writer: impl FnOnce(&mut [u8], api::TextureUploadLayout) -> bool,
    ) -> Result<(), Error> {
        super::texture_upload::write_with(encoder, dst, writer)
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
