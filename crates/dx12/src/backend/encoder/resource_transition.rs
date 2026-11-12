//! Explicit resource transitions and first-use native attachment initialization.
//!
//! Public layouts remain the caller-selected boundary states. An Undefined
//! placed attachment may require a private discard before its first native use;
//! this does not discard contents of a previously initialized subresource.

use super::*;

pub(super) fn record(
    encoder: &mut Dx12Encoder,
    ops: &[api::ResourceTransition<'_, Dx12>],
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
                        return Err(Error::Unsupported("DX12 upload buffer ownership transfer"));
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
                // COPY_DST color textures can carry a private native RTV flag for
                // blit lowering. Their uninitialized mips need the same first-use
                // metadata initialization as explicit attachment transitions.
                // SAFETY: the descriptor belongs to the retained native resource.
                let native_flags = unsafe { texture.0.resource.GetDesc() }.Flags;
                let initialize_copy_target = !encoder.is_compute
                    && !depth
                    && after == D3D12_RESOURCE_STATE_COPY_DEST
                    && native_flags.contains(D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
                let discard = (texture.0.sparse.is_some() || texture.0.backing_heap.is_some())
                    && from_layout == api::ImageLayout::Undefined
                    && barrier_op.ownership.is_none()
                    && (initialize_copy_target
                        || (!depth && after == D3D12_RESOURCE_STATE_RENDER_TARGET)
                        || (depth && after == D3D12_RESOURCE_STATE_DEPTH_WRITE));
                if discard && encoder.is_compute {
                    return Err(Error::Unsupported(
                        "DX12 attachment discard requires graphics queue",
                    ));
                }
                let initialization_state = if discard && initialize_copy_target {
                    D3D12_RESOURCE_STATE_RENDER_TARGET
                } else {
                    after
                };
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
                                initialization_state,
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
                                barrier_sub(
                                    &encoder.list,
                                    &texture.0.resource,
                                    subresource,
                                    initialization_state,
                                    after,
                                );
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
