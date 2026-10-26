//! Explicit Core resource-transition lowering for D3D12 barriers.
//!
//! The graph owns the before/after usage contract.  This module only maps that
//! supplied contract to D3D12 states and emits barriers for the stated range;
//! it does not remember a resource's last state, infer a later use, or perform
//! queue ownership scheduling.  Private upload and readback helpers use their
//! own fixed states outside this cross-operation contract.

use super::*;
pub(crate) fn barrier_sub(
    list: &ID3D12GraphicsCommandList,
    resource: &ID3D12Resource,
    subresource: u32,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) {
    if before == after {
        // A state transition is unnecessary, but dependent UAV accesses still
        // need ordering. Legacy UAV barriers cover the entire resource even
        // when the caller names only one texture subresource.
        if before.contains(D3D12_RESOURCE_STATE_UNORDERED_ACCESS) {
            let mut barrier = D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
                Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                Anonymous: D3D12_RESOURCE_BARRIER_0 {
                    UAV: ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                        pResource: ManuallyDrop::new(Some(resource.clone())),
                    }),
                },
            };
            // SAFETY: the descriptor owns a temporary COM reference; D3D12
            // consumes the borrowed descriptor synchronously while recording.
            unsafe {
                list.ResourceBarrier(std::slice::from_ref(&barrier));
                ManuallyDrop::drop(&mut (*barrier.Anonymous.UAV).pResource);
            }
        }
        return;
    }
    let mut b = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                pResource: ManuallyDrop::new(Some(resource.clone())),
                Subresource: subresource,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    };
    // SAFETY: `b` owns a cloned COM reference for the call; `ResourceBarrier`
    // consumes only the borrowed slice synchronously, then we drop precisely
    // that manually-managed reference before `b` leaves scope.
    unsafe {
        list.ResourceBarrier(std::slice::from_ref(&b));
        ManuallyDrop::drop(&mut (*b.Anonymous.Transition).pResource);
    }
}

/// Maps a graph-selected image layout to one DX12 resource state.  The
/// layout is mandatory: access masks validate the caller's claim but never
/// select a native state on their own.
pub(crate) fn texture_state(
    layout: api::ImageLayout,
    uses: api::TextureUses,
) -> Result<windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATES, Error> {
    match layout {
        // Undefined can describe a never-used resource, but a completed
        // upload has defined contents and must use Common instead.
        api::ImageLayout::Undefined => {
            if uses.accesses == api::AccessMask::EMPTY {
                Ok(D3D12_RESOURCE_STATE_COMMON)
            } else {
                Err(Error::Unsupported("DX12 Undefined layout access"))
            }
        }
        api::ImageLayout::Common => Ok(D3D12_RESOURCE_STATE_COMMON),
        api::ImageLayout::CopySrc => {
            if uses.accesses.contains(api::AccessMask::COPY_READ) {
                Ok(D3D12_RESOURCE_STATE_COPY_SOURCE)
            } else {
                Err(Error::Unsupported("DX12 CopySrc layout access"))
            }
        }
        api::ImageLayout::CopyDst => {
            if uses.accesses.contains(api::AccessMask::COPY_WRITE) {
                Ok(D3D12_RESOURCE_STATE_COPY_DEST)
            } else {
                Err(Error::Unsupported("DX12 CopyDst layout access"))
            }
        }
        api::ImageLayout::ResolveSrc => {
            if uses.stages.contains(api::StageMask::COLOR_OUTPUT)
                && uses.accesses.contains(api::AccessMask::COLOR_READ)
            {
                Ok(D3D12_RESOURCE_STATE_RESOLVE_SOURCE)
            } else {
                Err(Error::Unsupported("DX12 ResolveSrc layout access"))
            }
        }
        api::ImageLayout::ResolveDst => {
            if uses.stages.contains(api::StageMask::COLOR_OUTPUT)
                && uses.accesses.contains(api::AccessMask::COLOR_WRITE)
            {
                Ok(D3D12_RESOURCE_STATE_RESOLVE_DEST)
            } else {
                Err(Error::Unsupported("DX12 ResolveDst layout access"))
            }
        }
        api::ImageLayout::General => {
            let storage_access = api::AccessMask::SHADER_READ.union(api::AccessMask::SHADER_WRITE);
            if has_shader_stage(uses.stages)
                && uses.accesses.0 != 0
                && uses.accesses.0 & !storage_access.0 == 0
            {
                Ok(D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
            } else {
                Err(Error::Unsupported("DX12 General layout access"))
            }
        }
        api::ImageLayout::ColorAttachment => {
            if uses.accesses.contains(api::AccessMask::COLOR_WRITE)
                || uses.accesses.contains(api::AccessMask::COLOR_READ)
            {
                Ok(D3D12_RESOURCE_STATE_RENDER_TARGET)
            } else {
                Err(Error::Unsupported("DX12 ColorAttachment layout access"))
            }
        }
        api::ImageLayout::DepthStencilAttachment => {
            if uses.accesses.contains(api::AccessMask::DEPTH_WRITE)
                || uses.accesses.contains(api::AccessMask::STENCIL_WRITE)
            {
                Ok(D3D12_RESOURCE_STATE_DEPTH_WRITE)
            } else if uses.accesses.contains(api::AccessMask::DEPTH_READ)
                || uses.accesses.contains(api::AccessMask::STENCIL_READ)
            {
                Ok(D3D12_RESOURCE_STATE_DEPTH_READ)
            } else {
                Err(Error::Unsupported(
                    "DX12 DepthStencilAttachment layout access",
                ))
            }
        }
        api::ImageLayout::DepthReadOnly => {
            if !(uses.accesses.contains(api::AccessMask::DEPTH_READ)
                || uses.accesses.contains(api::AccessMask::STENCIL_READ)
                || uses.accesses.contains(api::AccessMask::SHADER_READ))
                || uses.accesses.contains(api::AccessMask::DEPTH_WRITE)
                || uses.accesses.contains(api::AccessMask::STENCIL_WRITE)
            {
                return Err(Error::Unsupported("DX12 DepthReadOnly layout access"));
            }
            let mut state = D3D12_RESOURCE_STATE_DEPTH_READ;
            if uses.accesses.contains(api::AccessMask::SHADER_READ) {
                if uses.stages.contains(api::StageMask::FRAGMENT) {
                    state |= D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
                }
                if uses.stages.contains(api::StageMask::VERTEX)
                    || uses.stages.contains(api::StageMask::TASK)
                    || uses.stages.contains(api::StageMask::MESH)
                    || uses.stages.contains(api::StageMask::COMPUTE)
                    || uses.stages.contains(api::StageMask::RAY)
                {
                    state |= D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
                }
                if state == D3D12_RESOURCE_STATE_DEPTH_READ {
                    return Err(Error::Unsupported("DX12 DepthReadOnly shader stage"));
                }
            }
            Ok(state)
        }
        api::ImageLayout::Present if uses.accesses == api::AccessMask::EMPTY => {
            Ok(D3D12_RESOURCE_STATE_PRESENT)
        }
        api::ImageLayout::ShaderReadOnly => {
            if !uses.accesses.contains(api::AccessMask::SHADER_READ) {
                return Err(Error::Unsupported("DX12 ShaderReadOnly layout access"));
            }
            let mut state = windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATES(0);
            if uses.stages.contains(api::StageMask::VERTEX)
                || uses.stages.contains(api::StageMask::TASK)
                || uses.stages.contains(api::StageMask::MESH)
                || uses.stages.contains(api::StageMask::COMPUTE)
                || uses.stages.contains(api::StageMask::RAY)
            {
                state |= windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            }
            if uses.stages.contains(api::StageMask::FRAGMENT) {
                state |= D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
            }
            if state.0 == 0 {
                Err(Error::Unsupported("DX12 ShaderReadOnly stage"))
            } else {
                Ok(state)
            }
        }
        _ => Err(Error::Unsupported("DX12 texture image layout")),
    }
}

fn has_shader_stage(stages: api::StageMask) -> bool {
    let shader_stages = api::StageMask::VERTEX
        .union(api::StageMask::FRAGMENT)
        .union(api::StageMask::COMPUTE)
        .union(api::StageMask::TASK)
        .union(api::StageMask::MESH)
        .union(api::StageMask::RAY);
    stages.0 & shader_stages.0 != 0
}

/// Lowers the graph-selected whole-buffer use into the corresponding D3D12
/// state.  Buffers do not have layouts, so callers must state both sides of
/// every cross-operation transition with [`api::BufferBarrier`].  Convenience
/// writes manage their own COMMON state; indirect execution requires an
/// explicit transition to INDIRECT_READ.
pub(crate) fn buffer_state(
    uses: api::BufferUses,
) -> Result<windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATES, Error> {
    let read = uses.accesses;
    if read == api::AccessMask::EMPTY {
        return Ok(D3D12_RESOURCE_STATE_COMMON);
    }
    if read == api::AccessMask::ACCEL_WRITE && uses.stages.contains(api::StageMask::ACCELERATION) {
        return Ok(D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    }
    if read == api::AccessMask::ACCEL_READ && uses.stages.contains(api::StageMask::ACCELERATION) {
        return Ok(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    }
    if read == api::AccessMask::RAY_TRACING_SHADER_DATA_READ
        && uses.stages.contains(api::StageMask::RAY)
    {
        return Ok(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    }
    let copy_read = read.contains(api::AccessMask::COPY_READ);
    let copy_write = read.contains(api::AccessMask::COPY_WRITE);
    if copy_write {
        if read != api::AccessMask::COPY_WRITE {
            return Err(Error::Unsupported("DX12 combined buffer copy access"));
        }
        return Ok(D3D12_RESOURCE_STATE_COPY_DEST);
    }
    if read.contains(api::AccessMask::SHADER_WRITE) {
        let storage_access = api::AccessMask::SHADER_READ.union(api::AccessMask::SHADER_WRITE);
        if read.0 & !storage_access.0 != 0 || !has_shader_stage(uses.stages) {
            return Err(Error::Unsupported("DX12 buffer shader write access"));
        }
        return Ok(D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    }
    let mut state = D3D12_RESOURCE_STATES(0);
    let mut known = 0u64;
    if copy_read {
        state |= D3D12_RESOURCE_STATE_COPY_SOURCE;
        known |= api::AccessMask::COPY_READ.0;
    }
    if read.contains(api::AccessMask::INDIRECT_READ) {
        state |= D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT;
        known |= api::AccessMask::INDIRECT_READ.0;
    }
    if read.contains(api::AccessMask::VERTEX_READ) || read.contains(api::AccessMask::UNIFORM_READ) {
        state |= D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER;
        known |= read.0 & (api::AccessMask::VERTEX_READ.0 | api::AccessMask::UNIFORM_READ.0);
    }
    if read.contains(api::AccessMask::INDEX_READ) {
        state |= D3D12_RESOURCE_STATE_INDEX_BUFFER;
        known |= api::AccessMask::INDEX_READ.0;
    }
    if read.contains(api::AccessMask::SHADER_READ) {
        let mut shader_state = D3D12_RESOURCE_STATES(0);
        if uses.stages.contains(api::StageMask::FRAGMENT) {
            shader_state |= D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        }
        if uses.stages.contains(api::StageMask::VERTEX)
            || uses.stages.contains(api::StageMask::COMPUTE)
            || uses.stages.contains(api::StageMask::TASK)
            || uses.stages.contains(api::StageMask::MESH)
            || uses.stages.contains(api::StageMask::RAY)
        {
            shader_state |= D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        }
        if shader_state.0 == 0 {
            return Err(Error::Unsupported("DX12 buffer shader-read stage"));
        }
        state |= shader_state;
        known |= api::AccessMask::SHADER_READ.0;
    }
    if known != read.0 || state.0 == 0 {
        return Err(Error::Unsupported("DX12 buffer access state"));
    }
    Ok(state)
}

pub(crate) fn copy_location(
    resource: &ID3D12Resource,
    footprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
    placed: bool,
    subresource: u32,
) -> D3D12_TEXTURE_COPY_LOCATION {
    D3D12_TEXTURE_COPY_LOCATION {
        pResource: ManuallyDrop::new(Some(resource.clone())),
        Type: if placed {
            D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT
        } else {
            D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX
        },
        Anonymous: if placed {
            D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: footprint,
            }
        } else {
            D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: subresource,
            }
        },
    }
}

pub(crate) fn texture_upload_buffer(
    device: &windows::Win32::Graphics::Direct3D12::ID3D12Device,
    size: u64,
) -> Result<ID3D12Resource, Error> {
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_UPLOAD,
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
        SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_NONE,
    };
    let mut output: Option<ID3D12Resource> = None;
    // SAFETY: the fully initialized heap/resource descriptors are local to the
    // call and D3D12 writes the returned COM resource into `output` only.
    unsafe {
        device.CreateCommittedResource(
            &heap,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            None,
            &mut output,
        )
    }
    .map_err(native_error)?;
    output
        .ok_or_else(|| Error::Backend("DX12 texture upload allocation returned no resource".into()))
}

#[cfg(test)]
#[path = "../../tests/backend/encoder/transition.rs"]
mod tests;
