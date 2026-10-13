//! Native D3D12 primitives shared by raster and upload recording.
//!
//! These helpers own only temporary FFI descriptors.  The caller retains every
//! actual resource in its encoder until fence retirement; this module creates
//! neither render passes nor resource-state history.

use std::mem::ManuallyDrop;

use fluxel_rhi_core::api::{Error, Result};
use windows::Win32::Graphics::Direct3D12::*;

use crate::backend::native_error;

pub(crate) fn barrier(
    list: &ID3D12GraphicsCommandList,
    resource: &ID3D12Resource,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) {
    if before == after {
        return;
    }
    let mut barrier = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                pResource: ManuallyDrop::new(Some(resource.clone())),
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    };
    // SAFETY: the union owns a cloned COM resource only for this synchronous
    // call. Drop exactly that clone once D3D12 consumed the barrier descriptor.
    unsafe {
        list.ResourceBarrier(std::slice::from_ref(&barrier));
        ManuallyDrop::drop(&mut (*barrier.Anonymous.Transition).pResource);
    }
}

pub(crate) fn upload_buffer(template: &ID3D12Resource, bytes: &[u8]) -> Result<ID3D12Resource> {
    let mut device = None;
    // SAFETY: `template` is live; D3D12 writes a cloned device COM reference
    // into the local output slot.
    unsafe { template.GetDevice::<ID3D12Device>(&mut device) }.map_err(native_error)?;
    let device = device.ok_or_else(|| Error::Backend("buffer has no D3D12 device".into()))?;
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
        Width: bytes.len() as u64,
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
    let mut upload: Option<ID3D12Resource> = None;
    // SAFETY: heap/desc are initialized locals and `upload` is a valid output
    // slot for the owned upload-resource interface.
    unsafe {
        device.CreateCommittedResource(
            &heap,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            None,
            &mut upload,
        )
    }
    .map_err(native_error)?;
    let upload = upload
        .ok_or_else(|| Error::Backend("upload resource creation returned no resource".into()))?;
    let mut mapped = std::ptr::null_mut();
    // SAFETY: upload memory is CPU writable and Map initializes `mapped` for
    // this write-only subresource range.
    unsafe {
        upload.Map(
            0,
            Some(&D3D12_RANGE { Begin: 0, End: 0 }),
            Some(&mut mapped),
        )
    }
    .map_err(native_error)?;
    // SAFETY: Map returned at least `bytes.len()` writable bytes; unmap matches
    // this single map and occurs after the non-overlapping copy.
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), mapped.cast::<u8>(), bytes.len());
        upload.Unmap(0, None);
    }
    Ok(upload)
}
