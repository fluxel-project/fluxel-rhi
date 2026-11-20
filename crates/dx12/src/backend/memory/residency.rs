//! Native residency calls for physical heaps, shared by placed and sparse memory.

use super::super::{Dx12Device, native_error};
use fluxel_rhi_core::api::Result;
use windows::Win32::Graphics::Direct3D12::{ID3D12Heap, ID3D12Pageable};
use windows::core::Interface;

pub(in crate::backend) fn make_resident(device: &Dx12Device, heap: &ID3D12Heap) -> Result<()> {
    let pageable = heap.cast::<ID3D12Pageable>().map_err(native_error)?;
    // SAFETY: the caller checked heap eligibility and device provenance. The
    // pageable reference remains valid through the synchronous operation.
    unsafe { device.0.native.MakeResident(&[Some(pageable)]) }.map_err(native_error)
}

/// The caller must own a residency reference and exclude conflicting GPU uses.
pub(in crate::backend) unsafe fn evict(device: &Dx12Device, heap: &ID3D12Heap) -> Result<()> {
    let pageable = heap.cast::<ID3D12Pageable>().map_err(native_error)?;
    // SAFETY: the live pageable belongs to this device. The caller establishes
    // native residency reference ownership and completion across all users.
    unsafe { device.0.native.Evict(&[Some(pageable)]) }.map_err(native_error)
}
