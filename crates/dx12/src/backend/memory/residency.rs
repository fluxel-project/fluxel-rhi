//! Native residency calls for eligible physical allocation owners.

use super::super::{Dx12Device, encoder, native_error};
use fluxel_rhi_core::api::{Error, ResidencyBudget, Result};
use windows::Win32::Graphics::Direct3D12::{
    D3D12_FENCE_FLAG_NONE, D3D12_RESIDENCY_FLAG_DENY_OVERBUDGET, D3D12_RESIDENCY_FLAG_NONE,
    ID3D12Fence, ID3D12Pageable,
};
use windows::core::Interface;

pub(in crate::backend) fn make_resident(device: &Dx12Device, owner: &impl Interface) -> Result<()> {
    let pageable = owner.cast::<ID3D12Pageable>().map_err(native_error)?;
    // SAFETY: the caller checked allocation eligibility and device provenance. The
    // pageable reference remains valid through the synchronous operation.
    unsafe { device.0.native.MakeResident(&[Some(pageable)]) }.map_err(native_error)
}

pub(in crate::backend) fn enqueue(
    device: &Dx12Device,
    owner: &impl Interface,
    budget: ResidencyBudget,
) -> Result<encoder::Dx12Completion> {
    let native = device
        .0
        .residency_device
        .as_ref()
        .ok_or(Error::Unsupported("asynchronous heap residency"))?;
    let flags = match budget {
        ResidencyBudget::AllowOverBudget => D3D12_RESIDENCY_FLAG_NONE,
        ResidencyBudget::DenyOverBudget => D3D12_RESIDENCY_FLAG_DENY_OVERBUDGET,
    };
    let pageable = owner.cast::<ID3D12Pageable>().map_err(native_error)?;
    // SAFETY: the live device creates a new fence private to this operation.
    // A residency completion can never advance a submission lane timeline.
    let fence: ID3D12Fence =
        unsafe { native.CreateFence(0, D3D12_FENCE_FLAG_NONE) }.map_err(native_error)?;
    let value = 1;
    // SAFETY: the pageable and fence belong to this device. Accepted work keeps both
    // objects alive independently of the caller's completion point.
    unsafe { native.EnqueueMakeResident(flags, &[Some(pageable.clone())], &fence, value) }
        .map_err(native_error)?;
    encoder::retirement::register_residency(device, fence.clone(), value, pageable);
    Ok(encoder::Dx12Completion { fence, value })
}

/// The caller must own a residency reference and exclude conflicting GPU uses.
pub(in crate::backend) unsafe fn evict(device: &Dx12Device, owner: &impl Interface) -> Result<()> {
    let pageable = owner.cast::<ID3D12Pageable>().map_err(native_error)?;
    // SAFETY: the live pageable belongs to this device. The caller establishes
    // native residency reference ownership and completion across all users.
    unsafe { device.0.native.Evict(&[Some(pageable)]) }.map_err(native_error)
}
