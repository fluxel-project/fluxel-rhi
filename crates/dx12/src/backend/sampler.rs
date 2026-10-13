//! Native immutable sampler descriptors for the DX12 backend.
//!
//! Each sampler owns one CPU-only descriptor heap so descriptor-copying bind
//! groups can outlive the construction call. This module maps only values that
//! have a defined D3D12 equivalent and rejects the rest rather than relying on
//! driver defaults.

use super::{Dx12Device, Dx12Sampler};
use fluxel_rhi_core::api::{self, Error, Result};
use std::sync::Arc;
use windows::Win32::Graphics::Direct3D12::*;

/// Owns the sole CPU sampler descriptor.  A D3D12 handle does not retain its
/// heap, so retaining both here is required for later bind-group copying.
pub(crate) struct SamplerState {
    pub(super) heap: ID3D12DescriptorHeap,
    pub(super) cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
    pub(super) native: D3D12_SAMPLER_DESC,
}

pub(crate) fn create_sampler(device: &Dx12Device, desc: &api::SamplerDesc) -> Result<Dx12Sampler> {
    validate(desc)?;
    let heap_desc = D3D12_DESCRIPTOR_HEAP_DESC {
        Type: D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
        NumDescriptors: 1,
        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
        NodeMask: 0,
    };
    // SAFETY: `heap_desc` is fully initialized and the returned COM heap is
    // retained by SamplerState for every later use of its CPU handle.
    let heap = unsafe {
        device
            .0
            .native
            .CreateDescriptorHeap::<ID3D12DescriptorHeap>(&heap_desc)
    }
    .map_err(native_error)?;
    // SAFETY: `heap` is a live one-descriptor sampler heap owned locally, so
    // its first CPU handle is valid until SamplerState drops the heap.
    let cpu = unsafe { heap.GetCPUDescriptorHandleForHeapStart() };
    let native = D3D12_SAMPLER_DESC {
        Filter: filter(desc),
        AddressU: address(desc.address[0]),
        AddressV: address(desc.address[1]),
        AddressW: address(desc.address[2]),
        MipLODBias: 0.0,
        MaxAnisotropy: u32::from(desc.anisotropy),
        ComparisonFunc: compare(desc.compare),
        BorderColor: border(desc.border),
        MinLOD: desc.lod.start,
        MaxLOD: desc.lod.end,
    };
    // SAFETY: validation made the descriptor representable, and `cpu` points
    // at the sole entry of the live CPU-only sampler heap.
    unsafe { device.0.native.CreateSampler(&native, cpu) };
    Ok(Dx12Sampler(Arc::new(SamplerState { heap, cpu, native })))
}
fn validate(d: &api::SamplerDesc) -> Result<()> {
    if !d.lod.start.is_finite() || !d.lod.end.is_finite() || d.lod.start > d.lod.end {
        return Err(Error::InvalidRange);
    }
    if d.anisotropy == 0 || d.anisotropy > 16 {
        return Err(Error::Unsupported("DX12 sampler anisotropy outside 1..=16"));
    }
    if d.anisotropy > 1
        && (!matches!(d.min_filter, api::FilterMode::Linear)
            || !matches!(d.mag_filter, api::FilterMode::Linear)
            || !matches!(d.mip_filter, api::FilterMode::Linear))
    {
        return Err(Error::Unsupported(
            "DX12 anisotropic sampler requires linear min/mag/mip filtering",
        ));
    }
    Ok(())
}
fn address(v: api::AddressMode) -> D3D12_TEXTURE_ADDRESS_MODE {
    match v {
        api::AddressMode::ClampToEdge => D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        api::AddressMode::Repeat => D3D12_TEXTURE_ADDRESS_MODE_WRAP,
        api::AddressMode::MirrorRepeat => D3D12_TEXTURE_ADDRESS_MODE_MIRROR,
        api::AddressMode::ClampToBorder | api::AddressMode::ClampToZero => {
            D3D12_TEXTURE_ADDRESS_MODE_BORDER
        }
    }
}
fn border(v: Option<api::BorderColor>) -> [f32; 4] {
    match v.unwrap_or(api::BorderColor::TransparentBlack) {
        api::BorderColor::TransparentBlack | api::BorderColor::ZeroInteger => [0.0; 4],
        api::BorderColor::OpaqueBlack => [0.0, 0.0, 0.0, 1.0],
        api::BorderColor::OpaqueWhite => [1.0; 4],
    }
}
fn compare(v: Option<api::CompareFunction>) -> D3D12_COMPARISON_FUNC {
    match v {
        None => D3D12_COMPARISON_FUNC_NONE,
        Some(api::CompareFunction::Never) => D3D12_COMPARISON_FUNC_NEVER,
        Some(api::CompareFunction::Less) => D3D12_COMPARISON_FUNC_LESS,
        Some(api::CompareFunction::Equal) => D3D12_COMPARISON_FUNC_EQUAL,
        Some(api::CompareFunction::LessEqual) => D3D12_COMPARISON_FUNC_LESS_EQUAL,
        Some(api::CompareFunction::Greater) => D3D12_COMPARISON_FUNC_GREATER,
        Some(api::CompareFunction::NotEqual) => D3D12_COMPARISON_FUNC_NOT_EQUAL,
        Some(api::CompareFunction::GreaterEqual) => D3D12_COMPARISON_FUNC_GREATER_EQUAL,
        Some(api::CompareFunction::Always) => D3D12_COMPARISON_FUNC_ALWAYS,
    }
}
fn filter(d: &api::SamplerDesc) -> D3D12_FILTER {
    if d.anisotropy > 1 {
        return D3D12_FILTER(if d.compare.is_some() { 213 } else { 85 });
    }
    let n = |x| matches!(x, api::FilterMode::Nearest) as i32;
    D3D12_FILTER(
        (1 - n(d.min_filter)) * 4
            + (1 - n(d.mag_filter)) * 16
            + (1 - n(d.mip_filter))
            + if d.compare.is_some() { 128 } else { 0 },
    )
}
fn native_error(e: windows::core::Error) -> Error {
    Error::Backend(format!("Direct3D 12: {e}"))
}
