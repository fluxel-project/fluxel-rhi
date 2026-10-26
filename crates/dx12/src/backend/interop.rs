//! Scoped access to the actual D3D12 device and recording command list.
//!
//! COM ownership keeps the exposed interface alive. Core ties encoder access
//! to an exclusive borrow; foreign commands must retain their own resources
//! through completion, obey explicit resource states, and restore native state
//! used by subsequent RHI commands. This module never submits foreign work.

use super::{Dx12, Dx12Device, encoder::Dx12Encoder};
use fluxel_rhi_core::api::{Error, NativeInteropRequest, Result};
use fluxel_rhi_core::backend::NativeInteropBackend;
use windows::{Win32::Graphics::Direct3D12::*, core::Interface};

/// Owned COM lease returned by Core's scoped native interoperation API.
///
/// Device pointers name `ID3D12Device`; encoder pointers name
/// `ID3D12GraphicsCommandList`. The pointer is borrowed, not an additional COM
/// reference for the caller to release. Do not close/reset the list, reset its
/// allocator, or submit it independently. The caller must restore pipeline,
/// root bindings, descriptor heaps and other modified recording state before
/// returning to ordinary RHI recording. Native command resources remain the
/// caller's responsibility until submission completion.
pub struct Dx12NativeInteropGuard {
    interface: NativeInterface,
}

enum NativeInterface {
    Device(ID3D12Device),
    Encoder(ID3D12GraphicsCommandList),
}

impl NativeInteropBackend for Dx12 {
    type NativeInteropGuard = Dx12NativeInteropGuard;

    fn acquire_native(
        device: &Dx12Device,
        request: NativeInteropRequest,
    ) -> Result<Self::NativeInteropGuard> {
        if !matches!(request, NativeInteropRequest::Device) {
            return Err(Error::InvalidState);
        }
        Ok(Dx12NativeInteropGuard {
            interface: NativeInterface::Device(device.0.native.clone()),
        })
    }

    fn acquire_encoder(encoder: &mut Dx12Encoder) -> Result<Self::NativeInteropGuard> {
        if encoder.closed {
            return Err(Error::InvalidState);
        }
        Ok(Dx12NativeInteropGuard {
            interface: NativeInterface::Encoder(encoder.list.clone()),
        })
    }

    unsafe fn raw_pointer(guard: &Self::NativeInteropGuard) -> *mut std::ffi::c_void {
        match &guard.interface {
            NativeInterface::Device(device) => device.as_raw(),
            NativeInterface::Encoder(list) => list.as_raw(),
        }
    }
}
