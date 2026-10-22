//! Narrow backend-native interoperation escape hatch.
//!
//! Calls are unsafe because callers cross the RHI ownership boundary. Core
//! neither validates foreign native commands nor tracks their state changes.
#![allow(missing_docs, dead_code, unused_variables, unused_imports)]
use super::*;
use std::{
    future::Future,
    marker::PhantomData,
    ops::Range,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

pub enum NativeInteropRequest {
    Device,
    Encoder,
}
pub trait NativeInteropBackend: CommandBackend {
    type NativeInteropGuard;
    fn acquire_native(
        device: &Self::NativeDevice,
        request: NativeInteropRequest,
    ) -> Result<Self::NativeInteropGuard>;
    /// Acquire the actual recording encoder. The caller of the public API
    /// keeps it exclusively borrowed for the lifetime of the returned guard.
    fn acquire_encoder(encoder: &mut Self::NativeEncoder) -> Result<Self::NativeInteropGuard>;
    /// # Safety
    /// The guard must remain alive and the backend must expose the returned
    /// pointer only under its native thread, lifetime, and synchronization rules.
    unsafe fn raw_pointer(guard: &Self::NativeInteropGuard) -> *mut std::ffi::c_void;
}
pub struct NativeInteropGuard<B: NativeInteropBackend> {
    device: DeviceId,
    native: B::NativeInteropGuard,
}
impl<B: NativeInteropBackend> NativeInteropGuard<B> {
    /// # Safety
    /// Caller maintains native synchronization, lifetime and backend state
    /// invalidation; raw access never bypasses the renderer's retirement rule.
    pub unsafe fn raw_pointer(&self) -> *mut std::ffi::c_void {
        // SAFETY: this public unsafe method transfers the backend guard's raw-access contract.
        unsafe { B::raw_pointer(&self.native) }
    }
}
pub struct NativeInteropDomain<'a, B: NativeInteropBackend> {
    device: &'a Device<B>,
}
impl<B: NativeInteropBackend> Device<B> {
    pub fn native_interop(&self) -> Option<NativeInteropDomain<'_, B>> {
        self.caps
            .native_interop
            .then_some(NativeInteropDomain { device: self })
    }
}
impl<B: NativeInteropBackend> NativeInteropDomain<'_, B> {
    /// # Safety
    /// Caller explicitly coordinates native side effects with RenderGraph.
    pub unsafe fn acquire(&self, request: NativeInteropRequest) -> Result<NativeInteropGuard<B>> {
        if matches!(request, NativeInteropRequest::Encoder) {
            return Err(Error::InvalidState);
        }
        Ok(NativeInteropGuard {
            device: self.device.id,
            native: B::acquire_native(&self.device.native, request)?,
        })
    }
}

/// Native encoder access tied to an exclusive borrow of the recording encoder.
pub struct NativeEncoderInteropGuard<'a, B: NativeInteropBackend> {
    native: B::NativeInteropGuard,
    _encoder: PhantomData<&'a mut CommandEncoder<B>>,
}
impl<B: NativeInteropBackend> NativeEncoderInteropGuard<'_, B> {
    /// # Safety
    /// The pointer is valid only while this guard is alive. The caller must
    /// obey native synchronization and command recording requirements.
    pub unsafe fn raw_pointer(&self) -> *mut std::ffi::c_void {
        // SAFETY: the guard keeps the encoder exclusively borrowed.
        unsafe { B::raw_pointer(&self.native) }
    }
}
impl<B: NativeInteropBackend> CommandEncoder<B> {
    /// Borrow the native recording encoder until the guard is dropped.
    ///
    /// # Safety
    /// The caller must restore native state required by subsequent RHI work,
    /// coordinate hazards and synchronization, and retain every resource used
    /// by native commands until their GPU execution has completed.
    ///
    /// ```compile_fail
    /// use fluxel_rhi_core::api::{CommandEncoder, NativeInteropBackend};
    /// unsafe fn cannot_finish<B: NativeInteropBackend>(mut encoder: CommandEncoder<B>) {
    ///     let guard = unsafe { encoder.native_interop().unwrap() };
    ///     let _work = encoder.finish();
    ///     let _ = unsafe { guard.raw_pointer() };
    /// }
    /// ```
    pub unsafe fn native_interop(&mut self) -> Result<NativeEncoderInteropGuard<'_, B>> {
        let native = B::acquire_encoder(self.ready()?)?;
        Ok(NativeEncoderInteropGuard {
            native,
            _encoder: PhantomData,
        })
    }
}
