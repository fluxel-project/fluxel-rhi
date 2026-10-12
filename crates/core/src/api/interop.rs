//! Statically typed RHI contract component.
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
pub trait NativeInteropBackend: Backend {
    type NativeInteropGuard;
    fn acquire_native(
        device: &Self::NativeDevice,
        request: NativeInteropRequest,
    ) -> Result<Self::NativeInteropGuard>;
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
        Ok(NativeInteropGuard {
            device: self.device.id,
            native: B::acquire_native(&self.device.native, request)?,
        })
    }
}
