//! Scoped, owner-thread access to the native GL dispatch table.
//!
//! The lease retains the context and rejects reentrant backend operations.
//! Native resource lifetime, hazards and foreign GL object ownership remain
//! the unsafe caller's responsibility.

use super::{Gl, GlDevice, types::GlEncoder};
use fluxel_rhi_core::api::{Error, NativeInteropBackend, NativeInteropRequest, Result};

/// Thread-affine native GL lease. The pointer is a borrowed `glow::Context`
/// dispatch table, not a WGL or EGL platform context handle. Cast it only to
/// `&glow::Context`, never to a mutable reference. It is valid only
/// while this guard is alive and its context is current. Native callers must
/// retain referenced resources through GPU completion, coordinate hazards,
/// never mutate or delete backend-owned objects, and restore foreign state
/// needed by subsequent RHI work that this guard does not restore, including
/// active transform feedback, query scopes, and extension-specific state.
pub struct GlNativeInteropGuard {
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    state: std::rc::Rc<super::types::GlDeviceState>,
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    _dispatch: std::rc::Rc<glow::Context>,
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    snapshot: crate::backend::gl::native::interop::NativeInteropSnapshot,
    pointer: *mut std::ffi::c_void,
}

impl NativeInteropBackend for Gl {
    type NativeInteropGuard = GlNativeInteropGuard;
    fn acquire_native(
        device: &GlDevice,
        request: NativeInteropRequest,
    ) -> Result<Self::NativeInteropGuard> {
        if !matches!(request, NativeInteropRequest::Device) {
            return Err(Error::InvalidState);
        }
        acquire(&device.0)
    }
    fn acquire_encoder(encoder: &mut GlEncoder) -> Result<Self::NativeInteropGuard> {
        if encoder.raster_extent.is_some() {
            return Err(Error::InvalidState);
        }
        let guard = acquire(&encoder.state)?;
        encoder.issued = true;
        Ok(guard)
    }
    unsafe fn raw_pointer(guard: &Self::NativeInteropGuard) -> *mut std::ffi::c_void {
        guard.pointer
    }
}

fn acquire(state: &std::rc::Rc<super::types::GlDeviceState>) -> Result<GlNativeInteropGuard> {
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    {
        if state.compute_pass_active.get() {
            return Err(Error::InvalidState);
        }
        let (dispatch, snapshot) = state
            .with_native("acquire GL native interop", |provider| {
                provider.interop_dispatch()
            })
            .map_err(super::spi::native_error)?;
        let pointer = std::rc::Rc::as_ptr(&dispatch).cast_mut().cast();
        state.native_lease.set(true);
        Ok(GlNativeInteropGuard {
            state: state.clone(),
            _dispatch: dispatch,
            snapshot,
            pointer,
        })
    }
    #[cfg(not(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl")))]
    {
        let _ = state;
        Err(Error::Unsupported("GL native interop platform"))
    }
}

#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
impl Drop for GlNativeInteropGuard {
    fn drop(&mut self) {
        use crate::backend::gl::native::NativeContextOwner as _;
        let restored = (|| {
            let mut context = self.state.context.try_borrow_mut().map_err(|_| ())?;
            context
                .make_current("restore GL native interop")
                .map_err(|_| ())?;
            let mut provider = self.state.provider.try_borrow_mut().map_err(|_| ())?;
            provider.end_interop(&self.snapshot).map_err(|_| ())
        })();
        if restored.is_err() {
            self.state.native_lease_failed.set(true);
        }
        self.state.native_lease.set(false);
    }
}
