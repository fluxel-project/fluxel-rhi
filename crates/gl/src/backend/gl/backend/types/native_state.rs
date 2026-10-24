//! Owner-thread native context access and scoped lease enforcement.

use super::GlDeviceState;

#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
impl Drop for GlDeviceState {
    fn drop(&mut self) {
        use crate::backend::gl::native::NativeContextOwner as _;
        // Rc confines this final owner to its context thread. The platform
        // owner is still alive here; no GL deletion may run unless this exact
        // context can be made current. Lost contexts reclaim their own names.
        if self
            .context
            .get_mut()
            .make_current("retire GL utilities")
            .is_ok()
        {
            // SAFETY: the exact retained owner context is current on this
            // thread, and final Rc destruction makes the provider exclusive.
            unsafe {
                let provider = self.provider.get_mut();
                provider.clear_indirect_count_utility();
                provider.clear_indirect_count_scratch();
            };
        }
    }
}

#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
impl GlDeviceState {
    /// Executes a platform-context operation without borrowing the GL object
    /// provider. Presentation uses this after an image lease is irrevocably
    /// handed off, so a swap failure cannot be reclassified as rejection.
    pub(crate) fn with_context<T>(
        &self,
        operation: &'static str,
        f: impl FnOnce(
            &mut crate::backend::gl::native::NativePlatformContext,
        ) -> Result<T, crate::backend::gl::api::GlError>,
    ) -> Result<T, crate::backend::gl::api::GlError> {
        self.check_native_lease(operation)?;
        let mut context = self.context.try_borrow_mut().map_err(|_| {
            crate::backend::gl::api::GlError::Driver {
                operation,
                message: "GL platform context is already in use".into(),
            }
        })?;
        f(&mut context)
    }

    /// Executes one native operation while this device's context is current.
    ///
    /// The borrow ordering is deliberate: currentness is established before a
    /// provider can resolve or issue any object name. `Rc` and `RefCell` keep
    /// this state thread-affine, so no other Rust thread can concurrently make
    /// the context current or mutate its object tables.
    pub(crate) fn with_native<T>(
        &self,
        operation: &'static str,
        f: impl FnOnce(
            &mut crate::backend::gl::native::provider::NativeGlProvider,
        ) -> Result<T, crate::backend::gl::api::GlError>,
    ) -> Result<T, crate::backend::gl::api::GlError> {
        self.check_native_lease(operation)?;
        self.with_native_current(operation, f)
    }

    /// Resource destructors may retire objects during a scoped native lease.
    /// The caller of the lease must retain objects used by foreign commands.
    /// A failed restore leaves context state uncertain, so even retirement is
    /// refused and the platform context will reclaim objects on teardown.
    pub(super) fn with_native_retirement<T>(
        &self,
        operation: &'static str,
        f: impl FnOnce(
            &mut crate::backend::gl::native::provider::NativeGlProvider,
        ) -> Result<T, crate::backend::gl::api::GlError>,
    ) -> Result<T, crate::backend::gl::api::GlError> {
        if self.native_lease_failed.get() {
            return Err(crate::backend::gl::api::GlError::Driver {
                operation,
                message: "GL native interop failed to restore".into(),
            });
        }
        self.with_native_current(operation, f)
    }

    fn with_native_current<T>(
        &self,
        operation: &'static str,
        f: impl FnOnce(
            &mut crate::backend::gl::native::provider::NativeGlProvider,
        ) -> Result<T, crate::backend::gl::api::GlError>,
    ) -> Result<T, crate::backend::gl::api::GlError> {
        use crate::backend::gl::native::NativeContextOwner as _;

        let mut context = self.context.try_borrow_mut().map_err(|_| {
            crate::backend::gl::api::GlError::Driver {
                operation,
                message: "GL context is already issuing a command".into(),
            }
        })?;
        context.make_current(operation)?;
        let mut provider = self.provider.try_borrow_mut().map_err(|_| {
            crate::backend::gl::api::GlError::Driver {
                operation,
                message: "GL provider is already issuing a command".into(),
            }
        })?;
        f(&mut provider)
    }

    pub(crate) fn check_native_lease(
        &self,
        operation: &'static str,
    ) -> Result<(), crate::backend::gl::api::GlError> {
        if self.native_lease.get() || self.native_lease_failed.get() {
            return Err(crate::backend::gl::api::GlError::Driver {
                operation,
                message: "GL native interop lease is active or failed to restore".into(),
            });
        }
        Ok(())
    }

    /// Names the single adapter represented by this adopted GL context.
    ///
    /// GL has no portable physical-device enumeration after context creation.
    /// Discovery's immutable driver strings are therefore the truthful facts
    /// used by the public instance enumeration result.
    pub(crate) fn adapter_name(&self) -> Result<String, crate::backend::gl::api::GlError> {
        use crate::backend::gl::api::GlFamilyApi as _;

        self.with_native("describe GL adapter", |provider| {
            let context = provider.discovery().context();
            Ok(format!(
                "{} {} ({})",
                context.vendor(),
                context.renderer(),
                context.version()
            ))
        })
    }
}
