//! Typed `glMultiDraw*Indirect` entry points for the current native context.
//!
//! Glow does not expose these wide pointer-width commands.  Keeping their ABI
//! here prevents the ordinary single-draw fallback from accidentally becoming
//! capability evidence for a batch command.

use crate::backend::api::{GlDiscoverySnapshot, GlFamilyProfile, GlKnownExtension, GlVersion};
use core::ffi::c_void;

#[derive(Clone, Copy)]
pub(crate) struct MultiDrawIndirectFns {
    pub(crate) arrays: unsafe extern "system" fn(u32, *const c_void, i32, i32),
    pub(crate) elements: unsafe extern "system" fn(u32, u32, *const c_void, i32, i32),
}

impl MultiDrawIndirectFns {
    /// Loads both entry points only when the discovery ledger admitted either
    /// desktop GL 4.3 core or `GL_ARB_multi_draw_indirect` for this context.
    pub(super) unsafe fn load(
        discovery: &GlDiscoverySnapshot,
        load: impl Fn(&str) -> *const c_void,
    ) -> Option<Self> {
        let profile = discovery.context().profile();
        let core = matches!(profile, GlFamilyProfile::Desktop { major, minor }
            if GlVersion::new(4, 3).is_met_by(GlVersion::new(major, minor)));
        let arb = matches!(profile, GlFamilyProfile::Desktop { .. })
            && discovery
                .extensions()
                .is_acquired(GlKnownExtension::ArbMultiDrawIndirect);
        if !core && !arb {
            return None;
        }
        // SAFETY: the caller established the route for this current context.
        unsafe { Self::load_symbols(load) }
    }

    /// Resolves the two canonical exports without treating either as a route.
    ///
    /// Native discovery must obtain loader evidence before it has built its
    /// snapshot. Its operation runner separately gates invocation on desktop
    /// 4.3/ARB acquisition, so merely resolving these names cannot publish or
    /// call a command outside the context contract.
    pub(super) unsafe fn load_symbols(load: impl Fn(&str) -> *const c_void) -> Option<Self> {
        // ARB_multi_draw_indirect deliberately removed the ARB suffix before
        // publication; extensions and GL 4.3 share these canonical symbols.
        let arrays = load("glMultiDrawArraysIndirect");
        let elements = load("glMultiDrawElementsIndirect");
        if arrays.is_null() || elements.is_null() {
            return None;
        }
        // SAFETY: the current-context loader returned these exact GL ABIs.
        Some(unsafe {
            Self {
                arrays: core::mem::transmute(arrays),
                elements: core::mem::transmute(elements),
            }
        })
    }
}
