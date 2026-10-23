//! Callable GL indirect-count entry points for the current desktop context.
use crate::backend::gl::api::{GlDiscoverySnapshot, GlFamilyProfile, GlVersion};
use core::ffi::c_void;

#[derive(Clone, Copy)]
pub(crate) struct IndirectCountFns {
    pub arrays: unsafe extern "system" fn(u32, *const c_void, isize, i32, i32),
    pub elements: unsafe extern "system" fn(u32, u32, *const c_void, isize, i32, i32),
}
impl IndirectCountFns {
    /// The loader must return callable symbols for the current context.
    pub(super) unsafe fn load(
        discovery: &GlDiscoverySnapshot,
        load: impl Fn(&str) -> *const c_void,
    ) -> Option<Self> {
        let desktop = matches!(
            discovery.context().profile(),
            GlFamilyProfile::Desktop { .. }
        );
        let core = matches!(discovery.context().profile(), GlFamilyProfile::Desktop { major, minor } if GlVersion::new(4, 6).is_met_by(GlVersion::new(major, minor)));
        // ARB_indirect_parameters is a desktop extension. Use acquired, typed
        // evidence rather than a raw driver string so an unrecognized or
        // profile-illegal name cannot authorize loading these entry points.
        let arb = desktop
            && discovery
                .extensions()
                .is_acquired(crate::backend::gl::api::GlKnownExtension::ArbIndirectParameters);
        if !core && !arb {
            return None;
        }
        let suffix = if core { "" } else { "ARB" };
        let arrays = load(&format!("glMultiDrawArraysIndirectCount{suffix}"));
        let elements = load(&format!("glMultiDrawElementsIndirectCount{suffix}"));
        if arrays.is_null() || elements.is_null() {
            return None;
        }
        // SAFETY: each non-null address belongs to the current GL context and has the specified GL ABI.
        Some(unsafe {
            Self {
                arrays: core::mem::transmute(arrays),
                elements: core::mem::transmute(elements),
            }
        })
    }
}
