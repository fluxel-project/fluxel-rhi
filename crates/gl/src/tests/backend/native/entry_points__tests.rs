use super::{NativeEntryPoint, NativeEntryPoints};
use crate::backend::api::{GlFamilyProfile, GlKnownExtension};
use core::ffi::c_void;

#[test]
fn one_missing_symbol_closes_only_its_group() {
    // SAFETY: the loader returns sentinel non-null values and no symbol is called.
    let table = unsafe {
        NativeEntryPoints::load(|name| {
            if name == "glMemoryBarrier" {
                core::ptr::null()
            } else {
                1usize as *const c_void
            }
        })
    };
    assert!(!table.has(NativeEntryPoint::StorageImage));
    assert!(table.has(NativeEntryPoint::DrawIndirect));
}

#[test]
fn typed_extension_cannot_cross_profile_family() {
    let table = NativeEntryPoints {
        compute: true,
        ..NativeEntryPoints::default()
    };
    assert!(!table.admits(
        NativeEntryPoint::Compute,
        GlFamilyProfile::Embedded { major: 3, minor: 1 },
        false,
        Some(GlKnownExtension::ArbComputeShader),
    ));
    // An acquired extension is not a substitute for actual acquisition:
    // `None` is the representation of reported-only evidence.
    assert!(!table.admits(
        NativeEntryPoint::Compute,
        GlFamilyProfile::Desktop { major: 4, minor: 2 },
        false,
        None,
    ));
}
