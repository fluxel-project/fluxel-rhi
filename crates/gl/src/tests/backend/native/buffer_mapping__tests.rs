use super::*;

#[test]
fn storage_requires_profile_extension_and_loaded_entry() {
    let present = |name: &str| match name {
        "glBufferStorage" | "glBufferStorageEXT" => core::ptr::dangling::<core::ffi::c_void>(),
        _ => core::ptr::null(),
    };
    let missing = |_: &str| core::ptr::null();
    let es = GlFamilyProfile::Embedded { major: 3, minor: 1 };
    assert!(storage_entry(es, false, true, present).is_some());
    assert!(storage_entry(es, false, false, present).is_none());
    assert!(storage_entry(es, false, true, missing).is_none());
    assert!(
        storage_entry(
            GlFamilyProfile::Embedded { major: 3, minor: 0 },
            false,
            true,
            present
        )
        .is_none()
    );
    assert!(storage_entry(GlFamilyProfile::WebGl2, true, true, present).is_none());
    let gl = GlFamilyProfile::Desktop { major: 4, minor: 3 };
    assert!(storage_entry(gl, true, false, present).is_some());
    assert!(storage_entry(gl, false, true, present).is_none());
    assert!(
        storage_entry(
            GlFamilyProfile::Desktop { major: 4, minor: 4 },
            false,
            false,
            present
        )
        .is_some()
    );
    assert!(
        storage_entry(
            GlFamilyProfile::Desktop { major: 4, minor: 4 },
            false,
            false,
            missing
        )
        .is_none()
    );
    assert!(
        storage_entry(es, false, true, |name| if name == "glBufferStorage" {
            present(name)
        } else {
            core::ptr::null()
        })
        .is_none()
    );
}
