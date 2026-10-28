//! GLES raw-copy readback exercises the RGB9E5 image-class alias when present.
use super::*;

#[cfg(feature = "native-gles-egl")]
#[test]
#[ignore = "requires an EGL GLES hardware context"]
fn gles_core_rgb9e5_exact_alias() {
    let d = block_on(open(
        GlInstance::gles_pbuffer([8, 8]).expect("EGL GLES pbuffer"),
    ));
    block_on(super::rgb9e5_readback::rgb9e5_readback(&d));
}
