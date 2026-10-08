//! OpenGL-family backend for Fluxel RHI.
//!
//! This crate contains the shared lowering and state engine for desktop OpenGL,
//! OpenGL ES, and WebGL2. Its public surface is composed by `fluxel-rhi`.

#![deny(missing_docs)]

pub use fluxel_rhi_core::api;

/// Fixture-only native GL hardware evidence reports.
///
/// This is intentionally excluded from the portable RHI vocabulary. It is
/// available only when a conformance harness selects `test-support`.
#[cfg(feature = "test-support")]
#[doc(hidden)]
pub mod test_support;

mod backend {
    #[cfg(all(
        test,
        any(
            feature = "native-gl-wgl",
            feature = "native-gles-egl",
            feature = "webgl2"
        )
    ))]
    #[path = "../../tests/common/mod.rs"]
    pub(crate) mod conformance;

    #[cfg(all(
        test,
        any(
            feature = "native-gl-wgl",
            feature = "native-gles-egl",
            feature = "webgl2"
        )
    ))]
    #[path = "../../tests/harness/mod.rs"]
    pub(crate) mod test_harness;

    pub(crate) mod gl;
}
