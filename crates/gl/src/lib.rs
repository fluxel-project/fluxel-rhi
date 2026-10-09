//! OpenGL-family backend for Fluxel RHI.
//!
//! This crate contains the shared lowering and state engine for desktop OpenGL,
//! OpenGL ES, and WebGL2. Its public surface is composed by `fluxel-rhi`.

#![deny(missing_docs)]

pub use fluxel_rhi_core::api;

/// The exact GLSL dialect selected by a GL-family context during discovery.
///
/// This is backend-specific discovery evidence. It lets a shader producer
/// select a native GLSL artifact before it asks the portable device to create
/// the module.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GlShaderTarget {
    /// Desktop core GLSL, for example version `460` for OpenGL 4.6.
    Desktop {
        /// GLSL version encoded without the decimal point, such as `460`.
        version: u16,
    },
    /// GLSL ES, for example version `310` for OpenGL ES 3.1.
    Embedded {
        /// GLSL ES version encoded without the decimal point, such as `310`.
        version: u16,
    },
    /// The WebGL 2 GLSL ES 3.00 target.
    WebGl2,
}

/// An owned desktop OpenGL provider and its single WGL presentation target.
///
/// A WGL provider owns the context and drawable route for the host window used
/// at construction. Keep the host window alive until this value is dropped.
/// The contained target is opaque and is valid only with the contained
/// provider and devices requested from it.
#[cfg(all(windows, feature = "native-gl-wgl"))]
pub struct WglProvider {
    provider: api::platform::PlatformProvider,
    target: api::presentation::PresentationTarget,
    resize: backend::gl::native::wgl::WglResizeHandle,
    shader_target: GlShaderTarget,
}

#[cfg(all(windows, feature = "native-gl-wgl"))]
impl WglProvider {
    /// Returns the portable provider used to request a device.
    pub fn provider(&self) -> &api::platform::PlatformProvider {
        &self.provider
    }

    /// Returns the opaque target for presentation preflight and device setup.
    pub fn presentation_target(&self) -> &api::presentation::PresentationTarget {
        &self.target
    }

    /// Records the current host drawable extent before presentation is
    /// reconfigured after a window resize.
    pub fn resize(&self, extent: [u32; 2]) -> api::error::RhiResult<()> {
        self.resize.resize(extent)
    }

    /// Returns the exact desktop GLSL target accepted by this WGL context.
    pub const fn shader_target(&self) -> GlShaderTarget {
        self.shader_target
    }

    /// Splits the provider and its target for callers that retain them in
    /// separate session fields.
    pub fn into_parts(
        self,
    ) -> (
        api::platform::PlatformProvider,
        api::presentation::PresentationTarget,
    ) {
        (self.provider, self.target)
    }
}

/// Opens a desktop-core WGL provider for a live Windows host window.
///
/// `extent` is the host's current drawable size. It becomes the initial
/// default-framebuffer extent. Call [`WglProvider::resize`] before
/// reconfiguring presentation after a host resize. The resulting provider
/// performs all OpenGL work on its dedicated WGL owner thread.
///
/// The returned [`WglProvider`] also carries the one presentation target bound
/// to that drawable. This avoids exposing a Win32 handle through the portable
/// RHI surface.
#[cfg(all(windows, feature = "native-gl-wgl"))]
pub fn create_wgl_provider<H>(host: &H, extent: [u32; 2]) -> api::error::RhiResult<WglProvider>
where
    H: raw_window_handle::HasWindowHandle + raw_window_handle::HasDisplayHandle + ?Sized,
{
    use std::sync::atomic::{AtomicU64, Ordering};

    use api::error::{RhiError, RhiErrorKind};
    use api::identity::{DeviceInstanceId, ObjectId};
    use api::platform::{BackendKind, PlatformProvider};
    use backend::gl::api::{ContextEpoch, ContextStamp, DeviceIdentity as GlDeviceIdentity};
    use backend::gl::native::wgl::{WglWorkerDescriptor, spawn_provider_with_control};

    // Providers normally enumerate a native adapter; WGL adopts one already
    // selected by the host, so allocate a distinct provider identity here.
    static NEXT_WGL_PROVIDER: AtomicU64 = AtomicU64::new(0x474C_0001);

    let instance = DeviceInstanceId::new(NEXT_WGL_PROVIDER.fetch_add(1, Ordering::Relaxed));
    let identity = GlDeviceIdentity::new(instance.as_u64()).ok_or_else(|| {
        RhiError::new(
            RhiErrorKind::BackendFailure,
            "WGL provider identity allocation produced zero",
        )
        .at("create_wgl_provider")
    })?;
    let stamp = ContextStamp::new(identity, ContextEpoch::INITIAL);
    let window = host.window_handle().map_err(|error| {
        RhiError::new(RhiErrorKind::InvalidUsage, error.to_string()).at("create_wgl_provider")
    })?;
    let display = host.display_handle().map_err(|error| {
        RhiError::new(RhiErrorKind::InvalidUsage, error.to_string()).at("create_wgl_provider")
    })?;
    let descriptor = WglWorkerDescriptor::new(stamp, window, display, extent).map_err(|error| {
        RhiError::new(
            RhiErrorKind::InvalidUsage,
            format!("WGL provider rejected the host window: {error:?}"),
        )
        .at("create_wgl_provider")
    })?;
    let (native, resize, shader_target) = spawn_provider_with_control(instance, descriptor)?;
    Ok(WglProvider {
        provider: PlatformProvider::new(BackendKind::OpenGl, instance, Box::new(native)),
        // The WGL provider owns exactly one drawable. Its native owner accepts
        // this opaque id and enforces that later configuration uses the same
        // target, so no Win32 handle needs to enter the portable target value.
        target: api::presentation::PresentationTarget::new(ObjectId::next()),
        resize,
        shader_target,
    })
}

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
