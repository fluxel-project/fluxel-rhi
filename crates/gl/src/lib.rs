//! Desktop OpenGL, OpenGL ES, and WebGL2 execution for Fluxel RHI.
//!
//! Each backend object has a static Core type. GL commands are issued on the
//! current context thread as they are encoded; presentation, native object
//! lifetime and driver state remain private to this crate. RenderGraph owns
//! work ordering, hazards, frame policy and resource retirement decisions.

#![deny(missing_docs)]

/// The portable RHI API shared with the other backends.
pub use fluxel_rhi_core::api;

mod backend;

/// The statically dispatched GL-family backend.
pub use backend::rhi::Gl;
/// A selected, current GL-family context and its adapter discovery entry.
pub use backend::rhi::GlInstance;

/// Native GL dispatch table borrowed through a scoped Core interop guard.
/// The guard's pointer may be cast only to a shared reference of this type.
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
pub use glow::Context as NativeGlContext;
/// Native GL operations available on [`NativeGlContext`]. Calls retain their
/// native unsafe synchronization, object-lifetime, and current-context rules.
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
pub use glow::HasContext as NativeGlApi;

#[cfg(all(
    test,
    any(
        all(windows, feature = "native-gl-wgl"),
        feature = "native-gles-egl",
        all(target_arch = "wasm32", feature = "webgl2")
    )
))]
#[path = "tests/gpu/mod.rs"]
mod gpu;

#[cfg(any(
    feature = "native-gl-wgl",
    feature = "native-gles-egl",
    feature = "webgl2"
))]
fn next_context_stamp() -> api::Result<backend::api::ContextStamp> {
    backend::api::ContextStamp::allocate().ok_or(api::Error::OutOfMemory)
}

#[cfg(all(windows, feature = "native-gl-wgl"))]
impl GlInstance {
    /// Adopts a live Win32 host window and opens the highest available desktop
    /// core context down to OpenGL 4.0 on the calling thread.
    ///
    /// # Safety
    /// The host window and its raw handles must outlive this instance, its
    /// device, surface, and every resource retained by that device. All GL
    /// calls and final drops must occur on this calling thread.
    pub unsafe fn from_wgl_window<H>(host: &H, extent: [u32; 2]) -> api::Result<Self>
    where
        H: raw_window_handle::HasWindowHandle + raw_window_handle::HasDisplayHandle + ?Sized,
    {
        use backend::native::{NativePlatformContext, wgl::WglContextSurface};

        let window = host
            .window_handle()
            .map_err(|error| api::Error::Backend(error.to_string()))?;
        let display = host
            .display_handle()
            .map_err(|error| api::Error::Backend(error.to_string()))?;
        // SAFETY: the caller keeps the host handles alive through all context
        // and surface use; WGL validates their actual Win32 kinds before use.
        let context = WglContextSurface::open(next_context_stamp()?, window, display, extent)
            .map_err(|error| api::Error::Backend(format!("{error:?}")))?;
        Self::from_current_context(NativePlatformContext::Wgl(context))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
impl GlInstance {
    /// Creates and owns a WebGL2 context for the supplied host canvas.
    ///
    /// The host keeps the canvas and its DOM lifecycle alive. Context loss is
    /// terminal for this instance; restoration opens a new instance and device.
    pub fn from_webgl2_canvas(canvas: web_sys::HtmlCanvasElement) -> api::Result<Self> {
        use backend::browser::{BrowserOwner, WebGl2BrowserDiscovery};

        let discovery = WebGl2BrowserDiscovery::open(next_context_stamp()?, canvas)
            .map_err(|error| api::Error::Backend(format!("{error:?}")))?;
        Ok(Self::from_browser_owner(BrowserOwner::new(discovery)))
    }

    /// Marks this WebGL2 device generation terminal after the host receives
    /// `webglcontextlost` (and calls `preventDefault`). Pending completion
    /// waiters are woken and all surface leases become stale. After
    /// `webglcontextrestored`, create a new `GlInstance`, Core `Instance`,
    /// device, and surface; the old objects cannot be reused.
    pub fn notify_context_lost(&self) {
        self.0.owner.mark_context_lost();
    }
}

#[cfg(feature = "native-gles-egl")]
impl GlInstance {
    fn retry_older_gles_version(error: &backend::native::egl::EglProviderError) -> bool {
        use backend::native::egl::EglProviderError;

        matches!(
            error,
            EglProviderError::ExactVersionUnavailable(_)
                | EglProviderError::BelowRequestedProfile { .. }
                | EglProviderError::Egl {
                    operation: "eglCreateContext",
                    ..
                }
        )
    }

    /// Adopts a live platform window through EGL, choosing the highest usable
    /// GLES 3.x context down to the GLES 3.0 floor.
    ///
    /// # Safety
    /// The display and window handles must remain valid until the EGL context,
    /// device, surfaces, and all retained GL resources have been dropped on the
    /// context owner thread.
    pub unsafe fn from_egl_window(
        display: raw_window_handle::RawDisplayHandle,
        window: raw_window_handle::RawWindowHandle,
    ) -> api::Result<Self> {
        use backend::native::{
            NativePlatformContext,
            egl::{EglGlesContext, EglGlesVersion},
        };

        let stamp = next_context_stamp()?;
        let mut last_error = None;
        for version in [
            EglGlesVersion::V3_2,
            EglGlesVersion::V3_1,
            EglGlesVersion::V3_0,
        ] {
            // SAFETY: the caller retains both native handles until every
            // adopted context object is retired; EGL validates the pair.
            match unsafe { EglGlesContext::new_window(stamp, display, window, version) } {
                Ok(context) => {
                    return Self::from_current_context(NativePlatformContext::Egl(context));
                }
                Err(error) if Self::retry_older_gles_version(&error) => last_error = Some(error),
                Err(error) => return Err(api::Error::Backend(format!("{error:?}"))),
            }
        }
        Err(api::Error::Backend(format!(
            "EGL GLES 3.x context: {last_error:?}"
        )))
    }

    /// Opens an offscreen GLES 3.x EGL pbuffer on the calling thread.
    pub fn gles_pbuffer(extent: [u32; 2]) -> api::Result<Self> {
        use backend::native::{
            NativePlatformContext,
            egl::{EglGlesContext, EglGlesVersion, EglPbufferSize},
        };

        let stamp = next_context_stamp()?;
        let size = EglPbufferSize {
            width: extent[0],
            height: extent[1],
        };
        let mut last_error = None;
        for version in [
            EglGlesVersion::V3_2,
            EglGlesVersion::V3_1,
            EglGlesVersion::V3_0,
        ] {
            match EglGlesContext::new_pbuffer(stamp, size, version) {
                Ok(context) => {
                    return Self::from_current_context(NativePlatformContext::Egl(context));
                }
                Err(error) if Self::retry_older_gles_version(&error) => last_error = Some(error),
                Err(error) => return Err(api::Error::Backend(format!("{error:?}"))),
            }
        }
        Err(api::Error::Backend(format!(
            "EGL GLES 3.x pbuffer: {last_error:?}"
        )))
    }
}
