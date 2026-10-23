//! EGL-owned GLES contexts for the native GL-family provider.
//!
//! The Host supplies raw display/window handles and keeps the underlying
//! native objects alive.  This provider owns every EGL object created from
//! those handles: display initialization, config, context, surface, current
//! binding, presentation, and teardown.  It deliberately does not create a
//! window, event loop, or an ANGLE/DX translation provider.
//!
//! This currently supports Linux Xlib/Wayland windows, Android NDK windows,
//! and headless pbuffers. Android maps its display-less handle to
//! `EGL_DEFAULT_DISPLAY`, the documented EGL route; other platform-display
//! extension ABIs are not guessed.
//!
//! It does not own Fluxel's GL object tables, the render pass, or the viewport
//! a pass renders with, and it never resizes the Host's window. The drawable
//! extent it observes is bounded by the context's recorded viewport limit; see
//! `presentation.rs` for why applying a viewport is not this layer's job.
//!
//! `presentation.rs` holds the family-owner and surface-presentation facets and
//! `tests/mod.rs` holds the pure-logic tests; this file is context and surface
//! construction, discovery wiring, and teardown.

use core::ffi::c_void;
use core::ptr;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::{Rc, Weak};

use raw_window_handle::{
    AndroidDisplayHandle, AndroidNdkWindowHandle, RawDisplayHandle, RawWindowHandle,
};

use super::discovery::discover_current_glow_identified;

use crate::backend::gl::api::{
    ContextStamp, GlContextLifecycle, GlDiscoverySnapshot, GlError, GlFamilyProfile,
    OwnerThreadIdentity,
};

type Egl = khronos_egl::DynamicInstance<khronos_egl::EGL1_4>;

thread_local! {
    /// EGL currentness is thread-local.  Keeping the display lease registry on
    /// that same thread prevents a second context from terminating a display
    /// still used by its sibling context, without claiming EGL is Send/Sync.
    static DISPLAY_LEASES: RefCell<HashMap<usize, Weak<EglDisplayLease>>> = RefCell::new(HashMap::new());
}

const EGL_CONTEXT_MAJOR_VERSION_KHR: i32 = 0x3098;
const EGL_CONTEXT_MINOR_VERSION_KHR: i32 = 0x30FB;
const EGL_KHR_CREATE_CONTEXT: &str = "EGL_KHR_create_context";

/// GLES context version requested from an EGL provider.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum EglGlesVersion {
    /// OpenGL ES 3.0.
    V3_0,
    /// OpenGL ES 3.1.
    V3_1,
    /// OpenGL ES 3.2.
    V3_2,
}

impl EglGlesVersion {
    const fn minor(self) -> i32 {
        match self {
            Self::V3_0 => 0,
            Self::V3_1 => 1,
            Self::V3_2 => 2,
        }
    }

    const fn profile(self) -> GlFamilyProfile {
        GlFamilyProfile::Embedded {
            major: 3,
            minor: self.minor() as u8,
        }
    }

    /// Whether this observed profile satisfies a fixture's minimum request.
    ///
    /// EGL's ES client-version selection is not an exact-version contract on
    /// every implementation.  Fluxel's Android evidence matrix therefore
    /// treats its requested profile as a floor: ES 3.1 satisfies an ES 3.0
    /// case, while ES 3.0 must never satisfy an ES 3.1 case.
    const fn satisfies(self, requested: Self) -> bool {
        self.minor() >= requested.minor()
    }
}

/// Pixel dimensions used for an EGL pbuffer probe surface.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct EglPbufferSize {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
}

impl EglPbufferSize {
    /// A small valid default suitable for capability probes.
    pub const PROBE: Self = Self {
        width: 16,
        height: 16,
    };

    fn checked(self) -> Result<(i32, i32), EglProviderError> {
        let width = i32::try_from(self.width).map_err(|_| EglProviderError::InvalidSize(self))?;
        let height = i32::try_from(self.height).map_err(|_| EglProviderError::InvalidSize(self))?;
        if width == 0 || height == 0 {
            return Err(EglProviderError::InvalidSize(self));
        }
        Ok((width, height))
    }
}

/// EGL provider setup or presentation failure.
#[derive(Debug)]
pub(crate) enum EglProviderError {
    /// The dynamic EGL implementation could not be loaded.
    Load(String),
    /// The supplied raw-handle pair has no documented `eglGetDisplay` mapping.
    UnsupportedHandlePair {
        display: &'static str,
        window: &'static str,
    },
    /// The requested EGL display could not be obtained.
    DisplayUnavailable,
    /// EGL did not expose the extension required to request a GLES minor version.
    ExactVersionUnavailable(EglGlesVersion),
    /// No EGL config satisfies the provider's GLES surface requirements.
    ConfigUnavailable,
    /// A raw handle omitted a field required by this provider's documented mapping.
    InvalidNativeHandle(&'static str),
    /// A pbuffer size was zero or outside EGL's signed attribute range.
    InvalidSize(EglPbufferSize),
    /// An EGL call failed.  The operation label is stable for Layer 2 logging.
    Egl {
        operation: &'static str,
        error: khronos_egl::Error,
    },
    /// The current GLES context was below the version requested by Fluxel.
    ///
    /// A newer observed profile is valid: Android's EGL selection rules permit
    /// a request for an ES 3.x floor to yield a newer ES 3.x context.
    BelowRequestedProfile {
        requested: EglGlesVersion,
        observed: String,
    },
    /// A surface extent this context cannot map to a viewport was observed.
    ///
    /// The extent is a real observation here -- EGL answers the live surface's
    /// dimensions, unlike WGL where the Host's window size is the only source.
    /// The maximum viewport dimensions that bound it come from the same
    /// discovery snapshot, and a surface above them could only be rejected later
    /// by an unrelated pass.
    DrawableExtentExceedsViewportLimit {
        requested: [u32; 2],
        limit: [u32; 2],
    },
    /// The provider was used from a thread other than its owner.
    Gl(GlError),
}

impl From<GlError> for EglProviderError {
    fn from(value: GlError) -> Self {
        Self::Gl(value)
    }
}

impl EglProviderError {
    /// Flattens back into the stable GL error vocabulary the presentation
    /// domain reports, keeping provider diagnostics in the message.
    pub(crate) fn into_gl_error(self) -> GlError {
        match self {
            Self::Gl(error) => error,
            Self::Egl { operation, error } => GlError::Driver {
                operation,
                message: format!("EGL error: {error:?}"),
            },
            // Several presentation entry points observe an extent, and this
            // flattening point is below all of them, so the label names the
            // fact that failed rather than guessing which caller found it.
            Self::DrawableExtentExceedsViewportLimit { requested, limit } => GlError::Validation {
                operation: "drawable-extent",
                message: format!(
                    "drawable extent {requested:?} exceeds the maximum viewport dimensions {limit:?}"
                ),
            },
            other => GlError::Driver {
                operation: "egl-provider",
                message: format!("EGL provider error: {other:?}"),
            },
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum EglSurfaceKind {
    Window { native_window: *mut c_void },
    Pbuffer(EglPbufferSize),
}

/// Shared initialized EGL display.  It terminates exactly once, when its last
/// context lease is dropped on the owner thread.
struct EglDisplayLease {
    egl: Egl,
    display: khronos_egl::Display,
}

impl Drop for EglDisplayLease {
    fn drop(&mut self) {
        // EGL has no safe cross-thread ownership model. Every safe constructor
        // and context is !Send, so this is the owning thread in normal use.
        let _ = self.egl.terminate(self.display);
    }
}

/// A native EGL GLES context and its owned presentation surface.
///
/// This type is deliberately `!Send` and `!Sync`: EGL context currentness is
/// thread-affine, and all calls verify the captured Rust thread identity before
/// entering EGL.  Destruction is best-effort because `Drop` cannot report a
/// driver error; explicit [`dispose`](Self::dispose) is available to observe it.
pub(crate) struct EglGlesContext {
    lease: Rc<EglDisplayLease>,
    config: khronos_egl::Config,
    context: khronos_egl::Context,
    surface: Option<khronos_egl::Surface>,
    kind: EglSurfaceKind,
    version: EglGlesVersion,
    stamp: ContextStamp,
    snapshot: GlDiscoverySnapshot,
    /// The single dispatch table created for discovery and transferred to the
    /// direct device once construction succeeds.
    glow: Option<glow::Context>,
    owner: OwnerThreadIdentity,
    lifecycle: GlContextLifecycle,
    _not_send_sync: core::marker::PhantomData<*mut ()>,
}

#[cfg(feature = "native-gles-egl")]
impl super::context::NativeContextOwner for EglGlesContext {
    fn gl_proc(&self, name: &str) -> *const core::ffi::c_void {
        EglGlesContext::gl_proc(self, name)
    }
    fn kind(&self) -> super::context::NativeContextKind {
        EglGlesContext::kind(self)
    }

    fn context_stamp(&self) -> ContextStamp {
        self.stamp
    }

    fn owner_thread(&self) -> OwnerThreadIdentity {
        EglGlesContext::owner_thread(self)
    }

    fn make_current(&mut self, operation: &'static str) -> Result<(), GlError> {
        self.make_current_for(operation)
    }

    fn load_glow(&mut self) -> Result<glow::Context, GlError> {
        EglGlesContext::load_glow(self).map_err(EglProviderError::into_gl_error)
    }

    fn load_unmap_buffer(&self) -> Result<super::context::UnmapBufferFn, GlError> {
        EglGlesContext::load_unmap_buffer(self)
    }

    fn swap_buffers(&mut self) -> Result<(), GlError> {
        EglGlesContext::swap_buffers(self)
    }

    fn resize_drawable(&mut self, extent: [u32; 2]) -> Result<(), GlError> {
        EglGlesContext::resize_drawable(self, extent)
    }

    fn drawable_extent(&self) -> Option<[u32; 2]> {
        EglGlesContext::drawable_extent(self)
    }

    fn supports_presentation(&self) -> bool {
        self.is_window_surface()
    }
}

/// Immutable EGL/GLES discovery projection for fixture-only evidence. Native
/// EGL handles and the `glow` table never leave this module.
#[derive(Clone, Debug)]
pub(crate) struct EglEvidence {
    pub(crate) profile: String,
    /// The GLES floor passed to EGL when this context was created.
    pub(crate) requested_version: String,
    /// The GL version string observed from the current context.
    pub(crate) observed_version: String,
    pub(crate) version: String,
    pub(crate) shading_language_version: String,
    pub(crate) vendor: String,
    pub(crate) renderer: String,
    pub(crate) driver_or_browser: String,
    pub(crate) extensions: Vec<String>,
    pub(crate) capabilities: Vec<(String, bool)>,
    pub(crate) limits: Vec<(String, String)>,
    pub(crate) extent: [u32; 2],
}

impl EglGlesContext {
    /// Returns EGL's live drawable extent.
    ///
    /// Window ownership remains with the Host, but EGL owns the present
    /// surface and `eglQuerySurface` observes its current dimensions.  Querying
    /// it here prevents a Core lease from naming a stale pre-resize extent.
    fn known_drawable_extent(&self) -> Option<[u32; 2]> {
        match self.kind {
            EglSurfaceKind::Pbuffer(size) => Some([size.width, size.height]),
            EglSurfaceKind::Window { .. } => {
                let surface = self.surface?;
                let width = self
                    .lease
                    .egl
                    .query_surface(self.lease.display, surface, khronos_egl::WIDTH)
                    .ok()
                    .and_then(|value| u32::try_from(value).ok())?;
                let height = self
                    .lease
                    .egl
                    .query_surface(self.lease.display, surface, khronos_egl::HEIGHT)
                    .ok()
                    .and_then(|value| u32::try_from(value).ok())?;
                Some([width, height])
            }
        }
    }
    /// Creates a presentable context from Host-owned native handles.
    ///
    /// # Safety
    /// The raw display and window must describe the same live native platform
    /// objects, remain valid until this context is disposed, and only be used
    /// from the calling thread.  The Host retains ownership of those objects;
    /// this method owns only EGL objects created from them.
    pub(crate) unsafe fn new_window(
        stamp: ContextStamp,
        display: RawDisplayHandle,
        window: RawWindowHandle,
        version: EglGlesVersion,
    ) -> Result<Self, EglProviderError> {
        let (native_display, native_window) = native_window_pair(display, window)?;
        // SAFETY: `native_display` is validated by native_window_pair and the
        // caller upholds its native lifetime/platform contract.
        unsafe {
            Self::new_inner(
                stamp,
                native_display,
                EglSurfaceKind::Window { native_window },
                version,
            )
        }
    }

    /// Creates an off-screen pbuffer context for probes and WSL development.
    ///
    /// The default EGL display is intentionally used; this does not fabricate a
    /// host window and cannot present to one.
    pub(crate) fn new_pbuffer(
        stamp: ContextStamp,
        size: EglPbufferSize,
        version: EglGlesVersion,
    ) -> Result<Self, EglProviderError> {
        size.checked()?;
        // SAFETY: EGL_DEFAULT_DISPLAY is the specified null native-display
        // sentinel, and this provider owns all EGL objects subsequently made.
        unsafe {
            Self::new_inner(
                stamp,
                ptr::null_mut(),
                EglSurfaceKind::Pbuffer(size),
                version,
            )
        }
    }

    /// # Safety
    ///
    /// `native_display` and any window carried by `kind` must belong to the
    /// same live platform connection and remain valid throughout construction.
    /// The caller must be on the intended EGL owner thread. On every fallible
    /// path after EGL object creation this function destroys its own objects;
    /// it never takes ownership of the host handles.
    unsafe fn new_inner(
        stamp: ContextStamp,
        native_display: *mut c_void,
        kind: EglSurfaceKind,
        version: EglGlesVersion,
    ) -> Result<Self, EglProviderError> {
        let lease = acquire_display(native_display)?;
        if let Err(error) = lease.egl.bind_api(khronos_egl::OPENGL_ES_API) {
            return Err(EglProviderError::Egl {
                operation: "eglBindAPI",
                error,
            });
        }
        let extensions = lease
            .egl
            .query_string(Some(lease.display), khronos_egl::EXTENSIONS)
            .map_err(|error| EglProviderError::Egl {
                operation: "eglQueryString(EGL_EXTENSIONS)",
                error,
            })?
            .to_string_lossy();
        if version.minor() != 0 && !has_extension(&extensions, EGL_KHR_CREATE_CONTEXT) {
            return Err(EglProviderError::ExactVersionUnavailable(version));
        }
        let surface_type = match kind {
            EglSurfaceKind::Window { .. } => khronos_egl::WINDOW_BIT,
            EglSurfaceKind::Pbuffer(_) => khronos_egl::PBUFFER_BIT,
        };
        let config_attributes = [
            khronos_egl::SURFACE_TYPE,
            surface_type,
            khronos_egl::RENDERABLE_TYPE,
            khronos_egl::OPENGL_ES3_BIT,
            khronos_egl::RED_SIZE,
            8,
            khronos_egl::GREEN_SIZE,
            8,
            khronos_egl::BLUE_SIZE,
            8,
            khronos_egl::ALPHA_SIZE,
            8,
            khronos_egl::DEPTH_SIZE,
            24,
            khronos_egl::STENCIL_SIZE,
            8,
            khronos_egl::NONE,
        ];
        let config = match lease
            .egl
            .choose_first_config(lease.display, &config_attributes)
        {
            Ok(Some(config)) => config,
            Ok(None) => return Err(EglProviderError::ConfigUnavailable),
            Err(error) => {
                return Err(EglProviderError::Egl {
                    operation: "eglChooseConfig",
                    error,
                });
            }
        };
        // EGL_CONTEXT_CLIENT_VERSION alone only selects the ES major family.
        // On this Android EGL implementation that means a request for ES 3.0
        // is legally upgraded to 3.1.  When EGL_KHR_create_context is present,
        // include an explicit minor value even for zero so the fixture can
        // pass the requested floor through to EGL.  The evidence path verifies
        // the observed profile is at least that floor, because EGL may legally
        // return a newer ES 3.x context.
        let context_attributes =
            exact_context_attributes(version, has_extension(&extensions, EGL_KHR_CREATE_CONTEXT));
        let context =
            match lease
                .egl
                .create_context(lease.display, config, None, &context_attributes)
            {
                Ok(context) => context,
                Err(error) => {
                    return Err(EglProviderError::Egl {
                        operation: "eglCreateContext",
                        error,
                    });
                }
            };
        let surface = match create_surface(&lease.egl, lease.display, config, kind) {
            Ok(surface) => surface,
            Err(error) => {
                let _ = lease.egl.destroy_context(lease.display, context);
                return Err(error);
            }
        };
        if let Err(error) =
            lease
                .egl
                .make_current(lease.display, Some(surface), Some(surface), Some(context))
        {
            let _ = lease.egl.destroy_surface(lease.display, surface);
            let _ = lease.egl.destroy_context(lease.display, context);
            return Err(EglProviderError::Egl {
                operation: "eglMakeCurrent",
                error,
            });
        }
        // Verification and discovery observe the exact current context before
        // construction completes; any failure drops the transactional result
        // and rolls the EGL objects back.  That rollback is why the remaining
        // steps run in one fallible scope rather than as bare `?` returns: from
        // here on the context is already current and the pair has no owner yet,
        // so an `Err` that returned without destroying them would leak both for
        // the life of the display and leave a context nothing can name
        // installed on the owner thread.
        let egl_loader = |name: &str| {
            lease
                .egl
                .get_proc_address(name)
                .map_or(ptr::null(), |function| {
                    function as *const () as *const c_void
                })
        };
        let prepared = (|| -> Result<(GlDiscoverySnapshot, glow::Context), EglProviderError> {
            // SAFETY: make_current above established this exact EGL context on
            // the owner thread and EGL owns the loader for glow's use.
            let glow = unsafe { glow::Context::from_loader_function(egl_loader) };
            use glow::HasContext as _;
            // SAFETY: as above; this queries only the current GLES context.
            let observed = unsafe { glow.get_parameter_string(glow::VERSION) };
            let observed_profile = parse_gles_version(&observed);
            if !observed_profile.is_some_and(|profile| profile.satisfies(version)) {
                return Err(EglProviderError::BelowRequestedProfile {
                    requested: version,
                    observed,
                });
            }
            // The EGL display's own identity strings exist only here, so they
            // are read while this display is initialized and this exact context
            // is current (audit P2-6).
            let driver_identity = egl_driver_identity(&lease.egl, lease.display);
            // SAFETY: the current context serves every discovery query and probe.
            let snapshot = unsafe {
                discover_current_glow_identified(&glow, stamp, egl_loader, &driver_identity)
            }
            .map_err(|error| {
                EglProviderError::Gl(GlError::Driver {
                    operation: "discover EGL context",
                    message: format!("native GL discovery failed: {error:?}"),
                })
            })?;
            // A pbuffer's extent is fixed by this construction, so it is
            // bounded before the context is handed back; window surfaces report
            // their live extent from EGL at each acquisition and are bounded
            // there.
            if let EglSurfaceKind::Pbuffer(size) = kind {
                check_drawable_extent(
                    snapshot.limits().max_viewport_dimensions,
                    [size.width, size.height],
                )?;
            }
            Ok((snapshot, glow))
        })();
        let (snapshot, glow) = match prepared {
            Ok(result) => result,
            Err(error) => {
                let _ = lease.egl.destroy_surface(lease.display, surface);
                let _ = lease.egl.destroy_context(lease.display, context);
                return Err(error);
            }
        };
        let result = Self {
            lease,
            config,
            context,
            surface: Some(surface),
            kind,
            version,
            stamp,
            snapshot,
            glow: Some(glow),
            owner: OwnerThreadIdentity::current(),
            lifecycle: GlContextLifecycle::Active,
            _not_send_sync: core::marker::PhantomData,
        };
        Ok(result)
    }
}

#[path = "egl/evidence.rs"]
mod evidence;
#[path = "egl/runtime.rs"]
mod runtime;
use evidence::{EVIDENCE_FRAGMENT, EVIDENCE_VERTEX, compile_evidence_shader};

fn current_binding_is_self<C: Eq, D: Eq>(
    current_context: Option<C>,
    current_display: Option<D>,
    owned_context: C,
    owned_display: D,
) -> bool {
    current_context == Some(owned_context) && current_display == Some(owned_display)
}

#[path = "egl/support.rs"]
mod support;
use support::*;
