//! Windows WGL context and window-surface ownership.
//!
//! The Host owns the Win32 window and keeps its borrowed raw handle alive.
//! RHI owns the device context lease, pixel format, WGL contexts, currentness,
//! and presentation.  This module deliberately has no event-loop dependency.
//!
//! This module does not own Fluxel's GL object tables, the render pass, or the
//! viewport any pass renders with, and it never resizes the Host's window.
//!
//! `gdi` below is the private Win32 ABI boundary this provider calls through;
//! it holds no provider state. Tests live in this module's own `tests/mod.rs`.

#![cfg(all(windows, feature = "native-gl-wgl"))]

use core::cell::Cell;
use core::ffi::{c_char, c_int, c_void};
use core::marker::PhantomData;
use std::ffi::CString;
use std::rc::Rc;

use raw_window_handle::{DisplayHandle, RawDisplayHandle, RawWindowHandle, WindowHandle};

use super::discovery::{
    REQUIRED_DESKTOP_VERSION, desktop_context_floor_marker, discover_current_glow_identified,
};
use crate::backend::gl::api::{
    ContextStamp, GlContextFlags, GlContextLifecycle, GlDiscoverySnapshot, GlError, GlVersion,
    OwnerThreadIdentity,
};

use glutin_wgl_sys::{wgl, wgl_extra};

mod gdi;

pub(super) type Hwnd = *mut c_void;
pub(super) type Hdc = *const c_void;

/// The RHI current-binding arbiter is thread-local because WGL currentness is.
/// It never substitutes bookkeeping for a driver bind: every executable seam
/// still calls `wglMakeCurrent` before touching GL or WGL.
#[derive(Default)]
struct CurrentBindingArbiter(Cell<Option<usize>>);

impl CurrentBindingArbiter {
    fn record(&self, context: wgl::types::HGLRC) {
        self.0.set(Some(context as usize));
    }

    fn clear_if(&self, context: wgl::types::HGLRC) {
        if self.0.get() == Some(context as usize) {
            self.0.set(None);
        }
    }

    #[cfg(test)]
    fn current(&self) -> Option<usize> {
        self.0.get()
    }
}

thread_local! {
    static RHI_CURRENT_WGL: CurrentBindingArbiter = CurrentBindingArbiter::default();
}

/// The desktop core version this provider asks a WGL driver for.
///
/// The request in [`core_context_attributes`], the early check on the actual
/// version string in [`meets_required_desktop_context`], the check on the
/// discovered profile in [`WglContextSurface::open`], and the recorded marker all read
/// this one value, so the four cannot ask for, enforce, accept, or record
/// different things (audit P2-12).
///
/// This used to be a second constant declared here beside
/// `native::discovery`'s, with a cross-check to keep the copies honest. The
/// copies are gone instead: `native` re-exports the constant and the marker
/// builder for exactly this reader, which is what the audit's residual asked
/// for, and a single value cannot drift from itself.
const REQUIRED_DESKTOP_CONTEXT: GlVersion = REQUIRED_DESKTOP_VERSION;

/// Failure while RHI owns a WGL context or its window surface.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum WglContextError {
    /// The Host did not supply a Win32 window handle.
    UnsupportedWindowHandle,
    /// The Host did not supply the Windows display kind.
    UnsupportedDisplayHandle,
    /// RHI could not lease an HDC for the still-live Host window.
    AcquireDeviceContext(String),
    /// Pixel formats are immutable for the lifetime of a Win32 window.
    PixelFormatAlreadyConfigured,
    /// The requested double-buffered RGBA pixel format is unavailable.
    ChoosePixelFormat(String),
    /// GDI rejected installation of the selected pixel format.
    SetPixelFormat(String),
    /// The bootstrap compatibility WGL context could not be created.
    CreateBootstrapContext(String),
    /// The bootstrap context could not be made current.
    MakeBootstrapCurrent(String),
    /// `wglCreateContextAttribsARB` was not exported by the driver.
    MissingCreateContextAttribs,
    /// The driver rejected every requested desktop core profile down to 4.0.
    CreateCoreContext(String),
    /// The final desktop core context could not be made current.
    MakeCoreCurrent(String),
    /// The requested context was created but did not report the required
    /// desktop core version and profile when queried from the actual current
    /// context.
    ActualContext { version: String, profile_mask: i32 },
    /// The Host reported a drawable extent this context cannot map to a viewport.
    ///
    /// Native GL has no query for the default framebuffer's extent, so the size
    /// is the Host's fact; the maximum viewport dimensions that bound it are a
    /// fact this provider did read, and a size above them could only be rejected
    /// later by an unrelated pass.
    DrawableExtentExceedsViewportLimit {
        requested: [u32; 2],
        limit: [u32; 2],
    },
    /// The context was usable but discovery could not gather complete facts.
    Discovery(String),
    /// A provider call was issued from another Rust thread.
    WrongThread {
        expected: OwnerThreadIdentity,
        actual: OwnerThreadIdentity,
    },
    /// The provider lifecycle cannot perform the requested operation.
    Lifecycle(GlContextLifecycle),
    /// The WGL driver rejected a currentness or presentation operation.
    Driver {
        operation: &'static str,
        message: String,
    },
}

impl WglContextError {
    fn driver(operation: &'static str) -> Self {
        Self::Driver {
            operation,
            message: std::io::Error::last_os_error().to_string(),
        }
    }

    fn as_gl_error(&self, operation: &'static str) -> GlError {
        match self {
            Self::WrongThread { expected, actual } => GlError::WrongThread {
                operation,
                expected: *expected,
                actual: *actual,
            },
            Self::Lifecycle(lifecycle) => GlError::InvalidLifecycle {
                operation,
                lifecycle: *lifecycle,
            },
            Self::Driver { message, .. } => GlError::Driver {
                operation,
                message: message.clone(),
            },
            Self::DrawableExtentExceedsViewportLimit { requested, limit } => GlError::Validation {
                operation,
                message: format!(
                    "drawable extent {requested:?} exceeds the maximum viewport dimensions {limit:?}"
                ),
            },
            other => GlError::Driver {
                operation,
                message: format!("WGL provider setup error: {other:?}"),
            },
        }
    }
}

/// RHI-owned current WGL context and double-buffered window surface.
///
/// The `Rc` marker makes the provider neither `Send` nor `Sync`: WGL's thread
/// currentness requirement is then enforced both structurally and at each
/// executable entry point.  The Host must keep the source window alive until
/// this value has been dropped.
pub(crate) struct WglContextSurface {
    stamp: ContextStamp,
    hwnd: Hwnd,
    hdc: Hdc,
    hglrc: wgl::types::HGLRC,
    /// The one dispatch table transferred into the direct device during
    /// construction. Fixture-only methods borrow it before that transfer.
    glow: Option<glow::Context>,
    snapshot: GlDiscoverySnapshot,
    owner_thread: OwnerThreadIdentity,
    lifecycle: Cell<GlContextLifecycle>,
    extent: Cell<[u32; 2]>,
    _not_send_sync: PhantomData<Rc<()>>,
}

/// Immutable observations exported to the fixture layer.  This intentionally
/// contains strings and numbers only; WGL handles and the GL dispatch table
/// remain confined to this module.
#[derive(Clone, Debug)]
pub(crate) struct WglEvidence {
    pub(crate) profile: String,
    pub(crate) version: String,
    pub(crate) shading_language_version: String,
    pub(crate) vendor: String,
    pub(crate) renderer: String,
    pub(crate) driver_or_browser: String,
    pub(crate) debug: bool,
    pub(crate) forward_compatible: bool,
    pub(crate) robust_access: bool,
    pub(crate) no_error: bool,
    pub(crate) other_flags: Vec<String>,
    pub(crate) extensions: Vec<String>,
    /// Typed counterparts of reported extension names. Raw names remain in
    /// `extensions`; this is diagnostic evidence only and never enables a
    /// capability by itself.
    pub(crate) typed_extensions: Vec<(String, String)>,
    pub(crate) limits: Vec<(String, String)>,
    pub(crate) capabilities: Vec<(String, bool)>,
    pub(crate) surface_facts: String,
    pub(crate) drawable_extent: [u32; 2],
}

impl WglContextSurface {
    pub(crate) const fn context_stamp(&self) -> ContextStamp {
        self.stamp
    }

    /// Creates a double-buffered WGL surface for the Host's borrowed window.
    ///
    /// A legacy context is only a bootstrap loader for
    /// `wglCreateContextAttribsARB`; it is deleted before this succeeds.  The
    /// retained context is the highest available desktop OpenGL core context
    /// down to 4.0, made current on
    /// this thread before the method returns.
    pub(crate) fn open(
        stamp: ContextStamp,
        window: WindowHandle<'_>,
        display: DisplayHandle<'_>,
        extent: [u32; 2],
    ) -> Result<Self, WglContextError> {
        let hwnd = host_hwnd(window, display)?;
        Self::open_from_hwnd(stamp, hwnd, extent, None)
    }

    /// Opens a desktop core context for a hardware fixture's minimum-version request.
    ///
    /// This is deliberately crate-private: choosing a WGL context version is a
    /// test-harness concern, not portable RHI policy. Unlike [`Self::open`],
    /// this sends one WGL version request rather than applying Fluxel's
    /// highest-to-lowest fallback policy. A driver may legally expose a newer
    /// context than requested; the actual current version is checked to be at
    /// least the requested minimum and is recorded by the fixture.
    pub(crate) fn open_with_minimum_desktop_version(
        stamp: ContextStamp,
        window: WindowHandle<'_>,
        display: DisplayHandle<'_>,
        extent: [u32; 2],
        version: GlVersion,
    ) -> Result<Self, WglContextError> {
        if !is_supported_desktop_test_version(version) {
            return Err(WglContextError::Discovery(format!(
                "WGL fixture only accepts desktop core versions 4.0 through 4.6, got {}.{}",
                version.major, version.minor
            )));
        }
        let hwnd = host_hwnd(window, display)?;
        Self::open_from_hwnd(stamp, hwnd, extent, Some(version))
    }

    /// Opens from a descriptor which was validated before it crossed to the
    /// worker.  Kept private so raw HWND values cannot become a general API.
    fn open_from_hwnd(
        stamp: ContextStamp,
        hwnd: Hwnd,
        extent: [u32; 2],
        requested_version: Option<GlVersion>,
    ) -> Result<Self, WglContextError> {
        // SAFETY: `hwnd` came from a borrowed, live `Win32WindowHandle`; the
        // caller's Host-lifetime contract keeps it valid until our Drop.
        let hdc = unsafe { gdi::get_dc(hwnd) };
        if hdc.is_null() {
            return Err(WglContextError::AcquireDeviceContext(last_os_error()));
        }

        let mut cleanup = OpenCleanup::new(hwnd, hdc);
        let format = gdi::PixelFormatDescriptor::window_rgba();
        // SAFETY: HDC is leased above and `format` is a valid C-layout PFD for
        // the duration of this call.
        let installed = unsafe { gdi::get_pixel_format(hdc) };
        if installed != 0 {
            return Err(WglContextError::PixelFormatAlreadyConfigured);
        }
        // SAFETY: same HDC/PFD validity as above; this only queries GDI.
        let pixel_format = unsafe { gdi::choose_pixel_format(hdc, &format) };
        if pixel_format == 0 {
            return Err(WglContextError::ChoosePixelFormat(last_os_error()));
        }
        // SAFETY: SetPixelFormat is called once for this newly configured
        // window HDC, using the format returned from ChoosePixelFormat.
        if unsafe { gdi::set_pixel_format(hdc, pixel_format, &format) } == 0 {
            return Err(WglContextError::SetPixelFormat(last_os_error()));
        }

        // SAFETY: HDC has a compatible installed pixel format.  This legacy
        // WGL context is retained only long enough to load extension pointers.
        let bootstrap = unsafe { wgl::CreateContext(hdc) };
        if bootstrap.is_null() {
            return Err(WglContextError::CreateBootstrapContext(last_os_error()));
        }
        cleanup.bootstrap = bootstrap;
        // SAFETY: both handles are owned by this construction transaction.
        if unsafe { wgl::MakeCurrent(hdc, bootstrap) } == 0 {
            return Err(WglContextError::MakeBootstrapCurrent(last_os_error()));
        }

        let extensions = wgl_extra::Wgl::load_with(load_wgl_symbol);
        if !extensions.CreateContextAttribsARB.is_loaded() {
            return Err(WglContextError::MissingCreateContextAttribs);
        }
        // WGL has no portable "best version" request. The production path
        // explicitly walks down from 4.6 to the v13 4.0 floor; the fixture path
        // deliberately makes one minimum-version request so each supported
        // floor can be independently exercised on a GL 4.6-capable Windows
        // machine. The driver may return a newer context for that request.
        let create = |version| {
            let attributes = core_context_attributes(version);
            // SAFETY: bootstrap current; attributes are a terminated WGL list.
            let context = unsafe {
                extensions.CreateContextAttribsARB(hdc, core::ptr::null(), attributes.as_ptr())
            };
            (!context.is_null()).then_some(context)
        };
        let core = match requested_version {
            Some(version) => create(version),
            None => DESKTOP_CONTEXT_REQUESTS.into_iter().find_map(create),
        };
        let core = core.unwrap_or(core::ptr::null_mut());
        if core.is_null() {
            return Err(WglContextError::CreateCoreContext(last_os_error()));
        }
        cleanup.core = core;

        // SAFETY: detach the bootstrap before destroying it; both calls use
        // only owned handles and obey WGL's make-current/delete sequence.
        if unsafe { wgl::MakeCurrent(hdc, core) } == 0 {
            return Err(WglContextError::MakeCoreCurrent(last_os_error()));
        }
        // SAFETY: the legacy context is no longer current and is construction
        // owned.  It is never exposed to a caller.
        unsafe { wgl::DeleteContext(bootstrap) };
        cleanup.bootstrap = core::ptr::null();

        // SAFETY: the core context is current on this thread. `glow` only
        // stores function pointers; command use remains behind `with_current`.
        let glow = unsafe { glow::Context::from_loader_function(load_wgl_symbol) };
        verify_actual_desktop_core_context(&glow, requested_version)?;
        // The driver identity is the platform layer's string and exists only
        // here, so it is read while this exact context is current (audit P2-6).
        let driver_identity = wgl_driver_identity(&glow);
        // Discovery (including the capability operation probes) observes the
        // exact current context before construction completes; a failure here
        // keeps the transaction roll-backed by `cleanup`.
        // SAFETY: the final core HGLRC is current on this construction thread;
        // its loader and identity strings are derived from that exact context.
        // Discovery failure remains inside `cleanup`, which tears down both
        // native handles before the error leaves this transaction.
        let snapshot = unsafe {
            discover_current_glow_identified(&glow, stamp, load_wgl_symbol, &driver_identity)
        }
        .map_err(|error| {
            WglContextError::Discovery(format!("native GL discovery failed: {error:?}"))
        })?;
        if !snapshot
            .context()
            .profile()
            .meets(Some(REQUIRED_DESKTOP_CONTEXT), None)
        {
            return Err(WglContextError::Discovery(format!(
                "WGL context did not report the required desktop core version {:?}",
                REQUIRED_DESKTOP_CONTEXT
            )));
        }
        verify_recorded_desktop_floor(snapshot.context().flags())?;
        // The Host's initial drawable size is bounded at the same point as every
        // later resize, so a provider is never handed back for a drawable whose
        // viewport could not exist.
        check_drawable_extent(snapshot.limits().max_viewport_dimensions, extent)?;
        let result = Self {
            stamp,
            hwnd,
            hdc,
            hglrc: core,
            glow: Some(glow),
            snapshot,
            owner_thread: OwnerThreadIdentity::current(),
            lifecycle: Cell::new(if extent.contains(&0) {
                GlContextLifecycle::Suspended
            } else {
                GlContextLifecycle::Active
            }),
            extent: Cell::new(extent),
            _not_send_sync: PhantomData,
        };
        // Only transfer final-context destruction after its actual driver
        // profile has been checked; a rejected profile remains transactional.
        cleanup.core = core::ptr::null();
        cleanup.disarm();
        Ok(result)
    }
}

#[path = "wgl/runtime.rs"]
mod runtime;

#[cfg(all(windows, feature = "native-gl-wgl"))]
impl super::context::NativeContextOwner for WglContextSurface {
    fn kind(&self) -> super::context::NativeContextKind {
        super::context::NativeContextKind::Desktop
    }

    fn context_stamp(&self) -> ContextStamp {
        self.context_stamp()
    }

    fn owner_thread(&self) -> OwnerThreadIdentity {
        WglContextSurface::owner_thread(self)
    }

    fn make_current(&mut self, operation: &'static str) -> Result<(), GlError> {
        self.bind_current(operation)
    }

    fn load_glow(&mut self) -> Result<glow::Context, GlError> {
        WglContextSurface::load_glow(self)
    }

    fn swap_buffers(&mut self) -> Result<(), GlError> {
        WglContextSurface::present(self)
    }

    fn resize_drawable(&mut self, extent: [u32; 2]) -> Result<(), GlError> {
        WglContextSurface::resize(self, extent)
    }

    fn drawable_extent(&self) -> Option<[u32; 2]> {
        self.client_extent()
    }

    fn supports_presentation(&self) -> bool {
        true
    }
}

// At 4x4 these coordinates cover exactly the two pixel centres at row 1,
// columns 1 and 2 in GL's bottom-left readback order. Keep this in lockstep
// with the GLES fixture and scripts/gl_raster_picture.py.
const EVIDENCE_VERTEX: &str = "#version 400 core\nvoid main() { const vec2 p[3]=vec2[3](vec2(-0.6,-0.5),vec2(0.6,-0.5),vec2(0.0,0.0)); gl_Position=vec4(p[gl_VertexID],0.0,1.0); }";
const EVIDENCE_FRAGMENT: &str = "#version 400 core\nout vec4 o; void main() { o=vec4(48.0/255.0,176.0/255.0,112.0/255.0,1.0); }";

#[path = "wgl/support.rs"]
mod support;
use support::*;
