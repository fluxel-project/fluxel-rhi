//! WGL teardown, evidence, and context-open support.
//!
//! This private module retains only helper code for `WglContextSurface`; it
//! never owns a dispatcher or moves an HGLRC across threads.

use super::*;

pub(super) fn compile_evidence_shader(
    gl: &glow::Context,
    kind: u32,
    source: &str,
    operation: &'static str,
) -> Result<glow::NativeShader, GlError> {
    use glow::HasContext as _;
    // SAFETY: caller holds this WGL context current for this complete helper.
    unsafe {
        let shader = gl
            .create_shader(kind)
            .map_err(|message| GlError::Driver { operation, message })?;
        gl.shader_source(shader, source);
        gl.compile_shader(shader);
        if gl.get_shader_compile_status(shader) {
            Ok(shader)
        } else {
            let message = gl.get_shader_info_log(shader);
            gl.delete_shader(shader);
            Err(GlError::Driver { operation, message })
        }
    }
}

impl Drop for WglContextSurface {
    fn drop(&mut self) {
        self.lifecycle.set(GlContextLifecycle::Disposed);
        // SAFETY: !Send prevents safe cross-thread destruction.  If unsafe
        // foreign code nevertheless made this HGLRC current elsewhere,
        // DeleteContext fails; in that case we leak rather than release its
        // HDC or pretend teardown succeeded.  On this thread detach only our
        // exact current context before deletion.
        unsafe {
            if wgl::GetCurrentContext() == self.hglrc
                && wgl::MakeCurrent(self.hdc, core::ptr::null()) == 0
            {
                self.lifecycle.set(GlContextLifecycle::Poisoned);
                return;
            }
            if wgl::DeleteContext(self.hglrc) == 0 {
                self.lifecycle.set(GlContextLifecycle::Poisoned);
                return;
            }
            RHI_CURRENT_WGL.with(|arbiter| arbiter.clear_if(self.hglrc));
            gdi::release_dc(self.hwnd, self.hdc);
        }
    }
}

pub(super) struct OpenCleanup {
    pub(super) hwnd: Hwnd,
    pub(super) hdc: Hdc,
    pub(super) bootstrap: wgl::types::HGLRC,
    pub(super) core: wgl::types::HGLRC,
    armed: bool,
}

impl OpenCleanup {
    pub(super) fn new(hwnd: Hwnd, hdc: Hdc) -> Self {
        Self {
            hwnd,
            hdc,
            bootstrap: core::ptr::null(),
            core: core::ptr::null(),
            armed: true,
        }
    }

    pub(super) fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for OpenCleanup {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        // SAFETY: these handles were created in this transaction and have not
        // escaped. If the driver refuses deletion, leak the dependent HDC
        // rather than releasing a potentially still-bound surface.
        unsafe {
            if !self.core.is_null()
                && wgl::GetCurrentContext() == self.core
                && wgl::MakeCurrent(self.hdc, core::ptr::null()) == 0
            {
                return;
            }
            if !self.core.is_null() && wgl::DeleteContext(self.core) == 0 {
                return;
            }
            if !self.bootstrap.is_null() {
                if wgl::GetCurrentContext() == self.bootstrap
                    && wgl::MakeCurrent(self.hdc, core::ptr::null()) == 0
                {
                    return;
                }
                if wgl::DeleteContext(self.bootstrap) == 0 {
                    return;
                }
            }
            gdi::release_dc(self.hwnd, self.hdc);
        }
    }
}

pub(super) fn host_hwnd(
    window: WindowHandle<'_>,
    display: DisplayHandle<'_>,
) -> Result<Hwnd, WglContextError> {
    if !matches!(display.as_raw(), RawDisplayHandle::Windows(_)) {
        return Err(WglContextError::UnsupportedDisplayHandle);
    }
    match window.as_raw() {
        RawWindowHandle::Win32(handle) => Ok(handle.hwnd.get() as Hwnd),
        _ => Err(WglContextError::UnsupportedWindowHandle),
    }
}

pub(super) const DESKTOP_CONTEXT_REQUESTS: [GlVersion; 7] = [
    GlVersion::new(4, 6),
    GlVersion::new(4, 5),
    GlVersion::new(4, 4),
    GlVersion::new(4, 3),
    GlVersion::new(4, 2),
    GlVersion::new(4, 1),
    REQUIRED_DESKTOP_CONTEXT,
];

/// The Windows evidence fixture exercises the complete v13 desktop range.
/// Keep this separate from the production fallback array: the latter is an
/// ordered policy, whereas this predicate makes an explicit fixture request
/// refuse unsupported spelling rather than silently testing some other path.
pub(super) const fn is_supported_desktop_test_version(version: GlVersion) -> bool {
    version.major == 4 && version.minor <= 6
}

pub(super) fn core_context_attributes(version: GlVersion) -> [c_int; 7] {
    [
        wgl_extra::CONTEXT_MAJOR_VERSION_ARB as c_int,
        c_int::from(version.major),
        wgl_extra::CONTEXT_MINOR_VERSION_ARB as c_int,
        c_int::from(version.minor),
        wgl_extra::CONTEXT_PROFILE_MASK_ARB as c_int,
        wgl_extra::CONTEXT_CORE_PROFILE_BIT_ARB as c_int,
        0,
    ]
}

/// The exact string [`REQUIRED_DESKTOP_CONTEXT`] is recorded as.
///
/// Built by the module that records it, so the marker this provider requires is
/// the marker discovery stamps rather than a second spelling of it (audit
/// P2-12).
pub(super) fn recorded_desktop_floor() -> String {
    desktop_context_floor_marker()
}

/// Requires the floor this provider asked for to be the one discovery recorded.
///
/// Both sides of the comparison now come from one source, so this can no longer
/// fail because two constants drifted apart.  It is kept because it still
/// proves something the constant cannot: that the snapshot this provider was
/// handed actually carries the marker, rather than being a snapshot whose
/// profile took a path that never stamped it.  That is a fact about the value
/// in hand, and reading the recorded flags keeps the check pure and free of any
/// context.
pub(super) fn verify_recorded_desktop_floor(flags: &GlContextFlags) -> Result<(), WglContextError> {
    let expected = recorded_desktop_floor();
    if flags.other.contains(&expected) {
        Ok(())
    } else {
        Err(WglContextError::Discovery(format!(
            "WGL provider requires {expected}, but the discovery record does not carry it"
        )))
    }
}

/// The Windows platform layer's driver identity for the current context.
///
/// Unlike an EGL display, Windows' WGL layer answers no query that names the
/// driver it dispatched to, and the version string is already its own field in
/// the discovery record -- repeating it here is exactly the P2-6 defect.  The
/// installable client driver is the layer that runs these commands, and its
/// vendor and renderer pair is the only identity it publishes, so the pair is
/// what this provider supplies as the platform's driver identity.
///
/// Both reads are this function's own queries, so the error they may raise is
/// consumed here rather than left pending for discovery to misread as a failure
/// of the context it is about to describe.  A blank answer returns an empty
/// string, which discovery records as an unavailable identity instead of as a
/// driver name nobody observed.
pub(super) fn wgl_driver_identity(glow: &glow::Context) -> String {
    use glow::HasContext as _;

    // SAFETY: the caller made this exact context current on this thread and
    // keeps it current for the whole call; both queries only read.
    let vendor = unsafe { glow.get_parameter_string(glow::VENDOR) };
    // SAFETY: same current-context invariant as the preceding query.
    let renderer = unsafe { glow.get_parameter_string(glow::RENDERER) };
    // SAFETY: same current-context invariant; this only consumes the error the
    // two queries above may have produced.
    let _ = unsafe { glow.get_error() };
    driver_identity_from(&vendor, &renderer)
}

/// Composes the platform driver identity from the driver's own two strings.
pub(super) fn driver_identity_from(vendor: &str, renderer: &str) -> String {
    match (vendor.trim(), renderer.trim()) {
        ("", "") => String::new(),
        ("", renderer) => renderer.to_owned(),
        (vendor, "") => vendor.to_owned(),
        (vendor, renderer) => format!("{vendor} {renderer}"),
    }
}

/// Whether a Host-reported drawable extent can be a viewport on this context.
///
/// The drawable's size is the Host's fact on this family, and the viewport any
/// pass must map into it is bounded by the driver's queried maximum viewport
/// dimensions.  Recording a size the context cannot map is what would otherwise
/// surface as an unrelated driver error in the first pass that draws into the
/// drawable, so the bound is checked where the size is recorded.
///
/// A zero-area extent is not a rejection: it is how a Host reports a minimized
/// or not-yet-shown window and becomes suspension instead.  A limit of zero
/// means the context reported none, so nothing can be measured against it and
/// the check cannot fail closed on a fact that was never read.
pub(super) fn within_viewport_limit(limit: [u32; 2], extent: [u32; 2]) -> bool {
    if extent.contains(&0) || limit.contains(&0) {
        return true;
    }
    extent[0] <= limit[0] && extent[1] <= limit[1]
}

/// Resolves a drawable extent against the context's recorded viewport limit.
pub(super) fn check_drawable_extent(
    limit: [u32; 2],
    extent: [u32; 2],
) -> Result<(), WglContextError> {
    if within_viewport_limit(limit, extent) {
        return Ok(());
    }
    Err(WglContextError::DrawableExtentExceedsViewportLimit {
        requested: extent,
        limit,
    })
}

pub(super) fn load_wgl_symbol(symbol: &str) -> *const c_void {
    let Ok(name) = CString::new(symbol) else {
        return core::ptr::null();
    };
    // SAFETY: CString supplies a NUL-terminated symbol name. WGL documents
    // this query for a current context; bootstrap/core is current at each
    // loader construction site. The returned pointer is only inspected here.
    let proc = unsafe { wgl::GetProcAddress(name.as_ptr() as *const c_char) } as *const c_void;
    if usable_wgl_proc(proc) {
        return proc;
    }
    // WGL deliberately does not return OpenGL 1.1 exports. `glow` needs both
    // that base ABI and WGL-loaded modern entry points, so resolve the former
    // from the already-linked system OpenGL module.
    // SAFETY: both byte strings are NUL terminated. `opengl32` is linked by
    // `glutin_wgl_sys`; no module ownership is acquired or released here.
    let module = unsafe { gdi::opengl32_module() };
    if module.is_null() {
        return core::ptr::null();
    }
    // SAFETY: `module` is a live borrowed module handle and `name` remains
    // alive until the OS has finished copying/reading the symbol string.
    unsafe { gdi::module_proc_address(module, name.as_ptr()) }
}

pub(super) fn usable_wgl_proc(proc: *const c_void) -> bool {
    let value = proc as isize;
    !proc.is_null() && !matches!(value, 1 | 2 | 3 | -1)
}

pub(super) fn verify_actual_desktop_core_context(
    glow: &glow::Context,
    requested_version: Option<GlVersion>,
) -> Result<(), WglContextError> {
    use glow::HasContext as _;

    // SAFETY: construction made the owned final HGLRC current on this thread.
    // The query checks the driver's actual context, not WGL request attributes.
    let version = unsafe { glow.get_parameter_string(glow::VERSION) };
    // SAFETY: same current-context invariant as the preceding query.
    if unsafe { glow.get_error() } != glow::NO_ERROR {
        return Err(WglContextError::ActualContext {
            version,
            profile_mask: 0,
        });
    }
    // SAFETY: GL_CONTEXT_PROFILE_MASK is valid on the requested desktop core
    // context. A query error is treated as rejection, never as a core claim.
    let profile_mask = unsafe { glow.get_parameter_i32(0x9126) };
    // SAFETY: consumes the error produced by precisely the preceding query.
    let actual_version = parse_version(&version);
    let version_meets_request = requested_version
        .map(|requested| actual_version.is_some_and(|actual| requested.is_met_by(actual)))
        .unwrap_or_else(|| meets_required_desktop_context(&version));
    if unsafe { glow.get_error() } != glow::NO_ERROR
        || (profile_mask & 0x0000_0001) == 0
        || !version_meets_request
    {
        return Err(WglContextError::ActualContext {
            version,
            profile_mask,
        });
    }
    Ok(())
}

/// Whether an actual version string meets [`REQUIRED_DESKTOP_CONTEXT`].
///
/// The requirement is a floor, not a spelling: it compares parsed components
/// against the same constant the context was requested with, so a relaxation
/// cannot move the request without moving the check (audit P2-12).  Which
/// desktop majors this family accepts at all stays where it belongs -- in the
/// profile parser, which answers it for every domain at once.
pub(super) fn meets_required_desktop_context(version: &str) -> bool {
    parse_version(version).is_some_and(|actual| REQUIRED_DESKTOP_CONTEXT.is_met_by(actual))
}

pub(super) fn parse_version(version: &str) -> Option<GlVersion> {
    let prefix = version.split_whitespace().next()?;
    let mut components = prefix.split('.');
    let major = components
        .next()
        .and_then(|component| component.parse::<u8>().ok())?;
    let minor = components
        .next()
        .and_then(|component| component.parse::<u8>().ok())?;
    Some(GlVersion::new(major, minor))
}

pub(super) fn last_os_error() -> String {
    std::io::Error::last_os_error().to_string()
}
