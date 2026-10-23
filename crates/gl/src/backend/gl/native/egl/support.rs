//! EGL teardown and construction helpers.
//!
//! These helpers are called only by the thread-affine `EglGlesContext`; they
//! create or release EGL objects but never dispatch work to another thread.

use super::*;

pub(super) fn acquire_display(
    native_display: *mut c_void,
) -> Result<Rc<EglDisplayLease>, EglProviderError> {
    // SAFETY: khronos-egl validates required symbol loading; the system EGL
    // library is trusted by the OS loader boundary.
    #[cfg(windows)]
    let egl = unsafe { Egl::load_required_from_filename("libEGL.dll") };
    #[cfg(not(windows))]
    let egl = unsafe { Egl::load_required() };
    let egl = egl.map_err(|error| EglProviderError::Load(error.to_string()))?;
    // SAFETY: the caller has checked that this native display belongs to the
    // selected raw-handle platform and stays live for the context lease.
    let display =
        unsafe { egl.get_display(native_display) }.ok_or(EglProviderError::DisplayUnavailable)?;
    let key = display.as_ptr() as usize;
    if let Some(existing) = DISPLAY_LEASES.with(|leases| {
        let mut leases = leases.borrow_mut();
        let existing = leases.get(&key).and_then(Weak::upgrade);
        if existing.is_none() {
            leases.remove(&key);
        }
        existing
    }) {
        return Ok(existing);
    }
    egl.initialize(display)
        .map_err(|error| EglProviderError::Egl {
            operation: "eglInitialize",
            error,
        })?;
    let lease = Rc::new(EglDisplayLease { egl, display });
    DISPLAY_LEASES.with(|leases| {
        leases.borrow_mut().insert(key, Rc::downgrade(&lease));
    });
    Ok(lease)
}

impl Drop for EglGlesContext {
    fn drop(&mut self) {
        if self.lifecycle != GlContextLifecycle::Disposed
            && OwnerThreadIdentity::current() == self.owner
        {
            let _ = self.dispose();
        }
    }
}

pub(super) fn native_window_pair(
    display: RawDisplayHandle,
    window: RawWindowHandle,
) -> Result<(*mut c_void, *mut c_void), EglProviderError> {
    match (display, window) {
        // Android's raw display intentionally carries no native pointer. EGL's
        // documented route is EGL_DEFAULT_DISPLAY plus the ANativeWindow used
        // by eglCreateWindowSurface; do not invent an EGL platform extension
        // call for it.
        (RawDisplayHandle::Android(_), RawWindowHandle::AndroidNdk(window)) => {
            Ok((ptr::null_mut(), window.a_native_window.as_ptr()))
        }
        (RawDisplayHandle::Xlib(display), RawWindowHandle::Xlib(window)) => {
            let native_display = display
                .display
                .ok_or(EglProviderError::InvalidNativeHandle(
                    "Xlib display pointer",
                ))?;
            if window.window == 0 {
                return Err(EglProviderError::InvalidNativeHandle("Xlib window"));
            }
            Ok((
                native_display.as_ptr(),
                window.window as usize as *mut c_void,
            ))
        }
        (RawDisplayHandle::Wayland(display), RawWindowHandle::Wayland(window)) => {
            Ok((display.display.as_ptr(), window.surface.as_ptr()))
        }
        (display, window) => Err(EglProviderError::UnsupportedHandlePair {
            display: display_name(display),
            window: window_name(window),
        }),
    }
}

pub(super) fn create_surface(
    egl: &Egl,
    display: khronos_egl::Display,
    config: khronos_egl::Config,
    kind: EglSurfaceKind,
) -> Result<khronos_egl::Surface, EglProviderError> {
    match kind {
        EglSurfaceKind::Window { native_window } => {
            // SAFETY: native_window is paired with display by new_window's
            // explicit unsafe contract and stays Host-owned/live.
            unsafe { egl.create_window_surface(display, config, native_window, None) }.map_err(
                |error| EglProviderError::Egl {
                    operation: "eglCreateWindowSurface",
                    error,
                },
            )
        }
        EglSurfaceKind::Pbuffer(size) => {
            let attributes = pbuffer_attributes(size)?;
            egl.create_pbuffer_surface(display, config, &attributes)
                .map_err(|error| EglProviderError::Egl {
                    operation: "eglCreatePbufferSurface",
                    error,
                })
        }
    }
}

pub(super) fn exact_context_attributes(
    version: EglGlesVersion,
    has_create_context: bool,
) -> Vec<i32> {
    match version {
        // EGL 1.0's ES client-version attribute is enough for the whole 3.0
        // profile.  Do not pass KHR attributes unless their extension was
        // checked, because several otherwise-valid EGL 1.4 implementations
        // reject unknown attributes.
        EglGlesVersion::V3_0 if has_create_context => vec![
            EGL_CONTEXT_MAJOR_VERSION_KHR,
            3,
            EGL_CONTEXT_MINOR_VERSION_KHR,
            0,
            khronos_egl::NONE,
        ],
        EglGlesVersion::V3_0 => {
            vec![khronos_egl::CONTEXT_CLIENT_VERSION, 3, khronos_egl::NONE]
        }
        EglGlesVersion::V3_1 | EglGlesVersion::V3_2 => vec![
            EGL_CONTEXT_MAJOR_VERSION_KHR,
            3,
            EGL_CONTEXT_MINOR_VERSION_KHR,
            version.minor(),
            khronos_egl::NONE,
        ],
    }
}

pub(super) fn parse_gles_version(value: &str) -> Option<EglGlesVersion> {
    let profile = super::super::probe::parse_native_profile(value).ok()?;
    match profile {
        GlFamilyProfile::Embedded { major, minor } if major > 3 || minor >= 2 => {
            Some(EglGlesVersion::V3_2)
        }
        GlFamilyProfile::Embedded { major: 3, minor: 1 } => Some(EglGlesVersion::V3_1),
        GlFamilyProfile::Embedded { major: 3, minor: 0 } => Some(EglGlesVersion::V3_0),
        _ => None,
    }
}

pub(super) fn pbuffer_attributes(size: EglPbufferSize) -> Result<[i32; 5], EglProviderError> {
    let (width, height) = size.checked()?;
    Ok([
        khronos_egl::WIDTH,
        width,
        khronos_egl::HEIGHT,
        height,
        khronos_egl::NONE,
    ])
}

pub(super) fn has_extension(extensions: &str, extension: &str) -> bool {
    extensions
        .split_ascii_whitespace()
        .any(|item| item == extension)
}

/// The EGL display's own vendor and version strings.
///
/// The driver identity is the one fact discovery cannot read from GL: on this
/// family it belongs to the platform layer that owns the display, and EGL is
/// the only layer that publishes it (audit P2-6).  The GL vendor/renderer pair
/// is already recorded as its own field, and the GL version string is recorded
/// as `version`, so repeating either here would reproduce exactly the defect
/// this wiring closes; what is read is the EGL implementation's own two
/// strings, queried from the display this provider initialized.
///
/// A failed query or a blank answer yields an empty string, which discovery
/// records as an unavailable identity instead of as a driver nobody observed.
pub(super) fn egl_driver_identity(egl: &Egl, display: khronos_egl::Display) -> String {
    let read = |name: khronos_egl::Int| {
        egl.query_string(Some(display), name)
            .map(|value| value.to_string_lossy().into_owned())
            .unwrap_or_default()
    };
    compose_egl_driver_identity(&read(khronos_egl::VENDOR), &read(khronos_egl::VERSION))
}

/// Composes the platform identity from the EGL implementation's two strings.
///
/// `EGL` names the layer the version belongs to, so the recorded identity
/// cannot read as a GL version even when the numbers coincide.
fn compose_egl_driver_identity(vendor: &str, version: &str) -> String {
    match (vendor.trim(), version.trim()) {
        ("", "") => String::new(),
        ("", version) => version.to_owned(),
        (vendor, "") => vendor.to_owned(),
        (vendor, version) => format!("{vendor} EGL {version}"),
    }
}

/// Whether an observed surface extent can be a viewport on this context.
///
/// The extent is a real EGL observation (`eglQuerySurface` for the live
/// surface), and the maximum viewport dimensions that bound it are a recorded
/// discovery fact; recording a size above them is what would otherwise surface
/// as an unrelated driver error in the first pass that draws into the surface.
///
/// A zero-area extent is not a rejection: it is how a minimized or
/// not-yet-shown surface reports itself and becomes suspension instead.  A
/// limit of zero means the context reported none, so nothing can be measured
/// against it and the check cannot fail closed on a fact that was never read.
fn within_viewport_limit(limit: [u32; 2], extent: [u32; 2]) -> bool {
    if extent.contains(&0) || limit.contains(&0) {
        return true;
    }
    extent[0] <= limit[0] && extent[1] <= limit[1]
}

/// Resolves an observed surface extent against the context's viewport limit.
pub(super) fn check_drawable_extent(
    limit: [u32; 2],
    extent: [u32; 2],
) -> Result<(), EglProviderError> {
    if within_viewport_limit(limit, extent) {
        return Ok(());
    }
    Err(EglProviderError::DrawableExtentExceedsViewportLimit {
        requested: extent,
        limit,
    })
}

fn display_name(handle: RawDisplayHandle) -> &'static str {
    match handle {
        RawDisplayHandle::Xlib(_) => "Xlib",
        RawDisplayHandle::Xcb(_) => "Xcb",
        RawDisplayHandle::Wayland(_) => "Wayland",
        RawDisplayHandle::Drm(_) => "DRM",
        RawDisplayHandle::Gbm(_) => "GBM",
        RawDisplayHandle::Windows(_) => "Windows",
        RawDisplayHandle::Android(_) => "Android",
        _ => "other",
    }
}

fn window_name(handle: RawWindowHandle) -> &'static str {
    match handle {
        RawWindowHandle::Xlib(_) => "Xlib",
        RawWindowHandle::Xcb(_) => "Xcb",
        RawWindowHandle::Wayland(_) => "Wayland",
        RawWindowHandle::Drm(_) => "DRM",
        RawWindowHandle::Gbm(_) => "GBM",
        RawWindowHandle::Win32(_) => "Win32",
        RawWindowHandle::AndroidNdk(_) => "AndroidNdk",
        _ => "other",
    }
}
#[cfg(test)]
mod version_tests {
    use super::{EglGlesVersion, parse_gles_version};

    #[test]
    fn angle_patch_version_meets_requested_floor() {
        assert_eq!(
            parse_gles_version("OpenGL ES 3.0.0 (ANGLE 2.1)"),
            Some(EglGlesVersion::V3_0)
        );
        assert_eq!(
            parse_gles_version("OpenGL ES 3.2.1 driver"),
            Some(EglGlesVersion::V3_2)
        );
    }
}
