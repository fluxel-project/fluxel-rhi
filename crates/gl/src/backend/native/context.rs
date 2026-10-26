//! Thread-affine native context carrier.

use crate::backend::api::{GlContextLifecycle, GlError, OwnerThreadIdentity};
use core::marker::PhantomData;
use std::rc::Rc;

/// Native `glUnmapBuffer` signature used where `glow` does not expose its
/// corruption status return value.
pub(crate) type UnmapBufferFn = unsafe extern "system" fn(u32) -> u8;
pub(crate) type GetProgramResourceFn =
    unsafe extern "system" fn(u32, u32, u32, i32, *const u32, i32, *mut i32, *mut i32);

/// Which platform family owns the current native context.
///
/// This is diagnostic provenance only.  It deliberately does not carry a WGL
/// or EGL handle: those handles have platform-specific destruction/currentness
/// rules and must remain in the platform adapter.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum NativeContextKind {
    /// A desktop context adopted from WGL/GLX/CGL or an equivalent host.
    Desktop,
    /// An ES context adopted from EGL or an equivalent host.
    Embedded,
}

/// Platform-owned native context used by the immediate GL backend.
///
/// The enum deliberately owns a concrete WGL or EGL value instead of hiding it
/// behind a `Send + Sync` driver object.  A GL call is valid only after the
/// owning calling thread has made this exact context current; callers perform
/// that transition and issue the native call in one synchronous operation.
/// No command is queued, replayed, or forwarded to an owner thread.
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
pub(crate) enum NativePlatformContext {
    #[cfg(all(windows, feature = "native-gl-wgl"))]
    Wgl(super::wgl::WglContextSurface),
    #[cfg(feature = "native-gles-egl")]
    Egl(super::egl::EglGlesContext),
}

/// Private contract shared by concrete WGL and EGL owners.
///
/// Implementors remain `!Send + !Sync`; this trait exists solely so the
/// concrete enum can forward immediate calls without exposing raw platform
/// handles beyond `native`.
pub(crate) trait NativeContextOwner {
    fn kind(&self) -> NativeContextKind;
    fn context_stamp(&self) -> crate::backend::api::ContextStamp;
    fn owner_thread(&self) -> OwnerThreadIdentity;
    fn make_current(&mut self, operation: &'static str) -> Result<(), GlError>;
    fn load_glow(&mut self) -> Result<glow::Context, GlError>;
    fn load_unmap_buffer(&self) -> Result<UnmapBufferFn, GlError>;
    fn gl_proc(&self, name: &str) -> *const core::ffi::c_void;
    fn load_program_resource(&self) -> Option<GetProgramResourceFn> {
        None
    }
    fn load_indexed_blend(&self) -> super::indexed_blend::IndexedBlendExtensions {
        Default::default()
    }
    fn swap_buffers(&mut self) -> Result<(), GlError>;
    /// Records or recreates the drawable at the Host-confirmed extent.
    fn resize_drawable(&mut self, extent: [u32; 2]) -> Result<(), GlError>;
    fn drawable_extent(&self) -> Option<[u32; 2]>;
    fn supports_presentation(&self) -> bool;
}

#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
impl NativeContextOwner for NativePlatformContext {
    fn kind(&self) -> NativeContextKind {
        match self {
            #[cfg(all(windows, feature = "native-gl-wgl"))]
            Self::Wgl(context) => context.kind(),
            #[cfg(feature = "native-gles-egl")]
            Self::Egl(context) => context.kind(),
        }
    }

    fn context_stamp(&self) -> crate::backend::api::ContextStamp {
        match self {
            #[cfg(all(windows, feature = "native-gl-wgl"))]
            Self::Wgl(context) => context.context_stamp(),
            #[cfg(feature = "native-gles-egl")]
            Self::Egl(context) => context.context_stamp(),
        }
    }

    fn owner_thread(&self) -> OwnerThreadIdentity {
        match self {
            #[cfg(all(windows, feature = "native-gl-wgl"))]
            Self::Wgl(context) => context.owner_thread(),
            #[cfg(feature = "native-gles-egl")]
            Self::Egl(context) => context.owner_thread(),
        }
    }

    fn make_current(&mut self, operation: &'static str) -> Result<(), GlError> {
        match self {
            #[cfg(all(windows, feature = "native-gl-wgl"))]
            Self::Wgl(context) => context.make_current(operation),
            #[cfg(feature = "native-gles-egl")]
            Self::Egl(context) => context.make_current_for(operation),
        }
    }

    fn load_glow(&mut self) -> Result<glow::Context, GlError> {
        match self {
            #[cfg(all(windows, feature = "native-gl-wgl"))]
            Self::Wgl(context) => context.load_glow(),
            #[cfg(feature = "native-gles-egl")]
            Self::Egl(context) => context.load_glow().map_err(|error| error.into_gl_error()),
        }
    }

    fn load_unmap_buffer(&self) -> Result<UnmapBufferFn, GlError> {
        match self {
            #[cfg(all(windows, feature = "native-gl-wgl"))]
            Self::Wgl(context) => context.load_unmap_buffer(),
            #[cfg(feature = "native-gles-egl")]
            Self::Egl(context) => context.load_unmap_buffer(),
        }
    }
    fn load_program_resource(&self) -> Option<GetProgramResourceFn> {
        match self {
            #[cfg(all(windows, feature = "native-gl-wgl"))]
            Self::Wgl(context) => {
                let ptr = NativeContextOwner::gl_proc(context, "glGetProgramResourceiv");
                if ptr.is_null() {
                    None
                } else {
                    // SAFETY: the platform loader resolves this exact GL entry point.
                    Some(unsafe {
                        core::mem::transmute::<*const core::ffi::c_void, GetProgramResourceFn>(ptr)
                    })
                }
            }
            #[cfg(feature = "native-gles-egl")]
            Self::Egl(context) => context.load_program_resource(),
        }
    }

    fn gl_proc(&self, name: &str) -> *const core::ffi::c_void {
        match self {
            #[cfg(all(windows, feature = "native-gl-wgl"))]
            Self::Wgl(context) => NativeContextOwner::gl_proc(context, name),
            #[cfg(feature = "native-gles-egl")]
            Self::Egl(context) => context.gl_proc(name),
        }
    }

    fn load_indexed_blend(&self) -> super::indexed_blend::IndexedBlendExtensions {
        match self {
            #[cfg(all(windows, feature = "native-gl-wgl"))]
            Self::Wgl(_) => Default::default(),
            #[cfg(feature = "native-gles-egl")]
            Self::Egl(context) => context.load_indexed_blend(),
        }
    }

    fn swap_buffers(&mut self) -> Result<(), GlError> {
        match self {
            #[cfg(all(windows, feature = "native-gl-wgl"))]
            Self::Wgl(context) => context.present(),
            #[cfg(feature = "native-gles-egl")]
            Self::Egl(context) => context.swap_buffers(),
        }
    }

    fn resize_drawable(&mut self, extent: [u32; 2]) -> Result<(), GlError> {
        match self {
            #[cfg(all(windows, feature = "native-gl-wgl"))]
            Self::Wgl(context) => context.resize(extent),
            #[cfg(feature = "native-gles-egl")]
            Self::Egl(context) => context.resize_drawable(extent),
        }
    }

    fn drawable_extent(&self) -> Option<[u32; 2]> {
        match self {
            #[cfg(all(windows, feature = "native-gl-wgl"))]
            Self::Wgl(context) => context.client_extent(),
            #[cfg(feature = "native-gles-egl")]
            Self::Egl(context) => context.drawable_extent(),
        }
    }

    fn supports_presentation(&self) -> bool {
        match self {
            #[cfg(all(windows, feature = "native-gl-wgl"))]
            Self::Wgl(_) => true,
            #[cfg(feature = "native-gles-egl")]
            Self::Egl(context) => context.is_window_surface(),
        }
    }
}

/// Proof, supplied by the platform adapter, that its native context is current
/// on the calling thread for the duration of the GL executor's use.
///
/// The proof is `!Send` and can only be created through an unsafe constructor:
/// Rust cannot inspect WGL/EGL currentness, and making an ordinary constructor
/// would turn an external platform invariant into a guess.  The host keeps
/// ownership of adopted contexts; an owned context is represented by the same
/// guard after its platform owner has made it current.
#[derive(Debug)]
pub(crate) struct CurrentContextGuard {
    owner: OwnerThreadIdentity,
    kind: NativeContextKind,
    _thread_affine: PhantomData<Rc<()>>,
}

impl CurrentContextGuard {
    /// # Safety
    ///
    /// `kind`'s native context must be current on this thread, exclusively
    /// usable by Fluxel until the guard is dropped, and remain alive longer
    /// than every `NativeContext` created from this guard.
    pub(crate) unsafe fn assume_current(kind: NativeContextKind) -> Self {
        Self {
            owner: OwnerThreadIdentity::current(),
            kind,
            _thread_affine: PhantomData,
        }
    }

    pub(crate) const fn kind(&self) -> NativeContextKind {
        self.kind
    }

    fn assert_owner(&self, operation: &'static str) -> Result<(), GlError> {
        let actual = OwnerThreadIdentity::current();
        if actual == self.owner {
            Ok(())
        } else {
            Err(GlError::WrongThread {
                operation,
                expected: self.owner,
                actual,
            })
        }
    }
}

/// A `glow` dispatch table plus the non-transferable current-context proof.
///
/// This does not call `wglMakeCurrent` or `eglMakeCurrent`.  Platform adapters
/// own that transition and must construct this object only while current.  The
/// explicit guard prevents a native GL object from accidentally gaining the
/// browser/WebGL object's cross-thread shape.
#[cfg(any(feature = "native-gl-wgl", feature = "native-gles-egl"))]
pub(crate) struct NativeContext<'a> {
    gl: &'a glow::Context,
    guard: CurrentContextGuard,
    lifecycle: GlContextLifecycle,
}

#[cfg(any(feature = "native-gl-wgl", feature = "native-gles-egl"))]
impl<'a> NativeContext<'a> {
    /// Adopts a host-owned or platform-owned current context.
    ///
    /// The context stays borrowed: Fluxel cannot destroy a WGL/EGL context it
    /// did not create.  A future platform owner may retain its own handle and
    /// use this same adopted carrier after making it current.
    pub(crate) fn adopt_current(gl: &'a glow::Context, guard: CurrentContextGuard) -> Self {
        Self {
            gl,
            guard,
            lifecycle: GlContextLifecycle::Active,
        }
    }

    pub(crate) fn glow(&self, operation: &'static str) -> Result<&glow::Context, GlError> {
        self.assert_ready(operation)?;
        Ok(self.gl)
    }

    pub(crate) const fn kind(&self) -> NativeContextKind {
        self.guard.kind()
    }

    pub(crate) const fn lifecycle(&self) -> GlContextLifecycle {
        self.lifecycle
    }

    /// Records terminal loss observed by a native call.  It is intentionally
    /// explicit: GL's error flag alone cannot distinguish an ordinary command
    /// error from a platform-declared context reset.
    pub(crate) fn mark_lost(&mut self) {
        self.lifecycle = GlContextLifecycle::Lost;
    }

    pub(crate) fn assert_ready(&self, operation: &'static str) -> Result<(), GlError> {
        self.guard.assert_owner(operation)?;
        if self.lifecycle.accepts_commands() {
            Ok(())
        } else {
            Err(self.lifecycle.refusal(operation))
        }
    }
}

#[cfg(test)]
#[path = "../../tests/backend/native/context__tests.rs"]
mod tests;
