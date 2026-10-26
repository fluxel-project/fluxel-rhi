//! OVR_multiview2 entry point for the current native GL context.
//!
//! The provider owns the function pointer and its context lifetime. A reported
//! extension and callable entry point alone are not a rendering capability:
//! discovery must also query the view limit and the complete path must be tested.

use crate::backend::api::GlDiscoverySnapshot;
use core::ffi::c_void;

/// `MAX_VIEWS_OVR`, valid only after GL_OVR_multiview2 is reported.
pub(super) const MAX_VIEWS_OVR: u32 = 0x9631;

#[derive(Clone, Copy)]
pub(super) struct NativeMultiview {
    attach: unsafe extern "system" fn(u32, u32, u32, i32, i32, i32),
}

impl NativeMultiview {
    /// Loads only for the exact context that reports the extension.
    ///
    /// # Safety
    /// The loader must return callable symbols of the current context and that
    /// context must remain current whenever the returned function is invoked.
    pub(super) unsafe fn load(
        discovery: &GlDiscoverySnapshot,
        load: impl Fn(&str) -> *const c_void,
    ) -> Option<Self> {
        if !discovery
            .extensions()
            .raw_reported_names()
            .any(|name| name == "GL_OVR_multiview2")
        {
            return None;
        }
        // SAFETY: forwarded from this method's loader contract.
        unsafe { Self::load_entry(load) }
    }

    /// Load the entry before discovery; the caller must separately gate calls
    /// on the extension being reported by the current context.
    pub(super) unsafe fn load_entry(load: impl Fn(&str) -> *const c_void) -> Option<Self> {
        let ptr = load("glFramebufferTextureMultiviewOVR");
        if ptr.is_null() {
            return None;
        }
        // SAFETY: the current context reports the extension and the caller
        // guarantees that the non-null symbol has its registry-defined ABI.
        Some(Self {
            attach: unsafe { core::mem::transmute(ptr) },
        })
    }

    /// Attach a validated consecutive range of array layers to the bound FBO.
    ///
    /// # Safety
    /// The owning context and target FBO must be current. The texture must be
    /// a live single-sample 2D array and the entire range must exist. The caller
    /// must check the extension view limit, i32 bounds and resulting GL error.
    pub(super) unsafe fn attach(
        self,
        point: u32,
        texture: glow::NativeTexture,
        mip: i32,
        base: i32,
        count: i32,
    ) {
        // SAFETY: caller establishes the context, attachment and range invariants.
        unsafe { (self.attach)(glow::FRAMEBUFFER, point, texture.0.get(), mip, base, count) };
    }
}

/// Exercises a two-layer attachment and framebuffer completeness on the current
/// context, restoring every binding it changes before deleting scratch objects.
/// The caller must have established the extension is reported.
pub(super) unsafe fn probe(
    gl: &glow::Context,
    entry: NativeMultiview,
) -> super::probes::ProbeAnswer {
    use super::probes::ProbeAnswer;
    use glow::HasContext as _;
    use std::num::NonZeroU32;
    // SAFETY: discovery owns this current context throughout the operation.
    unsafe {
        let read = NonZeroU32::new(gl.get_parameter_i32(glow::READ_FRAMEBUFFER_BINDING) as u32)
            .map(glow::NativeFramebuffer);
        let draw = NonZeroU32::new(gl.get_parameter_i32(glow::DRAW_FRAMEBUFFER_BINDING) as u32)
            .map(glow::NativeFramebuffer);
        let binding = NonZeroU32::new(gl.get_parameter_i32(glow::TEXTURE_BINDING_2D_ARRAY) as u32)
            .map(glow::NativeTexture);
        let texture = match gl.create_texture() {
            Ok(x) => x,
            Err(_) => {
                return if gl.get_error() == glow::CONTEXT_LOST {
                    ProbeAnswer::ContextLost
                } else {
                    ProbeAnswer::Unavailable
                };
            }
        };
        let framebuffer = match gl.create_framebuffer() {
            Ok(x) => x,
            Err(_) => {
                let error = gl.get_error();
                gl.delete_texture(texture);
                let cleanup = gl.get_error();
                return if error == glow::CONTEXT_LOST || cleanup == glow::CONTEXT_LOST {
                    ProbeAnswer::ContextLost
                } else {
                    ProbeAnswer::Unavailable
                };
            }
        };
        gl.bind_texture(glow::TEXTURE_2D_ARRAY, Some(texture));
        gl.tex_storage_3d(glow::TEXTURE_2D_ARRAY, 1, glow::RGBA8, 2, 2, 2);
        gl.bind_framebuffer(glow::FRAMEBUFFER, Some(framebuffer));
        entry.attach(glow::COLOR_ATTACHMENT0, texture, 0, 0, 2);
        let complete = gl.check_framebuffer_status(glow::FRAMEBUFFER) == glow::FRAMEBUFFER_COMPLETE;
        let operation_error = gl.get_error();
        gl.bind_framebuffer(glow::READ_FRAMEBUFFER, read);
        gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, draw);
        gl.bind_texture(glow::TEXTURE_2D_ARRAY, binding);
        gl.delete_framebuffer(framebuffer);
        gl.delete_texture(texture);
        let cleanup_error = gl.get_error();
        if operation_error == glow::CONTEXT_LOST || cleanup_error == glow::CONTEXT_LOST {
            ProbeAnswer::ContextLost
        } else if complete && operation_error == glow::NO_ERROR && cleanup_error == glow::NO_ERROR {
            ProbeAnswer::Passed
        } else {
            ProbeAnswer::Failed
        }
    }
}
