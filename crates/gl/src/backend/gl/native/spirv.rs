//! Desktop `GL_ARB_gl_spirv` / GL 4.6 entry points.
//!
//! The provider loads both commands from its exact WGL/EGL context.  The
//! extension string alone is never sufficient: a missing pointer closes this
//! route before any binary reaches the driver.

use core::ffi::{c_char, c_void};

use crate::backend::gl::api::{GlDiscoverySnapshot, GlFamilyProfile};

pub(super) const GL_SHADER_BINARY_FORMAT_SPIR_V: u32 = 0x9551;
pub(super) type ShaderBinaryFn =
    unsafe extern "system" fn(i32, *const u32, u32, *const c_void, i32);
pub(super) type SpecializeShaderFn =
    unsafe extern "system" fn(u32, *const c_char, u32, *const u32, *const u32);

#[derive(Clone, Copy)]
pub(crate) struct NativeSpirvFns {
    pub(super) shader_binary: ShaderBinaryFn,
    pub(super) specialize_shader: SpecializeShaderFn,
}

impl NativeSpirvFns {
    /// # Safety
    /// The loader must resolve symbols for `discovery`'s current context.
    pub(crate) unsafe fn load(
        discovery: &GlDiscoverySnapshot,
        load: &impl Fn(&str) -> *const c_void,
    ) -> Option<Self> {
        let core = matches!(discovery.context().profile(), GlFamilyProfile::Desktop { major: 4, minor } if minor >= 6);
        let extension = discovery
            .extensions()
            .raw_reported_names()
            .any(|name| name == "GL_ARB_gl_spirv");
        if !core && !extension {
            return None;
        }
        let binary = load("glShaderBinary");
        // The ARB registry promises the suffixed spelling; it need not alias
        // GL 4.6's unsuffixed command on an extension-only driver.
        let specialize = {
            let core_name = load("glSpecializeShader");
            if core_name.is_null() && extension {
                load("glSpecializeShaderARB")
            } else {
                core_name
            }
        };
        if binary.is_null() || specialize.is_null() {
            return None;
        }
        // SAFETY: the extension/core registry fixes both signatures, and the
        // loader is tied to the current provider context.
        Some(unsafe {
            Self {
                shader_binary: core::mem::transmute(binary),
                specialize_shader: core::mem::transmute(specialize),
            }
        })
    }
}
