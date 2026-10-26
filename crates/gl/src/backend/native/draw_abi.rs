//! Pointer-width ABI for native draw commands.
//!
//! `glow` models several byte offsets as `i32` for portability. Native GL
//! instead takes either `const void *` (element and indirect commands) or a
//! `GLintptr` (indirect dispatch). Keep these functions at the native boundary
//! so the public GL-family contract stays platform-neutral.

use core::ffi::c_void;

#[derive(Clone, Copy, Default)]
pub(crate) struct NativeDrawFunctions {
    pub(crate) elements: Option<unsafe extern "system" fn(u32, i32, u32, *const c_void)>,
    pub(crate) elements_instanced:
        Option<unsafe extern "system" fn(u32, i32, u32, *const c_void, i32)>,
    pub(crate) arrays_indirect: Option<unsafe extern "system" fn(u32, *const c_void)>,
    pub(crate) elements_indirect: Option<unsafe extern "system" fn(u32, u32, *const c_void)>,
    pub(crate) dispatch_indirect: Option<unsafe extern "system" fn(isize)>,
}

impl NativeDrawFunctions {
    /// # Safety
    ///
    /// The loader resolves names for the provider's current WGL/EGL context.
    pub(crate) unsafe fn load(mut load: impl FnMut(&str) -> *const c_void) -> Self {
        macro_rules! optional {
            ($name:literal, $ty:ty) => {{
                let pointer = load($name);
                (!pointer.is_null())
                    .then(|| unsafe { core::mem::transmute::<*const c_void, $ty>(pointer) })
            }};
        }
        Self {
            elements: optional!(
                "glDrawElements",
                unsafe extern "system" fn(u32, i32, u32, *const c_void)
            ),
            elements_instanced: optional!(
                "glDrawElementsInstanced",
                unsafe extern "system" fn(u32, i32, u32, *const c_void, i32)
            ),
            arrays_indirect: optional!(
                "glDrawArraysIndirect",
                unsafe extern "system" fn(u32, *const c_void)
            ),
            elements_indirect: optional!(
                "glDrawElementsIndirect",
                unsafe extern "system" fn(u32, u32, *const c_void)
            ),
            dispatch_indirect: optional!(
                "glDispatchComputeIndirect",
                unsafe extern "system" fn(isize)
            ),
        }
    }
}

#[cfg(test)]
#[path = "../../tests/backend/native/draw_abi__tests.rs"]
mod tests;
