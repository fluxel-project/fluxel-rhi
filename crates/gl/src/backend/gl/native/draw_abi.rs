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
mod tests {
    use super::NativeDrawFunctions;
    use core::ffi::c_void;

    #[cfg(target_pointer_width = "64")]
    use core::sync::atomic::{AtomicI64, AtomicU64, Ordering};

    #[cfg(target_pointer_width = "64")]
    static POINTER_OFFSET: AtomicU64 = AtomicU64::new(0);
    #[cfg(target_pointer_width = "64")]
    static INTEGER_OFFSET: AtomicI64 = AtomicI64::new(0);

    #[cfg(target_pointer_width = "64")]
    unsafe extern "system" fn capture_elements(_: u32, _: i32, _: u32, pointer: *const c_void) {
        POINTER_OFFSET.store(pointer as usize as u64, Ordering::Relaxed);
    }
    #[cfg(target_pointer_width = "64")]
    unsafe extern "system" fn capture_elements_instanced(
        _: u32,
        _: i32,
        _: u32,
        pointer: *const c_void,
        _: i32,
    ) {
        POINTER_OFFSET.store(pointer as usize as u64, Ordering::Relaxed);
    }
    #[cfg(target_pointer_width = "64")]
    unsafe extern "system" fn capture_arrays_indirect(_: u32, pointer: *const c_void) {
        POINTER_OFFSET.store(pointer as usize as u64, Ordering::Relaxed);
    }
    #[cfg(target_pointer_width = "64")]
    unsafe extern "system" fn capture_elements_indirect(_: u32, _: u32, pointer: *const c_void) {
        POINTER_OFFSET.store(pointer as usize as u64, Ordering::Relaxed);
    }
    #[cfg(target_pointer_width = "64")]
    unsafe extern "system" fn capture_dispatch_indirect(offset: isize) {
        INTEGER_OFFSET.store(offset as i64, Ordering::Relaxed);
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn raw_draw_abi_captures_offsets_above_four_gib() {
        let functions = NativeDrawFunctions {
            elements: Some(capture_elements),
            elements_instanced: Some(capture_elements_instanced),
            arrays_indirect: Some(capture_arrays_indirect),
            elements_indirect: Some(capture_elements_indirect),
            dispatch_indirect: Some(capture_dispatch_indirect),
        };
        let offset = 4_u64 * 1024 * 1024 * 1024 + 71;
        let pointer = offset as usize as *const c_void;
        // SAFETY: each target is a local capture mock.
        unsafe {
            (functions.elements.expect("test mock"))(0, 1, 0, pointer);
            assert_eq!(POINTER_OFFSET.load(Ordering::Relaxed), offset);
            (functions.elements_instanced.expect("test mock"))(0, 1, 0, pointer, 1);
            assert_eq!(POINTER_OFFSET.load(Ordering::Relaxed), offset);
            (functions.arrays_indirect.expect("test mock"))(0, pointer);
            assert_eq!(POINTER_OFFSET.load(Ordering::Relaxed), offset);
            (functions.elements_indirect.expect("test mock"))(0, 0, pointer);
            assert_eq!(POINTER_OFFSET.load(Ordering::Relaxed), offset);
            (functions.dispatch_indirect.expect("test mock"))(offset as isize);
        }
        assert_eq!(INTEGER_OFFSET.load(Ordering::Relaxed), offset as i64);
    }
}
