//! Optional GLES indexed draw-buffer entry points.
//!
//! A complete callable symbol group is retained per extension spelling. The
//! provider separately verifies the corresponding advertised extension name.
use core::ffi::c_void;

#[derive(Clone, Copy)]
pub(crate) struct IndexedBlendFns {
    pub enable: unsafe extern "system" fn(u32, u32),
    pub disable: unsafe extern "system" fn(u32, u32),
    pub mask: unsafe extern "system" fn(u32, u8, u8, u8, u8),
    pub func: unsafe extern "system" fn(u32, u32, u32, u32, u32),
    pub equation: unsafe extern "system" fn(u32, u32, u32),
}

#[derive(Clone, Copy, Default)]
pub(crate) struct IndexedBlendExtensions {
    pub oes: Option<IndexedBlendFns>,
    pub ext: Option<IndexedBlendFns>,
}

impl IndexedBlendFns {
    /// The loader must resolve callable symbols for the current EGL context.
    pub(super) unsafe fn load(
        mut load: impl FnMut(&str) -> *const c_void,
        suffix: &str,
    ) -> Option<Self> {
        let pointers = [
            "glEnablei",
            "glDisablei",
            "glColorMaski",
            "glBlendFuncSeparatei",
            "glBlendEquationSeparatei",
        ]
        .map(|name| load(&format!("{name}{suffix}")));
        if pointers.iter().any(|p| p.is_null()) {
            return None;
        }
        // SAFETY: each non-null address came from the current context loader,
        // and the signatures are the GL extension ABI for the named entry points.
        Some(unsafe {
            Self {
                enable: core::mem::transmute(pointers[0]),
                disable: core::mem::transmute(pointers[1]),
                mask: core::mem::transmute(pointers[2]),
                func: core::mem::transmute(pointers[3]),
                equation: core::mem::transmute(pointers[4]),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::IndexedBlendFns;

    #[test]
    fn incomplete_symbol_group_is_not_admitted() {
        // SAFETY: no pointers are called by this test.
        let result = unsafe {
            IndexedBlendFns::load(
                |name| {
                    if name == "glColorMaskiOES" {
                        core::ptr::null()
                    } else {
                        1usize as *const core::ffi::c_void
                    }
                },
                "OES",
            )
        };
        assert!(result.is_none());
    }
}
