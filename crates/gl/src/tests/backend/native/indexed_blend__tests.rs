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
