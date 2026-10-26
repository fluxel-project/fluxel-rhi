    use super::{CurrentContextGuard, NativeContextKind};

    #[test]
    fn currentness_guard_preserves_platform_provenance() {
        // SAFETY: this test does not issue GL; it tests only thread affinity.
        let guard = unsafe { CurrentContextGuard::assume_current(NativeContextKind::Embedded) };
        assert_eq!(guard.kind(), NativeContextKind::Embedded);
        assert!(guard.assert_owner("test").is_ok());
    }
