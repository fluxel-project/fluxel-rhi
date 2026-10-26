    use super::*;
    #[test]
    fn undeclared_raw_access_is_never_trusted() {
        assert_eq!(
            StateEvent::ScopedRawAccess(ScopedRawAccess::all()).domains(),
            DirtyDomains::ALL
        );
    }
