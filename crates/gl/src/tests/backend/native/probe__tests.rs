    use super::{NativeProfileError, parse_native_profile};
    use crate::backend::api::GlFamilyProfile;

    #[test]
    fn keeps_newer_versions_instead_of_downgrading_to_floor() {
        assert_eq!(
            parse_native_profile("4.6.0 NVIDIA 555"),
            Ok(GlFamilyProfile::Desktop { major: 4, minor: 6 })
        );
        assert_eq!(
            parse_native_profile("OpenGL ES 3.2 V@1"),
            Ok(GlFamilyProfile::Embedded { major: 3, minor: 2 })
        );
    }

    #[test]
    fn rejects_older_native_profiles_before_any_command_path() {
        assert_eq!(
            parse_native_profile("3.3"),
            Err(NativeProfileError::BelowDesktopFloor { major: 3, minor: 3 })
        );
        assert_eq!(
            parse_native_profile("OpenGL ES 2.0"),
            Err(NativeProfileError::BelowEmbeddedFloor { major: 2, minor: 0 })
        );
    }

    #[test]
    fn malformed_es_prefix_does_not_become_desktop_gl() {
        assert!(matches!(
            parse_native_profile("OpenGL ES-CM 1.1"),
            Err(NativeProfileError::Malformed(_))
        ));
    }
