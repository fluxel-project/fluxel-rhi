    use super::{EglGlesVersion, parse_gles_version};

    #[test]
    fn angle_patch_version_meets_requested_floor() {
        assert_eq!(
            parse_gles_version("OpenGL ES 3.0.0 (ANGLE 2.1)"),
            Some(EglGlesVersion::V3_0)
        );
        assert_eq!(
            parse_gles_version("OpenGL ES 3.2.1 driver"),
            Some(EglGlesVersion::V3_2)
        );
    }
