use super::*;

#[test]
fn multisampled_array_selects_gles_310_only_when_available() {
    use crate::backend::profile::GlProfile;
    use naga::back::glsl::Version;
    let stage = api::ShaderStage::Fragment;
    assert_eq!(
        naga_version(
            GlProfile::Embedded { major: 3, minor: 1 },
            stage,
            false,
            false,
            true
        )
        .unwrap(),
        Version::Embedded {
            version: 310,
            is_webgl: false
        },
    );
    assert_eq!(
        naga_version(
            GlProfile::Embedded { major: 3, minor: 2 },
            stage,
            false,
            false,
            true
        )
        .unwrap(),
        Version::Embedded {
            version: 320,
            is_webgl: false
        },
    );
    assert_eq!(
        naga_version(
            GlProfile::Embedded { major: 3, minor: 0 },
            stage,
            false,
            false,
            false
        )
        .unwrap(),
        Version::Embedded {
            version: 300,
            is_webgl: false
        },
    );
    assert!(
        naga_version(
            GlProfile::Embedded { major: 3, minor: 0 },
            stage,
            false,
            false,
            true
        )
        .is_err()
    );
    assert!(naga_version(GlProfile::WebGl2, stage, false, false, true).is_err());
}
