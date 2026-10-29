//! DX12 backend feature admission and address-space limit regressions.

use super::*;
#[test]
fn validates_ray_query_and_known_legacy_feature() {
    let ray = vec![api::SHADER_FEATURE_RAY_QUERY.to_owned()];
    assert!(!shader_features_supported(&ray, false, false));
    assert!(shader_features_supported(&ray, true, false));
    assert!(shader_features_supported(
        &["shaderClipDistance".into()],
        false,
        false,
    ));
    assert!(!shader_features_supported(
        &["subgroups".into()],
        false,
        false
    ));
    assert!(shader_features_supported(
        &["subgroups".into()],
        false,
        true
    ));
    assert!(!shader_features_supported(
        &["unknown-feature".into()],
        true,
        true,
    ));
}

#[test]
fn committed_buffer_limit_uses_gpu_address_bits() {
    assert_eq!(
        max_committed_buffer_bytes(32).unwrap(),
        (1u64 << 32) - 65536
    );
    assert_eq!(
        max_committed_buffer_bytes(40).unwrap(),
        (1u64 << 40) - 65536
    );
    assert!(max_committed_buffer_bytes(64).is_err());
}
