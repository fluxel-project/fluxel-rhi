
use super::GlLimits;

#[test]
fn push_limit_uses_each_available_stage_and_shared_locations() {
    let mut limits = GlLimits::unavailable();
    limits.max_default_uniform_vectors = [512, 224, 256];
    assert_eq!(limits.push_constant_bytes(true), 896);
    limits.max_uniform_locations = Some(300);
    assert_eq!(limits.push_constant_bytes(true), 600);
    limits.max_default_uniform_vectors[2] = 0;
    assert_eq!(limits.push_constant_bytes(true), 0);
    assert_eq!(limits.push_constant_bytes(false), 600);
}
