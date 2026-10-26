//! Unit tests for the corresponding implementation module.

use super::valid_aabb_range;

#[test]
fn bounding_boxes_require_valid_native_alignment_and_complete_buffer_range() {
    assert!(valid_aabb_range(88, 32, 32, 2));
    assert!(valid_aabb_range(56, 0, 28, 2));
    assert!(!valid_aabb_range(71, 24, 24, 2));
    assert!(!valid_aabb_range(80, 8, 24, 2));
    assert!(!valid_aabb_range(u64::MAX, u64::MAX - 7, 24, u32::MAX));
    assert!(!valid_aabb_range(80, 0, 24, 0));
}
