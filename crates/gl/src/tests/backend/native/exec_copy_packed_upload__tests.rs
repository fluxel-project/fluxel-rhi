use super::pbo_range_fits;
#[test]
fn pbo_bounds_are_validated_before_any_upload() {
    assert!(pbo_range_fits(16, i32::MAX as u64 - 16));
    assert!(!pbo_range_fits(16, i32::MAX as u64 - 15));
    assert!(!pbo_range_fits(u64::MAX, 1));
}
