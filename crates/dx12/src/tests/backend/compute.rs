use super::*;

#[test]
fn dynamic_address_validates_alignment_range_and_overflow() {
    assert_eq!(
        dynamic_address(4096, 256, 512, 1024, 256, 256).unwrap(),
        4608
    );
    assert!(dynamic_address(4096, 256, 512, 1024, 1, 256).is_err());
    assert!(dynamic_address(4096, 256, 512, 512, 256, 256).is_err());
    assert!(dynamic_address(4096, 0, 1024, 1024, 256, 256).is_err());
    assert!(dynamic_address(u64::MAX, 256, 512, 1024, 256, 256).is_err());
}

#[test]
fn indirect_record_requires_usage_alignment_and_twelve_bytes() {
    assert!(validate_indirect(true, 16, 4).is_ok());
    assert!(validate_indirect(false, 16, 4).is_err());
    assert!(validate_indirect(true, 16, 2).is_err());
    assert!(validate_indirect(true, 15, 4).is_err());
    assert!(validate_indirect(true, u64::MAX, u64::MAX - 3).is_err());
}
