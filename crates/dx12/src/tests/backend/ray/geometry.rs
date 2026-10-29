//! Ray-tracing geometry GPU-address, stride, and range validation.

use super::validate_address;
#[test]
fn rejects_misaligned_gpu_addresses_strides_and_ranges() {
    assert_eq!(
        validate_address(0x1000, 128, 4, 3, 12, 12, 4).unwrap(),
        0x1004
    );
    assert!(validate_address(0x1001, 128, 4, 3, 12, 12, 4).is_err());
    assert!(validate_address(0x1000, 128, 0, 2, 25, 24, 8).is_err());
    assert!(validate_address(0x1000, u64::MAX, 0, 2, u64::from(u32::MAX) + 1, 4, 4).is_err());
    assert!(validate_address(0x1000, 32, 0, 3, 12, 12, 4).is_err());
    assert!(validate_address(0x1000, 128, 1, 3, 2, 2, 2).is_err());
}
