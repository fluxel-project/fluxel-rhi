//! Unit tests for the corresponding implementation module.

use super::encoded_array_len;
#[test]
fn layout_checks_alignment_count_and_overflow() {
    assert_eq!(encoded_array_len(64, 16, 3, 192), Ok(192));
    assert!(encoded_array_len(65, 16, 1, 1024).is_err());
    assert!(encoded_array_len(64, 3, 1, 1024).is_err());
    assert!(encoded_array_len(64, 16, 0, 1024).is_err());
    assert!(encoded_array_len(64, 16, 3, 191).is_err());
    assert!(encoded_array_len(usize::MAX, 1, 2, u64::MAX).is_err());
}
