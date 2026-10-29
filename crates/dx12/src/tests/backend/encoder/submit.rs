//! Submission owner retention when the native device is removed.

use super::retain_after;

#[test]
fn device_removed_keeps_all_submitted_native_owners() {
    assert!(retain_after(u64::MAX, 1));
    assert!(retain_after(u64::MAX, u64::MAX));
    assert!(retain_after(10, u64::MAX));
    assert!(retain_after(10, 11));
    assert!(!retain_after(10, 10));
}
