//! Upload-page addressability limit validation.

use super::*;

#[test]
fn page_sizes_reject_unaddressable_allocations() {
    assert_eq!(allocation_size(1).unwrap(), PAGE_SIZE);
    assert_eq!(allocation_size(PAGE_SIZE + 1).unwrap(), PAGE_SIZE + 1);
    assert!(matches!(
        allocation_size(u64::MAX),
        Err(Error::InvalidRange)
    ));
}
