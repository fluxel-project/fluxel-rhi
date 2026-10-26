//! Unit tests for the corresponding implementation module.

#[cfg(not(target_vendor = "apple"))]
use super::{MetalReadback, map};

#[cfg(not(target_vendor = "apple"))]
#[test]
fn empty_readback_has_no_row_or_slice_pitch() {
    let data = map(MetalReadback { range: 0..0 }).unwrap();
    assert!(data.bytes.is_empty());
    assert_eq!(data.row_pitch, 0);
    assert_eq!(data.slice_pitch, 0);
}
