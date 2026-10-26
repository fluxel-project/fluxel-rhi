//! Unit tests for the corresponding implementation module.

use super::metal_binding_slot;
use fluxel_rhi_core::api::Error;

#[test]
fn binding_slots_use_disjoint_fixed_group_ranges() {
    assert_eq!(metal_binding_slot(0, 0).unwrap(), 0);
    assert_eq!(metal_binding_slot(1, 0).unwrap(), 7);
    assert_eq!(metal_binding_slot(3, 6).unwrap(), 27);
    assert!(matches!(metal_binding_slot(4, 0), Err(Error::InvalidRange)));
    assert!(matches!(metal_binding_slot(0, 7), Err(Error::InvalidRange)));
}
