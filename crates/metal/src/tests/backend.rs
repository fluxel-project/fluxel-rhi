//! Unit tests for the corresponding implementation module.

use super::lease_generation_is_current;

#[test]
fn surface_reconfiguration_invalidates_old_image_leases() {
    assert!(lease_generation_is_current(4, 4, 4, false));
    assert!(!lease_generation_is_current(4, 4, 5, false));
    assert!(!lease_generation_is_current(5, 4, 5, false));
    assert!(!lease_generation_is_current(5, 5, 5, true));
}
