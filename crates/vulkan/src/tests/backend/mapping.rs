//! Unit tests for the corresponding implementation module.

use super::atom_range;
#[test]
fn noncoherent_ranges_include_whole_atoms_and_allocation_end() {
    assert_eq!(atom_range(257, 300, 1024, 256), (256, 256));
    assert_eq!(atom_range(900, 999, 1000, 256), (768, 232));
}
