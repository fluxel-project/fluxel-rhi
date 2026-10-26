//! Unit tests for the corresponding implementation module.

use super::*;

#[test]
fn reads_the_selected_mesh_entry_topology() {
    let bytes = include_bytes!("../../../../tests/assets/mesh/mesh.spv");
    let words = read_spv(&mut Cursor::new(bytes.as_slice())).unwrap();
    assert_eq!(mesh_output_mode(&words, "main"), Some(5298));
    assert_eq!(mesh_output_mode(&words, "missing"), None);
}
