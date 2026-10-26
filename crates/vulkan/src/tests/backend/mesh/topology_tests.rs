//! Unit tests for the corresponding implementation module.

use super::*;

#[test]
fn mesh_topology_rejects_descriptor_shader_mismatch_and_strips() {
    assert!(topology_matches(api::PrimitiveTopology::TriangleList, 5298));
    assert!(!topology_matches(api::PrimitiveTopology::PointList, 5298));
    assert!(!topology_matches(
        api::PrimitiveTopology::TriangleStrip,
        5298
    ));
    assert!(topology_matches(api::PrimitiveTopology::LineList, 5269));
}
