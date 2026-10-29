//! Direct, indirect, and task mesh dispatch entry cases.

use super::scenario::{Dispatch, exercise};

#[test]
fn mesh_direct_depth_readback() {
    exercise(Dispatch::Direct, false);
}

#[test]
fn mesh_indirect_depth_readback() {
    exercise(Dispatch::Indirect, false);
}

#[test]
fn task_mesh_depth_readback() {
    exercise(Dispatch::Direct, true);
}
