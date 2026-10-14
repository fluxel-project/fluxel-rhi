//! Pipeline-group ABI checks for immediate GL raster binding.
//!
//! Shader optimization may omit declarations, but that does not relax the
//! Core layout used to create a pipeline. This comparison therefore uses the
//! complete immutable group declaration before the binding module issues any
//! native calls.

use super::super::types::GlBindGroupLayout;

/// Returns whether two group layouts describe the same complete Core ABI.
pub(crate) fn matches(expected: &GlBindGroupLayout, actual: &GlBindGroupLayout) -> bool {
    std::rc::Rc::ptr_eq(&expected.state, &actual.state)
        && expected.entries.len() == actual.entries.len()
        && expected
            .entries
            .iter()
            .zip(actual.entries.iter())
            .all(|(left, right)| left.binding == right.binding && left.slot == right.slot)
}
