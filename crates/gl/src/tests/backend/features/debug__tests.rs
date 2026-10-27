use super::*;
use crate::backend::{features, profile::GlProfile};
use std::collections::BTreeSet;

#[test]
fn debug_route_needs_push_pop_and_insert() {
    let profile = GlProfile::Desktop { major: 4, minor: 3 };
    let partial = BTreeSet::from([Fn::PushDebugGroup, Fn::PopDebugGroup]);
    assert!(!features::supports(
        F::DebugMarkers,
        profile,
        &BTreeSet::new(),
        &partial
    ));
    assert_eq!(
        features::missing_function(F::DebugMarkers, profile, &BTreeSet::new(), &partial),
        Some(Fn::DebugMessageInsert)
    );
}

#[test]
fn webgl2_does_not_infer_khr_debug_from_browser_profile() {
    let functions = BTreeSet::from([
        Fn::PushDebugGroup,
        Fn::PopDebugGroup,
        Fn::DebugMessageInsert,
    ]);
    assert!(!features::supports(
        F::DebugMarkers,
        GlProfile::WebGl2,
        &BTreeSet::new(),
        &functions
    ));
}
