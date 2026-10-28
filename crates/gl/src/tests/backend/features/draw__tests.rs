use super::*;
use crate::backend::{features, profile::GlProfile};
use std::collections::BTreeSet;

#[test]
fn indirect_draw_obeys_profile_floor_and_both_tokens() {
    let one = BTreeSet::from([Fn::DrawArraysIndirect]);
    let both = BTreeSet::from([Fn::DrawArraysIndirect, Fn::DrawElementsIndirect]);
    let none = BTreeSet::new();
    assert!(!features::supports(
        F::IndirectDraw,
        GlProfile::Desktop { major: 3, minor: 3 },
        &none,
        &both
    ));
    assert!(!features::supports(
        F::IndirectDraw,
        GlProfile::Desktop { major: 4, minor: 0 },
        &none,
        &one
    ));
    assert!(features::supports(
        F::IndirectDraw,
        GlProfile::Desktop { major: 4, minor: 0 },
        &none,
        &both
    ));
}

#[test]
fn indirect_count_is_never_in_gles_core() {
    let functions = BTreeSet::from([
        Fn::MultiDrawArraysIndirectCount,
        Fn::MultiDrawElementsIndirectCount,
    ]);
    assert!(!features::supports(
        F::IndirectCount,
        GlProfile::Embedded { major: 3, minor: 2 },
        &BTreeSet::new(),
        &functions
    ));
}
