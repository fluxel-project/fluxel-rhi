//! Scalar-type boundaries for the Direct SPIR-V atomic routes.

use super::{float32_atomic_pointer_matches, int64_atomic_pointer_matches};
use std::collections::HashSet;

#[test]
fn float32_atomic_route_rejects_other_scalar_widths_and_storage_classes() {
    let float32 = HashSet::from([10]);
    assert!(float32_atomic_pointer_matches(
        Some(&(12, 10)),
        12,
        &float32
    ));
    assert!(!float32_atomic_pointer_matches(
        Some(&(12, 11)),
        12,
        &float32
    ));
    assert!(!float32_atomic_pointer_matches(
        Some(&(4, 10)),
        12,
        &float32
    ));
    assert!(!float32_atomic_pointer_matches(None, 12, &float32));
}

#[test]
fn int64_signed_minmax_route_rejects_unsigned_int64_types() {
    let int64 = HashSet::from([20, 21]);
    let signed_int64 = HashSet::from([20]);
    let unsigned_int64 = HashSet::from([21]);
    assert!(int64_atomic_pointer_matches(
        Some(&(12, 20)),
        12,
        &int64,
        &signed_int64,
        &unsigned_int64,
        true,
        false,
    ));
    assert!(!int64_atomic_pointer_matches(
        Some(&(12, 21)),
        12,
        &int64,
        &signed_int64,
        &unsigned_int64,
        true,
        false,
    ));
    assert!(int64_atomic_pointer_matches(
        Some(&(12, 21)),
        12,
        &int64,
        &signed_int64,
        &unsigned_int64,
        false,
        false,
    ));
}

#[test]
fn int64_unsigned_minmax_route_rejects_signed_int64_types() {
    let int64 = HashSet::from([20, 21]);
    let signed_int64 = HashSet::from([20]);
    let unsigned_int64 = HashSet::from([21]);
    assert!(int64_atomic_pointer_matches(
        Some(&(12, 21)),
        12,
        &int64,
        &signed_int64,
        &unsigned_int64,
        false,
        true,
    ));
    assert!(!int64_atomic_pointer_matches(
        Some(&(12, 20)),
        12,
        &int64,
        &signed_int64,
        &unsigned_int64,
        false,
        true,
    ));
}

#[test]
fn int64_workgroup_unsigned_minmax_requires_unsigned_workgroup_pointer() {
    let int64 = HashSet::from([20, 21]);
    let signed_int64 = HashSet::from([20]);
    let unsigned_int64 = HashSet::from([21]);
    assert!(int64_atomic_pointer_matches(
        Some(&(4, 21)),
        4,
        &int64,
        &signed_int64,
        &unsigned_int64,
        false,
        true,
    ));
    assert!(!int64_atomic_pointer_matches(
        Some(&(4, 20)),
        4,
        &int64,
        &signed_int64,
        &unsigned_int64,
        false,
        true,
    ));
    assert!(!int64_atomic_pointer_matches(
        Some(&(12, 21)),
        4,
        &int64,
        &signed_int64,
        &unsigned_int64,
        false,
        true,
    ));
}
