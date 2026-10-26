//! Unit tests for the corresponding implementation module.

use super::{acceleration_scope_supported, api};

#[test]
fn accepts_build_and_supported_shader_read_scopes() {
    let scope = |stages, accesses| api::AccessScope { stages, accesses };
    assert!(acceleration_scope_supported(scope(
        api::StageMask::ACCELERATION,
        api::AccessMask::ACCEL_WRITE
    )));
    assert!(acceleration_scope_supported(scope(
        api::StageMask::COMPUTE,
        api::AccessMask::ACCEL_READ
    )));
    assert!(acceleration_scope_supported(scope(
        api::StageMask::FRAGMENT,
        api::AccessMask::ACCEL_READ
    )));
    assert!(!acceleration_scope_supported(scope(
        api::StageMask::EMPTY,
        api::AccessMask::ACCEL_WRITE
    )));
}
