//! Workgroup Int64 atomic opcode readback cases.
use super::common::run_int64_atomic_route;
use fluxel_rhi_core::api::*;

#[test]
fn int64_workgroup_atomic_add_accumulates_and_reads_back() {
    run_int64_atomic_route(
        SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_ADD,
        include_bytes!("../assets/int64_atomic_workgroup_add.comp.spv"),
        0,
        37,
        0x164a_3204,
        [37, 1, 1],
        [1, 1, 1],
    );
}

#[test]
fn int64_workgroup_atomic_min_accumulates_and_reads_back() {
    run_int64_atomic_route(
        SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_MIN,
        include_bytes!("../assets/int64_atomic_workgroup_min.comp.spv"),
        0,
        1,
        0x164a_3205,
        [37, 1, 1],
        [1, 1, 1],
    );
}

#[test]
fn int64_workgroup_atomic_max_accumulates_and_reads_back() {
    run_int64_atomic_route(
        SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_MAX,
        include_bytes!("../assets/int64_atomic_workgroup_max.comp.spv"),
        0,
        37,
        0x164a_3206,
        [37, 1, 1],
        [1, 1, 1],
    );
}

#[test]
fn int64_workgroup_atomic_xor_accumulates_and_reads_back() {
    run_int64_atomic_route(
        SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_XOR,
        include_bytes!("../assets/int64_atomic_workgroup_xor.comp.spv"),
        0,
        1,
        0x164a_3207,
        [37, 1, 1],
        [1, 1, 1],
    );
}

#[test]
fn int64_workgroup_atomic_and_accumulates_and_reads_back() {
    run_int64_atomic_route(
        SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_AND,
        include_bytes!("../assets/int64_atomic_workgroup_and.comp.spv"),
        -1,
        -64,
        0x164a_3208,
        [37, 1, 1],
        [1, 1, 1],
    );
}

#[test]
fn int64_workgroup_atomic_or_accumulates_and_reads_back() {
    run_int64_atomic_route(
        SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_OR,
        include_bytes!("../assets/int64_atomic_workgroup_or.comp.spv"),
        0,
        63,
        0x164a_3209,
        [37, 1, 1],
        [1, 1, 1],
    );
}

#[test]
fn int64_workgroup_atomic_umin_accumulates_and_reads_back() {
    run_int64_atomic_route(
        SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_UMIN,
        include_bytes!("../assets/int64_atomic_workgroup_umin.comp.spv"),
        -1,
        0,
        0x164a_320a,
        [37, 1, 1],
        [1, 1, 1],
    );
}

#[test]
fn int64_workgroup_atomic_umax_accumulates_and_reads_back() {
    run_int64_atomic_route(
        SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_UMAX,
        include_bytes!("../assets/int64_atomic_workgroup_umax.comp.spv"),
        0,
        -1,
        0x164a_3210,
        [37, 1, 1],
        [1, 1, 1],
    );
}

#[test]
fn int64_workgroup_atomic_exchange_returns_each_replaced_value() {
    run_int64_atomic_route(
        SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_EXCHANGE,
        include_bytes!("../assets/int64_atomic_workgroup_exchange.comp.spv"),
        0,
        1,
        0x164a_3211,
        [37, 1, 1],
        [1, 1, 1],
    );
}

#[test]
fn int64_workgroup_compare_exchange_returns_each_observed_value() {
    run_int64_atomic_route(
        SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_COMPARE_EXCHANGE,
        include_bytes!("../assets/int64_atomic_workgroup_compare_exchange.comp.spv"),
        0,
        1,
        0x164a_3212,
        [37, 1, 1],
        [1, 1, 1],
    );
}
