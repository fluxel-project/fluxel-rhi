//! StorageBuffer Int64 atomic opcode readback cases.
use super::common::run_int64_atomic_route;
use fluxel_rhi_core::api::*;

#[test]
fn int64_storage_buffer_atomic_add_accumulates_and_reads_back() {
    run_int64_atomic_route(
        SHADER_FEATURE_INT64_ATOMIC_BUFFER_ADD,
        include_bytes!("../assets/int64_atomic_add.comp.spv"),
        0,
        37,
        0x164a_3201,
        [1, 1, 1],
        [37, 1, 1],
    );
}

#[test]
fn int64_storage_buffer_atomic_min_accumulates_and_reads_back() {
    run_int64_atomic_route(
        SHADER_FEATURE_INT64_ATOMIC_BUFFER_MIN,
        include_bytes!("../assets/int64_atomic_min.comp.spv"),
        37,
        1,
        0x164a_3202,
        [1, 1, 1],
        [37, 1, 1],
    );
}

#[test]
fn int64_storage_buffer_atomic_max_accumulates_and_reads_back() {
    run_int64_atomic_route(
        SHADER_FEATURE_INT64_ATOMIC_BUFFER_MAX,
        include_bytes!("../assets/int64_atomic_max.comp.spv"),
        -37,
        1,
        0x164a_3203,
        [1, 1, 1],
        [37, 1, 1],
    );
}

#[test]
fn int64_storage_buffer_atomic_umin_accumulates_and_reads_back() {
    run_int64_atomic_route(
        SHADER_FEATURE_INT64_ATOMIC_BUFFER_UMIN,
        include_bytes!("../assets/int64_atomic_umin.comp.spv"),
        -1,
        0,
        0x164a_3209,
        [37, 1, 1],
        [1, 1, 1],
    );
}

#[test]
fn int64_storage_buffer_atomic_umax_accumulates_and_reads_back() {
    run_int64_atomic_route(
        SHADER_FEATURE_INT64_ATOMIC_BUFFER_UMAX,
        include_bytes!("../assets/int64_atomic_umax.comp.spv"),
        0,
        -1,
        0x164a_320f,
        [37, 1, 1],
        [1, 1, 1],
    );
}

#[test]
fn int64_storage_buffer_atomic_xor_accumulates_and_reads_back() {
    run_int64_atomic_route(
        SHADER_FEATURE_INT64_ATOMIC_BUFFER_XOR,
        include_bytes!("../assets/int64_atomic_xor.comp.spv"),
        0,
        1,
        0x164a_3207,
        [37, 1, 1],
        [1, 1, 1],
    );
}

#[test]
fn int64_storage_buffer_compare_exchange_returns_each_observed_value() {
    run_int64_atomic_route(
        SHADER_FEATURE_INT64_ATOMIC_BUFFER_COMPARE_EXCHANGE,
        include_bytes!("../assets/int64_atomic_compare_exchange.comp.spv"),
        0,
        1,
        0x164a_3208,
        [37, 1, 1],
        [1, 1, 1],
    );
}

#[test]
fn int64_storage_buffer_exchange_returns_each_replaced_value() {
    run_int64_atomic_route(
        SHADER_FEATURE_INT64_ATOMIC_BUFFER_EXCHANGE,
        include_bytes!("../assets/int64_atomic_exchange.comp.spv"),
        0,
        1,
        0x164a_320b,
        [37, 1, 1],
        [1, 1, 1],
    );
}

#[test]
fn int64_storage_buffer_atomic_and_accumulates_and_reads_back() {
    run_int64_atomic_route(
        SHADER_FEATURE_INT64_ATOMIC_BUFFER_AND,
        include_bytes!("../assets/int64_atomic_and.comp.spv"),
        -1,
        -64,
        0x164a_320c,
        [37, 1, 1],
        [1, 1, 1],
    );
}

#[test]
fn int64_storage_buffer_atomic_or_accumulates_and_reads_back() {
    run_int64_atomic_route(
        SHADER_FEATURE_INT64_ATOMIC_BUFFER_OR,
        include_bytes!("../assets/int64_atomic_or.comp.spv"),
        0,
        63,
        0x164a_320d,
        [37, 1, 1],
        [1, 1, 1],
    );
}
