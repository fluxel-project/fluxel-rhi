use super::*;
#[test]
fn enhanced_scope_maps_copy_and_shader_caches() {
    let (sync, access) = scope(
        api::AccessScope {
            stages: api::StageMask::COPY | api::StageMask::COMPUTE,
            accesses: api::AccessMask::COPY_WRITE | api::AccessMask::SHADER_WRITE,
        },
        false,
    )
    .unwrap();
    assert!(sync.0 & D3D12_BARRIER_SYNC_COPY.0 != 0);
    assert!(sync.0 & D3D12_BARRIER_SYNC_COMPUTE_SHADING.0 != 0);
    assert!(access.0 & D3D12_BARRIER_ACCESS_COPY_DEST.0 != 0);
    assert!(access.0 & D3D12_BARRIER_ACCESS_UNORDERED_ACCESS.0 != 0);
}

#[test]
fn vertex_input_covers_vertex_fetch_and_indices() {
    let (sync, _) = scope(
        api::AccessScope {
            stages: api::StageMask::VERTEX_INPUT,
            accesses: api::AccessMask::VERTEX_READ | api::AccessMask::INDEX_READ,
        },
        false,
    )
    .unwrap();
    assert!(sync.0 & D3D12_BARRIER_SYNC_INDEX_INPUT.0 != 0);
    assert!(sync.0 & D3D12_BARRIER_SYNC_VERTEX_SHADING.0 != 0);
}

#[test]
fn all_is_gpu_wide_but_explicit_host_is_rejected() {
    let (sync, access) = scope(
        api::AccessScope {
            stages: api::StageMask::ALL_COMMANDS,
            accesses: api::AccessMask::ALL,
        },
        false,
    )
    .unwrap();
    assert!(sync.0 & D3D12_BARRIER_SYNC_ALL.0 != 0);
    assert_eq!(access, D3D12_BARRIER_ACCESS_COMMON);
    assert!(
        scope(
            api::AccessScope {
                stages: api::StageMask::HOST,
                accesses: api::AccessMask::EMPTY,
            },
            false
        )
        .is_err()
    );
}

#[test]
fn ambiguous_core_accesses_widen_without_layout_changes() {
    let (query_sync, query_access) = scope(
        api::AccessScope {
            stages: api::StageMask::COMPUTE,
            accesses: api::AccessMask::QUERY_RESOLVE_WRITE,
        },
        false,
    )
    .unwrap();
    assert!(query_sync.0 & D3D12_BARRIER_SYNC_COPY.0 != 0);
    assert!(query_access.0 & D3D12_BARRIER_ACCESS_COPY_DEST.0 != 0);
    assert_eq!(query_access.0 & D3D12_BARRIER_ACCESS_RESOLVE_DEST.0, 0);

    let (sync, access) = scope(
        api::AccessScope {
            stages: api::StageMask::FRAGMENT,
            accesses: api::AccessMask::SHADER_READ
                | api::AccessMask::COLOR_WRITE
                | api::AccessMask::DEPTH_READ,
        },
        false,
    )
    .unwrap();
    assert!(sync.0 & D3D12_BARRIER_SYNC_RENDER_TARGET.0 != 0);
    assert!(sync.0 & D3D12_BARRIER_SYNC_RESOLVE.0 != 0);
    assert!(sync.0 & D3D12_BARRIER_SYNC_DEPTH_STENCIL.0 != 0);
    assert!(access.0 & D3D12_BARRIER_ACCESS_UNORDERED_ACCESS.0 != 0);
    assert!(access.0 & D3D12_BARRIER_ACCESS_RESOLVE_SOURCE.0 != 0);
    assert!(access.0 & D3D12_BARRIER_ACCESS_RESOLVE_DEST.0 != 0);
}

#[test]
fn empty_access_maps_to_no_access() {
    let (_, access) = scope(
        api::AccessScope {
            stages: api::StageMask::COMPUTE,
            accesses: api::AccessMask::EMPTY,
        },
        false,
    )
    .unwrap();
    assert_eq!(access, D3D12_BARRIER_ACCESS_NO_ACCESS);
}

#[test]
fn compute_lane_all_is_queue_compatible_and_graphics_is_rejected() {
    let (sync, access) = scope(
        api::AccessScope {
            stages: api::StageMask::ALL_COMMANDS,
            accesses: api::AccessMask::ALL,
        },
        true,
    )
    .unwrap();
    assert_eq!(sync, D3D12_BARRIER_SYNC_ALL);
    assert_eq!(access, D3D12_BARRIER_ACCESS_COMMON);
    assert!(
        scope(
            api::AccessScope {
                stages: api::StageMask::RAY | api::StageMask::ACCELERATION,
                accesses: api::AccessMask::SHADER_READ | api::AccessMask::ACCEL_WRITE,
            },
            true,
        )
        .is_ok()
    );
    assert!(
        scope(
            api::AccessScope {
                stages: api::StageMask::FRAGMENT,
                accesses: api::AccessMask::SHADER_READ,
            },
            true,
        )
        .is_err()
    );
}

#[test]
fn nonempty_access_needs_a_gpu_stage() {
    assert!(
        scope(
            api::AccessScope {
                stages: api::StageMask::EMPTY,
                accesses: api::AccessMask::COPY_WRITE,
            },
            false,
        )
        .is_err()
    );
}
#[test]
fn host_and_unknown_masks_are_rejected() {
    assert!(
        scope(
            api::AccessScope {
                stages: api::StageMask::HOST,
                accesses: api::AccessMask::EMPTY
            },
            false
        )
        .is_err()
    );
    assert!(
        scope(
            api::AccessScope {
                stages: api::StageMask(1 << 40),
                accesses: api::AccessMask::EMPTY
            },
            false
        )
        .is_err()
    );
}
#[test]
fn legacy_is_only_a_uav_write_dependency() {
    let write = api::AccessScope {
        stages: api::StageMask::COMPUTE,
        accesses: api::AccessMask::SHADER_WRITE,
    };
    let read = api::AccessScope {
        stages: api::StageMask::COMPUTE,
        accesses: api::AccessMask::SHADER_READ,
    };
    assert!(legacy_uav(write, read));
    assert!(!legacy_uav(
        api::AccessScope {
            stages: api::StageMask::COPY,
            accesses: api::AccessMask::COPY_WRITE
        },
        read
    ));
}
