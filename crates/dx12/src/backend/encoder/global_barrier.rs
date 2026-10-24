//! Lower Core's resource-less memory dependency to D3D12 barriers.
//!
//! A Core global transition has no resource or layout. Enhanced barriers
//! express that directly; older runtimes only have a null UAV fallback, which
//! is deliberately limited to an actual UAV producer/consumer dependency.

use super::*;

const GPU_STAGES: u64 = api::StageMask::COPY.0
    | api::StageMask::INDIRECT.0
    | api::StageMask::VERTEX_INPUT.0
    | api::StageMask::VERTEX.0
    | api::StageMask::FRAGMENT.0
    | api::StageMask::COLOR_OUTPUT.0
    | api::StageMask::COMPUTE.0
    | api::StageMask::TASK.0
    | api::StageMask::MESH.0
    | api::StageMask::RAY.0
    | api::StageMask::ACCELERATION.0;
const CORE_ACCESSES: u64 = api::AccessMask::HOST_READ.0
    | api::AccessMask::HOST_WRITE.0
    | api::AccessMask::COPY_READ.0
    | api::AccessMask::COPY_WRITE.0
    | api::AccessMask::VERTEX_READ.0
    | api::AccessMask::INDEX_READ.0
    | api::AccessMask::UNIFORM_READ.0
    | api::AccessMask::SHADER_READ.0
    | api::AccessMask::SHADER_WRITE.0
    | api::AccessMask::INDIRECT_READ.0
    | api::AccessMask::COLOR_READ.0
    | api::AccessMask::COLOR_WRITE.0
    | api::AccessMask::DEPTH_READ.0
    | api::AccessMask::DEPTH_WRITE.0
    | api::AccessMask::STENCIL_READ.0
    | api::AccessMask::STENCIL_WRITE.0
    | api::AccessMask::ACCEL_READ.0
    | api::AccessMask::ACCEL_WRITE.0
    | api::AccessMask::QUERY_RESOLVE_WRITE.0
    | api::AccessMask::RAY_TRACING_SHADER_DATA_READ.0;

fn scope(
    scope: api::AccessScope,
    compute_lane: bool,
) -> Result<(D3D12_BARRIER_SYNC, D3D12_BARRIER_ACCESS), Error> {
    if (scope.stages != api::StageMask::ALL_COMMANDS && scope.stages.contains(api::StageMask::HOST))
        || (scope.accesses != api::AccessMask::ALL
            && (scope.accesses.contains(api::AccessMask::HOST_READ)
                || scope.accesses.contains(api::AccessMask::HOST_WRITE)))
    {
        // A command-list barrier cannot establish CPU completion/visibility.
        return Err(Error::Unsupported("DX12 global barrier host scope"));
    }
    if scope.stages != api::StageMask::ALL_COMMANDS && scope.stages.0 & !GPU_STAGES != 0 {
        return Err(Error::Unsupported("DX12 global barrier stage mask"));
    }
    if scope.accesses != api::AccessMask::ALL && scope.accesses.0 & !CORE_ACCESSES != 0 {
        return Err(Error::Unsupported("DX12 global barrier access mask"));
    }
    if scope.stages == api::StageMask::EMPTY && scope.accesses != api::AccessMask::EMPTY {
        // Native enhanced barriers require a synchronization scope for every
        // cache access. Core must name a GPU stage instead of recording a
        // zero-sync/nonnull-access barrier.
        return Err(Error::Unsupported(
            "DX12 global barrier access requires a GPU stage",
        ));
    }
    if compute_lane {
        let compute_stages = api::StageMask::COPY
            | api::StageMask::INDIRECT
            | api::StageMask::COMPUTE
            | api::StageMask::RAY
            | api::StageMask::ACCELERATION;
        let graphics_accesses = api::AccessMask::VERTEX_READ
            | api::AccessMask::INDEX_READ
            | api::AccessMask::COLOR_READ
            | api::AccessMask::COLOR_WRITE
            | api::AccessMask::DEPTH_READ
            | api::AccessMask::DEPTH_WRITE
            | api::AccessMask::STENCIL_READ
            | api::AccessMask::STENCIL_WRITE;
        if (scope.stages != api::StageMask::ALL_COMMANDS && scope.stages.0 & !compute_stages.0 != 0)
            || (scope.accesses != api::AccessMask::ALL
                && scope.accesses.0 & graphics_accesses.0 != 0)
        {
            return Err(Error::Unsupported(
                "DX12 global barrier scope is unavailable on compute lane",
            ));
        }
    }

    let stages = scope.stages;
    if compute_lane && scope.accesses == api::AccessMask::ALL {
        // `ALL` is Core's all-GPU sentinel. COMMON avoids graphics-only
        // accesses while full synchronization conservatively includes legal
        // compute-queue ray tracing and acceleration-structure commands.
        return Ok((D3D12_BARRIER_SYNC_ALL, D3D12_BARRIER_ACCESS_COMMON));
    }
    if scope.accesses == api::AccessMask::ALL {
        // Do not manufacture every optional native access (notably DXR) for
        // Core's all-GPU sentinel. A global COMMON barrier with full sync is
        // the legal conservative memory-only form and has no layout effect.
        return Ok((D3D12_BARRIER_SYNC_ALL, D3D12_BARRIER_ACCESS_COMMON));
    }
    let mut sync = if stages == api::StageMask::ALL_COMMANDS {
        D3D12_BARRIER_SYNC_ALL
    } else {
        D3D12_BARRIER_SYNC_NONE
    };
    let add_sync = |out: &mut D3D12_BARRIER_SYNC, stage, native| {
        if stages.contains(stage) {
            *out |= native;
        }
    };
    if stages != api::StageMask::ALL_COMMANDS {
        add_sync(&mut sync, api::StageMask::COPY, D3D12_BARRIER_SYNC_COPY);
        add_sync(
            &mut sync,
            api::StageMask::INDIRECT,
            D3D12_BARRIER_SYNC_EXECUTE_INDIRECT,
        );
        add_sync(
            &mut sync,
            api::StageMask::VERTEX_INPUT,
            D3D12_BARRIER_SYNC_INDEX_INPUT | D3D12_BARRIER_SYNC_VERTEX_SHADING,
        );
        add_sync(
            &mut sync,
            api::StageMask::VERTEX,
            D3D12_BARRIER_SYNC_VERTEX_SHADING,
        );
        add_sync(
            &mut sync,
            api::StageMask::FRAGMENT,
            D3D12_BARRIER_SYNC_PIXEL_SHADING,
        );
        add_sync(
            &mut sync,
            api::StageMask::COLOR_OUTPUT,
            D3D12_BARRIER_SYNC_RENDER_TARGET,
        );
        add_sync(
            &mut sync,
            api::StageMask::COMPUTE,
            D3D12_BARRIER_SYNC_COMPUTE_SHADING,
        );
        // Task/mesh have no narrower D3D12 sync bit, so non-pixel shading is
        // conservative rather than an invalid narrowing.
        add_sync(
            &mut sync,
            api::StageMask::TASK,
            D3D12_BARRIER_SYNC_NON_PIXEL_SHADING,
        );
        add_sync(
            &mut sync,
            api::StageMask::MESH,
            D3D12_BARRIER_SYNC_NON_PIXEL_SHADING,
        );
        add_sync(
            &mut sync,
            api::StageMask::RAY,
            D3D12_BARRIER_SYNC_RAYTRACING,
        );
        add_sync(
            &mut sync,
            api::StageMask::ACCELERATION,
            D3D12_BARRIER_SYNC_BUILD_RAYTRACING_ACCELERATION_STRUCTURE,
        );
    }

    let all = scope.accesses == api::AccessMask::ALL;
    let has = |value| all || scope.accesses.contains(value);
    if has(api::AccessMask::QUERY_RESOLVE_WRITE) {
        // ResolveQueryData writes a buffer through the copy engine, not the
        // MSAA texture-resolve path represented by RESOLVE_DEST.
        sync |= D3D12_BARRIER_SYNC_COPY;
    }
    if has(api::AccessMask::DEPTH_READ)
        || has(api::AccessMask::DEPTH_WRITE)
        || has(api::AccessMask::STENCIL_READ)
        || has(api::AccessMask::STENCIL_WRITE)
    {
        // Core has no separate depth pipeline stage. Include the native stage
        // so a fragment/color scope cannot accidentally under-synchronize it.
        sync |= D3D12_BARRIER_SYNC_DEPTH_STENCIL;
    }
    if has(api::AccessMask::COLOR_READ) || has(api::AccessMask::COLOR_WRITE) {
        // A global scope has no image layout to distinguish RTV load/store
        // from an MSAA resolve. Widen both synchronization and cache access.
        sync |= D3D12_BARRIER_SYNC_RENDER_TARGET | D3D12_BARRIER_SYNC_RESOLVE;
    }
    let mut access = if scope.accesses == api::AccessMask::EMPTY {
        D3D12_BARRIER_ACCESS_NO_ACCESS
    } else {
        D3D12_BARRIER_ACCESS_COMMON
    };
    let add_access = |out: &mut D3D12_BARRIER_ACCESS, present, native| {
        if present {
            *out |= native;
        }
    };
    add_access(
        &mut access,
        has(api::AccessMask::COPY_READ),
        D3D12_BARRIER_ACCESS_COPY_SOURCE,
    );
    add_access(
        &mut access,
        has(api::AccessMask::COPY_WRITE),
        D3D12_BARRIER_ACCESS_COPY_DEST,
    );
    add_access(
        &mut access,
        has(api::AccessMask::VERTEX_READ),
        D3D12_BARRIER_ACCESS_VERTEX_BUFFER,
    );
    add_access(
        &mut access,
        has(api::AccessMask::INDEX_READ),
        D3D12_BARRIER_ACCESS_INDEX_BUFFER,
    );
    add_access(
        &mut access,
        has(api::AccessMask::UNIFORM_READ),
        D3D12_BARRIER_ACCESS_CONSTANT_BUFFER,
    );
    add_access(
        &mut access,
        has(api::AccessMask::SHADER_READ) || has(api::AccessMask::RAY_TRACING_SHADER_DATA_READ),
        D3D12_BARRIER_ACCESS_SHADER_RESOURCE,
    );
    // Core has no separate read-only/UAV-read bit. Include UAV for
    // SHADER_READ so this resource-less barrier remains conservative.
    add_access(
        &mut access,
        has(api::AccessMask::SHADER_READ),
        D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
    );
    add_access(
        &mut access,
        has(api::AccessMask::SHADER_WRITE),
        D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
    );
    add_access(
        &mut access,
        has(api::AccessMask::INDIRECT_READ),
        D3D12_BARRIER_ACCESS_INDIRECT_ARGUMENT,
    );
    add_access(
        &mut access,
        has(api::AccessMask::COLOR_READ) || has(api::AccessMask::COLOR_WRITE),
        D3D12_BARRIER_ACCESS_RENDER_TARGET
            | D3D12_BARRIER_ACCESS_RESOLVE_SOURCE
            | D3D12_BARRIER_ACCESS_RESOLVE_DEST,
    );
    add_access(
        &mut access,
        has(api::AccessMask::DEPTH_READ) || has(api::AccessMask::STENCIL_READ),
        D3D12_BARRIER_ACCESS_DEPTH_STENCIL_READ,
    );
    add_access(
        &mut access,
        has(api::AccessMask::DEPTH_WRITE) || has(api::AccessMask::STENCIL_WRITE),
        D3D12_BARRIER_ACCESS_DEPTH_STENCIL_WRITE,
    );
    add_access(
        &mut access,
        has(api::AccessMask::ACCEL_READ),
        D3D12_BARRIER_ACCESS_RAYTRACING_ACCELERATION_STRUCTURE_READ,
    );
    add_access(
        &mut access,
        has(api::AccessMask::ACCEL_WRITE),
        D3D12_BARRIER_ACCESS_RAYTRACING_ACCELERATION_STRUCTURE_WRITE,
    );
    add_access(
        &mut access,
        has(api::AccessMask::QUERY_RESOLVE_WRITE),
        D3D12_BARRIER_ACCESS_COPY_DEST,
    );
    Ok((sync, access))
}

fn enhanced_supported(device: &Dx12Device) -> bool {
    let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS12::default();
    // SAFETY: D3D12 fills exactly this stack-owned structure synchronously.
    unsafe {
        device.0.native.CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS12,
            (&mut options as *mut D3D12_FEATURE_DATA_D3D12_OPTIONS12).cast(),
            std::mem::size_of_val(&options) as u32,
        )
    }
    .is_ok()
        && options.EnhancedBarriersSupported.as_bool()
}

fn legacy_uav(before: api::AccessScope, after: api::AccessScope) -> bool {
    let shader_stages = api::StageMask::VERTEX
        | api::StageMask::FRAGMENT
        | api::StageMask::COMPUTE
        | api::StageMask::TASK
        | api::StageMask::MESH
        | api::StageMask::RAY
        | api::StageMask::ACCELERATION;
    let shader_accesses = api::AccessMask::SHADER_READ | api::AccessMask::SHADER_WRITE;
    before.stages.0 & !shader_stages.0 == 0
        && after.stages.0 & !shader_stages.0 == 0
        && before.accesses.0 & !shader_accesses.0 == 0
        && after.accesses.0 & !shader_accesses.0 == 0
        && before.accesses.contains(api::AccessMask::SHADER_WRITE)
        && after.accesses.0 != 0
}

pub(super) fn record(
    encoder: &Dx12Encoder,
    before: api::AccessScope,
    after: api::AccessScope,
) -> Result<(), Error> {
    let (sync_before, access_before) = scope(before, encoder.is_compute)?;
    let (sync_after, access_after) = scope(after, encoder.is_compute)?;
    if enhanced_supported(&encoder.device) {
        let list = encoder
            .list
            .cast::<ID3D12GraphicsCommandList7>()
            .map_err(|_| Error::Unsupported("DX12 enhanced barriers command-list interface"))?;
        let global = D3D12_GLOBAL_BARRIER {
            SyncBefore: sync_before,
            SyncAfter: sync_after,
            AccessBefore: access_before,
            AccessAfter: access_after,
        };
        let group = D3D12_BARRIER_GROUP {
            Type: D3D12_BARRIER_TYPE_GLOBAL,
            NumBarriers: 1,
            Anonymous: D3D12_BARRIER_GROUP_0 {
                pGlobalBarriers: std::ptr::addr_of!(global),
            },
        };
        // SAFETY: the command list reads this stack-owned descriptor synchronously.
        // A global barrier carries no resource pointer and never changes layout.
        unsafe { list.Barrier(std::slice::from_ref(&group)) };
        return Ok(());
    }
    if !legacy_uav(before, after) {
        return Err(Error::Unsupported(
            "DX12 global barrier requires enhanced barriers for this scope",
        ));
    }
    let barrier = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                pResource: ManuallyDrop::new(None),
            }),
        },
    };
    // SAFETY: D3D12 borrows the null-UAV barrier while recording only.
    unsafe { encoder.list.ResourceBarrier(std::slice::from_ref(&barrier)) };
    Ok(())
}

#[cfg(test)]
mod tests {
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
}
