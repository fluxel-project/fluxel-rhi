//! Resource-state and UAV-ordering lowering admission regressions.

use super::*;

#[test]
fn same_state_uav_ordering_requires_a_write_dependency() {
    let read = api::AccessMask::SHADER_READ;
    let write = api::AccessMask::SHADER_WRITE;
    let uav = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    for (from, to, expected) in [
        (read, read, false),
        (write, read, true),
        (read, write, true),
        (write, write, true),
        (read | write, read | write, true),
        (
            api::AccessMask::ACCEL_WRITE,
            api::AccessMask::ACCEL_WRITE,
            true,
        ),
    ] {
        assert_eq!(needs_uav_barrier(uav, uav, from, to), expected);
    }
    assert!(!needs_uav_barrier(
        uav,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
        write,
        read
    ));
    assert!(!needs_uav_barrier(
        D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_RESOURCE_STATE_COPY_DEST,
        api::AccessMask::COPY_WRITE,
        api::AccessMask::COPY_WRITE,
    ));
}

#[test]
fn general_layout_read_scopes_do_not_require_uav_ordering() {
    let uses = api::TextureUses {
        stages: api::StageMask::COMPUTE,
        accesses: api::AccessMask::SHADER_READ,
    };
    let state = texture_state(api::ImageLayout::General, uses).unwrap();
    assert_eq!(state, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    assert!(!needs_uav_barrier(
        state,
        state,
        uses.accesses,
        uses.accesses
    ));
}

#[test]
fn texture_shader_reads_cover_all_requested_shader_stages() {
    for stage in [
        api::StageMask::VERTEX,
        api::StageMask::TASK,
        api::StageMask::MESH,
    ] {
        let uses = |stages| api::TextureUses {
            stages,
            accesses: api::AccessMask::SHADER_READ,
        };
        assert_eq!(
            texture_state(api::ImageLayout::ShaderReadOnly, uses(stage)).unwrap(),
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
        );
        assert_eq!(
            texture_state(
                api::ImageLayout::ShaderReadOnly,
                uses(stage | api::StageMask::FRAGMENT)
            )
            .unwrap(),
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
                | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE
        );
    }
}

#[test]
fn texture_resolve_layouts_require_read_and_write_access() {
    let uses = |accesses| api::TextureUses {
        stages: api::StageMask::COLOR_OUTPUT,
        accesses,
    };
    assert_eq!(
        texture_state(
            api::ImageLayout::ResolveSrc,
            uses(api::AccessMask::COLOR_READ)
        )
        .unwrap(),
        D3D12_RESOURCE_STATE_RESOLVE_SOURCE
    );
    assert_eq!(
        texture_state(
            api::ImageLayout::ResolveDst,
            uses(api::AccessMask::COLOR_WRITE)
        )
        .unwrap(),
        D3D12_RESOURCE_STATE_RESOLVE_DEST
    );
    assert!(texture_state(api::ImageLayout::ResolveSrc, uses(api::AccessMask::EMPTY)).is_err());
    assert!(texture_state(api::ImageLayout::ResolveDst, uses(api::AccessMask::EMPTY)).is_err());
}

#[test]
fn depth_read_layout_can_combine_depth_and_shader_reads() {
    let uses = api::TextureUses {
        stages: api::StageMask::FRAGMENT | api::StageMask::VERTEX,
        accesses: api::AccessMask::DEPTH_READ | api::AccessMask::SHADER_READ,
    };
    assert_eq!(
        texture_state(api::ImageLayout::DepthReadOnly, uses).unwrap(),
        D3D12_RESOURCE_STATE_DEPTH_READ
            | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE
            | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
    );
    assert_eq!(
        texture_state(
            api::ImageLayout::ColorAttachment,
            api::TextureUses {
                stages: api::StageMask::COLOR_OUTPUT,
                accesses: api::AccessMask::COLOR_READ,
            }
        )
        .unwrap(),
        D3D12_RESOURCE_STATE_RENDER_TARGET
    );
}

#[test]
fn buffer_state_accepts_individual_vertex_and_uniform_reads() {
    for access in [api::AccessMask::VERTEX_READ, api::AccessMask::UNIFORM_READ] {
        assert_eq!(
            buffer_state(api::BufferUses {
                stages: api::StageMask::VERTEX,
                accesses: access
            })
            .expect("DX12 vertex/constant-buffer state"),
            D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
        );
    }
}

#[test]
fn buffer_state_rejects_shader_read_without_a_shader_stage() {
    assert!(
        buffer_state(api::BufferUses {
            stages: api::StageMask::VERTEX_INPUT,
            accesses: api::AccessMask::SHADER_READ,
        })
        .is_err()
    );
}

#[test]
fn storage_accesses_cover_graphics_shader_stages_and_read_write() {
    for stage in [
        api::StageMask::VERTEX,
        api::StageMask::FRAGMENT,
        api::StageMask::TASK,
        api::StageMask::MESH,
        api::StageMask::COMPUTE,
        api::StageMask::RAY,
    ] {
        let accesses = api::AccessMask::SHADER_READ | api::AccessMask::SHADER_WRITE;
        assert_eq!(
            buffer_state(api::BufferUses {
                stages: stage,
                accesses
            })
            .unwrap(),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS
        );
        for accesses in [api::AccessMask::SHADER_READ, accesses] {
            assert_eq!(
                texture_state(
                    api::ImageLayout::General,
                    api::TextureUses {
                        stages: stage,
                        accesses
                    }
                )
                .unwrap(),
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS
            );
        }
    }
    assert_eq!(
        buffer_state(api::BufferUses {
            stages: api::StageMask::COPY | api::StageMask::FRAGMENT,
            accesses: api::AccessMask::COPY_READ | api::AccessMask::SHADER_READ,
        })
        .unwrap(),
        D3D12_RESOURCE_STATE_COPY_SOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE
    );
}
