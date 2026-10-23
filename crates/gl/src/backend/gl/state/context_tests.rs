//! State-authority regression tests.
use super::*;
use crate::backend::gl::api::{
    BufferId, ContextEpoch, ContextStamp, DeviceIdentity, GlCullMode, GlFrontFace,
    GlMultisampleState, GlPrimitiveTopology, GlRasterPipeline, GlRasterState, GlViewport,
    ProgramId, SamplerId, VertexArrayId,
};

fn stamp(epoch: ContextEpoch) -> ContextStamp {
    ContextStamp::new(DeviceIdentity::new(7).unwrap(), epoch)
}
fn object(value: u32) -> GlObjectKey {
    GlObjectKey::new(stamp(ContextEpoch::INITIAL), value, 1)
}
fn packet(identity: u32, base: u64) -> RasterPipelinePacket {
    RasterPipelinePacket {
        identity: object(identity),
        blocks: RasterPipelineBlocks {
            program: CanonicalBlockId::new(base),
            raster: CanonicalBlockId::new(base + 1),
            depth_stencil: CanonicalBlockId::new(base + 2),
            blend: CanonicalBlockId::new(base + 3),
            multisample: CanonicalBlockId::new(base + 4),
        },
    }
}
fn pipeline(program_slot: u32, blend_constant: [u32; 4]) -> GlRasterPipeline {
    let context = stamp(ContextEpoch::INITIAL);
    GlRasterPipeline {
        program: ProgramId::new(context, program_slot, 1),
        vertex_array: VertexArrayId::new(context, program_slot, 1),
        state: GlRasterState {
            rasterization: Default::default(),
            topology: GlPrimitiveTopology::Triangles,
            cull_mode: GlCullMode::None,
            front_face: GlFrontFace::CounterClockwise,
            depth_stencil: None,
            color_targets: vec![],
            multisample: GlMultisampleState {
                sample_count: 1,
                alpha_to_coverage_enabled: false,
                sample_mask: u64::MAX,
            },
            viewport: GlViewport {
                x: 0,
                y: 0,
                width: 1,
                height: 1,
                min_depth: 0.0f32.to_bits(),
                max_depth: 1.0f32.to_bits(),
            },
            scissor: None,
            blend_constant,
        },
    }
}
#[test]
fn pipeline_identity_is_a_collision_free_first_gate() {
    let mut state = ContextState::new(ExecutionMode::Optimized);
    state.register_pipeline(packet(1, 10)).unwrap();
    assert!(!state.prepare_pipeline(packet(1, 10)).unwrap().is_empty());
    state.commit_pipeline(packet(1, 10));
    assert_eq!(
        state.prepare_pipeline(packet(1, 99)),
        Err(ContextStateError::PipelineIdentityConflict)
    );
}
#[test]
fn pipeline_miss_diffs_canonical_blocks_not_the_whole_pipeline() {
    let mut state = ContextState::new(ExecutionMode::Optimized);
    state.register_pipeline(packet(1, 10)).unwrap();
    let replacement = RasterPipelinePacket {
        identity: object(2),
        blocks: RasterPipelineBlocks {
            blend: CanonicalBlockId::new(99),
            ..packet(1, 10).blocks
        },
    };
    state.register_pipeline(replacement).unwrap();
    state.commit_pipeline(packet(1, 10));
    let diff = state.prepare_pipeline(replacement).unwrap();
    assert_eq!(
        diff,
        RasterPipelineDiff {
            blend: true,
            ..RasterPipelineDiff::default()
        }
    );
}
#[test]
fn structurally_equal_pipeline_leaves_share_canonical_ids() {
    let mut interner = RasterPipelineBlockInterner::new();
    let first = pipeline(1, [0; 4]);
    // Distinct pipeline object/VAO carriers must not prevent sharing the
    // immutable program and fixed-function leaves.
    let mut second = pipeline(1, [0; 4]);
    second.vertex_array = VertexArrayId::new(stamp(ContextEpoch::INITIAL), 99, 1);
    let a = interner.intern(&first, "test exhausted").unwrap();
    let b = interner.intern(&second, "test exhausted").unwrap();
    assert_eq!(a, b);
}
#[test]
fn one_changed_leaf_produces_one_lowering_call_and_identity_hit_none() {
    let mut interner = RasterPipelineBlockInterner::new();
    let first = pipeline(1, [0; 4]);
    let changed_blend = pipeline(1, [1, 0, 0, 0]);
    let first_packet = RasterPipelinePacket {
        identity: object(41),
        blocks: interner.intern(&first, "test exhausted").unwrap(),
    };
    let changed_packet = RasterPipelinePacket {
        identity: object(42),
        blocks: interner.intern(&changed_blend, "test exhausted").unwrap(),
    };
    let mut state = ContextState::new(ExecutionMode::Optimized);
    state.register_pipeline(first_packet).unwrap();
    state.register_pipeline(changed_packet).unwrap();
    state.commit_pipeline(first_packet);
    let diff = state.prepare_pipeline(changed_packet).unwrap();
    assert_eq!(
        diff,
        RasterPipelineDiff {
            blend: true,
            ..RasterPipelineDiff::default()
        }
    );
    state.commit_pipeline(changed_packet);
    assert_eq!(
        state.prepare_pipeline(changed_packet).unwrap(),
        RasterPipelineDiff::default(),
        "exact pipeline identity fast path must issue no leaf install"
    );
}
#[test]
fn dynamic_draw_invalidation_cannot_take_a_stale_pipeline_identity_hit() {
    let mut state = ContextState::new(ExecutionMode::Optimized);
    let raster = packet(1, 10);
    state.register_pipeline(raster).unwrap();
    state.commit_pipeline(raster);
    assert!(state.prepare_pipeline(raster).unwrap().is_empty());
    // A viewport/scissor/blend/stencil value belongs to a draw, not the
    // immutable public pipeline packet. The native lowering invalidates
    // this domain before installing the dynamically modified state.
    state.event(StateEvent::DomainFailed(StateDomain::RasterPipeline));
    let diff = state.prepare_pipeline(raster).unwrap();
    assert!(diff.program && diff.raster && diff.depth_stencil && diff.blend && diff.multisample);
}
#[test]
fn compute_program_change_forces_raster_program_restore() {
    let mut state = ContextState::new(ExecutionMode::Optimized);
    let raster = packet(1, 10);
    state.register_pipeline(raster).unwrap();
    state.commit_pipeline(raster);
    state.commit_compute_program(CanonicalBlockId::new(80));
    assert!(state.prepare_pipeline(raster).unwrap().program);
}
#[test]
fn compute_domain_invalidation_forces_raster_program_restore() {
    let mut state = ContextState::new(ExecutionMode::Optimized);
    let raster = packet(1, 10);
    state.register_pipeline(raster).unwrap();
    state.commit_pipeline(raster);
    state.event(StateEvent::DomainFailed(StateDomain::Compute));
    assert!(state.prepare_pipeline(raster).unwrap().program);
}
#[test]
fn current_program_invalidation_preserves_fixed_raster_skip_decisions() {
    let mut state = ContextState::new(ExecutionMode::Optimized);
    let raster = packet(1, 10);
    state.register_pipeline(raster).unwrap();
    state.commit_pipeline(raster);

    // Linking temporarily uses a program for reflection then leaves no program
    // selected. It changes the shared current-program register but none of the
    // already installed raster fixed-function leaves.
    state.event(StateEvent::DomainFailed(StateDomain::Compute));
    assert_eq!(
        state.prepare_pipeline(raster).unwrap(),
        RasterPipelineDiff {
            program: true,
            ..RasterPipelineDiff::default()
        }
    );
}
#[test]
fn bind_groups_flush_only_after_acknowledgement() {
    let mut state = ContextState::new(ExecutionMode::Optimized);
    state.stage_bind_group(BoundGroupPacket {
        group: BindGroupKey::new(stamp(ContextEpoch::INITIAL), 3, 1),
        index: 0,
        dynamic_offsets: vec![4],
        program_identity: CanonicalBlockId::new(1),
        dependencies: BTreeSet::new(),
    });
    let flush = state.binding_flush();
    assert_eq!(flush.groups.len(), 1);
    assert_eq!(state.binding_flush(), flush);
    state.acknowledge_bindings(&flush);
    assert!(state.binding_flush().is_empty());
}
#[test]
fn pass_begin_is_never_elided() {
    let mut state = ContextState::new(ExecutionMode::Optimized);
    let pass = PassPacket {
        draw_framebuffer: CanonicalBlockId::new(1),
        read_framebuffer: CanonicalBlockId::new(2),
        draw_buffers: CanonicalBlockId::new(3),
    };
    assert!(state.prepare_pass(pass));
    state.commit_pass(pass);
    assert!(state.prepare_pass(pass));
    assert!(state.pass.is_known());
}
#[test]
fn uniform_cache_key_includes_buffer_generation_and_range() {
    let mut state = ContextState::new(ExecutionMode::Optimized);
    let first = BufferId::new(stamp(ContextEpoch::INITIAL), 4, 1);
    let same_slot_new_generation = BufferId::new(stamp(ContextEpoch::INITIAL), 4, 2);
    let first_range = CanonicalBlockId::uniform_range(first, 256, 64);
    state.commit_uniform_slot(0, first_range);
    assert!(!state.prepare_uniform_slot(0, first_range));
    assert!(state.prepare_uniform_slot(0, CanonicalBlockId::uniform_range(first, 320, 64)));
    assert!(state.prepare_uniform_slot(0, CanonicalBlockId::uniform_range(first, 256, 128)));
    assert!(state.prepare_uniform_slot(
        0,
        CanonicalBlockId::uniform_range(same_slot_new_generation, 256, 64)
    ));
}
#[test]
fn sampler_cache_is_unit_scoped_and_generation_safe() {
    let mut state = ContextState::new(ExecutionMode::Optimized);
    let first = SamplerId::new(stamp(ContextEpoch::INITIAL), 5, 1);
    let recycled = SamplerId::new(stamp(ContextEpoch::INITIAL), 5, 2);
    let first_key = CanonicalBlockId::object(first);
    state.commit_sampler_slot(3, first_key);
    assert!(
        !state.prepare_sampler_slot(3, first_key),
        "identical sampler/unit is the only bind elision"
    );
    assert!(state.prepare_sampler_slot(4, first_key));
    assert!(state.prepare_sampler_slot(3, CanonicalBlockId::object(recycled)));
    state.event(StateEvent::SamplerRetired(first));
    assert!(state.prepare_sampler_slot(3, first_key));
}
#[test]
fn raw_binding_access_forces_sampler_rebind() {
    let mut state = ContextState::new(ExecutionMode::Optimized);
    let sampler = SamplerId::new(stamp(ContextEpoch::INITIAL), 5, 1);
    let key = CanonicalBlockId::object(sampler);
    state.commit_sampler_slot(3, key);
    state.event(StateEvent::DomainFailed(StateDomain::Bindings));
    assert!(state.prepare_sampler_slot(3, key));
}
#[test]
fn retirement_erases_cached_object_binding_and_dependent_group() {
    let mut state = ContextState::new(ExecutionMode::Optimized);
    let buffer = BufferId::new(stamp(ContextEpoch::INITIAL), 8, 3);
    let key = CanonicalBlockId::uniform_range(buffer, 0, 64);
    state.commit_uniform_slot(2, key);
    let mut dependencies = BTreeSet::new();
    dependencies.insert(ResourceRef::Buffer(buffer));
    state.stage_bind_group(BoundGroupPacket {
        group: BindGroupKey::new(stamp(ContextEpoch::INITIAL), 11, 1),
        index: 0,
        dynamic_offsets: vec![],
        program_identity: CanonicalBlockId::new(1),
        dependencies,
    });
    state.acknowledge_bindings(&state.binding_flush());
    state.event(StateEvent::BufferRetired(buffer));
    assert!(state.prepare_uniform_slot(2, key));
    assert!(state.binding_flush().is_empty());
}
