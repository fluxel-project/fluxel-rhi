//! Public snapshot and interval-rate semantics for device RHI counters.
use super::*;
use std::sync::Arc;

#[test]
fn snapshots_are_cumulative_and_present_rate_uses_caller_interval() {
    let stats = RhiStats::new();
    let before = stats.snapshot();
    stats.record_surface_acquire();
    stats.record_present_accepted();
    stats.record_present_accepted();
    stats.record_pipeline_bind();
    stats.record_draw();
    let after = stats.snapshot();

    assert_eq!(before.surface_acquires, 0);
    assert_eq!(after.surface_acquires, 1);
    assert_eq!(after.presents_accepted, 2);
    assert_eq!(after.pipeline_binds, 1);
    assert_eq!(after.draw, 1);
    let delta = after.delta_since(before).unwrap();
    assert_eq!(delta.presents_accepted, 2);
    assert_eq!(delta.draw, 1);
    assert!(before.delta_since(after).is_none());
    assert_eq!(
        after.present_rate_hz_since(before, Duration::from_millis(40)),
        Some(50.0)
    );
    assert_eq!(after.present_rate_hz_since(before, Duration::ZERO), None);
    assert_eq!(
        before.present_rate_hz_since(after, Duration::from_secs(1)),
        None
    );
}

#[test]
fn every_counter_is_independently_observable() {
    let stats = RhiStats::new();
    stats.record_draw_indexed();
    stats.record_dispatch();
    stats.record_bind_group_bind();
    stats.record_vertex_buffer_bind();
    stats.record_index_buffer_bind();
    stats.record_push_constant_write();
    stats.record_viewport_set();
    stats.record_scissor_set();
    stats.record_blend_constant_set();
    stats.record_stencil_reference_set();
    stats.record_ray_trace();
    stats.record_copy();
    stats.record_blit();
    stats.record_clear();
    stats.record_write();
    stats.record_submit();
    let snapshot = stats.snapshot();
    assert_eq!(snapshot.draw_indexed, 1);
    assert_eq!(snapshot.dispatch, 1);
    assert_eq!(snapshot.bind_group_binds, 1);
    assert_eq!(snapshot.vertex_buffer_binds, 1);
    assert_eq!(snapshot.index_buffer_binds, 1);
    assert_eq!(snapshot.push_constant_writes, 1);
    assert_eq!(snapshot.viewport_sets, 1);
    assert_eq!(snapshot.scissor_sets, 1);
    assert_eq!(snapshot.blend_constant_sets, 1);
    assert_eq!(snapshot.stencil_reference_sets, 1);
    assert_eq!(snapshot.ray_traces, 1);
    assert_eq!(snapshot.copies, 1);
    assert_eq!(snapshot.blits, 1);
    assert_eq!(snapshot.clears, 1);
    assert_eq!(snapshot.writes, 1);
    assert_eq!(snapshot.submits, 1);
}

#[test]
fn encoder_local_counts_merge_only_when_the_encoder_ends() {
    let stats = Arc::new(RhiStats::new());
    let mut local = stats.local();
    local.record_draw();
    local.record_draw();
    assert_eq!(stats.snapshot().draw, 0);
    drop(local);
    assert_eq!(stats.snapshot().draw, 2);
}
