//! Low-overhead, device-scoped counters for accepted RHI operations.
//!
//! Counters describe successful API requests. They do not imply a native
//! state change, GPU completion, or a displayed frame.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

/// Shared counters exposed by [`Device::stats`].
///
/// Shared updates use relaxed atomics and never take a lock. Command encoders
/// accumulate locally, then merge on finish/drop to avoid contended atomics in
/// per-command paths. Read a copy with [`snapshot`](Self::snapshot).
pub struct RhiStats {
    surface_acquires: AtomicU64,
    presents_accepted: AtomicU64,
    draw: AtomicU64,
    draw_indexed: AtomicU64,
    dispatch: AtomicU64,
    pipeline_binds: AtomicU64,
    bind_group_binds: AtomicU64,
    vertex_buffer_binds: AtomicU64,
    index_buffer_binds: AtomicU64,
    push_constant_writes: AtomicU64,
    viewport_sets: AtomicU64,
    scissor_sets: AtomicU64,
    blend_constant_sets: AtomicU64,
    stencil_reference_sets: AtomicU64,
    ray_traces: AtomicU64,
    copies: AtomicU64,
    blits: AtomicU64,
    clears: AtomicU64,
    writes: AtomicU64,
    submits: AtomicU64,
}

impl RhiStats {
    pub(crate) fn new() -> Self {
        Self {
            surface_acquires: AtomicU64::new(0),
            presents_accepted: AtomicU64::new(0),
            draw: AtomicU64::new(0),
            draw_indexed: AtomicU64::new(0),
            dispatch: AtomicU64::new(0),
            pipeline_binds: AtomicU64::new(0),
            bind_group_binds: AtomicU64::new(0),
            vertex_buffer_binds: AtomicU64::new(0),
            index_buffer_binds: AtomicU64::new(0),
            push_constant_writes: AtomicU64::new(0),
            viewport_sets: AtomicU64::new(0),
            scissor_sets: AtomicU64::new(0),
            blend_constant_sets: AtomicU64::new(0),
            stencil_reference_sets: AtomicU64::new(0),
            ray_traces: AtomicU64::new(0),
            copies: AtomicU64::new(0),
            blits: AtomicU64::new(0),
            clears: AtomicU64::new(0),
            writes: AtomicU64::new(0),
            submits: AtomicU64::new(0),
        }
    }

    pub(crate) fn local(self: &std::sync::Arc<Self>) -> RhiStatsLocal {
        RhiStatsLocal {
            shared: self.clone(),
            counts: RhiStatsSnapshot::default(),
        }
    }

    pub(crate) fn record_surface_acquire(&self) {
        self.surface_acquires.fetch_add(1, Ordering::Relaxed);
    }
    pub(crate) fn record_present_accepted(&self) {
        self.presents_accepted.fetch_add(1, Ordering::Relaxed);
    }
    pub(crate) fn record_draw(&self) {
        self.draw.fetch_add(1, Ordering::Relaxed);
    }
    pub(crate) fn record_draw_indexed(&self) {
        self.draw_indexed.fetch_add(1, Ordering::Relaxed);
    }
    pub(crate) fn record_dispatch(&self) {
        self.dispatch.fetch_add(1, Ordering::Relaxed);
    }
    pub(crate) fn record_pipeline_bind(&self) {
        self.pipeline_binds.fetch_add(1, Ordering::Relaxed);
    }
    pub(crate) fn record_bind_group_bind(&self) {
        self.bind_group_binds.fetch_add(1, Ordering::Relaxed);
    }
    pub(crate) fn record_vertex_buffer_bind(&self) {
        self.vertex_buffer_binds.fetch_add(1, Ordering::Relaxed);
    }
    pub(crate) fn record_index_buffer_bind(&self) {
        self.index_buffer_binds.fetch_add(1, Ordering::Relaxed);
    }
    pub(crate) fn record_push_constant_write(&self) {
        self.push_constant_writes.fetch_add(1, Ordering::Relaxed);
    }
    pub(crate) fn record_viewport_set(&self) {
        self.viewport_sets.fetch_add(1, Ordering::Relaxed);
    }
    pub(crate) fn record_scissor_set(&self) {
        self.scissor_sets.fetch_add(1, Ordering::Relaxed);
    }
    pub(crate) fn record_blend_constant_set(&self) {
        self.blend_constant_sets.fetch_add(1, Ordering::Relaxed);
    }
    pub(crate) fn record_stencil_reference_set(&self) {
        self.stencil_reference_sets.fetch_add(1, Ordering::Relaxed);
    }
    pub(crate) fn record_ray_trace(&self) {
        self.ray_traces.fetch_add(1, Ordering::Relaxed);
    }
    pub(crate) fn record_copy(&self) {
        self.copies.fetch_add(1, Ordering::Relaxed);
    }
    pub(crate) fn record_blit(&self) {
        self.blits.fetch_add(1, Ordering::Relaxed);
    }
    pub(crate) fn record_clear(&self) {
        self.clears.fetch_add(1, Ordering::Relaxed);
    }
    pub(crate) fn record_write(&self) {
        self.writes.fetch_add(1, Ordering::Relaxed);
    }
    pub(crate) fn record_submit(&self) {
        self.submits.fetch_add(1, Ordering::Relaxed);
    }

    /// Reads published counters without locking or querying the GPU.
    ///
    /// Counters held locally by currently live command encoders become visible
    /// when those encoders finish or are dropped.
    pub fn snapshot(&self) -> RhiStatsSnapshot {
        let read = |counter: &AtomicU64| counter.load(Ordering::Relaxed);
        RhiStatsSnapshot {
            surface_acquires: read(&self.surface_acquires),
            presents_accepted: read(&self.presents_accepted),
            draw: read(&self.draw),
            draw_indexed: read(&self.draw_indexed),
            dispatch: read(&self.dispatch),
            pipeline_binds: read(&self.pipeline_binds),
            bind_group_binds: read(&self.bind_group_binds),
            vertex_buffer_binds: read(&self.vertex_buffer_binds),
            index_buffer_binds: read(&self.index_buffer_binds),
            push_constant_writes: read(&self.push_constant_writes),
            viewport_sets: read(&self.viewport_sets),
            scissor_sets: read(&self.scissor_sets),
            blend_constant_sets: read(&self.blend_constant_sets),
            stencil_reference_sets: read(&self.stencil_reference_sets),
            ray_traces: read(&self.ray_traces),
            copies: read(&self.copies),
            blits: read(&self.blits),
            clears: read(&self.clears),
            writes: read(&self.writes),
            submits: read(&self.submits),
        }
    }
}

/// Per-encoder counters. Operations update plain integers; one atomic merge
/// per nonzero counter happens when the encoder is finished or abandoned.
pub(crate) struct RhiStatsLocal {
    shared: std::sync::Arc<RhiStats>,
    counts: RhiStatsSnapshot,
}

impl RhiStatsLocal {
    pub(crate) fn record_draw(&mut self) {
        self.counts.draw += 1;
    }
    pub(crate) fn record_draw_indexed(&mut self) {
        self.counts.draw_indexed += 1;
    }
    pub(crate) fn record_dispatch(&mut self) {
        self.counts.dispatch += 1;
    }
    pub(crate) fn record_pipeline_bind(&mut self) {
        self.counts.pipeline_binds += 1;
    }
    pub(crate) fn record_bind_group_bind(&mut self) {
        self.counts.bind_group_binds += 1;
    }
    pub(crate) fn record_vertex_buffer_bind(&mut self) {
        self.counts.vertex_buffer_binds += 1;
    }
    pub(crate) fn record_index_buffer_bind(&mut self) {
        self.counts.index_buffer_binds += 1;
    }
    pub(crate) fn record_push_constant_write(&mut self) {
        self.counts.push_constant_writes += 1;
    }
    pub(crate) fn record_viewport_set(&mut self) {
        self.counts.viewport_sets += 1;
    }
    pub(crate) fn record_scissor_set(&mut self) {
        self.counts.scissor_sets += 1;
    }
    pub(crate) fn record_blend_constant_set(&mut self) {
        self.counts.blend_constant_sets += 1;
    }
    pub(crate) fn record_stencil_reference_set(&mut self) {
        self.counts.stencil_reference_sets += 1;
    }
    pub(crate) fn record_ray_trace(&mut self) {
        self.counts.ray_traces += 1;
    }
    pub(crate) fn record_copy(&mut self) {
        self.counts.copies += 1;
    }
    pub(crate) fn record_blit(&mut self) {
        self.counts.blits += 1;
    }
    pub(crate) fn record_clear(&mut self) {
        self.counts.clears += 1;
    }
    pub(crate) fn record_write(&mut self) {
        self.counts.writes += 1;
    }
}

impl Drop for RhiStatsLocal {
    fn drop(&mut self) {
        macro_rules! merge {
            ($field:ident) => {
                if self.counts.$field != 0 {
                    self.shared
                        .$field
                        .fetch_add(self.counts.$field, Ordering::Relaxed);
                }
            };
        }
        merge!(draw);
        merge!(draw_indexed);
        merge!(dispatch);
        merge!(pipeline_binds);
        merge!(bind_group_binds);
        merge!(vertex_buffer_binds);
        merge!(index_buffer_binds);
        merge!(push_constant_writes);
        merge!(viewport_sets);
        merge!(scissor_sets);
        merge!(blend_constant_sets);
        merge!(stencil_reference_sets);
        merge!(ray_traces);
        merge!(copies);
        merge!(blits);
        merge!(clears);
        merge!(writes);
    }
}

/// Copy of cumulative RHI request counters at one observation point.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RhiStatsSnapshot {
    /// Successfully acquired surface images.
    pub surface_acquires: u64,
    /// Surface images whose native present handoff was accepted.
    pub presents_accepted: u64,
    /// Accepted non-indexed draw requests, including secondary/indirect routes where observable.
    pub draw: u64,
    /// Accepted indexed draw requests.
    pub draw_indexed: u64,
    /// Accepted dispatch requests.
    pub dispatch: u64,
    /// Accepted pipeline bind requests. Repeated identical binds count again.
    pub pipeline_binds: u64,
    /// Accepted bind-group requests. Repeated identical binds count again.
    pub bind_group_binds: u64,
    /// Accepted vertex-buffer bind requests.
    pub vertex_buffer_binds: u64,
    /// Accepted index-buffer bind requests.
    pub index_buffer_binds: u64,
    /// Accepted push-constant update requests.
    pub push_constant_writes: u64,
    /// Accepted viewport requests.
    pub viewport_sets: u64,
    /// Accepted scissor requests.
    pub scissor_sets: u64,
    /// Accepted blend-constant requests.
    pub blend_constant_sets: u64,
    /// Accepted stencil-reference requests.
    pub stencil_reference_sets: u64,
    /// Accepted ray-tracing dispatch requests.
    pub ray_traces: u64,
    /// Accepted copy requests.
    pub copies: u64,
    /// Accepted blit requests.
    pub blits: u64,
    /// Accepted buffer or texture clear requests.
    pub clears: u64,
    /// Accepted buffer or texture write requests.
    pub writes: u64,
    /// Fully or partially accepted deferred submissions.
    pub submits: u64,
}

impl RhiStatsSnapshot {
    /// Returns the counter differences since an earlier snapshot.
    ///
    /// `None` indicates that a counter wrapped or the snapshots were reversed.
    pub fn delta_since(self, earlier: Self) -> Option<Self> {
        Some(Self {
            surface_acquires: self
                .surface_acquires
                .checked_sub(earlier.surface_acquires)?,
            presents_accepted: self
                .presents_accepted
                .checked_sub(earlier.presents_accepted)?,
            draw: self.draw.checked_sub(earlier.draw)?,
            draw_indexed: self.draw_indexed.checked_sub(earlier.draw_indexed)?,
            dispatch: self.dispatch.checked_sub(earlier.dispatch)?,
            pipeline_binds: self.pipeline_binds.checked_sub(earlier.pipeline_binds)?,
            bind_group_binds: self
                .bind_group_binds
                .checked_sub(earlier.bind_group_binds)?,
            vertex_buffer_binds: self
                .vertex_buffer_binds
                .checked_sub(earlier.vertex_buffer_binds)?,
            index_buffer_binds: self
                .index_buffer_binds
                .checked_sub(earlier.index_buffer_binds)?,
            push_constant_writes: self
                .push_constant_writes
                .checked_sub(earlier.push_constant_writes)?,
            viewport_sets: self.viewport_sets.checked_sub(earlier.viewport_sets)?,
            scissor_sets: self.scissor_sets.checked_sub(earlier.scissor_sets)?,
            blend_constant_sets: self
                .blend_constant_sets
                .checked_sub(earlier.blend_constant_sets)?,
            stencil_reference_sets: self
                .stencil_reference_sets
                .checked_sub(earlier.stencil_reference_sets)?,
            ray_traces: self.ray_traces.checked_sub(earlier.ray_traces)?,
            copies: self.copies.checked_sub(earlier.copies)?,
            blits: self.blits.checked_sub(earlier.blits)?,
            clears: self.clears.checked_sub(earlier.clears)?,
            writes: self.writes.checked_sub(earlier.writes)?,
            submits: self.submits.checked_sub(earlier.submits)?,
        })
    }

    /// Computes accepted present cadence between snapshots.
    ///
    /// The duration comes from the caller's monotonic clock. `None` means the
    /// snapshots were reversed or elapsed time is zero. This is not display
    /// refresh rate or GPU frame time.
    pub fn present_rate_hz_since(self, earlier: Self, elapsed: Duration) -> Option<f64> {
        let frames = self
            .presents_accepted
            .checked_sub(earlier.presents_accepted)?;
        let seconds = elapsed.as_secs_f64();
        (seconds > 0.0).then_some(frames as f64 / seconds)
    }
}

#[cfg(test)]
#[path = "../tests/api/stats.rs"]
mod tests;
