//! Submission-bound ownership for mutable Vulkan TLAS storage.
//!
//! A Vulkan TLAS stores BLAS device addresses in native memory.  `committed`
//! therefore changes only after the exact command-buffer segment that writes
//! those addresses was accepted by `vkQueueSubmit`.  The per-submit pin is
//! conservative: any acceleration descriptor retained by a primary or
//! secondary command can be consumed by an inline ray query.

use super::VulkanAcceleration;
use std::sync::Arc;

/// Metadata for one recorded TLAS build.  This is ownership publication
/// metadata, never an interpreted GPU command for replay.
pub(crate) struct RecordedTlasBuild {
    pub(crate) target: VulkanAcceleration,
    pub(crate) blas: Vec<VulkanAcceleration>,
    pub(crate) segment: usize,
}

pub(crate) fn extend_unique(
    retained: &mut Vec<VulkanAcceleration>,
    incoming: impl IntoIterator<Item = VulkanAcceleration>,
) {
    for value in incoming {
        if !retained
            .iter()
            .any(|existing| Arc::ptr_eq(&existing.0, &value.0))
        {
            retained.push(value);
        }
    }
}

pub(crate) fn committed(acceleration: &VulkanAcceleration) -> Vec<VulkanAcceleration> {
    acceleration
        .0
        .committed_blas
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone()
}

/// Publishes only builds in a native command-buffer segment accepted by Vulkan.
///
/// The caller invokes this strictly after the corresponding `queue_submit`
/// returned success.  Cancelled recorded work and failed/uncertain submission
/// keep the previously accepted set unchanged.
pub(crate) fn publish_segment(work: &super::super::submit::RecordedWork, segment: usize) {
    for build in work
        .encoder
        .ray_builds
        .iter()
        .filter(|build| build.segment == segment)
    {
        let mut committed = build
            .target
            .0
            .committed_blas
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        *committed = build.blas.clone();
    }
}

/// Pins a batch-sized union in one fence-retired recorded work item.
///
/// `groups` and secondary-group accessors are intentionally used rather than
/// trying to inspect shader instructions: any bound acceleration descriptor
/// may be read by raster, compute, ray-pipeline, or secondary inline queries.
pub(crate) fn pin_batch(work: &mut [super::super::submit::RecordedWork]) {
    if work.is_empty() {
        return;
    }
    let mut dependencies = Vec::new();
    for record in work.iter() {
        for group in &record.encoder.groups {
            for acceleration in group.accelerations() {
                if acceleration.0.top {
                    extend_unique(&mut dependencies, committed(acceleration));
                }
            }
        }
        for secondary in &record.encoder.secondary {
            for group in secondary.groups() {
                for acceleration in group.accelerations() {
                    if acceleration.0.top {
                        extend_unique(&mut dependencies, committed(acceleration));
                    }
                }
            }
        }
        for build in &record.encoder.ray_builds {
            extend_unique(&mut dependencies, build.blas.clone());
        }
    }
    // This encoder is retained by the terminal fence (or leaked for an
    // indeterminate handoff), bounding the union to this native batch rather
    // than accumulating it in TLAS state forever.
    extend_unique(&mut work[0].encoder.accelerations, dependencies);
}
