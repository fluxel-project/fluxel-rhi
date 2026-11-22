//! Handoff publication and fence-scoped ownership of acceleration structures.

use std::{
    collections::HashSet,
    sync::{Arc, atomic::Ordering},
};
use windows::Win32::Graphics::Direct3D12::ID3D12Fence;

use super::{
    CopySource, Dx12Acceleration, committed, committed_signature, extend_unique, prepared_input,
};
use crate::backend::encoder::Dx12RecordedWork;

/// Pins the conservative dependency union for one native Execute batch.
///
/// `bound_groups` covers raster and compute inline ray queries as well as
/// dispatch rays: the backend cannot infer which shader instruction consumes
/// an acceleration structure, so every retained binding is a possible use.
pub(crate) fn pin_batch(work: &mut [Dx12RecordedWork]) {
    if work.is_empty() {
        return;
    }
    let mut dependencies = Vec::new();
    let mut known = None;
    let mut pin = |owners: Vec<_>| {
        if owners.is_empty() {
            return;
        }
        let known = known.get_or_insert_with(HashSet::new);
        dependencies.extend(
            owners
                .into_iter()
                .filter(|owner: &Dx12Acceleration| known.insert(Arc::as_ptr(&owner.0))),
        );
    };
    for record in work.iter() {
        for group in &record.encoder.bound_groups {
            for acceleration in group.0.accelerations() {
                if acceleration.0.is_top_level {
                    pin(committed(acceleration));
                }
            }
        }
        for secondary in &record.encoder.secondary {
            for group in &secondary.groups {
                for acceleration in group.0.accelerations() {
                    if acceleration.0.is_top_level {
                        pin(committed(acceleration));
                    }
                }
            }
        }
        for build in &record.encoder.ray_builds {
            if build.refit && build.target.0.is_top_level {
                pin(committed(&build.target));
            }
            pin(build.blas.clone());
            if let Some(CopySource::Clone(source)) = &build.source {
                if source.0.is_top_level {
                    pin(committed(source));
                }
            }
        }
    }
    // One record in this accepted batch owns the union through the enclosing
    // fence-retired `issued` list. This is bounded by that batch, never TLAS
    // history, and also protects an indeterminate handoff quarantine.
    extend_unique(&mut work[0].encoder.accelerations, dependencies);
}

/// Publishes just the builds whose native command-list segment was accepted.
pub(crate) fn publish_segment(work: &Dx12RecordedWork, segment: usize) {
    prepared_input::publish(&work.encoder.ray_inputs, segment);
    for build in work
        .encoder
        .ray_builds
        .iter()
        .filter(|build| build.segment == segment)
    {
        let (blas, signature) = match &build.source {
            Some(CopySource::Clone(source)) => (committed(source), committed_signature(source)),
            _ => (build.blas.clone(), build.signature.clone()),
        };
        if build.target.0.is_top_level {
            *build
                .target
                .0
                .committed_blas
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()) = blas;
        }
        *build
            .target
            .0
            .committed_signature
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = signature;
        *build
            .target
            .0
            .committed_revision
            .lock()
            .unwrap_or_else(|p| p.into_inner()) = build
            .receipt
            .as_ref()
            .map(|receipt| receipt.revision.clone());
        // Both publication and admission read this bit under the submission gate.
        build
            .target
            .0
            .committed_built
            .store(true, Ordering::Relaxed);
    }
}

pub(crate) fn seal_compaction_receipts(work: &Dx12RecordedWork, fence: &ID3D12Fence, value: u64) {
    for receipt in work
        .encoder
        .ray_builds
        .iter()
        .filter_map(|build| build.receipt.as_ref())
    {
        receipt.seal(fence, value);
    }
}
