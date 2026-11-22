//! Immutable DXR refit shapes and ordered admission at native handoff.

use std::{
    collections::HashMap,
    sync::{Arc, atomic::Ordering},
};

use fluxel_rhi_core::api::{self, Error, Result};

use super::{CopySource, Dx12, RecordedAccelerationBuild, committed_signature};

#[derive(Eq, PartialEq)]
pub(super) struct RefitSignature {
    fast_trace: bool,
    compactable: bool,
    shape: Shape,
}

#[derive(Eq, PartialEq)]
enum Shape {
    Tlas(usize),
    Blas(Vec<GeometryShape>),
}

#[derive(Eq, PartialEq)]
enum GeometryShape {
    Triangles {
        vertex_count: u32,
        vertex_format: api::VertexFormat,
        stride: u64,
        opaque: bool,
        transformed: bool,
    },
    Aabbs {
        count: u32,
        stride: u64,
        opaque: bool,
    },
}

impl RefitSignature {
    pub(super) fn from_desc(
        desc: &api::AccelerationDesc<Dx12>,
        compactable: bool,
    ) -> Option<Arc<Self>> {
        if !desc.allow_update {
            return None;
        }
        let shape = match &desc.input {
            api::AccelerationInput::Tlas(instances) => Shape::Tlas(instances.len()),
            api::AccelerationInput::Blas(geometries) => Shape::Blas(
                geometries
                    .iter()
                    .map(GeometryShape::from_geometry)
                    .collect::<Option<Vec<_>>>()?,
            ),
        };
        Some(Arc::new(Self {
            fast_trace: desc.fast_trace,
            compactable,
            shape,
        }))
    }
}

impl GeometryShape {
    fn from_geometry(geometry: &api::AccelerationGeometry<Dx12>) -> Option<Self> {
        match geometry {
            api::AccelerationGeometry::Triangles {
                vertex_count,
                vertex_format,
                stride,
                indices: None,
                opaque,
                ..
            }
            | api::AccelerationGeometry::TransformedTriangles {
                vertex_count,
                vertex_format,
                stride,
                indices: None,
                opaque,
                ..
            } => Some(Self::Triangles {
                vertex_count: *vertex_count,
                vertex_format: *vertex_format,
                stride: *stride,
                opaque: *opaque,
                transformed: matches!(
                    geometry,
                    api::AccelerationGeometry::TransformedTriangles { .. }
                ),
            }),
            api::AccelerationGeometry::Triangles { .. }
            | api::AccelerationGeometry::TransformedTriangles { .. } => None,
            api::AccelerationGeometry::Aabbs {
                stride,
                count,
                opaque,
                ..
            } => Some(Self::Aabbs {
                count: *count,
                stride: *stride,
                opaque: *opaque,
            }),
        }
    }
}

/// Checks source identity in recorded order while the submission gate is
/// held. These signatures model build compatibility only. Queue waits and
/// UAV barriers must still come from the caller's plan.
pub(crate) fn validate_sequence<'a>(
    builds: impl IntoIterator<Item = &'a RecordedAccelerationBuild>,
) -> Result<()> {
    struct PendingBuild<'a> {
        signature: Option<Arc<RefitSignature>>,
        revision: Option<&'a Arc<()>>,
    }
    let mut pending = HashMap::<_, PendingBuild<'_>>::new();
    for build in builds {
        let identity = Arc::as_ptr(&build.target.0);
        let mut signature = build.signature.clone();
        if build.refit {
            let required = build.signature.as_ref().ok_or(Error::InvalidState)?;
            let compatible = match pending.get(&identity) {
                Some(previous) => previous.signature.as_deref() == Some(required.as_ref()),
                None => committed_signature(&build.target).as_deref() == Some(required.as_ref()),
            };
            if !compatible {
                return Err(Error::InvalidState);
            }
        }
        match &build.source {
            Some(CopySource::Clone(source)) => {
                signature = match pending.get(&Arc::as_ptr(&source.0)) {
                    Some(previous) => previous.signature.clone(),
                    None if source.0.committed_built.load(Ordering::Relaxed) => {
                        committed_signature(source)
                    }
                    None => return Err(Error::InvalidState),
                };
            }
            Some(CopySource::Compaction(source, required)) => {
                let compatible = match pending.get(&Arc::as_ptr(&source.0)) {
                    Some(previous) => previous
                        .revision
                        .is_some_and(|revision| Arc::ptr_eq(revision, required)),
                    None => source
                        .0
                        .committed_revision
                        .lock()
                        .unwrap_or_else(|p| p.into_inner())
                        .as_ref()
                        .is_some_and(|actual| Arc::ptr_eq(actual, required)),
                };
                if !compatible {
                    return Err(Error::InvalidState);
                }
            }
            None => {}
        }
        pending.insert(
            identity,
            PendingBuild {
                signature,
                revision: build.receipt.as_ref().map(|receipt| &receipt.revision),
            },
        );
    }
    Ok(())
}
