//! Immutable DXR refit shapes and ordered admission at native handoff.

use std::{collections::HashMap, sync::Arc};

use fluxel_rhi_core::api::{self, Error, Result};

use super::{Dx12, RecordedAccelerationBuild};

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
    let mut pending: HashMap<_, &RecordedAccelerationBuild> = HashMap::new();
    for build in builds {
        let identity = Arc::as_ptr(&build.target.0);
        if build.refit {
            let required = build.signature.as_ref().ok_or(Error::InvalidState)?;
            let compatible = match pending.get(&identity) {
                Some(previous) => previous.signature.as_deref() == Some(required.as_ref()),
                None => {
                    build
                        .target
                        .0
                        .committed_signature
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .as_deref()
                        == Some(required.as_ref())
                }
            };
            if !compatible {
                return Err(Error::InvalidState);
            }
        }
        if let Some((source, required)) = &build.required_source {
            let compatible = match pending.get(&Arc::as_ptr(&source.0)) {
                Some(previous) => previous
                    .receipt
                    .as_ref()
                    .is_some_and(|receipt| Arc::ptr_eq(&receipt.revision, required)),
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
        pending.insert(identity, build);
    }
    Ok(())
}
