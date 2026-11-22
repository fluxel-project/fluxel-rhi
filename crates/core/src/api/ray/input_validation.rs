//! Core provenance and shape checks for acceleration-structure inputs.

use super::*;

impl<B: Backend> AccelerationInput<B> {
    pub(super) fn validate(&self, device: DeviceId, limits: &DeviceLimits) -> Result<()> {
        let foreign = match self {
            Self::Blas(geometries) => geometries.iter().any(|geometry| match geometry {
                AccelerationGeometry::Triangles {
                    vertices, indices, ..
                } | AccelerationGeometry::TransformedTriangles { vertices, indices, .. } => {
                    vertices.device != device
                        || indices
                            .as_ref()
                            .is_some_and(|indices| indices.buffer.device != device)
                        || matches!(geometry, AccelerationGeometry::TransformedTriangles { transform, .. } if transform.device != device)
                }
                AccelerationGeometry::SnapshotTriangles { vertices, indices, transform, .. } => {
                    vertices.device != device
                        || indices.device != device
                        || transform.as_ref().is_some_and(|(buffer, _)| buffer.device != device)
                }
                AccelerationGeometry::Aabbs { data, .. } => data.device != device,
            }),
            Self::Tlas(instances) => instances
                .iter()
                .any(|instance| instance.blas.device != device),
            Self::PackedTlas(instances) => instances.device != device,
        };
        if foreign {
            return Err(Error::WrongDevice);
        }
        if let Self::Tlas(instances) = self {
            if limits.max_tlas_instance_count != 0
                && instances.len() > limits.max_tlas_instance_count as usize
            {
                return Err(Error::InvalidRange);
            }
            if instances.iter().any(|instance| !instance.flags.valid()) {
                return Err(Error::InvalidState);
            }
        }
        if let Self::PackedTlas(instances) = self {
            validate_tlas_count(instances.count(), limits.max_tlas_instance_count)?;
        }
        if let Self::Blas(geometries) = self {
            if limits.max_blas_geometry_count != 0
                && geometries.len() > limits.max_blas_geometry_count as usize
            {
                return Err(Error::InvalidRange);
            }
            let mut primitive_count = 0u64;
            for geometry in geometries {
                if limits.max_blas_primitive_count != 0 {
                    let count = match geometry {
                        AccelerationGeometry::Triangles {
                            vertex_count,
                            indices,
                            ..
                        }
                        | AccelerationGeometry::TransformedTriangles {
                            vertex_count,
                            indices,
                            ..
                        } => {
                            indices
                                .as_ref()
                                .map_or(*vertex_count, |indices| indices.count)
                                / 3
                        }
                        AccelerationGeometry::Aabbs { count, .. } => *count,
                        AccelerationGeometry::SnapshotTriangles { indices, .. } => {
                            indices.count() / 3
                        }
                    };
                    primitive_count += u64::from(count);
                    if primitive_count > u64::from(limits.max_blas_primitive_count) {
                        return Err(Error::InvalidRange);
                    }
                }
                let valid = match geometry {
                    AccelerationGeometry::Triangles {
                        vertex_offset,
                        vertex_count,
                        stride,
                        ..
                    }
                    | AccelerationGeometry::TransformedTriangles {
                        vertex_offset,
                        vertex_count,
                        stride,
                        ..
                    }
                    | AccelerationGeometry::SnapshotTriangles {
                        vertex_offset,
                        vertex_count,
                        stride,
                        ..
                    } => {
                        *vertex_count != 0
                            && *stride != 0
                            && u64::from(*vertex_count - 1)
                                .checked_mul(*stride)
                                .and_then(|n| vertex_offset.checked_add(n))
                                .is_some()
                    }
                    AccelerationGeometry::Aabbs {
                        offset,
                        stride,
                        count,
                        ..
                    } => {
                        *count != 0
                            && *stride >= 24
                            && *stride % 8 == 0
                            && *offset % 8 == 0
                            && u64::from(*count - 1)
                                .checked_mul(*stride)
                                .and_then(|n| offset.checked_add(n))
                                .and_then(|n| n.checked_add(24))
                                .is_some()
                    }
                };
                if !valid {
                    return Err(Error::InvalidRange);
                }
                if matches!(geometry,
                    AccelerationGeometry::Triangles { indices: Some(indices), .. }
                    | AccelerationGeometry::TransformedTriangles { indices: Some(indices), .. }
                    if indices.byte_range().is_none())
                {
                    return Err(Error::InvalidRange);
                }
                let transform_offset = match geometry {
                    AccelerationGeometry::TransformedTriangles {
                        transform_offset, ..
                    } => Some(*transform_offset),
                    AccelerationGeometry::SnapshotTriangles { transform, .. } => {
                        transform.as_ref().map(|(_, offset)| *offset)
                    }
                    _ => None,
                };
                if transform_offset
                    .is_some_and(|offset| offset % 16 != 0 || offset.checked_add(48).is_none())
                {
                    return Err(Error::InvalidRange);
                }
            }
        }
        Ok(())
    }
}
