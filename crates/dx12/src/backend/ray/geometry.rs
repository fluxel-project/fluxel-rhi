//! Validation and lowering of public acceleration-build inputs to DXR inputs.
//!
//! The returned geometry array owns the memory referenced by `inputs` until
//! the synchronous prebuild or build call has returned. Buffer lifetimes are
//! retained separately by the command encoder when a build is recorded.
use fluxel_rhi_core::api::{self, Error, Result};
use windows::Win32::Graphics::{Direct3D12::*, Dxgi::Common::*};

use super::super::{Dx12, Dx12Buffer};

pub(super) struct GeometryInputs {
    geometries: Vec<D3D12_RAYTRACING_GEOMETRY_DESC>,
    pub(super) buffers: Vec<Dx12Buffer>,
    kind: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE,
    count: u32,
    flags: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS,
}

fn validate_address(
    base: u64,
    size: u64,
    offset: u64,
    count: u32,
    stride: u64,
    item: u64,
    alignment: u64,
) -> Result<u64> {
    if count == 0 || stride < item || stride > u64::from(u32::MAX) || stride % alignment != 0 {
        return Err(Error::InvalidRange);
    }
    let end = u64::from(count - 1)
        .checked_mul(stride)
        .and_then(|n| n.checked_add(item))
        .and_then(|n| n.checked_add(offset))
        .ok_or(Error::InvalidRange)?;
    let address = base.checked_add(offset).ok_or(Error::InvalidRange)?;
    if end > size || address % alignment != 0 {
        return Err(Error::InvalidRange);
    }
    Ok(address)
}

fn address(
    buffer: &Dx12Buffer,
    offset: u64,
    count: u32,
    stride: u64,
    item: u64,
    alignment: u64,
) -> Result<u64> {
    if !buffer.usage.contains(api::BufferUsage::ACCEL_BUILD_INPUT) || count == 0 || stride < item {
        return Err(Error::InvalidRange);
    }
    // SAFETY: the borrowed buffer retains a valid native resource throughout
    // this call; its GPU virtual address is stable for its lifetime.
    validate_address(
        unsafe { buffer.resource.GetGPUVirtualAddress() },
        buffer.size,
        offset,
        count,
        stride,
        item,
        alignment,
    )
}

impl GeometryInputs {
    pub(super) fn new(desc: &api::AccelerationDesc<Dx12>) -> Result<Self> {
        let mut buffers = Vec::new();
        let mut geometries = Vec::new();
        let (kind, count) = match &desc.input {
            api::AccelerationInput::Tlas(instances) => (
                D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL,
                u32::try_from(instances.len()).map_err(|_| Error::InvalidRange)?,
            ),
            api::AccelerationInput::Blas(items) => {
                for item in items {
                    let (geometry, opaque) = match item {
                        api::AccelerationGeometry::Triangles {
                            vertices,
                            vertex_offset,
                            vertex_count,
                            vertex_format,
                            stride,
                            indices,
                            opaque,
                        } => {
                            let (format, bytes, alignment) = match vertex_format {
                                api::VertexFormat::Float32(3) => {
                                    (DXGI_FORMAT_R32G32B32_FLOAT, 12, 4)
                                }
                                api::VertexFormat::Float32(2) => (DXGI_FORMAT_R32G32_FLOAT, 8, 4),
                                api::VertexFormat::Float16(4) => {
                                    (DXGI_FORMAT_R16G16B16A16_FLOAT, 8, 2)
                                }
                                api::VertexFormat::Float16(2) => (DXGI_FORMAT_R16G16_FLOAT, 4, 2),
                                _ => {
                                    return Err(Error::Unsupported(
                                        "DX12 ray triangle vertex format",
                                    ));
                                }
                            };
                            if vertex_offset % alignment != 0
                                || stride % alignment != 0
                                || indices.as_ref().map_or(*vertex_count, |i| i.count) % 3 != 0
                            {
                                return Err(Error::InvalidRange);
                            }
                            // SAFETY: Core checks resource identity and the
                            // native reference is borrowed only during lowering.
                            let vb = unsafe { vertices.native_ref() };
                            let vertex_address = address(
                                vb,
                                *vertex_offset,
                                *vertex_count,
                                *stride,
                                bytes,
                                alignment,
                            )?;
                            buffers.push(vb.clone());
                            let (index_format, index_count, index_address) =
                                if let Some(indices) = indices {
                                    let (format, bytes) = match indices.format {
                                        api::IndexFormat::Uint16 => (DXGI_FORMAT_R16_UINT, 2),
                                        api::IndexFormat::Uint32 => (DXGI_FORMAT_R32_UINT, 4),
                                    };
                                    // SAFETY: same Core identity invariant as the vertex buffer.
                                    let ib = unsafe { indices.buffer.native_ref() };
                                    let addr = address(
                                        ib,
                                        indices.offset,
                                        indices.count,
                                        bytes,
                                        bytes,
                                        bytes,
                                    )?;
                                    buffers.push(ib.clone());
                                    (format, indices.count, addr)
                                } else {
                                    (DXGI_FORMAT_UNKNOWN, 0, 0)
                                };
                            (
                                D3D12_RAYTRACING_GEOMETRY_DESC_0 {
                                    Triangles: D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC {
                                        Transform3x4: 0,
                                        IndexFormat: index_format,
                                        VertexFormat: format,
                                        IndexCount: index_count,
                                        VertexCount: *vertex_count,
                                        IndexBuffer: index_address,
                                        VertexBuffer: D3D12_GPU_VIRTUAL_ADDRESS_AND_STRIDE {
                                            StartAddress: vertex_address,
                                            StrideInBytes: *stride,
                                        },
                                    },
                                },
                                *opaque,
                            )
                        }
                        api::AccelerationGeometry::Aabbs {
                            data,
                            offset,
                            stride,
                            count,
                            opaque,
                        } => {
                            // SAFETY: Core checked device identity for the input.
                            let buffer = unsafe { data.native_ref() };
                            let start = address(buffer, *offset, *count, *stride, 24, 8)?;
                            buffers.push(buffer.clone());
                            (
                                D3D12_RAYTRACING_GEOMETRY_DESC_0 {
                                    AABBs: D3D12_RAYTRACING_GEOMETRY_AABBS_DESC {
                                        AABBCount: u64::from(*count),
                                        AABBs: D3D12_GPU_VIRTUAL_ADDRESS_AND_STRIDE {
                                            StartAddress: start,
                                            StrideInBytes: *stride,
                                        },
                                    },
                                },
                                *opaque,
                            )
                        }
                    };
                    geometries.push(D3D12_RAYTRACING_GEOMETRY_DESC {
                        Type: match item {
                            api::AccelerationGeometry::Triangles { .. } => {
                                D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES
                            }
                            _ => D3D12_RAYTRACING_GEOMETRY_TYPE_PROCEDURAL_PRIMITIVE_AABBS,
                        },
                        Flags: if opaque {
                            D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE
                        } else {
                            D3D12_RAYTRACING_GEOMETRY_FLAG_NONE
                        },
                        Anonymous: geometry,
                    });
                }
                (
                    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL,
                    u32::try_from(items.len()).map_err(|_| Error::InvalidRange)?,
                )
            }
        };
        if count == 0 {
            return Err(Error::InvalidRange);
        }
        let mut flags = if desc.fast_trace {
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE
        } else {
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD
        };
        if desc.allow_update {
            flags |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE;
        }
        Ok(Self {
            geometries,
            buffers,
            kind,
            count,
            flags,
        })
    }

    pub(super) fn native(
        &self,
        instances_address: u64,
    ) -> D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
            Type: self.kind,
            Flags: self.flags,
            NumDescs: self.count,
            DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
            Anonymous: if self.kind == D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL {
                D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
                    pGeometryDescs: self.geometries.as_ptr(),
                }
            } else {
                D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
                    InstanceDescs: instances_address,
                }
            },
        }
    }
}

#[cfg(test)]
#[path = "../../tests/backend/ray/geometry.rs"]
mod tests;
