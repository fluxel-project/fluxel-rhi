//! Maps single-entry DXIL metadata to the existing DX12 descriptor ABI.

use fluxel_rhi_core::api::{self, Error, Result};
use windows::Win32::Graphics::{Direct3D::*, Direct3D12::*};

pub(super) fn reflect(
    bytes: &[u8],
    stage: api::ShaderStage,
    runtime_arrays: &[api::RuntimeBindingCapacity],
) -> Result<Vec<(u32, api::BindGroupLayoutEntry)>> {
    let version_type = match stage {
        api::ShaderStage::Vertex => D3D12_SHVER_VERTEX_SHADER,
        api::ShaderStage::Fragment => D3D12_SHVER_PIXEL_SHADER,
        api::ShaderStage::Compute => D3D12_SHVER_COMPUTE_SHADER,
        _ => return Err(Error::Unsupported("DX12 native layout inference stage")),
    };
    super::dxc::reflect_dxil(bytes, |reflection| {
        let mut shader = D3D12_SHADER_DESC::default();
        // SAFETY: reflection is live and the descriptor is writable.
        unsafe { reflection.GetDesc(&mut shader) }.map_err(reflection_error)?;
        if shader.Version >> 16 != version_type.0 as u32 {
            return Err(Error::Shader(
                "DXIL shader stage disagrees with the selected entry".into(),
            ));
        }
        let mut entries = Vec::new();
        for index in 0..shader.BoundResources {
            let mut resource = D3D12_SHADER_INPUT_BIND_DESC::default();
            // SAFETY: the index is in the reported binding span and output is writable.
            unsafe { reflection.GetResourceBindingDesc(index, &mut resource) }
                .map_err(reflection_error)?;
            let count = match resource.BindCount {
                0 | u32::MAX => {
                    let capacity = runtime_arrays
                        .iter()
                        .find(|item| {
                            item.group == resource.Space && item.binding == resource.BindPoint
                        })
                        .ok_or(Error::Unsupported(
                            "runtime array layout inference requires a capacity",
                        ))?;
                    api::BindingCount::RuntimeSized {
                        max_count: capacity.max_count,
                    }
                }
                1 => api::BindingCount::One,
                count => api::BindingCount::Fixed(count),
            };
            entries.push((
                resource.Space,
                api::BindGroupLayoutEntry {
                    binding: resource.BindPoint,
                    visibility: vec![stage],
                    ty: binding_type(reflection, &resource)?,
                    count,
                    partially_bound: false,
                    nonuniform_indexing: count != api::BindingCount::One,
                },
            ));
        }
        Ok(entries)
    })
}

fn binding_type(
    reflection: &ID3D12ShaderReflection,
    resource: &D3D12_SHADER_INPUT_BIND_DESC,
) -> Result<api::BindingType> {
    use api::{BindingType, StorageAccess, TextureSampleType, TextureViewDimension};
    Ok(match resource.Type {
        D3D_SIT_CBUFFER => {
            if resource.Name.is_null() {
                return Err(Error::Shader(
                    "DXIL constant buffer name unavailable".into(),
                ));
            }
            // SAFETY: the resource name and returned child belong to this live reflection.
            let buffer =
                unsafe { reflection.GetConstantBufferByName(resource.Name) }.ok_or_else(|| {
                    Error::Shader("DXIL constant buffer reflection unavailable".into())
                })?;
            let mut desc = D3D12_SHADER_BUFFER_DESC::default();
            // SAFETY: the child remains valid with its parent and output is writable.
            unsafe { buffer.GetDesc(&mut desc) }.map_err(reflection_error)?;
            BindingType::UniformBuffer {
                dynamic_offset: false,
                min_size: u64::from(desc.Size),
            }
        }
        D3D_SIT_STRUCTURED | D3D_SIT_BYTEADDRESS => BindingType::StorageBuffer {
            dynamic_offset: false,
            min_size: 0,
            access: StorageAccess::Read,
        },
        D3D_SIT_UAV_RWSTRUCTURED | D3D_SIT_UAV_RWBYTEADDRESS => BindingType::StorageBuffer {
            dynamic_offset: false,
            min_size: 0,
            access: StorageAccess::ReadWrite,
        },
        D3D_SIT_SAMPLER => BindingType::Sampler {
            filtering: true,
            comparison: resource.uFlags & D3D_SIF_COMPARISON_SAMPLER.0 as u32 != 0,
        },
        D3D_SIT_RTACCELERATIONSTRUCTURE => BindingType::AccelerationStructure,
        D3D_SIT_TEXTURE => {
            let sample = match resource.ReturnType {
                D3D_RETURN_TYPE_SINT => TextureSampleType::Sint,
                D3D_RETURN_TYPE_UINT => TextureSampleType::Uint,
                _ => {
                    return Err(Error::Unsupported(
                        "DXIL sampled resource requires an explicit sample type",
                    ));
                }
            };
            let (dimension, multisampled) = match resource.Dimension {
                D3D_SRV_DIMENSION_TEXTURE1D => (TextureViewDimension::D1, false),
                D3D_SRV_DIMENSION_TEXTURE2D => (TextureViewDimension::D2, false),
                D3D_SRV_DIMENSION_TEXTURE2DARRAY => (TextureViewDimension::D2Array, false),
                D3D_SRV_DIMENSION_TEXTURE2DMS => (TextureViewDimension::D2, true),
                D3D_SRV_DIMENSION_TEXTURE2DMSARRAY => (TextureViewDimension::D2Array, true),
                D3D_SRV_DIMENSION_TEXTURE3D => (TextureViewDimension::D3, false),
                D3D_SRV_DIMENSION_TEXTURECUBE => (TextureViewDimension::Cube, false),
                D3D_SRV_DIMENSION_TEXTURECUBEARRAY => (TextureViewDimension::CubeArray, false),
                _ => {
                    return Err(Error::Unsupported(
                        "DXIL inferred sampled texture dimension",
                    ));
                }
            };
            BindingType::SampledTexture {
                sample,
                dimension,
                multisampled,
            }
        }
        D3D_SIT_UAV_RWTYPED => {
            return Err(Error::Unsupported(
                "DXIL typed UAV requires an explicit binding format",
            ));
        }
        _ => return Err(Error::Unsupported("DXIL layout inference resource type")),
    })
}

fn reflection_error(error: windows::core::Error) -> Error {
    Error::Shader(format!("DXIL resource reflection: {error}"))
}
