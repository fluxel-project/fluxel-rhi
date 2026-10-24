//! DXR state-object creation and shader identifier ownership.
//!
//! All descriptor pointers below are valid only through CreateStateObject;
//! owned COM state and copied identifiers outlive that synchronous call.
use super::super::{Dx12, Dx12Device, Dx12RayPipeline, pipeline::PipelineLayoutState};
use fluxel_rhi_core::api::{self, Error, Result};
use std::{mem::ManuallyDrop, sync::Arc};
use windows::{
    Win32::Graphics::Direct3D12::*,
    core::{Interface, PCWSTR},
};

pub(crate) struct RayPipelineState {
    pub(crate) state: ID3D12StateObject,
    pub(crate) layout: Arc<PipelineLayoutState>,
    identifiers: Vec<Vec<u8>>,
}

fn wide(value: &str) -> Result<Vec<u16>> {
    if value.is_empty() || value.contains('\0') {
        return Err(Error::InvalidRange);
    }
    Ok(value.encode_utf16().chain(std::iter::once(0)).collect())
}
fn ptr(value: &[u16]) -> PCWSTR {
    PCWSTR(value.as_ptr())
}

fn local_root(
    device: &Dx12Device,
    constants: api::RayLocalConstants,
) -> Result<ID3D12RootSignature> {
    if constants.size_bytes == 0 || constants.size_bytes % 4 != 0 || constants.size_bytes > 256 {
        return Err(Error::InvalidRange);
    }
    let parameter = D3D12_ROOT_PARAMETER {
        ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
        Anonymous: D3D12_ROOT_PARAMETER_0 {
            Constants: D3D12_ROOT_CONSTANTS {
                ShaderRegister: constants.binding,
                RegisterSpace: constants.group,
                Num32BitValues: constants.size_bytes / 4,
            },
        },
        ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
    };
    let desc = D3D12_ROOT_SIGNATURE_DESC {
        NumParameters: 1,
        pParameters: &parameter,
        NumStaticSamplers: 0,
        pStaticSamplers: std::ptr::null(),
        Flags: D3D12_ROOT_SIGNATURE_FLAG_LOCAL_ROOT_SIGNATURE,
    };
    let mut blob = None;
    // SAFETY: desc points to the stack parameter for the duration of serialization.
    unsafe { D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut blob, None) }
        .map_err(super::super::native_error)?;
    let blob = blob.ok_or(Error::OutOfMemory)?;
    // SAFETY: the serialized blob owns this byte range during the synchronous call.
    let bytes =
        unsafe { std::slice::from_raw_parts(blob.GetBufferPointer().cast(), blob.GetBufferSize()) };
    // SAFETY: the valid serialized root signature remains live through creation.
    unsafe { device.0.native.CreateRootSignature(0, bytes) }.map_err(super::super::native_error)
}

pub(crate) fn create_pipeline(
    device: &Dx12Device,
    desc: &api::RayPipelineDesc<Dx12>,
) -> Result<Dx12RayPipeline> {
    let ray_device = super::ray_device(device)?;
    if desc.cache.is_some() {
        // The existing cache persists native ID3D12PipelineState blobs. DXR
        // creates ID3D12StateObject instead; ignoring the supplied cache or
        // storing shader source would falsely claim native cache support.
        return Err(Error::Unsupported("DX12 ray state object cache"));
    }
    if desc.shaders.is_empty()
        || desc.groups.is_empty()
        || desc.max_recursion == 0
        || desc.max_recursion > 31
        || desc.hit_attribute_bytes > 32
    {
        return Err(Error::Unsupported("DX12 ray pipeline descriptor"));
    }
    let stages: Vec<_> = desc
        .shaders
        .iter()
        .map(|stage| {
            if !stage.constants.is_empty() {
                return Err(Error::Unsupported("DX12 ray specialization"));
            }
            // SAFETY: Core validated that all shader modules belong to this device.
            let shader = unsafe { stage.module.native_ref() };
            let entry = shader
                .entries
                .iter()
                .find(|e| e.name == stage.entry)
                .ok_or(Error::InvalidState)?;
            match entry.stage {
                api::ShaderStage::RayGeneration
                | api::ShaderStage::Miss
                | api::ShaderStage::Callable
                | api::ShaderStage::ClosestHit
                | api::ShaderStage::AnyHit
                | api::ShaderStage::Intersection => (),
                _ => return Err(Error::Unsupported("DX12 ray shader stage")),
            }
            Ok((shader, entry.stage))
        })
        .collect::<Result<_>>()?;
    let originals: Vec<_> = desc
        .shaders
        .iter()
        .map(|stage| wide(&stage.entry))
        .collect::<Result<_>>()?;
    let aliases: Vec<_> = (0..stages.len())
        .map(|i| wide(&format!("fluxel_shader_{i}")))
        .collect::<Result<_>>()?;
    let exports: Vec<_> = aliases
        .iter()
        .zip(&originals)
        .map(|(alias, original)| D3D12_EXPORT_DESC {
            Name: ptr(alias),
            ExportToRename: ptr(original),
            Flags: D3D12_EXPORT_FLAG_NONE,
        })
        .collect();
    let libraries: Vec<_> = stages
        .iter()
        .enumerate()
        .map(|(i, (shader, _))| D3D12_DXIL_LIBRARY_DESC {
            DXILLibrary: D3D12_SHADER_BYTECODE {
                pShaderBytecode: shader.bytes.as_ptr().cast(),
                BytecodeLength: shader.bytes.len(),
            },
            NumExports: 1,
            pExports: &exports[i],
        })
        .collect();
    let names: Vec<_> = (0..desc.groups.len())
        .map(|i| wide(&format!("fluxel_group_{i}")))
        .collect::<Result<_>>()?;
    let mut hit_groups = Vec::new();
    let mut group_exports = Vec::with_capacity(desc.groups.len());
    let stage_index = |index: u32, expected: api::ShaderStage| -> Result<usize> {
        let i = usize::try_from(index).map_err(|_| Error::InvalidRange)?;
        if stages.get(i).is_none_or(|(_, stage)| *stage != expected) {
            return Err(Error::InvalidState);
        }
        Ok(i)
    };
    for (i, group) in desc.groups.iter().enumerate() {
        if let Some(index) = group.general {
            if group.closest_hit.is_some()
                || group.any_hit.is_some()
                || group.intersection.is_some()
            {
                return Err(Error::InvalidState);
            }
            let stage = stages.get(index as usize).ok_or(Error::InvalidRange)?.1;
            if !matches!(
                stage,
                api::ShaderStage::RayGeneration
                    | api::ShaderStage::Miss
                    | api::ShaderStage::Callable
            ) {
                return Err(Error::InvalidState);
            }
            group_exports.push(ptr(&aliases[index as usize]));
        } else {
            if group.closest_hit.is_none()
                && group.any_hit.is_none()
                && group.intersection.is_none()
            {
                return Err(Error::InvalidState);
            }
            let import = |index: Option<u32>, stage| -> Result<PCWSTR> {
                Ok(match index {
                    Some(index) => ptr(&aliases[stage_index(index, stage)?]),
                    None => PCWSTR::null(),
                })
            };
            hit_groups.push(D3D12_HIT_GROUP_DESC {
                HitGroupExport: ptr(&names[i]),
                Type: if group.intersection.is_some() {
                    D3D12_HIT_GROUP_TYPE_PROCEDURAL_PRIMITIVE
                } else {
                    D3D12_HIT_GROUP_TYPE_TRIANGLES
                },
                AnyHitShaderImport: import(group.any_hit, api::ShaderStage::AnyHit)?,
                ClosestHitShaderImport: import(group.closest_hit, api::ShaderStage::ClosestHit)?,
                IntersectionShaderImport: import(
                    group.intersection,
                    api::ShaderStage::Intersection,
                )?,
            });
            group_exports.push(ptr(&names[i]));
        }
    }
    let local_roots: Vec<_> = desc
        .groups
        .iter()
        .map(|g| g.local_constants.map(|c| local_root(device, c)).transpose())
        .collect::<Result<_>>()?;
    let local_descs: Vec<_> = local_roots
        .iter()
        .map(|root| D3D12_LOCAL_ROOT_SIGNATURE {
            pLocalRootSignature: ManuallyDrop::new(root.clone()),
        })
        .collect();
    // SAFETY: Core validated the pipeline layout's device identity.
    let layout = unsafe { desc.layout.native_ref() }.0.clone();
    if !layout.native_abi_compatible {
        return Err(Error::Unsupported("DX12 DXIL descriptor register overlap"));
    }
    let global = D3D12_GLOBAL_ROOT_SIGNATURE {
        pGlobalRootSignature: ManuallyDrop::new(Some(layout.compute_root.clone())),
    };
    let shader_config = D3D12_RAYTRACING_SHADER_CONFIG {
        MaxPayloadSizeInBytes: desc.payload_bytes,
        MaxAttributeSizeInBytes: desc.hit_attribute_bytes,
    };
    let pipeline_config = D3D12_RAYTRACING_PIPELINE_CONFIG {
        MaxTraceRecursionDepth: desc.max_recursion,
    };
    let mut subobjects =
        Vec::with_capacity(libraries.len() + hit_groups.len() + 3 + 2 * local_roots.len());
    for library in &libraries {
        subobjects.push(D3D12_STATE_SUBOBJECT {
            Type: D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY,
            pDesc: (library as *const D3D12_DXIL_LIBRARY_DESC).cast(),
        });
    }
    for hit in &hit_groups {
        subobjects.push(D3D12_STATE_SUBOBJECT {
            Type: D3D12_STATE_SUBOBJECT_TYPE_HIT_GROUP,
            pDesc: (hit as *const D3D12_HIT_GROUP_DESC).cast(),
        });
    }
    subobjects.push(D3D12_STATE_SUBOBJECT {
        Type: D3D12_STATE_SUBOBJECT_TYPE_GLOBAL_ROOT_SIGNATURE,
        pDesc: (&global as *const D3D12_GLOBAL_ROOT_SIGNATURE).cast(),
    });
    subobjects.push(D3D12_STATE_SUBOBJECT {
        Type: D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_SHADER_CONFIG,
        pDesc: (&shader_config as *const D3D12_RAYTRACING_SHADER_CONFIG).cast(),
    });
    subobjects.push(D3D12_STATE_SUBOBJECT {
        Type: D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_PIPELINE_CONFIG,
        pDesc: (&pipeline_config as *const D3D12_RAYTRACING_PIPELINE_CONFIG).cast(),
    });
    let mut local_indices = Vec::new();
    for (i, root) in local_roots.iter().enumerate() {
        if root.is_some() {
            local_indices.push((i, subobjects.len()));
            subobjects.push(D3D12_STATE_SUBOBJECT {
                Type: D3D12_STATE_SUBOBJECT_TYPE_LOCAL_ROOT_SIGNATURE,
                pDesc: (&local_descs[i] as *const D3D12_LOCAL_ROOT_SIGNATURE).cast(),
            });
        }
    }
    let associations: Vec<_> = local_indices
        .iter()
        .map(|(group, index)| D3D12_SUBOBJECT_TO_EXPORTS_ASSOCIATION {
            pSubobjectToAssociate: &subobjects[*index],
            NumExports: 1,
            pExports: &group_exports[*group],
        })
        .collect();
    for association in &associations {
        subobjects.push(D3D12_STATE_SUBOBJECT {
            Type: D3D12_STATE_SUBOBJECT_TYPE_SUBOBJECT_TO_EXPORTS_ASSOCIATION,
            pDesc: (association as *const D3D12_SUBOBJECT_TO_EXPORTS_ASSOCIATION).cast(),
        });
    }
    let native_desc = D3D12_STATE_OBJECT_DESC {
        Type: D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE,
        NumSubobjects: subobjects.len() as u32,
        pSubobjects: subobjects.as_ptr(),
    };
    // SAFETY: all native descriptors, names, bytecode and association pointers
    // remain stable and alive for this synchronous call.
    let result = unsafe { ray_device.CreateStateObject::<ID3D12StateObject>(&native_desc) }
        .map_err(super::super::native_error);
    // SAFETY: the COM references were manually held in the temporary ABI wrappers.
    unsafe {
        let mut global = global;
        ManuallyDrop::drop(&mut global.pGlobalRootSignature);
        for mut root in local_descs {
            ManuallyDrop::drop(&mut root.pLocalRootSignature);
        }
    }
    let state = result?;
    let properties: ID3D12StateObjectProperties =
        state.cast().map_err(super::super::native_error)?;
    let mut identifiers = Vec::with_capacity(group_exports.len());
    for name in group_exports {
        // SAFETY: export names are live; returned identifier belongs to state.
        let identifier = unsafe { properties.GetShaderIdentifier(name) };
        if identifier.is_null() {
            return Err(Error::Backend("DX12 shader identifier unavailable".into()));
        }
        // SAFETY: DXR identifiers have exactly 32 bytes for this interface.
        identifiers.push(
            unsafe {
                std::slice::from_raw_parts(
                    identifier.cast::<u8>(),
                    D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES as usize,
                )
            }
            .to_vec(),
        );
    }
    Ok(Dx12RayPipeline(Arc::new(RayPipelineState {
        state,
        layout,
        identifiers,
    })))
}

pub(crate) fn identifier(pipeline: &Dx12RayPipeline, group: u32) -> Result<Vec<u8>> {
    pipeline
        .0
        .identifiers
        .get(group as usize)
        .cloned()
        .ok_or(Error::InvalidRange)
}
