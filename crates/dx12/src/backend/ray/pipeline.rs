//! DXR state-object creation and shader identifier ownership.
//!
//! All descriptor pointers below are valid only through CreateStateObject;
//! owned COM state and copied identifiers outlive that synchronous call.
use super::super::{
    Dx12, Dx12Device, Dx12RayPipeline, Dx12Shader, Dx12ShaderSource, pipeline::PipelineLayoutState,
    shader_naga,
};
use fluxel_rhi_core::api::{self, Error, Result};
use std::{collections::HashMap, mem::ManuallyDrop, sync::Arc};
use windows::{
    Win32::Graphics::Direct3D12::*,
    core::{Interface, PCWSTR},
};

pub(crate) struct RayPipelineState {
    pub(crate) state: ID3D12StateObject,
    pub(crate) layout: Arc<PipelineLayoutState>,
    identifiers: Vec<Vec<u8>>,
}

fn wide(value: &str) -> Vec<u16> {
    value.encode_utf16().chain(std::iter::once(0)).collect()
}
fn ptr(value: &[u16]) -> PCWSTR {
    PCWSTR(value.as_ptr())
}

fn local_root(
    device: &Dx12Device,
    constants: api::RayLocalConstants,
) -> Result<ID3D12RootSignature> {
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

struct AdmittedRay<'a> {
    layout: &'a Arc<PipelineLayoutState>,
    stages: Vec<(&'a Dx12Shader, api::ShaderStage)>,
}

pub(in crate::backend) fn is_ray_stage(stage: api::ShaderStage) -> bool {
    matches!(
        stage,
        api::ShaderStage::RayGeneration
            | api::ShaderStage::Miss
            | api::ShaderStage::Callable
            | api::ShaderStage::ClosestHit
            | api::ShaderStage::AnyHit
            | api::ShaderStage::Intersection
    )
}

pub(super) fn pipeline_support(
    device: &Dx12Device,
    desc: &api::RayPipelineDesc<Dx12>,
) -> Result<()> {
    super::ray_device(device)?;
    admit_pipeline(device, desc).map(drop)
}

fn admit_pipeline<'a>(
    device: &Dx12Device,
    desc: &'a api::RayPipelineDesc<Dx12>,
) -> Result<AdmittedRay<'a>> {
    device.require_active()?;
    if desc.cache.is_some() {
        // The existing cache persists native ID3D12PipelineState blobs. DXR
        // creates ID3D12StateObject instead; ignoring the supplied cache or
        // storing shader source would falsely claim native cache support.
        return Err(Error::Unsupported("DX12 ray state object cache"));
    }
    if desc.shaders.is_empty()
        || desc.groups.is_empty()
        || desc.max_recursion > D3D12_RAYTRACING_MAX_DECLARABLE_TRACE_RECURSION_DEPTH
        || desc.hit_attribute_bytes > D3D12_RAYTRACING_MAX_ATTRIBUTE_SIZE_IN_BYTES
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
            if stage.entry.is_empty() || stage.entry.contains('\0') {
                return Err(Error::InvalidRange);
            }
            let entry = shader
                .entries
                .iter()
                .find(|e| e.name == stage.entry)
                .ok_or(Error::InvalidState)?;
            if !is_ray_stage(entry.stage) {
                return Err(Error::Unsupported("DX12 ray shader stage"));
            }
            match &shader.source {
                None => shader_naga::validate_dxil_library_export(
                    &shader.bytes,
                    &stage.entry,
                    entry.stage,
                )?,
                Some(Dx12ShaderSource::Hlsl(source))
                    if matches!(source.format, api::ShaderFormat::Hlsl(model) if (63..=device.0.shader_model.min(69)).contains(&model)) => {}
                Some(_) => return Err(Error::Unsupported("DX12 ray shader source")),
            }
            Ok((shader, entry.stage))
        })
        .collect::<Result<_>>()?;
    for group in &desc.groups {
        if group.local_constants.is_some_and(|constants| {
            constants.size_bytes == 0 || constants.size_bytes % 4 != 0 || constants.size_bytes > 256
        }) {
            return Err(Error::InvalidRange);
        }
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
        } else {
            let imports = [
                (group.closest_hit, api::ShaderStage::ClosestHit),
                (group.any_hit, api::ShaderStage::AnyHit),
                (group.intersection, api::ShaderStage::Intersection),
            ];
            if imports.iter().all(|(index, _)| index.is_none()) {
                return Err(Error::InvalidState);
            }
            for (index, expected) in imports {
                if index.is_some_and(|index| {
                    stages
                        .get(index as usize)
                        .is_none_or(|(_, stage)| *stage != expected)
                }) {
                    return Err(Error::InvalidState);
                }
            }
        }
    }
    // SAFETY: Core checked that the layout belongs to the pipeline device.
    let layout = &unsafe { desc.layout.native_ref() }.0;
    if !layout.native_abi_compatible {
        return Err(Error::Unsupported("DX12 DXIL descriptor register overlap"));
    }
    Ok(AdmittedRay { layout, stages })
}

pub(crate) fn create_pipeline(
    device: &Dx12Device,
    desc: &api::RayPipelineDesc<Dx12>,
) -> Result<Dx12RayPipeline> {
    let ray_device = super::ray_device(device)?;
    let AdmittedRay { layout, stages } = admit_pipeline(device, desc)?;
    // Multiple entries from one HLSL module share a single compiled DXIL library.
    let mut compiled = HashMap::<(*const [u8], u16, bool), Arc<[u8]>>::new();
    let artifacts: Vec<_> = stages
        .iter()
        .zip(&desc.shaders)
        .map(|(&(shader, stage), desc)| {
            let Some(Dx12ShaderSource::Hlsl(source)) = &shader.source else {
                return Ok(shader.bytes.clone());
            };
            let api::ShaderFormat::Hlsl(model) = source.format else {
                return Err(Error::InvalidState);
            };
            let native_16bit = shader_naga::requires_native_16bit(source);
            // Shader handles can be cloned by value. Their source allocation
            // remains shared, and the other fields determine the DXC options.
            let key = (Arc::as_ptr(&source.bytes), model, native_16bit);
            let artifact = match compiled.entry(key) {
                std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
                std::collections::hash_map::Entry::Vacant(entry) => {
                    let text = std::str::from_utf8(&source.bytes)
                        .map_err(|_| Error::Shader("HLSL is not UTF-8".into()))?;
                    entry.insert(Arc::from(shader_naga::compile_ray_library(
                        text,
                        model,
                        native_16bit,
                    )?))
                }
            };
            shader_naga::validate_dxil_library_export(artifact, &desc.entry, stage)?;
            Ok(artifact.clone())
        })
        .collect::<Result<_>>()?;
    let originals: Vec<_> = desc
        .shaders
        .iter()
        .map(|stage| wide(&stage.entry))
        .collect();
    let aliases: Vec<_> = (0..stages.len())
        .map(|i| wide(&format!("fluxel_shader_{i}")))
        .collect();
    let mut library_indices = HashMap::new();
    let mut library_exports: Vec<(&[u8], Vec<D3D12_EXPORT_DESC>)> = Vec::new();
    for ((bytes, alias), original) in artifacts.iter().zip(&aliases).zip(&originals) {
        let index = *library_indices
            .entry(Arc::as_ptr(bytes))
            .or_insert_with(|| {
                library_exports.push((bytes, Vec::new()));
                library_exports.len() - 1
            });
        library_exports[index].1.push(D3D12_EXPORT_DESC {
            Name: ptr(alias),
            ExportToRename: ptr(original),
            Flags: D3D12_EXPORT_FLAG_NONE,
        });
    }
    let libraries: Vec<_> = library_exports
        .iter()
        .map(|(bytes, exports)| D3D12_DXIL_LIBRARY_DESC {
            DXILLibrary: D3D12_SHADER_BYTECODE {
                pShaderBytecode: bytes.as_ptr().cast(),
                BytecodeLength: bytes.len(),
            },
            NumExports: exports.len() as u32,
            pExports: exports.as_ptr(),
        })
        .collect();
    let names: Vec<_> = (0..desc.groups.len())
        .map(|i| wide(&format!("fluxel_group_{i}")))
        .collect();
    let mut hit_groups = Vec::new();
    let mut group_exports = Vec::with_capacity(desc.groups.len());
    for (i, group) in desc.groups.iter().enumerate() {
        if let Some(index) = group.general {
            group_exports.push(ptr(&aliases[index as usize]));
        } else {
            let import = |index: Option<u32>| {
                index.map_or(PCWSTR::null(), |index| ptr(&aliases[index as usize]))
            };
            hit_groups.push(D3D12_HIT_GROUP_DESC {
                HitGroupExport: ptr(&names[i]),
                Type: if group.intersection.is_some() {
                    D3D12_HIT_GROUP_TYPE_PROCEDURAL_PRIMITIVE
                } else {
                    D3D12_HIT_GROUP_TYPE_TRIANGLES
                },
                AnyHitShaderImport: import(group.any_hit),
                ClosestHitShaderImport: import(group.closest_hit),
                IntersectionShaderImport: import(group.intersection),
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
        layout: layout.clone(),
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
