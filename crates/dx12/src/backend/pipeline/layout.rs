//! Root-signature and pipeline-layout lowering.
//!
//! Core layout groups are lowered to DXIL register spaces and root parameters.
//! The resulting state is immutable and shared by pipelines. This module keeps
//! the backing range vectors alive until serialization has copied them, then
//! owns only the native root signature and its mapping metadata; it does not
//! allocate descriptors or choose dynamic bindings at record time.
use fluxel_rhi_core::api::{self, Error, Result};
use std::sync::Arc;
use windows::Win32::Graphics::Direct3D12::*;

use super::super::{Dx12, Dx12Device, Dx12PipelineLayout, descriptor};
pub(crate) struct DynamicCbvRoot {
    pub(crate) binding: u32,
    pub(crate) root: u32,
    pub(crate) min_size: u64,
}
pub(crate) struct GroupTables {
    pub(crate) views: Option<u32>,
    pub(crate) samplers: Option<u32>,
    pub(crate) dynamic_cbvs: Vec<DynamicCbvRoot>,
    /// All dynamic bindings in ascending binding order, including storage.
    pub(crate) dynamic_bindings: Vec<(u32, descriptor::DescriptorClass)>,
}
pub(crate) struct PipelineLayoutState {
    pub(crate) root: ID3D12RootSignature,
    pub(crate) compute_root: ID3D12RootSignature,
    pub(crate) root_bytes: Vec<u8>,
    pub(crate) compute_root_bytes: Vec<u8>,
    pub(crate) vertex_index_root: Option<(ID3D12RootSignature, Vec<u8>, u32)>,
    pub(crate) num_workgroups_root: Option<(ID3D12RootSignature, Vec<u8>, u32)>,
    pub(crate) register_map: Vec<(u32, u32, u32, u8, u32)>,
    pub(crate) binding_types: Vec<(u32, u32, api::BindingType)>,
    /// Preserve exact array admission metadata through WGSL pipeline lowering.
    pub(crate) storage_entries: Vec<(u32, api::BindGroupLayoutEntry)>,
    /// Explicit capacities for public runtime-sized arrays.  This stays
    /// separate from `register_map`: a fixed array can reserve the same number
    /// of descriptors, but WGSL's dynamic binding-array syntax must not be
    /// accepted for that different public contract.
    pub(crate) runtime_binding_capacities: Vec<(u32, u32, u32)>,
    pub(crate) native_abi_compatible: bool,
    pub(crate) groups: Vec<Arc<descriptor::BindGroupLayoutState>>,
    /// One root CBV table parameter per logical group.  Group ordinal is both
    /// this vector's index and the HLSL register space used for its ranges.
    pub(crate) tables: Vec<GroupTables>,
    pub(crate) push_constants: Option<PushConstantsRoot>,
}
pub(crate) struct PushConstantsRoot {
    graphics: Vec<(api::StageMask, u32)>,
    compute: Option<u32>,
    ranges: Vec<(std::ops::Range<u32>, api::StageMask)>,
}
impl PushConstantsRoot {
    fn covers(&self, stage: api::StageMask, start: u32, end: u32) -> bool {
        (start..end).step_by(4).all(|offset| {
            self.ranges.iter().any(|(range, mask)| {
                mask.0 & stage.0 != 0 && range.start <= offset && offset + 4 <= range.end
            })
        })
    }
    pub(crate) fn graphics_writes(
        &self,
        stages: api::StageMask,
        start: u32,
        end: u32,
    ) -> Result<Vec<u32>> {
        if stages.0 == 0 || start >= end || start % 4 != 0 || end % 4 != 0 {
            return Err(Error::InvalidRange);
        }
        let mut remaining = stages.0;
        let mut roots = Vec::new();
        for (stage, root) in &self.graphics {
            if stages.0 & stage.0 != 0 {
                if !self.covers(*stage, start, end) {
                    return Err(Error::InvalidRange);
                }
                roots.push(*root);
                remaining &= !stage.0;
            }
        }
        if remaining != 0 {
            return Err(Error::InvalidRange);
        }
        Ok(roots)
    }
    pub(crate) fn compute_write(&self, start: u32, end: u32) -> Result<u32> {
        if start >= end
            || start % 4 != 0
            || end % 4 != 0
            || !self.covers(api::StageMask::COMPUTE, start, end)
        {
            return Err(Error::InvalidRange);
        }
        self.compute.ok_or(Error::InvalidRange)
    }
}
pub(crate) fn create_pipeline_layout(
    device: &Dx12Device,
    desc: &api::PipelineLayoutDesc<'_, Dx12>,
) -> Result<Dx12PipelineLayout> {
    // Layout group position is the DXIL register space. View and sampler
    // tables are separate; dynamic uniform bindings use root descriptors.
    let mut ranges: Vec<Vec<D3D12_DESCRIPTOR_RANGE>> = Vec::with_capacity(desc.groups.len() * 2);
    let mut parameters: Vec<D3D12_ROOT_PARAMETER> = Vec::with_capacity(desc.groups.len());
    let mut tables = Vec::with_capacity(desc.groups.len());
    let mut groups = Vec::with_capacity(desc.groups.len());
    let mut register_map = Vec::new();
    let mut binding_types = Vec::new();
    let mut storage_entries = Vec::new();
    let mut runtime_binding_capacities = Vec::new();
    let mut native_abi_compatible = true;
    for (group, layout) in desc.groups.iter().enumerate() {
        // SAFETY: the Core descriptor is tied to this device; this borrow is
        // used only while synchronously building an independent root signature.
        let native = unsafe { layout.native_ref() };
        groups.push(native.0.clone());
        let state = &native.0;
        storage_entries.extend(
            state
                .storage_entries
                .iter()
                .cloned()
                .map(|entry| (group as u32, entry)),
        );
        let mut next_register = [0u32; 4];
        for slot in &state.slots {
            binding_types.push((group as u32, slot.binding, slot.binding_type.clone()));
            let class = match slot.class {
                descriptor::DescriptorClass::Cbv => 0,
                descriptor::DescriptorClass::Srv => 1,
                descriptor::DescriptorClass::Uav => 2,
                descriptor::DescriptorClass::Sampler => 3,
            };
            let register = slot.binding.max(next_register[class]);
            next_register[class] = register
                .checked_add(slot.descriptor_count)
                .ok_or(Error::InvalidRange)?;
            native_abi_compatible &= register == slot.binding;
            register_map.push((
                group as u32,
                slot.binding,
                register,
                class as u8,
                slot.descriptor_count,
            ));
            if slot.runtime_sized {
                runtime_binding_capacities.push((
                    group as u32,
                    slot.binding,
                    slot.descriptor_count,
                ));
            }
        }
        let mapped = |binding| {
            register_map
                .iter()
                .find(|&&(g, b, _, _, _)| g == group as u32 && b == binding)
                .map(|&(_, _, register, _, _)| register)
                .expect("layout slot mapped")
        };
        let view_ranges: Vec<D3D12_DESCRIPTOR_RANGE> = state
            .slots
            .iter()
            .filter(|slot| {
                slot.class != descriptor::DescriptorClass::Sampler
                    && !(slot.dynamic_offset && slot.class == descriptor::DescriptorClass::Cbv)
            })
            .map(|slot| D3D12_DESCRIPTOR_RANGE {
                RangeType: match slot.class {
                    descriptor::DescriptorClass::Cbv => D3D12_DESCRIPTOR_RANGE_TYPE_CBV,
                    descriptor::DescriptorClass::Srv => D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
                    descriptor::DescriptorClass::Uav => D3D12_DESCRIPTOR_RANGE_TYPE_UAV,
                    descriptor::DescriptorClass::Sampler => unreachable!(),
                },
                NumDescriptors: slot.descriptor_count,
                BaseShaderRegister: mapped(slot.binding),
                RegisterSpace: group as u32,
                OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
            })
            .collect();
        let sampler_ranges: Vec<D3D12_DESCRIPTOR_RANGE> = state
            .slots
            .iter()
            .filter(|slot| slot.class == descriptor::DescriptorClass::Sampler)
            .map(|slot| D3D12_DESCRIPTOR_RANGE {
                RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER,
                NumDescriptors: slot.descriptor_count,
                BaseShaderRegister: mapped(slot.binding),
                RegisterSpace: group as u32,
                OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
            })
            .collect();
        let mut view = None;
        let mut samplers = None;
        if !view_ranges.is_empty() {
            ranges.push(view_ranges);
            let table = ranges.last().unwrap();
            let index = parameters.len() as u32;
            parameters.push(D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                        NumDescriptorRanges: table.len() as u32,
                        pDescriptorRanges: table.as_ptr(),
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            });
            view = Some(index);
        }
        if !sampler_ranges.is_empty() {
            ranges.push(sampler_ranges);
            let table = ranges.last().unwrap();
            let index = parameters.len() as u32;
            parameters.push(D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                        NumDescriptorRanges: table.len() as u32,
                        pDescriptorRanges: table.as_ptr(),
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            });
            samplers = Some(index);
        }
        let mut dynamic_cbvs = Vec::new();
        for slot in state
            .slots
            .iter()
            .filter(|slot| slot.class == descriptor::DescriptorClass::Cbv && slot.dynamic_offset)
        {
            let root = parameters.len() as u32;
            parameters.push(D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_CBV,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    Descriptor: D3D12_ROOT_DESCRIPTOR {
                        ShaderRegister: mapped(slot.binding),
                        RegisterSpace: group as u32,
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            });
            dynamic_cbvs.push(DynamicCbvRoot {
                binding: slot.binding,
                root,
                min_size: slot.min_size,
            });
        }
        tables.push(GroupTables {
            views: view,
            samplers,
            dynamic_cbvs,
            dynamic_bindings: state
                .slots
                .iter()
                .filter(|slot| slot.dynamic_offset)
                .map(|slot| (slot.binding, slot.class))
                .collect(),
        });
    }
    let mut compute_parameters = parameters.clone();
    let push_constants = if desc.push_constants.is_empty() {
        None
    } else {
        let mut push_ranges = Vec::new();
        for entry in desc.push_constants {
            if entry.range.start >= entry.range.end
                || entry.range.end > 128
                || entry.range.start % 4 != 0
                || entry.range.end % 4 != 0
            {
                return Err(Error::InvalidRange);
            }
            let mut stages = api::StageMask::EMPTY;
            for stage in &entry.stages {
                let bit = match stage {
                    api::ShaderStage::Vertex => api::StageMask::VERTEX,
                    api::ShaderStage::Fragment => api::StageMask::FRAGMENT,
                    api::ShaderStage::Compute => api::StageMask::COMPUTE,
                    api::ShaderStage::Task => api::StageMask::TASK,
                    api::ShaderStage::Mesh => api::StageMask::MESH,
                    _ => return Err(Error::Unsupported("DX12 ray push constants")),
                };
                if stages.0 & bit.0 != 0 {
                    return Err(Error::InvalidRange);
                }
                stages = stages.union(bit);
            }
            if stages.0 == 0 {
                return Err(Error::InvalidRange);
            }
            push_ranges.push((entry.range.clone(), stages));
        }
        let max_end = |stage: api::StageMask| {
            push_ranges
                .iter()
                .filter(|(_, mask)| mask.0 & stage.0 != 0)
                .map(|(range, _)| range.end)
                .max()
        };
        let constant = |count, visibility| D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Constants: D3D12_ROOT_CONSTANTS {
                    ShaderRegister: 1,
                    RegisterSpace: 0,
                    Num32BitValues: count,
                },
            },
            ShaderVisibility: visibility,
        };
        let mut graphics = Vec::new();
        for (stage, visibility) in [
            (api::StageMask::VERTEX, D3D12_SHADER_VISIBILITY_VERTEX),
            (api::StageMask::FRAGMENT, D3D12_SHADER_VISIBILITY_PIXEL),
            (api::StageMask::TASK, D3D12_SHADER_VISIBILITY_AMPLIFICATION),
            (api::StageMask::MESH, D3D12_SHADER_VISIBILITY_MESH),
        ] {
            if let Some(end) = max_end(stage) {
                let root = parameters.len() as u32;
                parameters.push(constant(end / 4, visibility));
                graphics.push((stage, root));
            }
        }
        let compute = max_end(api::StageMask::COMPUTE).map(|end| {
            let root = compute_parameters.len() as u32;
            compute_parameters.push(constant(end / 4, D3D12_SHADER_VISIBILITY_ALL));
            root
        });
        Some(PushConstantsRoot {
            graphics,
            compute,
            ranges: push_ranges,
        })
    };
    let (root, root_bytes) = create_root(device, &parameters)?;
    let (compute_root, compute_root_bytes) = if push_constants.is_some() {
        create_root(device, &compute_parameters)?
    } else {
        (root.clone(), root_bytes.clone())
    };
    let num_workgroups_root = if groups.len() <= 255 {
        let index = compute_parameters.len() as u32;
        compute_parameters.push(D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Constants: D3D12_ROOT_CONSTANTS {
                    ShaderRegister: 0,
                    RegisterSpace: 255,
                    Num32BitValues: 3,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        });
        match create_root(device, &compute_parameters) {
            Ok((root, bytes)) => Some((root, bytes, index)),
            Err(Error::Unsupported(_)) => None,
            Err(error) => return Err(error),
        }
    } else {
        None
    };
    let vertex_index_root = if groups.len() <= 255 {
        let index = parameters.len() as u32;
        parameters.push(D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Constants: D3D12_ROOT_CONSTANTS {
                    ShaderRegister: 0,
                    RegisterSpace: 255,
                    Num32BitValues: 3,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_VERTEX,
        });
        match create_root(device, &parameters) {
            Ok((root, bytes)) => Some((root, bytes, index)),
            Err(Error::Unsupported(_)) => None,
            Err(error) => return Err(error),
        }
    } else {
        None
    };
    Ok(Dx12PipelineLayout(Arc::new(PipelineLayoutState {
        root,
        compute_root,
        root_bytes,
        compute_root_bytes,
        vertex_index_root,
        num_workgroups_root,
        register_map,
        binding_types,
        storage_entries,
        runtime_binding_capacities,
        native_abi_compatible,
        groups,
        tables,
        push_constants,
    })))
}
fn create_root(
    device: &Dx12Device,
    parameters: &[D3D12_ROOT_PARAMETER],
) -> Result<(ID3D12RootSignature, Vec<u8>)> {
    let mut root_dwords = 0u32;
    for parameter in parameters {
        root_dwords += match parameter.ParameterType {
            D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE => 1,
            // SAFETY: the discriminator identifies the initialized union member.
            D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS => unsafe {
                parameter.Anonymous.Constants.Num32BitValues
            },
            _ => 2,
        };
    }
    if root_dwords > 64 {
        return Err(Error::Unsupported("DX12 root signature DWORD budget"));
    }
    let root_desc = D3D12_ROOT_SIGNATURE_DESC {
        NumParameters: parameters.len() as u32,
        pParameters: parameters.as_ptr(),
        NumStaticSamplers: 0,
        pStaticSamplers: std::ptr::null(),
        Flags: D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT,
    };
    let mut blob = None;
    // SAFETY: parameters and descriptor ranges remain live throughout serialization.
    unsafe {
        D3D12SerializeRootSignature(&root_desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut blob, None)
    }
    .map_err(native_error)?;
    let blob = blob.ok_or_else(|| {
        Error::Backend("D3D12 root-signature serialization returned no blob".into())
    })?;
    // SAFETY: the blob owns its byte range throughout the synchronous call.
    let bytes =
        unsafe { std::slice::from_raw_parts(blob.GetBufferPointer().cast(), blob.GetBufferSize()) };
    // SAFETY: serialized bytes remain live while creating the signature.
    let signature = unsafe {
        device
            .0
            .native
            .CreateRootSignature::<ID3D12RootSignature>(0, bytes)
    }
    .map_err(native_error)?;
    Ok((signature, bytes.to_vec()))
}
fn native_error(e: windows::core::Error) -> Error {
    Error::Backend(format!("Direct3D 12: {e}"))
}
