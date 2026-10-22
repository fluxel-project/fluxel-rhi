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
}
pub(crate) struct PipelineLayoutState {
    pub(crate) root: ID3D12RootSignature,
    pub(crate) groups: Vec<Arc<descriptor::BindGroupLayoutState>>,
    /// One root CBV table parameter per logical group.  Group ordinal is both
    /// this vector's index and the HLSL register space used for its ranges.
    pub(crate) tables: Vec<GroupTables>,
    pub(crate) push_constants: Option<PushConstantsRoot>,
}
pub(crate) struct PushConstantsRoot {
    pub(crate) root: u32,
    pub(crate) range: std::ops::Range<u32>,
    pub(crate) stages: api::StageMask,
}
pub(crate) fn create_pipeline_layout(
    device: &Dx12Device,
    desc: &api::PipelineLayoutDesc<'_, Dx12>,
) -> Result<Dx12PipelineLayout> {
    // Layout group position is the DXIL register space.  Binding layout
    // validation accepts only static uniform buffers at this stage, hence every
    // range is a CBV table range and one group maps to one root parameter.
    let mut ranges: Vec<Vec<D3D12_DESCRIPTOR_RANGE>> = Vec::with_capacity(desc.groups.len() * 2);
    let mut parameters: Vec<D3D12_ROOT_PARAMETER> = Vec::with_capacity(desc.groups.len());
    let mut tables = Vec::with_capacity(desc.groups.len());
    let mut groups = Vec::with_capacity(desc.groups.len());
    for (group, layout) in desc.groups.iter().enumerate() {
        // SAFETY: the Core descriptor is tied to this device; this borrow is
        // used only while synchronously building an independent root signature.
        let native = unsafe { layout.native_ref() };
        groups.push(native.0.clone());
        let state = &native.0;
        let view_ranges: Vec<D3D12_DESCRIPTOR_RANGE> = state
            .slots
            .iter()
            .filter(|slot| {
                slot.class != descriptor::DescriptorClass::Sampler && !slot.dynamic_offset
            })
            .map(|slot| D3D12_DESCRIPTOR_RANGE {
                RangeType: match slot.class {
                    descriptor::DescriptorClass::Cbv => D3D12_DESCRIPTOR_RANGE_TYPE_CBV,
                    descriptor::DescriptorClass::Srv => D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
                    descriptor::DescriptorClass::Uav => D3D12_DESCRIPTOR_RANGE_TYPE_UAV,
                    descriptor::DescriptorClass::Sampler => unreachable!(),
                },
                NumDescriptors: if slot.runtime_sized {
                    u32::MAX
                } else {
                    slot.descriptor_count
                },
                BaseShaderRegister: slot.binding,
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
                BaseShaderRegister: slot.binding,
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
                        ShaderRegister: slot.binding,
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
        });
    }
    let push_constants = match desc.push_constants {
        [] => None,
        [range] => {
            if !matches!(
                range.stages.as_slice(),
                [api::ShaderStage::Vertex] | [api::ShaderStage::Mesh] | [api::ShaderStage::Task]
            ) || range.range.start >= range.range.end
                || range.range.end > 128
                || range.range.start % 4 != 0
                || range.range.end % 4 != 0
            {
                return Err(Error::Unsupported("DX12 vertex push-constant range"));
            }
            let root = parameters.len() as u32;
            parameters.push(D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    Constants: D3D12_ROOT_CONSTANTS {
                        ShaderRegister: 1,
                        RegisterSpace: 0,
                        Num32BitValues: (range.range.end - range.range.start) / 4,
                    },
                },
                ShaderVisibility: match range.stages[0] {
                    api::ShaderStage::Mesh => D3D12_SHADER_VISIBILITY_MESH,
                    api::ShaderStage::Task => D3D12_SHADER_VISIBILITY_AMPLIFICATION,
                    _ => D3D12_SHADER_VISIBILITY_VERTEX,
                },
            });
            Some(PushConstantsRoot {
                root,
                range: range.range.clone(),
                stages: match range.stages[0] {
                    api::ShaderStage::Mesh => api::StageMask::MESH,
                    api::ShaderStage::Task => api::StageMask::TASK,
                    _ => api::StageMask::VERTEX,
                },
            })
        }
        _ => {
            return Err(Error::Unsupported("DX12 multiple push-constant ranges"));
        }
    };
    let root_desc = D3D12_ROOT_SIGNATURE_DESC {
        NumParameters: parameters.len() as u32,
        pParameters: parameters.as_ptr(),
        NumStaticSamplers: 0,
        pStaticSamplers: std::ptr::null(),
        Flags: D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT,
    };
    let mut blob = None;
    // SAFETY: all pointers in `root_desc` point at vectors retained until this
    // synchronous serializer returns; D3D12 receives a fully initialized ABI
    // descriptor and writes only to the supplied COM out parameter.
    unsafe {
        D3D12SerializeRootSignature(&root_desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut blob, None)
    }
    .map_err(native_error)?;
    let blob = blob.ok_or_else(|| {
        Error::Backend("D3D12 root-signature serialization returned no blob".into())
    })?;
    // SAFETY: the serializer owns `blob`; its documented buffer pointer and
    // size remain valid while `blob` stays in scope for CreateRootSignature.
    let bytes =
        unsafe { std::slice::from_raw_parts(blob.GetBufferPointer().cast(), blob.GetBufferSize()) };
    // SAFETY: the blob byte range remains live for this synchronous creation
    // call, and the returned COM object takes ownership of native state.
    let root = unsafe {
        device
            .0
            .native
            .CreateRootSignature::<ID3D12RootSignature>(0, bytes)
    }
    .map_err(native_error)?;
    Ok(Dx12PipelineLayout(Arc::new(PipelineLayoutState {
        root,
        groups,
        tables,
        push_constants,
    })))
}
fn native_error(e: windows::core::Error) -> Error {
    Error::Backend(format!("Direct3D 12: {e}"))
}
