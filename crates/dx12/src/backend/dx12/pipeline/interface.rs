//! DX12 root-signature lowering for one portable pipeline interface.
//!
//! A root signature is backend-private state assembled at pipeline creation; it
//! is not a public `PipelineInterface` object. The frozen portable descriptor
//! already guarantees device identity and logical compatibility before this seam
//! is reached.

use std::slice;

use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::{
    D3D_ROOT_SIGNATURE_VERSION_1, D3D12_DESCRIPTOR_RANGE, D3D12_ROOT_CONSTANTS,
    D3D12_ROOT_DESCRIPTOR, D3D12_ROOT_DESCRIPTOR_TABLE, D3D12_ROOT_PARAMETER,
    D3D12_ROOT_PARAMETER_0, D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
    D3D12_ROOT_PARAMETER_TYPE_CBV, D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
    D3D12_ROOT_PARAMETER_TYPE_SRV, D3D12_ROOT_PARAMETER_TYPE_UAV, D3D12_ROOT_SIGNATURE_DESC,
    D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT, D3D12_ROOT_SIGNATURE_FLAG_NONE,
    D3D12_SHADER_VISIBILITY_ALL, D3D12SerializeRootSignature, ID3D12Device, ID3D12RootSignature,
};

use crate::api::binding::BindGroupLayout;
use crate::api::pipeline::ImmediateRange;
use crate::backend::dx12::failure::Dx12Failure;

/// The native root signature and the parameter indices produced while lowering
/// one portable pipeline interface.
///
/// D3D12 has no legal zero-range descriptor table.  Consequently empty groups
/// do not produce a parameter, and the parameter index cannot be recovered from
/// `group * 2`; the vector is the one authoritative mapping command lowering
/// must use.
pub(crate) struct Dx12RootSignature {
    handle: ID3D12RootSignature,
    view_parameters: Vec<Option<u32>>,
    sampler_parameters: Vec<Option<u32>>,
    dynamic_parameters: Vec<Vec<u32>>,
    /// One contiguous root-constant block backs the portable byte address
    /// space. Splitting ranges into native parameters would turn byte offsets
    /// into HLSL register numbers and can collide with a group-0 CBV.
    immediate_parameter: Option<u32>,
    immediate_ranges: Vec<ImmediateRange>,
}

impl Dx12RootSignature {
    pub(crate) fn handle(&self) -> &ID3D12RootSignature {
        &self.handle
    }

    /// The root-parameter index for a group's CBV/SRV/UAV descriptor table.
    /// `None` means that the group's layout has no view bindings.
    pub(crate) fn view_parameter(&self, group: u32) -> Option<u32> {
        self.view_parameters.get(group as usize).copied().flatten()
    }

    pub(crate) fn sampler_parameter(&self, group: u32) -> Option<u32> {
        self.sampler_parameters
            .get(group as usize)
            .copied()
            .flatten()
    }
    pub(crate) fn dynamic_parameters(&self, group: u32) -> Option<&[u32]> {
        self.dynamic_parameters
            .get(group as usize)
            .map(Vec::as_slice)
    }
    pub(crate) fn immediate_parameter(&self, offset: u32, size: u32) -> Option<(u32, u32)> {
        self.immediate_ranges
            .iter()
            .position(|range| {
                offset >= range.offset
                    && offset
                        .checked_add(size)
                        .is_some_and(|end| end <= range.offset + range.size)
            })
            .and(
                self.immediate_parameter
                    .map(|parameter| (parameter, offset / 4)),
            )
    }
}

/// Lowers the ordered portable group layouts into a DX12 root signature.
///
/// An empty group sequence is valid: it lowers to a root signature with no
/// parameters when this implementation is completed.
pub(crate) fn build_root_signature(
    device: &ID3D12Device,
    groups: &[BindGroupLayout],
    immediate_ranges: &[ImmediateRange],
    uses_input_assembler: bool,
) -> Result<Dx12RootSignature, Dx12Failure> {
    // The vectors own every pointer handed to the serializer.  They remain alive
    // until it has copied the description, which is the complete lifetime the
    // D3D12 call requires.
    let mut ranges: Vec<Vec<D3D12_DESCRIPTOR_RANGE>> = Vec::with_capacity(groups.len());
    let mut parameters = Vec::with_capacity(groups.len());
    let mut view_parameters = Vec::with_capacity(groups.len());
    let mut sampler_parameters = Vec::with_capacity(groups.len());
    let mut dynamic_parameters = Vec::with_capacity(groups.len());
    let mut root_descriptor_count = 0usize;

    for (group_index, group) in groups.iter().enumerate() {
        let plan = crate::backend::dx12::binding::layout::TablePlan::of(group.descriptor())?;

        if plan.views().is_empty() {
            view_parameters.push(None);
        } else {
            let table_ranges: Vec<D3D12_DESCRIPTOR_RANGE> = plan
                .views()
                .iter()
                .map(|range| D3D12_DESCRIPTOR_RANGE {
                    RangeType: range.class.range_type(),
                    NumDescriptors: range.count,
                    BaseShaderRegister: range.slot.get(),
                    RegisterSpace: group_index as u32,
                    OffsetInDescriptorsFromTableStart: range.first,
                })
                .collect();
            ranges.push(table_ranges);
            let table = ranges
                .last()
                .expect("a non-empty table range vector was just pushed");
            let parameter_index = parameters.len() as u32;
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
            view_parameters.push(Some(parameter_index));
        }
        if plan.samplers().is_empty() {
            sampler_parameters.push(None);
        } else {
            let table_ranges: Vec<D3D12_DESCRIPTOR_RANGE> = plan
                .samplers()
                .iter()
                .map(|range| D3D12_DESCRIPTOR_RANGE {
                    RangeType: range.class.range_type(),
                    NumDescriptors: range.count,
                    BaseShaderRegister: range.slot.get(),
                    RegisterSpace: group_index as u32,
                    OffsetInDescriptorsFromTableStart: range.first,
                })
                .collect();
            ranges.push(table_ranges);
            let table = ranges
                .last()
                .expect("a non-empty table range vector was just pushed");
            let parameter_index = parameters.len() as u32;
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
            sampler_parameters.push(Some(parameter_index));
        }

        let mut group_dynamic_parameters = Vec::new();
        for dynamic in plan.dynamics() {
            let parameter_type = match dynamic.class {
                crate::backend::dx12::binding::vocabulary::RegisterClass::ConstantBuffer => {
                    D3D12_ROOT_PARAMETER_TYPE_CBV
                }
                crate::backend::dx12::binding::vocabulary::RegisterClass::ShaderResource => {
                    D3D12_ROOT_PARAMETER_TYPE_SRV
                }
                crate::backend::dx12::binding::vocabulary::RegisterClass::UnorderedAccess => {
                    D3D12_ROOT_PARAMETER_TYPE_UAV
                }
                crate::backend::dx12::binding::vocabulary::RegisterClass::Sampler => {
                    unreachable!("dynamic offsets are buffers")
                }
            };
            for element in 0..dynamic.count {
                let parameter_index = parameters.len() as u32;
                parameters.push(D3D12_ROOT_PARAMETER {
                    ParameterType: parameter_type,
                    Anonymous: D3D12_ROOT_PARAMETER_0 {
                        Descriptor: D3D12_ROOT_DESCRIPTOR {
                            ShaderRegister: dynamic.slot.get() + element,
                            RegisterSpace: group_index as u32,
                        },
                    },
                    ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
                });
                group_dynamic_parameters.push(parameter_index);
                root_descriptor_count += 1;
            }
        }
        dynamic_parameters.push(group_dynamic_parameters);
    }
    let immediate_dwords = immediate_ranges
        .last()
        .map_or(0, |range| (range.offset + range.size) / 4);
    // A root signature has one shared 64-DWORD budget. Descriptor tables cost
    // one DWORD and root CBV/SRV/UAV descriptors cost two; the immediate ABI is
    // capped at 32 DWORDs. The public immediate-data fact therefore reports 128
    // bytes rather than the theoretical 256-byte empty-signature maximum.
    if immediate_dwords > 32
        || parameters
            .len()
            .saturating_add(root_descriptor_count)
            .saturating_add(immediate_dwords as usize)
            > 64
    {
        return Err(Dx12Failure::Unsupported {
            what: "a pipeline interface that exceeds the DX12 root-signature budget",
            why: "immediate data is limited to 32 DWORDs and shares DX12's 64-DWORD root-signature budget with descriptor tables",
        });
    }
    let immediate_parameter = (!immediate_ranges.is_empty()).then(|| parameters.len() as u32);
    if immediate_dwords != 0 {
        parameters.push(D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Constants: D3D12_ROOT_CONSTANTS {
                    // This is the Fluxel shader-ABI reserved root-constant
                    // binding (`b0, space=0xffff_ffff`), deliberately outside
                    // all logical bind-group spaces. `offset` is a byte offset
                    // inside this one block, never a CBV register number.
                    ShaderRegister: 0,
                    RegisterSpace: u32::MAX,
                    Num32BitValues: immediate_dwords,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        });
    }

    let description = D3D12_ROOT_SIGNATURE_DESC {
        NumParameters: parameters.len() as u32,
        pParameters: parameters.as_ptr(),
        NumStaticSamplers: 0,
        pStaticSamplers: std::ptr::null(),
        // A graphics PSO with an input layout is invalid unless this root
        // signature explicitly permits input-assembler bindings. Compute never
        // needs that permission, so it retains the narrower native contract.
        Flags: if uses_input_assembler {
            D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT
        } else {
            D3D12_ROOT_SIGNATURE_FLAG_NONE
        },
    };
    let mut serialized: Option<ID3DBlob> = None;
    // `description` points only at the vectors above, which stay live for this
    // synchronous serializer call. D3D12 returns an owned COM blob on success.
    unsafe {
        D3D12SerializeRootSignature(
            &description,
            D3D_ROOT_SIGNATURE_VERSION_1,
            &mut serialized,
            None,
        )
    }
    .map_err(|error| {
        Dx12Failure::Native(crate::backend::dx12::ffi::NativeError::new(
            &error,
            "Dx12Device::create_compute_pipeline",
        ))
    })?;
    let serialized = serialized.ok_or_else(|| {
        Dx12Failure::Native(
            crate::backend::dx12::ffi::NativeError::driver_contract_violation(
                "D3D12SerializeRootSignature succeeded without returning a root-signature blob",
                "Dx12Device::create_compute_pipeline",
            ),
        )
    })?;
    // The blob is an owned COM object. Its contents stay valid for the slice's
    // lifetime, and CreateRootSignature copies them before returning.
    let bytes = unsafe {
        slice::from_raw_parts(
            serialized.GetBufferPointer() as *const u8,
            serialized.GetBufferSize(),
        )
    };
    let handle = unsafe { device.CreateRootSignature::<ID3D12RootSignature>(0, bytes) }.map_err(
        |error| {
            Dx12Failure::Native(crate::backend::dx12::ffi::NativeError::new(
                &error,
                "Dx12Device::create_compute_pipeline",
            ))
        },
    )?;
    Ok(Dx12RootSignature {
        handle,
        view_parameters,
        sampler_parameters,
        dynamic_parameters,
        immediate_parameter,
        immediate_ranges: immediate_ranges.to_vec(),
    })
}
