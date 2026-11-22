//! Reads register visibility facts from the DXIL program and PSV0 parts.
//!
//! PSV0 is native pipeline validation metadata, independent of DXC's optional
//! reflection output. A library has no single selected program to validate here.

use fluxel_rhi_core::api::{BindingType, Error, Result, ShaderStage, StorageAccess, TextureFormat};
use windows::Win32::Graphics::Direct3D12::*;

use super::super::pipeline::PipelineLayoutState;
use super::dxil_container::{malformed, parts, word};

// Layouts follow DxilContainer.h and DxilPipelineStateValidation.h in DXC.
const RESOURCE_PREFIX_SIZE: usize = 16;
const RESOURCE_DETAILS_SIZE: usize = 24;
const UAV_TYPED: u32 = 6;
const TEXTURE_2D: u32 = 2;
const USED_BY_ATOMIC_64: u32 = 1;

pub(in crate::backend) fn validate(
    bytes: &[u8],
    stage: ShaderStage,
    layout: &PipelineLayoutState,
) -> Result<()> {
    if !layout.requires_native_binding_validation(stage) {
        return Ok(());
    }
    for row in resource_rows(bytes, stage)? {
        // PSVResourceType orders Sampler, CBV, SRVs, then UAVs.
        let class = match word(row, 0)? {
            1 => 3,
            2 => 0,
            3..=5 => 1,
            6..=9 => 2,
            _ => return Err(Error::Unsupported("DX12 DXIL resource register class")),
        };
        let space = word(row, 4)?;
        let start = u64::from(word(row, 8)?);
        let end = u64::from(word(row, 12)?) + 1;
        if start >= end {
            return Err(malformed());
        }
        layout.validate_native_binding(stage, space, class, start..end)?;
    }
    Ok(())
}

pub(in crate::backend) fn validate_r64uint_bindings(
    bytes: &[u8],
    layout: &PipelineLayoutState,
    declared: bool,
) -> Result<()> {
    let r64_binding = |group, binding| {
        layout.binding_entries.iter().any(|(space, entry)| {
            *space == group
                && entry.binding == binding
                && matches!(
                    entry.ty,
                    BindingType::StorageTexture {
                        format: TextureFormat::R64Uint,
                        access: StorageAccess::Atomic,
                        ..
                    }
                )
        })
    };
    let ranges = layout
        .register_map
        .iter()
        .filter(|&&(group, binding, _, class, _)| class == 2 && r64_binding(group, binding));
    if !declared && ranges.clone().next().is_none() {
        return Ok(());
    }
    if declared {
        super::validate_typed_int64_atomics(bytes)?;
    }
    let mut found_atomic = false;
    for row in resource_rows(bytes, ShaderStage::Compute)? {
        let resource_type = word(row, 0)?;
        if !(UAV_TYPED..=9).contains(&resource_type) {
            continue;
        }
        let space = word(row, 4)?;
        let start = u64::from(word(row, 8)?);
        let end = u64::from(word(row, 12)?) + 1;
        if start >= end {
            return Err(malformed());
        }
        let covered: u64 = ranges
            .clone()
            .filter(|&&(group, ..)| group == space)
            .map(|&(_, _, register, _, count)| {
                let register = u64::from(register);
                end.min(register + u64::from(count))
                    .saturating_sub(start.max(register))
            })
            .sum();
        let atomic = resource_type == UAV_TYPED
            && row.len() >= RESOURCE_DETAILS_SIZE
            && word(row, 20)? & USED_BY_ATOMIC_64 != 0;
        if covered != 0 || (declared && atomic) {
            if !declared || !atomic || word(row, 16)? != TEXTURE_2D || covered != end - start {
                return Err(Error::Unsupported(
                    "DX12 R64Uint requires a declared 2D Int64 atomic UAV binding",
                ));
            }
            found_atomic = true;
        }
    }
    if declared && !found_atomic {
        return Err(Error::Unsupported(
            "DX12 R64Uint atomic UAV metadata is missing",
        ));
    }
    Ok(())
}

fn resource_rows(bytes: &[u8], stage: ShaderStage) -> Result<core::slice::ChunksExact<'_, u8>> {
    let expected_kind = match stage {
        ShaderStage::Vertex => D3D12_SHVER_VERTEX_SHADER,
        ShaderStage::Fragment => D3D12_SHVER_PIXEL_SHADER,
        ShaderStage::Compute => D3D12_SHVER_COMPUTE_SHADER,
        ShaderStage::Mesh => D3D12_SHVER_MESH_SHADER,
        ShaderStage::Task => D3D12_SHVER_AMPLIFICATION_SHADER,
        _ => return Err(Error::Unsupported("DX12 DXIL library stage visibility")),
    };
    let (program, validation) = parts(bytes, *b"PSV0")?;
    let validation = validation.ok_or(Error::Unsupported(
        "DX12 DXIL stage visibility requires PSV0",
    ))?;
    if word(program, 0)? >> 16 != expected_kind.0 as u32 {
        return Err(Error::Shader(
            "DXIL program kind disagrees with the selected stage".into(),
        ));
    }
    let runtime_size = word(validation, 0)? as usize;
    let count_offset = 4usize.checked_add(runtime_size).ok_or_else(malformed)?;
    let count = word(validation, count_offset)? as usize;
    if count == 0 {
        return Ok([].chunks_exact(RESOURCE_PREFIX_SIZE));
    }
    let records = validation.get(count_offset..).ok_or_else(malformed)?;
    let stride = word(records, 4)? as usize;
    if stride < RESOURCE_PREFIX_SIZE {
        return Err(malformed());
    }
    let byte_count = count.checked_mul(stride).ok_or_else(malformed)?;
    let rows = records
        .get(8..)
        .and_then(|data| data.get(..byte_count))
        .ok_or_else(malformed)?;
    Ok(rows.chunks_exact(stride))
}
