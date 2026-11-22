//! Reads register visibility facts from the DXIL program and PSV0 parts.
//!
//! PSV0 is native pipeline validation metadata, independent of DXC's optional
//! reflection output. A library has no single selected program to validate here.

use fluxel_rhi_core::api::{Error, Result, ShaderStage};
use windows::Win32::Graphics::Direct3D12::*;

use super::super::pipeline::PipelineLayoutState;
use super::dxil_container::{malformed, parts, word};

// Layouts follow DxilContainer.h and DxilPipelineStateValidation.h in DXC.
const RESOURCE_PREFIX_SIZE: usize = 16;

pub(in crate::backend) fn validate(
    bytes: &[u8],
    stage: ShaderStage,
    layout: &PipelineLayoutState,
) -> Result<()> {
    if !layout.requires_native_binding_validation(stage) {
        return Ok(());
    }
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
        return Ok(());
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
    for row in rows.chunks_exact(stride) {
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
