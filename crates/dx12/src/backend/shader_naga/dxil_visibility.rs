//! Reads register visibility facts from the DXIL program and PSV0 parts.
//!
//! PSV0 is native pipeline validation metadata, independent of DXC's optional
//! reflection output. A library has no single selected program to validate here.

use fluxel_rhi_core::api::{
    BindingType, Error, Result, ShaderStage, StorageAccess, TextureFormat, TextureViewDimension,
};
use windows::Win32::Graphics::Direct3D12::*;

use super::super::pipeline::PipelineLayoutState;
use super::dxil_container::{
    FEEDBACK_TEXTURE_2D, FEEDBACK_TEXTURE_2D_ARRAY, malformed, parts, word,
};

// Layouts follow DxilContainer.h and DxilPipelineStateValidation.h in DXC.
const RESOURCE_PREFIX_SIZE: usize = 16;
const RESOURCE_DETAILS_SIZE: usize = 24;
const UAV_TYPED: u32 = 6;
const TEXTURE_2D: u32 = 2;
const ACCELERATION_STRUCTURE: u32 = 16;
const USED_BY_ATOMIC_64: u32 = 1;

fn atomic_texture_ranges(
    layout: &PipelineLayoutState,
    format: TextureFormat,
) -> impl Iterator<Item = (u32, u64, u64)> + Clone + '_ {
    layout
        .register_map
        .iter()
        .filter_map(move |&(space, binding, register, class, count)| {
            (class == 2
                && layout.binding_entries.iter().any(|(group, entry)| {
                    *group == space
                        && entry.binding == binding
                        && matches!(
                            entry.ty,
                            BindingType::StorageTexture {
                                format: actual,
                                dimension: TextureViewDimension::D2,
                                access: StorageAccess::Atomic,
                            } if actual == format
                        )
                }))
            .then_some((
                space,
                u64::from(register),
                u64::from(register) + u64::from(count),
            ))
        })
}

fn atomic_texture_coverage(
    ranges: impl Iterator<Item = (u32, u64, u64)>,
    space: u32,
    start: u64,
    end: u64,
) -> u64 {
    ranges
        .filter(|&(group, ..)| group == space)
        .map(|(_, first, limit)| end.min(limit).saturating_sub(start.max(first)))
        .sum()
}

pub(in crate::backend) fn validate(
    bytes: &[u8],
    stage: ShaderStage,
    layout: &PipelineLayoutState,
) -> Result<()> {
    let validate_visibility = layout.requires_native_binding_validation(stage);
    let has_feedback = layout
        .binding_entries
        .iter()
        .any(|(_, entry)| matches!(entry.ty, BindingType::SamplerFeedback { .. }));
    for row in resource_rows(bytes, stage)? {
        let kind = if row.len() >= RESOURCE_DETAILS_SIZE {
            word(row, 16)?
        } else {
            0
        };
        let acceleration = kind == ACCELERATION_STRUCTURE;
        let feedback = match kind {
            FEEDBACK_TEXTURE_2D => Some(TextureViewDimension::D2),
            FEEDBACK_TEXTURE_2D_ARRAY => Some(TextureViewDimension::D2Array),
            _ => None,
        };
        let resource_type = word(row, 0)?;
        if !validate_visibility
            && !acceleration
            && feedback.is_none()
            && !(has_feedback && (UAV_TYPED..=9).contains(&resource_type))
        {
            continue;
        }
        // PSVResourceType orders Sampler, CBV, SRVs, then UAVs.
        let class = match resource_type {
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
        if let Some(dimension) = feedback {
            if resource_type != UAV_TYPED {
                return Err(malformed());
            }
            let matches_layout = end - start == 1
                && feedback_bindings(layout, space, start..end).any(|(entry, actual)| {
                    actual == dimension && entry.visibility.contains(&stage)
                });
            if !matches_layout {
                return Err(Error::Shader(format!(
                    "DX12 sampler feedback UAV {start} in space {space} lacks a matching visible {dimension:?} binding"
                )));
            }
        } else if has_feedback
            && class == 2
            && feedback_bindings(layout, space, start..end)
                .next()
                .is_some()
        {
            return Err(Error::Shader(format!(
                "DX12 ordinary UAV range {start}..{end} in space {space} overlaps a sampler feedback binding"
            )));
        }
        if acceleration {
            if class != 1 {
                return Err(malformed());
            }
            layout.validate_native_acceleration_binding(stage, space, start..end)?;
        } else {
            layout.validate_native_binding(stage, space, class, start..end)?;
        }
    }
    Ok(())
}

fn feedback_bindings(
    layout: &PipelineLayoutState,
    space: u32,
    registers: std::ops::Range<u64>,
) -> impl Iterator<
    Item = (
        &fluxel_rhi_core::api::BindGroupLayoutEntry,
        TextureViewDimension,
    ),
> {
    layout
        .binding_entries
        .iter()
        .filter_map(move |(group, entry)| {
            let BindingType::SamplerFeedback { dimension, .. } = entry.ty else {
                return None;
            };
            (*group == space
                && layout
                    .register_map
                    .iter()
                    .any(|&(group, binding, register, class, _)| {
                        group == space
                            && binding == entry.binding
                            && class == 2
                            && registers.contains(&u64::from(register))
                    }))
            .then_some((entry, dimension))
        })
}

pub(in crate::backend) fn validate_r64uint_bindings(
    bytes: &[u8],
    layout: &PipelineLayoutState,
    declared: bool,
) -> Result<()> {
    let ranges = atomic_texture_ranges(layout, TextureFormat::R64Uint);
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
        let covered = atomic_texture_coverage(ranges.clone(), space, start, end);
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

pub(in crate::backend) fn validate_r32_atomic_bindings(
    bytes: &[u8],
    layout: &PipelineLayoutState,
    formats: &[TextureFormat],
) -> Result<()> {
    if formats.is_empty() {
        return Ok(());
    }
    let resources = resource_rows(bytes, ShaderStage::Compute)?;
    for &format in formats {
        let ranges = atomic_texture_ranges(layout, format);
        let mut found = false;
        for row in resources.clone() {
            if word(row, 0)? != UAV_TYPED
                || row.len() < RESOURCE_DETAILS_SIZE
                || word(row, 16)? != TEXTURE_2D
            {
                continue;
            }
            let space = word(row, 4)?;
            let start = u64::from(word(row, 8)?);
            let end = u64::from(word(row, 12)?) + 1;
            if start >= end {
                return Err(malformed());
            }
            let covered = atomic_texture_coverage(ranges.clone(), space, start, end);
            found |= covered == end - start;
        }
        if !found {
            return Err(Error::Unsupported(
                "DX12 R32 atomic shader requires a matching 2D atomic UAV",
            ));
        }
    }
    Ok(())
}
