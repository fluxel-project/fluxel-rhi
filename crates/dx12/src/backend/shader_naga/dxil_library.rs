//! Validates the selected DXR library export against native RDAT metadata.

use fluxel_rhi_core::api::{Error, Result, ShaderStage};
use windows::Win32::Graphics::Direct3D12::*;

use super::super::pipeline::PipelineLayoutState;
use super::dxil_container::{parts, shader_model, word};

// Prefixes from DXC's DxilRuntimeReflection.h and RDAT_LibraryTypes.inl.
const RDAT_VERSION: u32 = 0x10;
const FUNCTION_PREFIX_SIZE: usize = 20;
const RESOURCE_PREFIX_SIZE: usize = 24;
const SRV_CLASS: u32 = 0;
const ACCELERATION_STRUCTURE: u32 = 16;

pub(in crate::backend) fn validate_export(
    bytes: &[u8],
    export: &str,
    stage: ShaderStage,
    maximum_model: u16,
    layout: &PipelineLayoutState,
) -> Result<()> {
    let expected = match stage {
        ShaderStage::RayGeneration => D3D12_SHVER_RAY_GENERATION_SHADER,
        ShaderStage::Miss => D3D12_SHVER_MISS_SHADER,
        ShaderStage::Callable => D3D12_SHVER_CALLABLE_SHADER,
        ShaderStage::ClosestHit => D3D12_SHVER_CLOSEST_HIT_SHADER,
        ShaderStage::AnyHit => D3D12_SHVER_ANY_HIT_SHADER,
        ShaderStage::Intersection => D3D12_SHVER_INTERSECTION_SHADER,
        _ => return Err(Error::Unsupported("DX12 library export stage")),
    };
    let (program, rdat) = parts(bytes, *b"RDAT")?;
    let version = word(program, 0)?;
    if version >> 16 != D3D12_SHVER_LIBRARY.0 as u32 {
        return Err(Error::Shader(
            "DX12 ray pipeline requires a DXIL library".into(),
        ));
    }
    if !(63..=maximum_model).contains(&shader_model(version)) {
        return Err(Error::Unsupported("DX12 ray library shader model"));
    }
    let rdat = rdat.ok_or(Error::Unsupported(
        "DX12 library export admission requires RDAT",
    ))?;
    let tables = export_tables(rdat)?;
    let strings = tables.strings.ok_or_else(malformed)?;
    let mut selected = None;
    for function in table_rows(
        tables.functions.ok_or_else(malformed)?,
        FUNCTION_PREFIX_SIZE,
    )? {
        let name = string(strings, word(function, 0)?)?;
        let unmangled = string(strings, word(function, 4)?)?;
        if (export.as_bytes() == name || export.as_bytes() == unmangled)
            && selected.replace(function).is_some()
        {
            return Err(Error::Shader(format!(
                "ambiguous DXIL library export {export}"
            )));
        }
    }
    let function =
        selected.ok_or_else(|| Error::Shader(format!("missing DXIL library export {export}")))?;
    if word(function, 16)? != expected.0 as u32 {
        return Err(Error::Shader(format!(
            "DXIL library export {export} does not match {stage:?}"
        )));
    }
    let resource_list = word(function, 8)?;
    if resource_list != u32::MAX {
        validate_acceleration_resources(&tables, resource_list, stage, layout)?;
    }
    Ok(())
}

#[derive(Default)]
struct RuntimeTables<'a> {
    strings: Option<&'a [u8]>,
    indices: Option<&'a [u8]>,
    resources: Option<&'a [u8]>,
    functions: Option<&'a [u8]>,
}

fn validate_acceleration_resources(
    tables: &RuntimeTables<'_>,
    resource_list: u32,
    stage: ShaderStage,
    layout: &PipelineLayoutState,
) -> Result<()> {
    // RDAT array references are word offsets. Each row begins with its length.
    let offset = (resource_list as usize)
        .checked_mul(4)
        .ok_or_else(malformed)?;
    let row = tables
        .indices
        .and_then(|table| table.get(offset..))
        .ok_or_else(malformed)?;
    let bytes = (word(row, 0)? as usize)
        .checked_mul(4)
        .ok_or_else(malformed)?;
    let indices = row
        .get(4..)
        .and_then(|data| data.get(..bytes))
        .ok_or_else(malformed)?;
    if indices.is_empty() {
        return Ok(());
    }
    let resources = table_rows(
        tables.resources.ok_or_else(malformed)?,
        RESOURCE_PREFIX_SIZE,
    )?;
    for entry in indices.chunks_exact(4) {
        let resource = resources
            .clone()
            .nth(word(entry, 0)? as usize)
            .ok_or_else(malformed)?;
        if word(resource, 4)? == ACCELERATION_STRUCTURE {
            if word(resource, 0)? != SRV_CLASS {
                return Err(malformed());
            }
            let space = word(resource, 12)?;
            let start = u64::from(word(resource, 16)?);
            let end = u64::from(word(resource, 20)?) + 1;
            if start >= end {
                return Err(malformed());
            }
            layout.validate_native_acceleration_binding(stage, space, start..end)?;
        }
    }
    Ok(())
}

fn export_tables(rdat: &[u8]) -> Result<RuntimeTables<'_>> {
    if word(rdat, 0)? != RDAT_VERSION {
        return Err(Error::Unsupported("DX12 library RDAT version"));
    }
    let count = word(rdat, 4)? as usize;
    let bytes = count.checked_mul(4).ok_or_else(malformed)?;
    let offsets = rdat
        .get(8..)
        .and_then(|data| data.get(..bytes))
        .ok_or_else(malformed)?;
    let mut tables = RuntimeTables::default();
    for offset in offsets.chunks_exact(4) {
        let offset = word(offset, 0)? as usize;
        if offset < 8 + bytes || offset % 4 != 0 {
            return Err(malformed());
        }
        let part = rdat.get(offset..).ok_or_else(malformed)?;
        let tag = word(part, 0)?;
        let size = word(part, 4)? as usize;
        let data = part
            .get(8..)
            .and_then(|data| data.get(..size))
            .ok_or_else(malformed)?;
        let destination = match tag {
            1 => &mut tables.strings,
            2 => &mut tables.indices,
            3 => &mut tables.resources,
            4 => &mut tables.functions,
            _ => continue,
        };
        if destination.replace(data).is_some() {
            return Err(malformed());
        }
    }
    Ok(tables)
}

fn table_rows(table: &[u8], minimum_stride: usize) -> Result<std::slice::ChunksExact<'_, u8>> {
    let count = word(table, 0)? as usize;
    let stride = word(table, 4)? as usize;
    if stride < minimum_stride || stride % 4 != 0 {
        return Err(malformed());
    }
    let size = count.checked_mul(stride).ok_or_else(malformed)?;
    let data = table
        .get(8..)
        .and_then(|data| data.get(..size))
        .ok_or_else(malformed)?;
    Ok(data.chunks_exact(stride))
}

fn string(table: &[u8], offset: u32) -> Result<&[u8]> {
    let value = table.get(offset as usize..).ok_or_else(malformed)?;
    let end = value
        .iter()
        .position(|&byte| byte == 0)
        .ok_or_else(malformed)?;
    Ok(&value[..end])
}

fn malformed() -> Error {
    Error::Shader("invalid DXIL library runtime data".into())
}
