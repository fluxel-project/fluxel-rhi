//! Validates the selected DXR library export against native RDAT metadata.

use fluxel_rhi_core::api::{Error, Result, ShaderStage};
use windows::Win32::Graphics::Direct3D12::*;

use super::dxil_container::{parts, shader_model, word};

// Prefixes from DXC's DxilRuntimeReflection.h and RDAT_LibraryTypes.inl.
const RDAT_VERSION: u32 = 0x10;
const FUNCTION_PREFIX_SIZE: usize = 20;

pub(in crate::backend) fn validate_export(
    bytes: &[u8],
    export: &str,
    stage: ShaderStage,
    maximum_model: u16,
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
    let (strings, functions) = export_tables(rdat)?;
    let mut selected_kind = None;
    for function in function_rows(functions)? {
        let name = string(strings, word(function, 0)?)?;
        let unmangled = string(strings, word(function, 4)?)?;
        if (export.as_bytes() == name || export.as_bytes() == unmangled)
            && selected_kind.replace(word(function, 16)?).is_some()
        {
            return Err(Error::Shader(format!(
                "ambiguous DXIL library export {export}"
            )));
        }
    }
    let kind = selected_kind
        .ok_or_else(|| Error::Shader(format!("missing DXIL library export {export}")))?;
    if kind != expected.0 as u32 {
        return Err(Error::Shader(format!(
            "DXIL library export {export} does not match {stage:?}"
        )));
    }
    Ok(())
}

fn export_tables(rdat: &[u8]) -> Result<(&[u8], &[u8])> {
    if word(rdat, 0)? != RDAT_VERSION {
        return Err(Error::Unsupported("DX12 library RDAT version"));
    }
    let count = word(rdat, 4)? as usize;
    let bytes = count.checked_mul(4).ok_or_else(malformed)?;
    let offsets = rdat
        .get(8..)
        .and_then(|data| data.get(..bytes))
        .ok_or_else(malformed)?;
    let mut strings = None;
    let mut functions = None;
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
            1 => &mut strings,
            4 => &mut functions,
            _ => continue,
        };
        if destination.replace(data).is_some() {
            return Err(malformed());
        }
    }
    Ok((
        strings.ok_or_else(malformed)?,
        functions.ok_or_else(malformed)?,
    ))
}

fn function_rows(table: &[u8]) -> Result<std::slice::ChunksExact<'_, u8>> {
    let count = word(table, 0)? as usize;
    let stride = word(table, 4)? as usize;
    if stride < FUNCTION_PREFIX_SIZE || stride % 4 != 0 {
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
