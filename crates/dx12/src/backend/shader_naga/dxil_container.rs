//! Checked byte-level reads of DXIL container parts.

use fluxel_rhi_core::api::{Error, Result};

// The DXIL container and part headers are specified in DXC's DxilContainer.h.
const CONTAINER_HEADER_SIZE: usize = 32;
const PART_HEADER_SIZE: usize = 8;
// DXC's DXIL ResourceKind values shared by PSV0 and RDAT.
pub(super) const FEEDBACK_TEXTURE_2D: u32 = 17;
pub(super) const FEEDBACK_TEXTURE_2D_ARRAY: u32 = 18;

/// Verify the selected pixel program's written dual-source output registers.
pub(in crate::backend) fn validate_dual_source_outputs(
    bytes: Option<&[u8]>,
    required: [u8; 2],
) -> Result<()> {
    use windows::Win32::Graphics::Direct3D12::D3D12_SHVER_PIXEL_SHADER;
    // DxilProgramSignature and DxilProgramSignatureElement from DxilContainer.h.
    const SIGNATURE_HEADER_SIZE: usize = 8;
    const ELEMENT_SIZE: usize = 32;
    const TARGET: u32 = 64;
    let missing =
        || Error::Unsupported("DX12 dual-source blend requires written SV_Target0 and SV_Target1");
    let (program, signature) = parts(bytes.ok_or_else(missing)?, *b"OSG1")?;
    if word(program, 0)? >> 16 != D3D12_SHVER_PIXEL_SHADER.0 as u32 {
        return Err(Error::Shader(
            "DX12 dual-source blend requires a pixel program".into(),
        ));
    }
    let signature = signature.ok_or_else(missing)?;
    let count = word(signature, 0)? as usize;
    let offset = word(signature, 4)? as usize;
    if offset < SIGNATURE_HEADER_SIZE {
        return Err(malformed());
    }
    let size = count.checked_mul(ELEMENT_SIZE).ok_or_else(malformed)?;
    let elements = signature
        .get(offset..)
        .and_then(|data| data.get(..size))
        .ok_or_else(malformed)?;
    let mut written = [0u8; 2];
    for element in elements.chunks_exact(ELEMENT_SIZE) {
        let mask = element[24] & !element[25] & 0xf;
        if word(element, 12)? != TARGET || mask == 0 {
            continue;
        }
        let index = word(element, 8)?;
        if index > 1 {
            return Err(Error::Unsupported(
                "DX12 dual-source blend permits only color outputs 0 and 1",
            ));
        }
        written[index as usize] |= mask;
    }
    if written.contains(&0) {
        return Err(missing());
    }
    if written
        .into_iter()
        .zip(required)
        .any(|(written, required)| written & required != required)
    {
        return Err(Error::Unsupported(
            "DX12 dual-source fragment outputs omit a component used by blending",
        ));
    }
    Ok(())
}

/// Qualify a precompiled program declaring native SM 6.6 compute atomics.
pub(in crate::backend) fn validate_atomic_dxil(bytes: &[u8], maximum_model: u16) -> Result<()> {
    let (program, _) = parts(bytes, *b"PSV0")?;
    let version = word(program, 0)?;
    let stage = version >> 16;
    let compute = windows::Win32::Graphics::Direct3D12::D3D12_SHVER_COMPUTE_SHADER.0 as u32;
    if stage != compute || !(66..=maximum_model).contains(&shader_model(version)) {
        return Err(Error::Unsupported(
            "DX12 Int64 atomics require compute DXIL at SM 6.6 or newer",
        ));
    }
    Ok(())
}

pub(super) fn shader_model(program_version: u32) -> u16 {
    (((program_version >> 4) & 0xf) * 10 + (program_version & 0xf)) as u16
}

pub(in crate::backend) fn validate_sampler_feedback_model(
    bytes: &[u8],
    maximum_model: u16,
) -> Result<()> {
    let (program, _) = parts(bytes, *b"PSV0")?;
    if !(65..=maximum_model).contains(&shader_model(word(program, 0)?)) {
        return Err(Error::Unsupported("DX12 sampler feedback shader model"));
    }
    Ok(())
}

/// The native artifact must declare the typed-resource Int64 atomic feature.
pub(in crate::backend) fn validate_typed_int64_atomics(bytes: &[u8]) -> Result<()> {
    use windows::Win32::Graphics::Direct3D12::D3D_SHADER_REQUIRES_ATOMIC_INT64_ON_TYPED_RESOURCE;
    let (_, flags) = parts(bytes, *b"SFI0")?;
    let flags = flags.ok_or_else(malformed)?;
    if flags.len() < 8 {
        return Err(malformed());
    }
    if word(flags, 0)? & D3D_SHADER_REQUIRES_ATOMIC_INT64_ON_TYPED_RESOURCE == 0 {
        return Err(Error::Unsupported(
            "DX12 artifact does not use typed Int64 atomics",
        ));
    }
    Ok(())
}

pub(super) fn parts(bytes: &[u8], tag: [u8; 4]) -> Result<(&[u8], Option<&[u8]>)> {
    if !bytes.starts_with(b"DXBC") {
        return Err(malformed());
    }
    let size = word(bytes, 24)? as usize;
    let bytes = bytes.get(..size).ok_or_else(malformed)?;
    let count = word(bytes, 28)? as usize;
    let index_size = count.checked_mul(4).ok_or_else(malformed)?;
    let offsets = bytes
        .get(CONTAINER_HEADER_SIZE..)
        .and_then(|data| data.get(..index_size))
        .ok_or_else(malformed)?;
    let mut program = None;
    let mut validation = None;
    for offset in offsets.chunks_exact(4) {
        let offset = word(offset, 0)? as usize;
        if offset < CONTAINER_HEADER_SIZE + index_size {
            return Err(malformed());
        }
        let part = bytes.get(offset..).ok_or_else(malformed)?;
        let size = word(part, 4)? as usize;
        let data = part
            .get(PART_HEADER_SIZE..)
            .and_then(|data| data.get(..size))
            .ok_or_else(malformed)?;
        let destination = match &part[..4] {
            b"DXIL" => &mut program,
            candidate if candidate == tag => &mut validation,
            _ => continue,
        };
        if destination.replace(data).is_some() {
            return Err(malformed());
        }
    }
    Ok((program.ok_or_else(malformed)?, validation))
}

pub(super) fn word(bytes: &[u8], offset: usize) -> Result<u32> {
    let data = bytes
        .get(offset..)
        .and_then(|data| data.first_chunk::<4>())
        .ok_or_else(malformed)?;
    Ok(u32::from_le_bytes(*data))
}

pub(super) fn malformed() -> Error {
    Error::Shader("invalid DXIL program or pipeline validation data".into())
}
