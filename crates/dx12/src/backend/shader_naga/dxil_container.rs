//! Checked byte-level reads of DXIL container parts.

use fluxel_rhi_core::api::{Error, Result};

// The DXIL container and part headers are specified in DXC's DxilContainer.h.
const CONTAINER_HEADER_SIZE: usize = 32;
const PART_HEADER_SIZE: usize = 8;

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
