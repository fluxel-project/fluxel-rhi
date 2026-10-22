//! Serialized cached PSO blobs keyed by canonical pipeline descriptions.
//!
//! Keys are supplied by pipeline lowering and must encode the entire native
//! descriptor and root signature without process-specific pointers. Blob
//! compatibility is verified again by the native CreatePipelineState call.
use super::super::{Dx12, Dx12Device, Dx12PipelineCache};
use fluxel_rhi_core::api::{self, Error, Result};
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};
use windows::Win32::Graphics::Direct3D12::ID3D12PipelineState;

const MAGIC: &[u8; 8] = b"FXD12PC1";
const LIMIT: usize = 256 * 1024 * 1024;

/// Encode independent canonical components without boundary ambiguity.
pub(crate) fn key(parts: &[&[u8]]) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    for part in parts {
        let n = u64::try_from(part.len()).map_err(|_| Error::InvalidRange)?;
        let size = out
            .len()
            .checked_add(8)
            .and_then(|v| v.checked_add(part.len()))
            .ok_or(Error::InvalidRange)?;
        if size > LIMIT {
            return Err(Error::InvalidRange);
        }
        out.extend_from_slice(&n.to_le_bytes());
        out.extend_from_slice(part);
    }
    if out.is_empty() {
        return Err(Error::InvalidRange);
    }
    Ok(out)
}

/// Only native adapter/driver incompatibility permits uncached retry.
pub(crate) fn is_compatibility_error(error: &windows::core::Error) -> bool {
    use windows::Win32::Foundation::{
        D3D12_ERROR_ADAPTER_NOT_FOUND, D3D12_ERROR_DRIVER_VERSION_MISMATCH,
    };
    matches!(
        error.code(),
        D3D12_ERROR_ADAPTER_NOT_FOUND | D3D12_ERROR_DRIVER_VERSION_MISMATCH
    )
}

#[derive(Default)]
pub(crate) struct PipelineCacheState {
    entries: Mutex<BTreeMap<Vec<u8>, Vec<u8>>>,
}
impl PipelineCacheState {
    pub(crate) fn lookup(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        Ok(self
            .entries
            .lock()
            .map_err(|_| Error::InvalidState)?
            .get(key)
            .cloned())
    }
    pub(crate) fn store(&self, key: &[u8], pso: &ID3D12PipelineState) -> Result<()> {
        // SAFETY: the PSO remains alive for the duration of the synchronous call.
        let blob = unsafe { pso.GetCachedBlob() }.map_err(super::super::native_error)?;
        // SAFETY: D3D owns a valid byte buffer for the lifetime of the blob.
        let bytes = unsafe {
            std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize())
        };
        if key.is_empty() || key.len() > LIMIT || bytes.is_empty() || bytes.len() > LIMIT {
            return Err(Error::InvalidRange);
        }
        self.entries
            .lock()
            .map_err(|_| Error::InvalidState)?
            .insert(key.to_vec(), bytes.to_vec());
        Ok(())
    }
}
fn checksum(data: &[u8]) -> u32 {
    let mut crc = !0u32;
    for &byte in data {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            crc = (crc >> 1) ^ (0xedb88320 & 0u32.wrapping_sub(crc & 1));
        }
    }
    !crc
}
fn take_u32(data: &[u8], pos: &mut usize) -> Result<u32> {
    let end = pos.checked_add(4).ok_or(Error::InvalidState)?;
    let bytes: [u8; 4] = data
        .get(*pos..end)
        .ok_or(Error::InvalidState)?
        .try_into()
        .map_err(|_| Error::InvalidState)?;
    *pos = end;
    Ok(u32::from_le_bytes(bytes))
}
fn parse(data: &[u8]) -> Result<BTreeMap<Vec<u8>, Vec<u8>>> {
    if data.len() > LIMIT || data.len() < 16 || &data[..8] != MAGIC {
        return Err(Error::InvalidState);
    }
    let body_end = data.len() - 4;
    if checksum(&data[..body_end])
        != u32::from_le_bytes(
            data[body_end..]
                .try_into()
                .map_err(|_| Error::InvalidState)?,
        )
    {
        return Err(Error::InvalidState);
    }
    let mut pos = 8;
    let count = take_u32(data, &mut pos)? as usize;
    if count > (body_end.saturating_sub(pos)) / 10 {
        return Err(Error::InvalidState);
    }
    let mut entries = BTreeMap::new();
    for _ in 0..count {
        let k = take_u32(data, &mut pos)? as usize;
        let v = take_u32(data, &mut pos)? as usize;
        if k == 0 || v == 0 {
            return Err(Error::InvalidState);
        }
        let kend = pos.checked_add(k).ok_or(Error::InvalidState)?;
        let end = kend.checked_add(v).ok_or(Error::InvalidState)?;
        if end > body_end {
            return Err(Error::InvalidState);
        }
        if entries
            .insert(data[pos..kend].to_vec(), data[kend..end].to_vec())
            .is_some()
        {
            return Err(Error::InvalidState);
        }
        pos = end;
    }
    if pos != body_end {
        return Err(Error::InvalidState);
    }
    Ok(entries)
}
impl api::CacheBackend for Dx12 {
    fn create_cache(_: &Dx12Device, data: Option<&[u8]>) -> Result<Dx12PipelineCache> {
        Ok(Dx12PipelineCache(Arc::new(PipelineCacheState {
            entries: Mutex::new(data.map(parse).transpose()?.unwrap_or_default()),
        })))
    }
    fn serialize_cache(cache: &Dx12PipelineCache) -> Result<Vec<u8>> {
        let entries = cache.0.entries.lock().map_err(|_| Error::InvalidState)?;
        let mut out = MAGIC.to_vec();
        out.extend_from_slice(
            &u32::try_from(entries.len())
                .map_err(|_| Error::InvalidRange)?
                .to_le_bytes(),
        );
        for (key, value) in entries.iter() {
            let needed = out
                .len()
                .checked_add(8)
                .and_then(|n| n.checked_add(key.len()))
                .and_then(|n| n.checked_add(value.len()))
                .and_then(|n| n.checked_add(4))
                .ok_or(Error::InvalidRange)?;
            if needed > LIMIT {
                return Err(Error::InvalidRange);
            }
            out.extend_from_slice(
                &u32::try_from(key.len())
                    .map_err(|_| Error::InvalidRange)?
                    .to_le_bytes(),
            );
            out.extend_from_slice(
                &u32::try_from(value.len())
                    .map_err(|_| Error::InvalidRange)?
                    .to_le_bytes(),
            );
            out.extend_from_slice(key);
            out.extend_from_slice(value);
        }
        out.extend_from_slice(&checksum(&out).to_le_bytes());
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn serialized_entries_round_trip_and_reject_corruption() {
        let cache = Dx12PipelineCache(Arc::new(PipelineCacheState::default()));
        cache
            .0
            .entries
            .lock()
            .unwrap()
            .insert(b"stable-key".to_vec(), b"native-pso".to_vec());
        let data = <Dx12 as api::CacheBackend>::serialize_cache(&cache).unwrap();
        assert_eq!(
            parse(&data).unwrap().get(b"stable-key".as_slice()).unwrap(),
            b"native-pso"
        );
        let mut corrupted = data.clone();
        corrupted[15] ^= 1;
        assert!(matches!(parse(&corrupted), Err(Error::InvalidState)));
        for end in 0..data.len() {
            assert!(parse(&data[..end]).is_err());
        }
    }
    #[test]
    fn rejects_duplicate_keys_even_with_valid_checksum() {
        let mut data = MAGIC.to_vec();
        data.extend_from_slice(&2u32.to_le_bytes());
        for v in [1u8, 2] {
            data.extend_from_slice(&1u32.to_le_bytes());
            data.extend_from_slice(&1u32.to_le_bytes());
            data.extend_from_slice(&[3, v]);
        }
        data.extend_from_slice(&checksum(&data).to_le_bytes());
        assert!(matches!(parse(&data), Err(Error::InvalidState)));
    }
}
