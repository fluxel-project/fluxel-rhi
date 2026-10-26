//! Device-scoped GL program binaries and their versioned portable envelope.
//!
//! The cache contains driver artifacts, not recorded GPU work. A driver
//! rejection becomes a source-link miss; every accepted binary is reflected
//! against the requesting pipeline's exact layout again.

use super::types::{Gl, GlDevice, GlDeviceState, GlPipelineCache};
use fluxel_rhi_core::api::{self, Error, Result};
use std::{cell::RefCell, collections::BTreeMap, rc::Rc};

#[derive(Clone)]
pub(crate) struct CachedBinary {
    pub(crate) format: u32,
    pub(crate) bytes: Vec<u8>,
}
pub(crate) struct GlPipelineCacheInner {
    pub(crate) state: Rc<GlDeviceState>,
    identity: Vec<u8>,
    entries: RefCell<BTreeMap<String, CachedBinary>>,
    #[cfg(test)]
    pub(crate) hits: std::cell::Cell<u32>,
}
#[cfg(test)]
impl GlPipelineCacheInner {
    pub(crate) fn invalidate_binary_format(&self) {
        self.entries
            .borrow_mut()
            .values_mut()
            .next()
            .expect("one cached program")
            .format = 0;
    }
}

const MAGIC: &[u8; 8] = b"FLXGLC02";
const MAX_CACHE_BYTES: usize = 256 * 1024 * 1024;

// Detect accidental corruption or a swapped binary/key pair. This checksum is
// not authentication: a caller able to rewrite the cache can recompute it.
fn digest(key: &[u8], format: u32, binary: &[u8]) -> u128 {
    let mut hash = 0x6c62272e07bb014262b821756295c58du128;
    for byte in (key.len() as u64)
        .to_le_bytes()
        .into_iter()
        .chain(key.iter().copied())
        .chain(format.to_le_bytes())
        .chain((binary.len() as u64).to_le_bytes())
        .chain(binary.iter().copied())
    {
        hash = (hash ^ u128::from(byte)).wrapping_mul(0x0000000001000000000000000000013bu128);
    }
    hash
}

impl fluxel_rhi_core::backend::CacheBackend for Gl {
    fn create_cache(device: &GlDevice, data: Option<&[u8]>) -> Result<GlPipelineCache> {
        #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
        {
            let identity = device
                .0
                .with_native("GL program cache", |p| p.program_cache_identity())
                .map_err(super::spi::native_error)?;
            let entries = match data {
                Some(data) => decode(data, &identity)?,
                None => BTreeMap::new(),
            };
            Ok(GlPipelineCache(Rc::new(GlPipelineCacheInner {
                state: device.0.clone(),
                identity,
                entries: RefCell::new(entries),
                #[cfg(test)]
                hits: std::cell::Cell::new(0),
            })))
        }
        #[cfg(not(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl")))]
        {
            let _ = (device, data);
            Err(Error::Unsupported("GL program binaries"))
        }
    }
    fn serialize_cache(cache: &GlPipelineCache) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        append(&mut out, &cache.0.identity)?;
        let entries = cache.0.entries.borrow();
        out.extend_from_slice(
            &u32::try_from(entries.len())
                .map_err(|_| Error::InvalidRange)?
                .to_le_bytes(),
        );
        for (key, binary) in entries.iter() {
            append(&mut out, key.as_bytes())?;
            out.extend_from_slice(&binary.format.to_le_bytes());
            append(&mut out, &binary.bytes)?;
            out.extend_from_slice(
                &digest(key.as_bytes(), binary.format, &binary.bytes).to_le_bytes(),
            );
            if out.len() > MAX_CACHE_BYTES {
                return Err(Error::InvalidRange);
            }
        }
        Ok(out)
    }
}

fn append(out: &mut Vec<u8>, value: &[u8]) -> Result<()> {
    out.extend_from_slice(
        &u32::try_from(value.len())
            .map_err(|_| Error::InvalidRange)?
            .to_le_bytes(),
    );
    out.extend_from_slice(value);
    Ok(())
}
fn read_u32(data: &mut &[u8]) -> Result<u32> {
    if data.len() < 4 {
        return Err(Error::InvalidRange);
    }
    let (head, tail) = data.split_at(4);
    *data = tail;
    Ok(u32::from_le_bytes(head.try_into().expect("length checked")))
}
fn read_bytes<'a>(data: &mut &'a [u8]) -> Result<&'a [u8]> {
    let n = read_u32(data)? as usize;
    if data.len() < n {
        return Err(Error::InvalidRange);
    }
    let (head, tail) = data.split_at(n);
    *data = tail;
    Ok(head)
}
fn decode(mut data: &[u8], identity: &[u8]) -> Result<BTreeMap<String, CachedBinary>> {
    if data.len() > MAX_CACHE_BYTES || !data.starts_with(MAGIC) {
        return Err(Error::InvalidRange);
    }
    data = &data[MAGIC.len()..];
    if read_bytes(&mut data)? != identity {
        return Err(Error::InvalidRange);
    }
    let count = read_u32(&mut data)?;
    let mut entries = BTreeMap::new();
    for _ in 0..count {
        let key = std::str::from_utf8(read_bytes(&mut data)?)
            .map_err(|_| Error::InvalidRange)?
            .to_owned();
        let format = read_u32(&mut data)?;
        let bytes = read_bytes(&mut data)?.to_vec();
        if data.len() < 16 {
            return Err(Error::InvalidRange);
        }
        let (checksum, tail) = data.split_at(16);
        data = tail;
        if u128::from_le_bytes(checksum.try_into().expect("length checked"))
            != digest(key.as_bytes(), format, &bytes)
        {
            return Err(Error::InvalidRange);
        }
        if bytes.is_empty()
            || entries
                .insert(key, CachedBinary { format, bytes })
                .is_some()
        {
            return Err(Error::InvalidRange);
        }
    }
    if !data.is_empty() {
        return Err(Error::InvalidRange);
    }
    Ok(entries)
}

#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
pub(crate) fn create_program(
    provider: &mut crate::backend::native::provider::NativeGlProvider,
    descriptor: &crate::backend::api::GlProgramDescriptor,
    cache: Option<&GlPipelineCache>,
) -> std::result::Result<
    (
        crate::backend::api::ProgramId,
        crate::backend::api::GlProgramReflection,
    ),
    crate::backend::api::GlError,
> {
    use crate::backend::api::GlShaderApi as _;
    let Some(cache) = cache else {
        return provider.create_program(descriptor);
    };
    let key = format!("{descriptor:?}");
    let binary = cache.0.entries.borrow().get(&key).cloned();
    let (program, reflection, new_binary, hit) =
        provider.cached_program(descriptor, binary.as_ref())?;
    #[cfg(test)]
    if hit {
        cache.0.hits.set(cache.0.hits.get() + 1);
    }
    #[cfg(not(test))]
    let _ = hit;
    if !hit {
        let mut entries = cache.0.entries.borrow_mut();
        if let Some(binary) = new_binary {
            entries.insert(key, binary);
        } else {
            entries.remove(&key);
        }
    }
    Ok((program, reflection))
}
