//! Exclusive host access to shared Metal buffers.
//!
//! The renderer supplies GPU completion before mapping. A lease retains its
//! allocation and serializes host access across clones of the same buffer.

use super::{Metal, MetalBuffer, MetalDevice};
use fluxel_rhi_core::api::{self, Error, Result};
use std::{
    future::{Ready, ready},
    ops::Range,
};

pub struct MetalMap {
    #[cfg(target_vendor = "apple")]
    buffer: MetalBuffer,
    #[cfg(target_vendor = "apple")]
    offset: usize,
    #[cfg(target_vendor = "apple")]
    len: usize,
    #[cfg(target_vendor = "apple")]
    mode: api::MapMode,
}

impl Drop for MetalMap {
    fn drop(&mut self) {
        #[cfg(target_vendor = "apple")]
        self.buffer
            .mapped
            .store(false, std::sync::atomic::Ordering::Release);
    }
}

impl fluxel_rhi_core::backend::MappingBackend for Metal {
    type NativeMap = MetalMap;
    type MapFuture<'a> = Ready<Result<MetalMap>>;

    fn map<'a>(
        device: &'a MetalDevice,
        buffer: &'a MetalBuffer,
        range: Range<u64>,
        mode: api::MapMode,
    ) -> Self::MapFuture<'a> {
        ready(map(device, buffer, range, mode))
    }
    fn bytes(map: &MetalMap) -> &[u8] {
        #[cfg(target_vendor = "apple")]
        {
            use objc2_metal::MTLBuffer as _;
            // SAFETY: the retained shared buffer and exclusive lease cover this checked range.
            unsafe {
                std::slice::from_raw_parts(
                    map.buffer
                        .raw
                        .contents()
                        .cast::<u8>()
                        .as_ptr()
                        .add(map.offset),
                    map.len,
                )
            }
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = map;
            unreachable!()
        }
    }
    fn bytes_mut(map: &mut MetalMap) -> &mut [u8] {
        #[cfg(target_vendor = "apple")]
        {
            use objc2_metal::MTLBuffer as _;
            assert_eq!(map.mode, api::MapMode::Write);
            // SAFETY: this mutable lease is exclusive and its checked range is shared storage.
            unsafe {
                std::slice::from_raw_parts_mut(
                    map.buffer
                        .raw
                        .contents()
                        .cast::<u8>()
                        .as_ptr()
                        .add(map.offset),
                    map.len,
                )
            }
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = map;
            unreachable!()
        }
    }
    fn flush(map: &mut MetalMap, bytes: Range<u64>) -> Result<()> {
        checked_range(map, bytes)
    }
    fn invalidate(map: &mut MetalMap, bytes: Range<u64>) -> Result<()> {
        checked_range(map, bytes)
    }
    fn unmap(map: MetalMap) {
        drop(map)
    }
}

fn checked_range(map: &MetalMap, range: Range<u64>) -> Result<()> {
    #[cfg(target_vendor = "apple")]
    if range.start <= range.end && range.end <= map.len as u64 {
        return Ok(());
    }
    let _ = (map, range);
    Err(Error::InvalidRange)
}

fn map(
    device: &MetalDevice,
    buffer: &MetalBuffer,
    range: Range<u64>,
    mode: api::MapMode,
) -> Result<MetalMap> {
    #[cfg(target_vendor = "apple")]
    {
        use std::sync::{Arc, atomic::Ordering};
        if !Arc::ptr_eq(&device._state, &buffer.device) {
            return Err(Error::WrongDevice);
        }
        let usage = match mode {
            api::MapMode::Read => api::BufferUsage::MAP_READ,
            api::MapMode::Write => api::BufferUsage::MAP_WRITE,
        };
        if !buffer.usage.contains(usage) || buffer.memory == api::MemoryClass::Device {
            return Err(Error::Unsupported("Metal buffer mapping mode"));
        }
        if range.start >= range.end || range.end > buffer.size() || range.end > isize::MAX as u64 {
            return Err(Error::InvalidRange);
        }
        let offset = usize::try_from(range.start).map_err(|_| Error::InvalidRange)?;
        let len = usize::try_from(range.end - range.start).map_err(|_| Error::InvalidRange)?;
        if buffer.mapped.swap(true, Ordering::AcqRel) {
            return Err(Error::InvalidState);
        }
        Ok(MetalMap {
            buffer: buffer.clone(),
            offset,
            len,
            mode,
        })
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        let _ = (device, buffer, range, mode);
        Err(Error::Unsupported("Metal requires an Apple target"))
    }
}
