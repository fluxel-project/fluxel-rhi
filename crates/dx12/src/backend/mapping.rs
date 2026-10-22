//! CPU mapping for fixed-state DX12 upload and readback heaps.
use super::{Dx12, Dx12Buffer, Dx12Device};
use fluxel_rhi_core::api::{self, Error, Result};
use std::{
    future::{Ready, ready},
    ops::Range,
};
use windows::Win32::Graphics::Direct3D12::D3D12_RANGE;
pub struct Dx12Map {
    buffer: Dx12Buffer,
    ptr: *mut u8,
    range: Range<usize>,
    mode: api::MapMode,
}
impl api::MappingBackend for Dx12 {
    type NativeMap = Dx12Map;
    type MapFuture<'a> = Ready<Result<Dx12Map>>;
    fn map<'a>(
        _: &'a Dx12Device,
        buffer: &'a Dx12Buffer,
        range: Range<u64>,
        mode: api::MapMode,
    ) -> Self::MapFuture<'a> {
        ready((|| {
            let permitted = match mode {
                api::MapMode::Write => {
                    buffer.memory == api::MemoryClass::Upload
                        && buffer.usage.contains(api::BufferUsage::MAP_WRITE)
                }
                api::MapMode::Read => {
                    buffer.memory == api::MemoryClass::Readback
                        && buffer.usage.contains(api::BufferUsage::MAP_READ)
                }
            };
            if !permitted {
                return Err(Error::Unsupported("DX12 buffer mapping mode"));
            }
            if range.start >= range.end || range.end > buffer.size {
                return Err(Error::InvalidRange);
            }
            let start = usize::try_from(range.start).map_err(|_| Error::InvalidRange)?;
            let end = usize::try_from(range.end).map_err(|_| Error::InvalidRange)?;
            if buffer
                .mapped
                .swap(true, std::sync::atomic::Ordering::AcqRel)
            {
                return Err(Error::InvalidState);
            }
            let read = if mode == api::MapMode::Read {
                D3D12_RANGE {
                    Begin: start,
                    End: end,
                }
            } else {
                D3D12_RANGE { Begin: 0, End: 0 }
            };
            let mut ptr = std::ptr::null_mut();
            // SAFETY: the caller guarantees host/GPU access synchronization; ranges fit the resource.
            if let Err(error) = unsafe { buffer.resource.Map(0, Some(&read), Some(&mut ptr)) } {
                buffer
                    .mapped
                    .store(false, std::sync::atomic::Ordering::Release);
                return Err(super::native_error(error));
            }
            if ptr.is_null() {
                // SAFETY: Map succeeded, so its matching Unmap is required.
                unsafe { buffer.resource.Unmap(0, None) };
                buffer
                    .mapped
                    .store(false, std::sync::atomic::Ordering::Release);
                return Err(Error::Backend("DX12 Map returned null".into()));
            }
            Ok(Dx12Map {
                buffer: buffer.clone(),
                ptr: ptr.cast(),
                range: start..end,
                mode,
            })
        })())
    }
    fn bytes(map: &Dx12Map) -> &[u8] {
        // SAFETY: Map produced the pointer, the range is in-bounds, and the
        // per-buffer lease excludes a second mutable mapping.
        unsafe { std::slice::from_raw_parts(map.ptr.add(map.range.start), map.range.len()) }
    }
    fn bytes_mut(map: &mut Dx12Map) -> &mut [u8] {
        assert!(map.mode == api::MapMode::Write);
        // SAFETY: the exclusive lease and mutable borrow prevent aliases.
        unsafe { std::slice::from_raw_parts_mut(map.ptr.add(map.range.start), map.range.len()) }
    }
    fn flush(map: &mut Dx12Map, bytes: Range<u64>) -> Result<()> {
        validate(map, bytes)
    }
    fn invalidate(map: &mut Dx12Map, bytes: Range<u64>) -> Result<()> {
        validate(map, bytes)
    }
    fn unmap(map: Dx12Map) {
        let written = if map.mode == api::MapMode::Write {
            D3D12_RANGE {
                Begin: map.range.start,
                End: map.range.end,
            }
        } else {
            D3D12_RANGE { Begin: 0, End: 0 }
        };
        // SAFETY: this balances the successful Map held by this unique lease.
        unsafe {
            map.buffer.resource.Unmap(0, Some(&written));
        }
        map.buffer
            .mapped
            .store(false, std::sync::atomic::Ordering::Release);
    }
}
fn validate(map: &Dx12Map, bytes: Range<u64>) -> Result<()> {
    if bytes.start > bytes.end || bytes.end > map.range.len() as u64 {
        Err(Error::InvalidRange)
    } else {
        Ok(())
    }
}
