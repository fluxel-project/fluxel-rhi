//! Fence-retired upload pages for immediate command recording.
//!
//! An encoder exclusively owns its pages until dropped. Submitted encoders are
//! retained until fence completion, including uncertain submissions; dropping an
//! unsubmitted encoder returns its pages immediately.
use super::{Dx12Encoder, barrier};
use crate::backend::{Dx12Buffer, native_error};
use fluxel_rhi_core::api::{BufferUsage, Error, MemoryClass, Result};
use std::sync::{Arc, Mutex};
use windows::Win32::Graphics::{Direct3D12::*, Dxgi::Common::*};

const PAGE_SIZE: u64 = 64 * 1024;
const MAX_CACHED_BYTES: u64 = 8 * 1024 * 1024;
const ALIGNMENT: u64 = 16;

pub(super) fn write_buffer(
    encoder: &mut Dx12Encoder,
    dst: &Dx12Buffer,
    offset: u64,
    bytes: &[u8],
) -> Result<()> {
    let size = validate_destination(dst, offset, bytes.len())?;
    if size != 0 {
        let source = encoder
            .upload_arena
            .write(&encoder.device.0.native, bytes)?;
        record_copy(encoder, dst, offset, size, source);
    }
    Ok(())
}

pub(super) fn write_buffer_with(
    encoder: &mut Dx12Encoder,
    dst: &Dx12Buffer,
    offset: u64,
    size: usize,
    writer: impl FnOnce(&mut [u8]) -> bool,
) -> Result<()> {
    let len = validate_destination(dst, offset, size)?;
    if len == 0 {
        writer(&mut []);
    } else if let Some(source) =
        encoder
            .upload_arena
            .write_with(&encoder.device.0.native, size, ALIGNMENT, writer)?
    {
        record_copy(encoder, dst, offset, len, source);
    }
    Ok(())
}

fn validate_destination(dst: &Dx12Buffer, offset: u64, size: usize) -> Result<u64> {
    if dst.memory != MemoryClass::Device || !dst.usage.contains(BufferUsage::COPY_DST) {
        return Err(Error::Unsupported(
            "DX12 write_buffer requires a device COPY_DST buffer",
        ));
    }
    let size = u64::try_from(size).map_err(|_| Error::InvalidRange)?;
    if size != 0 && offset.checked_add(size).is_none_or(|end| end > dst.size) {
        return Err(Error::InvalidRange);
    }
    Ok(size)
}

fn record_copy(
    encoder: &mut Dx12Encoder,
    dst: &Dx12Buffer,
    offset: u64,
    size: u64,
    (source, source_offset): (ID3D12Resource, u64),
) {
    // The write operation starts and ends at the explicit COMMON boundary.
    barrier(
        &encoder.list,
        &dst.resource,
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_COPY_DEST,
    );
    // SAFETY: both ranges were checked. The encoder retains the source page
    // and destination buffer until the submitted copy has completed.
    unsafe {
        encoder
            .list
            .CopyBufferRegion(&dst.resource, offset, &source, source_offset, size);
    }
    barrier(
        &encoder.list,
        &dst.resource,
        D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_RESOURCE_STATE_COMMON,
    );
    encoder.buffers.push(dst.clone());
}

fn allocation_size(size: u64) -> Result<u64> {
    let size = PAGE_SIZE.max(size);
    if size > isize::MAX as u64 {
        return Err(Error::InvalidRange);
    }
    Ok(size)
}

struct Page {
    resource: ID3D12Resource,
    // Exclusive CPU write ownership while checked out; mapped until drop.
    mapped: usize,
    size: u64,
    cursor: u64,
}
impl Drop for Page {
    fn drop(&mut self) {
        // SAFETY: this page owns its map and cannot be accessed after drop.
        unsafe { self.resource.Unmap(0, None) };
    }
}
#[derive(Default)]
pub(crate) struct UploadPool {
    pages: Mutex<Vec<Page>>,
}
impl UploadPool {
    fn acquire(&self, device: &ID3D12Device, size: u64) -> Result<Page> {
        if size <= PAGE_SIZE {
            if let Some(mut page) = self.pages.lock().unwrap_or_else(|p| p.into_inner()).pop() {
                page.cursor = 0;
                return Ok(page);
            }
        }
        let size = allocation_size(size)?;
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_UPLOAD,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
            MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
            CreationNodeMask: 1,
            VisibleNodeMask: 1,
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Alignment: 0,
            Width: size,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_UNKNOWN,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_NONE,
        };
        let mut resource = None;
        crate::diagnostics::record(crate::diagnostics::Operation::UploadResources, 1);
        // SAFETY: descriptors and output slot are valid for the synchronous call.
        unsafe {
            device.CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_GENERIC_READ,
                None,
                &mut resource,
            )
        }
        .map_err(native_error)?;
        let resource: ID3D12Resource = resource.ok_or_else(|| {
            Error::Backend("DX12 upload page allocation returned no resource".into())
        })?;
        crate::diagnostics::record(crate::diagnostics::Operation::UploadCapacityBytes, size);
        let mut mapped = std::ptr::null_mut();
        crate::diagnostics::record(crate::diagnostics::Operation::UploadMaps, 1);
        // SAFETY: upload buffers are CPU writable; the empty range declares no CPU reads.
        unsafe {
            resource.Map(
                0,
                Some(&D3D12_RANGE { Begin: 0, End: 0 }),
                Some(&mut mapped),
            )
        }
        .map_err(native_error)?;
        if mapped.is_null() {
            // SAFETY: successful Map must be balanced.
            unsafe { resource.Unmap(0, None) };
            return Err(Error::Backend("DX12 upload page mapped to null".into()));
        }
        Ok(Page {
            resource,
            mapped: mapped as usize,
            size,
            cursor: 0,
        })
    }
    fn release(&self, mut pages: Vec<Page>) {
        let mut cache = self.pages.lock().unwrap_or_else(|p| p.into_inner());
        for page in pages.drain(..) {
            if page.size == PAGE_SIZE && (cache.len() as u64) < MAX_CACHED_BYTES / PAGE_SIZE {
                cache.push(page);
            }
        }
    }
}
pub(crate) struct UploadArena {
    pool: Arc<UploadPool>,
    pages: Vec<Page>,
}
impl UploadArena {
    pub(crate) fn new(pool: Arc<UploadPool>) -> Self {
        Self {
            pool,
            pages: Vec::new(),
        }
    }
    pub(crate) fn write(
        &mut self,
        device: &ID3D12Device,
        bytes: &[u8],
    ) -> Result<(ID3D12Resource, u64)> {
        let len = u64::try_from(bytes.len()).map_err(|_| Error::InvalidRange)?;
        let (page, start) = self.reserve(device, len, ALIGNMENT)?;
        // SAFETY: reserve checked the mapped range and caller memory cannot
        // alias this private arena.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), page.ptr(start)?, bytes.len());
        }
        Ok(page.finish_write(start, len))
    }

    pub(super) fn write_with(
        &mut self,
        device: &ID3D12Device,
        size: usize,
        alignment: u64,
        writer: impl FnOnce(&mut [u8]) -> bool,
    ) -> Result<Option<(ID3D12Resource, u64)>> {
        let len = u64::try_from(size).map_err(|_| Error::InvalidRange)?;
        let (page, start) = self.reserve(device, len, alignment)?;
        let ptr = page.ptr(start)?;
        // SAFETY: reserve checked this entire range is exclusively writable.
        // Clear recycled storage before exposing an initialized slice.
        let written = unsafe {
            std::ptr::write_bytes(ptr, 0, size);
            writer(std::slice::from_raw_parts_mut(ptr, size))
        };
        Ok(written.then(|| page.finish_write(start, len)))
    }

    fn reserve(
        &mut self,
        device: &ID3D12Device,
        len: u64,
        alignment: u64,
    ) -> Result<(&mut Page, u64)> {
        debug_assert!(alignment.is_power_of_two());
        if len == 0 {
            return Err(Error::InvalidRange);
        }
        let available = self.pages.last().and_then(|p| {
            p.cursor
                .checked_add(alignment - 1)
                .map(|n| n & !(alignment - 1))
                .and_then(|start| {
                    start
                        .checked_add(len)
                        .filter(|end| *end <= p.size)
                        .map(|_| start)
                })
        });
        let start = if let Some(start) = available {
            start
        } else {
            self.pages.push(self.pool.acquire(device, len)?);
            0
        };
        Ok((self.pages.last_mut().expect("page acquired"), start))
    }
}

impl Page {
    fn ptr(&self, start: u64) -> Result<*mut u8> {
        let start = usize::try_from(start).map_err(|_| Error::InvalidRange)?;
        // SAFETY: reserve selected a byte offset within the live mapping.
        Ok(unsafe { (self.mapped as *mut u8).add(start) })
    }

    fn finish_write(&mut self, start: u64, len: u64) -> (ID3D12Resource, u64) {
        self.cursor = start + len;
        crate::diagnostics::record(crate::diagnostics::Operation::UploadBytes, len);
        (self.resource.clone(), start)
    }
}
impl Drop for UploadArena {
    fn drop(&mut self) {
        self.pool.release(std::mem::take(&mut self.pages));
    }
}

#[cfg(test)]
#[path = "../../tests/backend/encoder/upload.rs"]
mod tests;
