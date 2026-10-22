//! Fence-retired upload pages for immediate command recording.
//!
//! An encoder exclusively owns its pages until dropped. Submitted encoders are
//! retained until fence completion, including uncertain submissions; dropping an
//! unsubmitted encoder returns its pages immediately.
use crate::backend::native_error;
use fluxel_rhi_core::api::{Error, Result};
use std::sync::{Arc, Mutex};
use windows::Win32::Graphics::{Direct3D12::*, Dxgi::Common::*};

const PAGE_SIZE: u64 = 64 * 1024;
const MAX_CACHED_BYTES: u64 = 8 * 1024 * 1024;
const ALIGNMENT: u64 = 16;

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
        if len == 0 {
            return Err(Error::InvalidRange);
        }
        let available = self.pages.last().and_then(|p| {
            p.cursor
                .checked_add(ALIGNMENT - 1)
                .map(|n| n & !(ALIGNMENT - 1))
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
        let page = self.pages.last_mut().expect("page acquired");
        let start_usize = usize::try_from(start).map_err(|_| Error::InvalidRange)?;
        // SAFETY: the checked range fits the exclusive mapped page, and the
        // source slice cannot alias this private mapping.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                (page.mapped as *mut u8).add(start_usize),
                bytes.len(),
            );
        }
        page.cursor = start + len;
        crate::diagnostics::record(crate::diagnostics::Operation::UploadBytes, len);
        Ok((page.resource.clone(), start))
    }
}
impl Drop for UploadArena {
    fn drop(&mut self) {
        self.pool.release(std::mem::take(&mut self.pages));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn page_sizes_reject_unaddressable_allocations() {
        assert_eq!(allocation_size(1).unwrap(), PAGE_SIZE);
        assert_eq!(allocation_size(PAGE_SIZE + 1).unwrap(), PAGE_SIZE + 1);
        assert!(matches!(
            allocation_size(u64::MAX),
            Err(Error::InvalidRange)
        ));
    }
}
