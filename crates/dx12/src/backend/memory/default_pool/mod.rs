//! Ordinary DX12 resource suballocation.
//!
//! This pool only owns native heap pages and free byte ranges. It never
//! aliases live resources or infers their use; explicit `MemoryBackend`
//! placement continues to use caller-owned heaps. Resource owners retain a
//! lease so a range returns only after the native resource field is dropped.

use super::*;
use fluxel_rhi_core::api::{
    AllocatorMemorySnapshot, AllocatorMemorySource, MemoryReportScope, MemoryReportSource,
    MemorySnapshot, NativeMemoryReport,
};
use std::sync::{Mutex, Weak};
use windows::Win32::{
    Foundation::E_OUTOFMEMORY,
    Graphics::Dxgi::{
        DXGI_MEMORY_SEGMENT_GROUP_LOCAL, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL,
        DXGI_QUERY_VIDEO_MEMORY_INFO, IDXGIAdapter3,
    },
};

// Initial fixed page policy; this is not an adapter-budget estimate. The pool
// retains at most four empty heaps per class and releases extras on final drop.
const PAGE_BYTES: u64 = 16 * 1024 * 1024;
// Keep enough pages to serve a repeat churn batch without rebuilding heaps.
// The fixed cap bounds idle retention to 64 MiB per native heap class.
const MAX_EMPTY_PAGES_PER_CLASS: usize = 4;

#[derive(Clone)]
pub(crate) struct DefaultResourcePool {
    inner: Arc<Mutex<PoolState>>,
}

struct PoolState {
    device: ID3D12Device,
    budget_adapter: Option<IDXGIAdapter3>,
    pages: Vec<Page>,
    next_id: u64,
    page_bytes: u64,
    peak_reserved_bytes: u64,
    peak_live_range_bytes: u64,
}

struct Page {
    id: u64,
    class: u32,
    heap: ID3D12Heap,
    size: u64,
    free: Vec<(u64, u64)>,
    live: usize,
}

#[derive(Clone)]
#[allow(dead_code)] // The lease's Arc ownership is its entire purpose.
pub(crate) struct DefaultAllocationLease(Arc<LeaseInner>);

struct LeaseInner {
    pool: Weak<Mutex<PoolState>>,
    page_id: u64,
    offset: u64,
    size: u64,
}

pub(crate) struct DefaultAllocation {
    pub(crate) heap: ID3D12Heap,
    pub(crate) offset: u64,
    pub(crate) lease: DefaultAllocationLease,
}

impl DefaultResourcePool {
    pub(crate) fn new(device: ID3D12Device, budget_adapter: Option<IDXGIAdapter3>) -> Self {
        Self {
            inner: Arc::new(Mutex::new(PoolState {
                device,
                budget_adapter,
                pages: Vec::new(),
                next_id: 1,
                page_bytes: PAGE_BYTES,
                peak_reserved_bytes: 0,
                peak_live_range_bytes: 0,
            })),
        }
    }

    pub(crate) fn allocate(
        &self,
        class: u32,
        size: u64,
        alignment: u64,
    ) -> Result<Option<DefaultAllocation>> {
        if size == 0 || alignment == 0 {
            return Ok(None);
        }
        let mut state = self.inner.lock().map_err(|_| Error::InvalidState)?;
        if size > state.page_bytes {
            return Ok(None);
        }
        for index in 0..state.pages.len() {
            let allocation = {
                let page = &mut state.pages[index];
                if page.class != class {
                    None
                } else if let Some(offset) = take_range(&mut page.free, size, alignment) {
                    page.live += 1;
                    Some((page.heap.clone(), page.id, offset))
                } else {
                    None
                }
            };
            if let Some((heap, page_id, offset)) = allocation {
                record_peaks(&mut state);
                return Ok(Some(DefaultAllocation {
                    heap,
                    offset,
                    lease: DefaultAllocationLease(Arc::new(LeaseInner {
                        pool: Arc::downgrade(&self.inner),
                        page_id,
                        offset,
                        size,
                    })),
                }));
            }
        }

        let page_size = state
            .page_bytes
            .max(align_up(size, alignment).ok_or(Error::InvalidRange)?);
        let properties = super::properties(class)?;
        let flags = super::heap_flags(class)?;
        let heap_alignment = super::heap_alignment(class)?;
        let page_size = align_up(page_size, heap_alignment).ok_or(Error::InvalidRange)?;
        if !budget_allows_page(&state, class, page_size) {
            return Ok(None);
        }
        let desc = D3D12_HEAP_DESC {
            SizeInBytes: page_size,
            Properties: properties,
            Alignment: heap_alignment,
            Flags: flags,
        };
        let mut heap = None;
        // SAFETY: the initialized heap descriptor is copied synchronously;
        // D3D12 writes one owned COM reference on success.
        if let Err(error) = unsafe { state.device.CreateHeap(&desc, &mut heap) } {
            if error.code() == E_OUTOFMEMORY {
                return Ok(None);
            }
            return Err(super::super::native_error(error));
        }
        let heap = heap.ok_or(Error::OutOfMemory)?;
        let id = state.next_id;
        state.next_id = state.next_id.checked_add(1).ok_or(Error::OutOfMemory)?;
        let mut page = Page {
            id,
            class,
            heap,
            size: page_size,
            free: vec![(0, page_size)],
            live: 0,
        };
        let offset = take_range(&mut page.free, size, alignment).ok_or(Error::OutOfMemory)?;
        page.live = 1;
        let allocation = DefaultAllocation {
            heap: page.heap.clone(),
            offset,
            lease: DefaultAllocationLease(Arc::new(LeaseInner {
                pool: Arc::downgrade(&self.inner),
                page_id: id,
                offset,
                size,
            })),
        };
        state.pages.push(page);
        record_peaks(&mut state);
        Ok(Some(allocation))
    }

    /// Returns exact suballocation facts for this pool and advisory DXGI
    /// process segment reports. It intentionally excludes every resource that
    /// is not backed by a retained default-pool page.
    pub(crate) fn snapshot(&self) -> Result<MemorySnapshot> {
        let mut state = self.inner.lock().map_err(|_| Error::InvalidState)?;
        let (
            reserved_bytes,
            live_range_bytes,
            free_bytes,
            page_count,
            idle_page_count,
            live_allocation_count,
        ) = pool_totals(&state);
        state.peak_reserved_bytes = state.peak_reserved_bytes.max(reserved_bytes);
        state.peak_live_range_bytes = state.peak_live_range_bytes.max(live_range_bytes);
        let allocator = AllocatorMemorySnapshot {
            source: AllocatorMemorySource::Dx12DefaultResourcePool,
            reserved_bytes,
            live_range_bytes,
            free_bytes,
            peak_reserved_bytes: state.peak_reserved_bytes,
            peak_live_range_bytes: state.peak_live_range_bytes,
            page_count,
            idle_page_count,
            live_allocation_count,
        };
        let native_reports = dxgi_reports(&state);
        Ok(MemorySnapshot {
            allocator: Some(allocator),
            native_reports,
        })
    }
}

fn pool_totals(state: &PoolState) -> (u64, u64, u64, u32, u32, u32) {
    let reserved_bytes = state
        .pages
        .iter()
        .fold(0u64, |total, page| total.saturating_add(page.size));
    let free_bytes = state
        .pages
        .iter()
        .flat_map(|page| page.free.iter().map(|(_, size)| *size))
        .fold(0u64, u64::saturating_add);
    let live_range_bytes = reserved_bytes.saturating_sub(free_bytes);
    let page_count = u32::try_from(state.pages.len()).unwrap_or(u32::MAX);
    let idle_page_count =
        u32::try_from(state.pages.iter().filter(|page| page.live == 0).count()).unwrap_or(u32::MAX);
    let live_allocation_count =
        u32::try_from(state.pages.iter().map(|page| page.live).sum::<usize>()).unwrap_or(u32::MAX);
    (
        reserved_bytes,
        live_range_bytes,
        free_bytes,
        page_count,
        idle_page_count,
        live_allocation_count,
    )
}

fn record_peaks(state: &mut PoolState) {
    let (reserved_bytes, live_range_bytes, ..) = pool_totals(state);
    state.peak_reserved_bytes = state.peak_reserved_bytes.max(reserved_bytes);
    state.peak_live_range_bytes = state.peak_live_range_bytes.max(live_range_bytes);
}

fn dxgi_reports(state: &PoolState) -> Vec<NativeMemoryReport> {
    [
        (
            DXGI_MEMORY_SEGMENT_GROUP_LOCAL,
            MemoryReportScope::DxgiLocalSegment,
        ),
        (
            DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL,
            MemoryReportScope::DxgiNonLocalSegment,
        ),
    ]
    .into_iter()
    .map(|(segment, scope)| {
        let mut info = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        let available = state.budget_adapter.as_ref().is_some_and(|adapter| {
            // SAFETY: the adapter is retained by the pool and the initialized
            // output structure remains valid for this synchronous DXGI query.
            unsafe { adapter.QueryVideoMemoryInfo(0, segment, &mut info) }.is_ok()
                && info.Budget > 0
        });
        NativeMemoryReport {
            source: MemoryReportSource::DxgiProcessSegment,
            scope,
            usage_bytes: available.then_some(info.CurrentUsage),
            budget_bytes: available.then_some(info.Budget),
        }
    })
    .collect()
}

/// A budget query is advisory: if the optional DXGI interface/query is
/// unavailable, keep the bounded cache policy and let native allocation decide.
/// `CurrentUsage` is process-wide segment usage, never a per-pool byte counter.
fn budget_allows_page(state: &PoolState, class: u32, page_size: u64) -> bool {
    let Some(adapter) = &state.budget_adapter else {
        return true;
    };
    let primary_segment = if class == UPLOAD_CLASS || class == READBACK_CLASS {
        DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL
    } else {
        DXGI_MEMORY_SEGMENT_GROUP_LOCAL
    };
    let fallback_to_local = primary_segment == DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL;
    for (index, segment) in [primary_segment, DXGI_MEMORY_SEGMENT_GROUP_LOCAL]
        .into_iter()
        .enumerate()
    {
        if index == 1 && !fallback_to_local {
            break;
        }
        let mut info = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        // SAFETY: the adapter is retained by the pool and the initialized
        // output structure remains valid for this synchronous DXGI query.
        if unsafe { adapter.QueryVideoMemoryInfo(0, segment, &mut info) }.is_ok() && info.Budget > 0
        {
            return within_budget(info.CurrentUsage, info.Budget, page_size);
        }
    }
    true
}

fn within_budget(current_usage: u64, budget: u64, additional: u64) -> bool {
    budget > 0
        && current_usage
            .checked_add(additional)
            .is_some_and(|projected| projected <= budget)
}

impl Drop for LeaseInner {
    fn drop(&mut self) {
        let Some(pool) = self.pool.upgrade() else {
            return;
        };
        let Ok(mut state) = pool.lock() else {
            return;
        };
        let Some(index) = state.pages.iter().position(|page| page.id == self.page_id) else {
            return;
        };
        let (class, is_empty) = {
            let page = &mut state.pages[index];
            free_range(&mut page.free, self.offset, self.size);
            page.live = page.live.saturating_sub(1);
            (page.class, page.live == 0)
        };
        if is_empty
            && state
                .pages
                .iter()
                .filter(|candidate| candidate.class == class && candidate.live == 0)
                .count()
                > MAX_EMPTY_PAGES_PER_CLASS
        {
            state.pages.swap_remove(index);
        }
    }
}

fn align_up(value: u64, alignment: u64) -> Option<u64> {
    value
        .checked_add(alignment - 1)
        .map(|value| value / alignment * alignment)
}

fn take_range(free: &mut Vec<(u64, u64)>, size: u64, alignment: u64) -> Option<u64> {
    for index in 0..free.len() {
        let (start, length) = free[index];
        let Some(aligned) = align_up(start, alignment) else {
            continue;
        };
        let Some(end) = aligned.checked_add(size) else {
            continue;
        };
        let Some(free_end) = start.checked_add(length) else {
            continue;
        };
        if end > free_end {
            continue;
        }
        free.remove(index);
        if end < free_end {
            free.insert(index, (end, free_end - end));
        }
        if aligned > start {
            free.insert(index, (start, aligned - start));
        }
        return Some(aligned);
    }
    None
}

fn free_range(ranges: &mut Vec<(u64, u64)>, start: u64, size: u64) {
    let Some(mut end) = start.checked_add(size) else {
        return;
    };
    let mut merged_start = start;
    let mut index = ranges.partition_point(|&(range_start, _)| range_start < start);
    if index > 0 {
        let (previous_start, previous_size) = ranges[index - 1];
        let Some(previous_end) = previous_start.checked_add(previous_size) else {
            return;
        };
        if previous_end >= start {
            merged_start = previous_start;
            end = end.max(previous_end);
            ranges.remove(index - 1);
            index -= 1;
        }
    }
    while index < ranges.len() && ranges[index].0 <= end {
        let (next_start, next_size) = ranges[index];
        let Some(next_end) = next_start.checked_add(next_size) else {
            return;
        };
        end = end.max(next_end);
        ranges.remove(index);
    }
    ranges.insert(index, (merged_start, end - merged_start));
}

#[cfg(test)]
#[path = "../../../tests/backend/memory/default_pool.rs"]
mod tests;
