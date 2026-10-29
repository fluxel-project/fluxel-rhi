//! Private pages for ordinary Vulkan resources.
//!
//! Pages are keyed by the exact memory type, device-address allocation flag,
//! and resource class. Buffers and optimal images never share a page, avoiding
//! buffer-image granularity hazards. The pool owns bounded idle pages; resource
//! owners keep weakly linked range leases until native destruction/retirement.

use super::super::{mapping::MappingState, vk_error};
use ash::vk;
use fluxel_rhi_core::api::{
    AllocatorMemorySnapshot, AllocatorMemorySource, Error, MemoryReportScope, MemoryReportSource,
    MemorySnapshot, NativeMemoryReport, Result,
};
use std::sync::{Arc, Mutex, Weak};

const PAGE_BYTES: u64 = 16 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ResourceClass {
    Buffer,
    OptimalImage,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PageKey {
    memory_type: u32,
    device_address: bool,
    class: ResourceClass,
    persistent_mapping: bool,
}

pub(crate) struct DefaultResourcePool(Arc<Mutex<PoolState>>);

struct PoolState {
    device: ash::Device,
    instance: ash::Instance,
    physical: vk::PhysicalDevice,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    non_coherent_atom_size: u64,
    memory_budget: bool,
    pages: Vec<Page>,
    next_id: u64,
    peak_reserved_bytes: u64,
    peak_live_range_bytes: u64,
}

struct Page {
    id: u64,
    key: PageKey,
    memory: vk::DeviceMemory,
    size: u64,
    mapped: Option<usize>,
    free: Vec<(u64, u64)>,
    live: usize,
    device: ash::Device,
}

#[allow(dead_code)] // The lease's Arc ownership is its entire purpose.
pub(crate) struct DefaultAllocationLease(Arc<LeaseInner>);
struct LeaseInner {
    pool: Weak<Mutex<PoolState>>,
    page_id: u64,
    offset: u64,
    size: u64,
}

pub(crate) struct Allocation {
    pub(crate) memory: vk::DeviceMemory,
    pub(crate) offset: u64,
    pub(crate) lease: DefaultAllocationLease,
}

impl DefaultResourcePool {
    pub(crate) fn new(
        device: ash::Device,
        memory_properties: vk::PhysicalDeviceMemoryProperties,
        instance: ash::Instance,
        physical: vk::PhysicalDevice,
        memory_budget: bool,
        non_coherent_atom_size: u64,
    ) -> Self {
        Self(Arc::new(Mutex::new(PoolState {
            device,
            instance,
            physical,
            memory_properties,
            non_coherent_atom_size: non_coherent_atom_size.max(1),
            memory_budget,
            pages: Vec::new(),
            next_id: 1,
            peak_reserved_bytes: 0,
            peak_live_range_bytes: 0,
        })))
    }

    pub(crate) fn allocate(
        &self,
        requirements: vk::MemoryRequirements,
        requested_flags: vk::MemoryPropertyFlags,
        device_address: bool,
        device_address_alignment: u64,
        class: ResourceClass,
        persistent_mapping: bool,
    ) -> Result<Option<(Allocation, Option<Arc<MappingState>>)>> {
        if requirements.size == 0 {
            return Ok(None);
        }
        let mut state = self.0.lock().map_err(|_| Error::InvalidState)?;
        let Some(memory_type) = (0..state.memory_properties.memory_type_count).find(|index| {
            requirements.memory_type_bits & (1 << index) != 0
                && state.memory_properties.memory_types[*index as usize]
                    .property_flags
                    .contains(requested_flags)
        }) else {
            return Ok(None);
        };
        let flags = state.memory_properties.memory_types[memory_type as usize].property_flags;
        let key = PageKey {
            memory_type,
            device_address,
            class,
            persistent_mapping,
        };
        let atom = if flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE) {
            state.non_coherent_atom_size
        } else {
            1
        };
        // Padding mapped ranges to a non-coherent atom prevents one resource's
        // flush/invalidate from touching a neighboring live resource.
        let range_size = align_up(requirements.size, atom).ok_or(Error::InvalidRange)?;
        let alignment = requirements.alignment.max(atom).max(if device_address {
            device_address_alignment.max(1)
        } else {
            1
        });
        if range_size > PAGE_BYTES {
            return Ok(None);
        }
        if let Some((page, offset)) = state
            .pages
            .iter_mut()
            .filter(|page| page.key == key)
            .find_map(|page| take_range(&mut page.free, range_size, alignment).map(|o| (page, o)))
        {
            page.live += 1;
            let allocation = make_allocation(&self.0, page, offset, range_size);
            let mapping = (class == ResourceClass::Buffer)
                .then(|| MappingState::new_resource(page.size, flags, page.mapped));
            update_peaks(&mut state);
            return Ok(Some((allocation, mapping)));
        }

        // Budget only gates page growth. Existing pages remain reusable even
        // when the process has reached its advisory heap budget.
        if !budget_allows_page(&state, memory_type, PAGE_BYTES) {
            return Ok(None);
        }
        let id = state.next_id;
        let next_id = id.checked_add(1).ok_or(Error::OutOfMemory)?;
        let page_size = PAGE_BYTES;
        let mut address_flags =
            vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        let mut info = vk::MemoryAllocateInfo::default()
            .allocation_size(page_size)
            .memory_type_index(memory_type);
        if device_address {
            info = info.push_next(&mut address_flags);
        }
        // SAFETY: memory type and requirement mask were checked against the
        // retained physical-device properties; the output is initialized.
        let memory = match unsafe { state.device.allocate_memory(&info, None) } {
            Ok(memory) => memory,
            Err(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY | vk::Result::ERROR_OUT_OF_HOST_MEMORY) => {
                return Ok(None);
            }
            Err(error) => return Err(vk_error(error)),
        };
        let mapped = if persistent_mapping {
            // Keep host-visible pages mapped for their entire page lifetime.
            // Each resource has its own map-state lock, while atom padding
            // keeps non-coherent cache operations disjoint.
            match unsafe {
                state
                    .device
                    .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
            } {
                Ok(ptr) if !ptr.is_null() => Some(ptr as usize),
                Ok(_) => {
                    unsafe {
                        state.device.unmap_memory(memory);
                        state.device.free_memory(memory, None);
                    }
                    return Ok(None);
                }
                Err(_) => {
                    unsafe { state.device.free_memory(memory, None) };
                    return Ok(None);
                }
            }
        } else {
            None
        };
        state.next_id = next_id;
        let device = state.device.clone();
        let mut page = Page {
            id,
            key,
            memory,
            size: page_size,
            mapped,
            free: vec![(0, page_size)],
            live: 0,
            device,
        };
        let offset = take_range(&mut page.free, range_size, alignment).ok_or(Error::OutOfMemory)?;
        page.live = 1;
        let allocation = make_allocation(&self.0, &page, offset, range_size);
        let mapping = (class == ResourceClass::Buffer)
            .then(|| MappingState::new_resource(page_size, flags, mapped));
        state.pages.push(page);
        update_peaks(&mut state);
        Ok(Some((allocation, mapping)))
    }

    pub(crate) fn clear(&self) {
        if let Ok(mut state) = self.0.lock() {
            state.pages.clear();
        }
    }

    /// Returns exact facts for retained default-pool pages plus the optional
    /// `VK_EXT_memory_budget` estimates for every physical heap. The latter
    /// are deliberately kept separate: they describe implementation-defined
    /// process usage, not this pool's allocations or residency.
    pub(crate) fn snapshot(&self) -> Result<MemorySnapshot> {
        let state = self.0.lock().map_err(|_| Error::InvalidState)?;
        let (reserved_bytes, live_range_bytes, free_bytes, idle_page_count, live_allocation_count) =
            current_totals(&state);
        let budget = query_budget(&state);
        let native_reports = (0..state.memory_properties.memory_heap_count as usize)
            .map(|heap_index| {
                let heap = state.memory_properties.memory_heaps[heap_index];
                let (usage_bytes, budget_bytes) = budget
                    .as_ref()
                    .map(|values| {
                        let budget = values.heap_budget[heap_index];
                        if budget == 0 {
                            (None, None)
                        } else {
                            (Some(values.heap_usage[heap_index]), Some(budget))
                        }
                    })
                    .unwrap_or((None, None));
                NativeMemoryReport {
                    source: MemoryReportSource::VulkanProcessHeapEstimate,
                    scope: MemoryReportScope::VulkanHeap {
                        heap_index: heap_index as u32,
                        heap_size_bytes: heap.size,
                        device_local: heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL),
                    },
                    usage_bytes,
                    budget_bytes,
                }
            })
            .collect();
        Ok(MemorySnapshot {
            allocator: Some(AllocatorMemorySnapshot {
                source: AllocatorMemorySource::VulkanDefaultResourcePool,
                reserved_bytes,
                live_range_bytes,
                free_bytes,
                peak_reserved_bytes: state.peak_reserved_bytes,
                peak_live_range_bytes: state.peak_live_range_bytes,
                page_count: u32::try_from(state.pages.len()).unwrap_or(u32::MAX),
                idle_page_count,
                live_allocation_count,
            }),
            native_reports,
        })
    }
}

fn current_totals(state: &PoolState) -> (u64, u64, u64, u32, u32) {
    let pages = state
        .pages
        .iter()
        .map(|page| {
            (
                page.size,
                page.free
                    .iter()
                    .fold(0u64, |total, (_, bytes)| total.saturating_add(*bytes)),
                page.live,
            )
        })
        .collect::<Vec<_>>();
    aggregate_totals(&pages)
}

fn aggregate_totals(pages: &[(u64, u64, usize)]) -> (u64, u64, u64, u32, u32) {
    let mut reserved = 0u64;
    let mut free = 0u64;
    let mut idle = 0u32;
    let mut allocations = 0u32;
    for &(size, page_free, live) in pages {
        reserved = reserved.saturating_add(size);
        free = free.saturating_add(page_free);
        if live == 0 {
            idle = idle.saturating_add(1);
        }
        allocations = allocations.saturating_add(u32::try_from(live).unwrap_or(u32::MAX));
    }
    (
        reserved,
        reserved.saturating_sub(free),
        free,
        idle,
        allocations,
    )
}

fn update_peaks(state: &mut PoolState) {
    let (reserved, live, _, _, _) = current_totals(state);
    apply_peaks(
        &mut state.peak_reserved_bytes,
        &mut state.peak_live_range_bytes,
        reserved,
        live,
    );
}

fn apply_peaks(peak_reserved: &mut u64, peak_live: &mut u64, reserved: u64, live: u64) {
    *peak_reserved = (*peak_reserved).max(reserved);
    *peak_live = (*peak_live).max(live);
}

/// The extension reports process-wide, implementation-dependent heap usage.
/// It only gates adding a cache page; it is not a pool-resident byte counter.
fn budget_allows_page(state: &PoolState, memory_type: u32, page_size: u64) -> bool {
    if !state.memory_budget {
        return true;
    }
    let heap = state.memory_properties.memory_types[memory_type as usize].heap_index as usize;
    let Some(budget) = query_budget(state) else {
        return true;
    };
    within_budget(budget.heap_usage[heap], budget.heap_budget[heap], page_size)
}

fn query_budget(state: &PoolState) -> Option<vk::PhysicalDeviceMemoryBudgetPropertiesEXT<'_>> {
    if !state.memory_budget {
        return None;
    }
    let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
    let mut properties = vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget);
    // SAFETY: the enabled extension structure and output chain stay valid for
    // this synchronous query; the physical device is retained with its instance.
    unsafe {
        state
            .instance
            .get_physical_device_memory_properties2(state.physical, &mut properties)
    };
    Some(budget)
}

fn within_budget(current_usage: u64, budget: u64, additional: u64) -> bool {
    budget == 0
        || current_usage
            .checked_add(additional)
            .is_some_and(|projected| projected <= budget)
}

fn make_allocation(
    pool: &Arc<Mutex<PoolState>>,
    page: &Page,
    offset: u64,
    size: u64,
) -> Allocation {
    Allocation {
        memory: page.memory,
        offset,
        lease: DefaultAllocationLease(Arc::new(LeaseInner {
            pool: Arc::downgrade(pool),
            page_id: page.id,
            offset,
            size,
        })),
    }
}

impl Drop for Page {
    fn drop(&mut self) {
        // SAFETY: the pool is the sole owner of page memory; all native
        // resources dropped their handles before returning the final lease.
        unsafe {
            if self.mapped.is_some() {
                self.device.unmap_memory(self.memory);
            }
            self.device.free_memory(self.memory, None);
        }
    }
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
        let (key, is_empty) = {
            let page = &mut state.pages[index];
            page.free.push((self.offset, self.size));
            coalesce(&mut page.free);
            page.live = page.live.saturating_sub(1);
            (page.key, page.live == 0)
        };
        if is_empty
            && state
                .pages
                .iter()
                .filter(|p| p.key == key && p.live == 0)
                .count()
                > 1
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
        free.swap_remove(index);
        if aligned > start {
            free.push((start, aligned - start));
        }
        if end < free_end {
            free.push((end, free_end - end));
        }
        return Some(aligned);
    }
    None
}

fn coalesce(ranges: &mut Vec<(u64, u64)>) {
    ranges.sort_unstable_by_key(|range| range.0);
    let mut write = 0;
    for read in 1..ranges.len() {
        let (start, size) = ranges[read];
        let end = ranges[write].0 + ranges[write].1;
        if start <= end {
            ranges[write].1 = end.max(start + size) - ranges[write].0;
        } else {
            write += 1;
            ranges[write] = (start, size);
        }
    }
    ranges.truncate(if ranges.is_empty() { 0 } else { write + 1 });
}

#[cfg(test)]
#[path = "../../tests/backend/memory/default_pool.rs"]
mod tests;
