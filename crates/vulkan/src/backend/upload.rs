//! Persistently mapped, queue-family-local staging pages.
//!
//! Pages are raw Vulkan objects rather than `VulkanBuffer`s.  In particular an
//! idle page must not retain `DeviceState`: the device owns this pool, so that
//! would form a device -> pool -> page -> device cycle.  An encoder receives
//! `UploadSlice`s; `RecordedWork` consequently keeps their leases alive until
//! its completion fence retires.  Dropping an unsubmitted encoder releases the
//! lease immediately, making the page eligible for another recording.

use super::{DeviceState, vk_error};
use ash::vk;
use fluxel_rhi_core::api::{Error, Result};
use std::sync::{
    Arc, Mutex, Weak,
    atomic::{AtomicU64, Ordering},
};

const DEFAULT_PAGE_BYTES: u64 = 4 * 1024 * 1024;

#[cfg(all(test, feature = "hardware-tests"))]
mod tests;

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct UploadDiagnostics {
    pub(super) creates: u64,
    pub(super) maps: u64,
    pub(super) page_bytes: u64,
}

/// A device-owned collection of *idle* pages.  Busy pages are held only by
/// encoder slices, never by this list.
#[derive(Clone, Default)]
pub(super) struct UploadPool(Arc<PoolInner>);

#[derive(Default)]
struct PoolInner {
    idle: Mutex<Vec<RawPage>>,
    creates: AtomicU64,
    maps: AtomicU64,
    page_bytes: AtomicU64,
}

struct RawPage {
    device: ash::Device,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    mapped: *mut u8,
    size: u64,
    family: u32,
}

// The page is accessed through the encoder's exclusive mutable recording
// path.  The mapped pointer is only written before the command that reads it
// is submitted, and a lease is never shared by two encoders.
unsafe impl Send for RawPage {}
unsafe impl Sync for RawPage {}

impl Drop for RawPage {
    fn drop(&mut self) {
        // SAFETY: RawPage owns this persistent map and its native handles.
        // `UploadPool::clear` is called before `vkDestroyDevice`; active pages
        // are retained by completion work, so they are also dropped first.
        unsafe {
            self.device.unmap_memory(self.memory);
            self.device.destroy_buffer(self.buffer, None);
            self.device.free_memory(self.memory, None);
        }
    }
}

struct Lease {
    pool: Weak<PoolInner>,
    page: Mutex<Option<RawPage>>,
    cursor: Mutex<u64>,
}

impl Drop for Lease {
    fn drop(&mut self) {
        let Some(page) = self.page.lock().unwrap_or_else(|p| p.into_inner()).take() else {
            return;
        };
        if let Some(pool) = self.pool.upgrade() {
            pool.idle
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .push(page);
        }
        // If the pool has already gone away, dropping `page` is intentional.
    }
}

/// One recorded copy's view of a page.  Cloning it is how consecutive uploads
/// from the same encoder share the same page and lease.
#[derive(Clone)]
pub(crate) struct UploadSlice {
    lease: Arc<Lease>,
    pub(super) buffer: vk::Buffer,
    pub(super) offset: u64,
}

impl UploadPool {
    #[cfg(test)]
    pub(super) fn diagnostics(&self) -> UploadDiagnostics {
        UploadDiagnostics {
            creates: self.0.creates.load(Ordering::Relaxed),
            maps: self.0.maps.load(Ordering::Relaxed),
            page_bytes: self.0.page_bytes.load(Ordering::Relaxed),
        }
    }

    /// Destroys only idle pages.  Device teardown calls this before destroying
    /// the logical device; live encoder/completion leases must already be gone.
    pub(super) fn clear(&self) {
        self.0
            .idle
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clear();
    }

    /// Copies `bytes` into a queue-family-local page. `bias` is the caller's
    /// logical buffer offset: `slice.offset + bias` is aligned for the native
    /// copy command (texture layouts can have a nonzero source offset).
    pub(super) fn write(
        &self,
        device: &DeviceState,
        family: u32,
        retained: &mut Vec<UploadSlice>,
        bytes: &[u8],
        alignment: u64,
        bias: u64,
    ) -> Result<UploadSlice> {
        if bytes.is_empty() || alignment == 0 || !alignment.is_power_of_two() {
            return Err(Error::InvalidRange);
        }
        if let Some(last) = retained.last() {
            if let Some(slice) = allocate_from(&last.lease, bytes.len() as u64, alignment, bias)? {
                copy_to_slice(&slice, bytes)?;
                retained.push(slice.clone());
                return Ok(slice);
            }
        }
        let minimum = (bytes.len() as u64)
            .checked_add(alignment)
            .and_then(|v| v.checked_add(bias))
            .ok_or(Error::InvalidRange)?;
        let page = self.acquire(device, family, minimum)?;
        let lease = Arc::new(Lease {
            pool: Arc::downgrade(&self.0),
            page: Mutex::new(Some(page)),
            cursor: Mutex::new(0),
        });
        let slice = allocate_from(&lease, bytes.len() as u64, alignment, bias)?
            .ok_or(Error::OutOfMemory)?;
        copy_to_slice(&slice, bytes)?;
        retained.push(slice.clone());
        Ok(slice)
    }

    fn acquire(&self, device: &DeviceState, family: u32, minimum: u64) -> Result<RawPage> {
        let mut idle = self.0.idle.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(index) = idle
            .iter()
            .position(|page| page.family == family && page.size >= minimum)
        {
            return Ok(idle.swap_remove(index));
        }
        drop(idle);
        create_page(device, family, minimum, &self.0)
    }
}

/// Vulkan's optimal buffer-copy offset alignment is a device limit.  The
/// fixed four-byte rule remains relevant for `vkCmdCopyBuffer` as well.
pub(super) fn copy_offset_alignment(device: &DeviceState) -> u64 {
    // SAFETY: the physical device remains owned by `device.instance`.
    let optimal = unsafe {
        device
            .instance
            .instance
            .get_physical_device_properties(device.physical)
    }
    .limits
    .optimal_buffer_copy_offset_alignment;
    4_u64.max(optimal).next_power_of_two()
}

fn allocate_from(
    lease: &Arc<Lease>,
    bytes: u64,
    alignment: u64,
    bias: u64,
) -> Result<Option<UploadSlice>> {
    let mut cursor = lease.cursor.lock().unwrap_or_else(|p| p.into_inner());
    let shifted = cursor.checked_add(bias).ok_or(Error::InvalidRange)?;
    let aligned = shifted
        .checked_add(alignment - 1)
        .ok_or(Error::InvalidRange)?
        & !(alignment - 1);
    let offset = aligned.checked_sub(bias).ok_or(Error::InvalidRange)?;
    let end = offset.checked_add(bytes).ok_or(Error::InvalidRange)?;
    let page = lease.page.lock().unwrap_or_else(|p| p.into_inner());
    let Some(page) = page.as_ref() else {
        return Ok(None);
    };
    if end > page.size {
        return Ok(None);
    }
    *cursor = end;
    Ok(Some(UploadSlice {
        lease: lease.clone(),
        buffer: page.buffer,
        offset,
    }))
}

fn copy_to_slice(slice: &UploadSlice, bytes: &[u8]) -> Result<()> {
    let page = slice.lease.page.lock().unwrap_or_else(|p| p.into_inner());
    let Some(page) = page.as_ref() else {
        return Err(Error::InvalidState);
    };
    // SAFETY: `offset..offset+size` was reserved from this encoder-exclusive
    // page; the persistent mapping covers the full allocation.
    unsafe {
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            page.mapped.add(slice.offset as usize),
            bytes.len(),
        )
    };
    Ok(())
}

fn create_page(
    device: &DeviceState,
    family: u32,
    minimum: u64,
    pool: &PoolInner,
) -> Result<RawPage> {
    let size = DEFAULT_PAGE_BYTES.max(minimum);
    let info = vk::BufferCreateInfo::default()
        .size(size)
        .usage(vk::BufferUsageFlags::TRANSFER_SRC)
        // Page ownership is fixed to the encoder's command-pool family.  This
        // avoids silently sharing EXCLUSIVE buffers between graphics/compute.
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    // SAFETY: DeviceState owns the live logical device.
    let buffer = unsafe { device.device.create_buffer(&info, None) }.map_err(vk_error)?;
    // SAFETY: the new buffer is live and unbound.
    let requirements = unsafe { device.device.get_buffer_memory_requirements(buffer) };
    // SAFETY: the physical device belongs to the still-live instance.
    let properties = unsafe {
        device
            .instance
            .instance
            .get_physical_device_memory_properties(device.physical)
    };
    let required = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
    let Some(memory_type_index) = (0..properties.memory_type_count).find(|&index| {
        requirements.memory_type_bits & (1 << index) != 0
            && properties.memory_types[index as usize]
                .property_flags
                .contains(required)
    }) else {
        // SAFETY: this unpublished buffer is owned locally after failed selection.
        unsafe { device.device.destroy_buffer(buffer, None) };
        return Err(Error::OutOfMemory);
    };
    let allocate = vk::MemoryAllocateInfo::default()
        .allocation_size(requirements.size)
        .memory_type_index(memory_type_index);
    // SAFETY: selected type satisfies this buffer's requirements.
    let memory = match unsafe { device.device.allocate_memory(&allocate, None) }.map_err(vk_error) {
        Ok(memory) => memory,
        Err(error) => {
            // SAFETY: allocation failed; the unpublished buffer is still local.
            unsafe { device.device.destroy_buffer(buffer, None) };
            return Err(error);
        }
    };
    // SAFETY: memory and buffer are local, compatible native objects.
    if let Err(error) =
        unsafe { device.device.bind_buffer_memory(buffer, memory, 0) }.map_err(vk_error)
    {
        // SAFETY: failed binding left both unpublished handles locally owned.
        unsafe {
            device.device.free_memory(memory, None);
            device.device.destroy_buffer(buffer, None)
        };
        return Err(error);
    }
    // SAFETY: the coherent host-visible allocation is mapped once for its lifetime.
    let mapped = match unsafe {
        device
            .device
            .map_memory(memory, 0, requirements.size, vk::MemoryMapFlags::empty())
    }
    .map_err(vk_error)
    {
        Ok(mapped) if !mapped.is_null() => mapped.cast(),
        Ok(_) => {
            // SAFETY: balance the successful map before destroying local handles.
            unsafe {
                device.device.unmap_memory(memory);
                device.device.destroy_buffer(buffer, None);
                device.device.free_memory(memory, None)
            };
            return Err(Error::Backend("Vulkan staging map returned null".into()));
        }
        Err(error) => {
            // SAFETY: failed mapping creates no borrowed pointer or GPU use.
            unsafe {
                device.device.destroy_buffer(buffer, None);
                device.device.free_memory(memory, None)
            };
            return Err(error);
        }
    };
    pool.creates.fetch_add(1, Ordering::Relaxed);
    pool.maps.fetch_add(1, Ordering::Relaxed);
    pool.page_bytes
        .fetch_add(requirements.size, Ordering::Relaxed);
    Ok(RawPage {
        device: device.device.clone(),
        buffer,
        memory,
        mapped,
        size: requirements.size,
        family,
    })
}
