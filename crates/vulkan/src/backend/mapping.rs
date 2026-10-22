//! Exclusive host mappings of Vulkan device-memory allocations.
//!
//! Placed buffers sharing an allocation share its mapping state. A lease maps
//! the whole allocation so atom-aligned cache operations can cover its edges;
//! only the requested buffer range is exposed to the caller.
use super::{Vulkan, VulkanBuffer, VulkanDevice, vk_error};
use ash::vk;
use fluxel_rhi_core::api::{self, Error, Result};
use std::{
    future::{Ready, ready},
    ops::Range,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

pub(super) struct MappingState {
    mapped: AtomicBool,
    size: u64,
    flags: vk::MemoryPropertyFlags,
}
impl MappingState {
    pub(super) fn new(size: u64, flags: vk::MemoryPropertyFlags) -> Arc<Self> {
        Arc::new(Self {
            mapped: AtomicBool::new(false),
            size,
            flags,
        })
    }
}

pub struct VulkanMap {
    buffer: VulkanBuffer,
    ptr: *mut u8,
    offset: usize,
    len: usize,
    mode: api::MapMode,
    atom: u64,
}
impl Drop for VulkanMap {
    fn drop(&mut self) {
        // SAFETY: this lease owns the only successful mapping of this allocation;
        // the retained buffer keeps the allocation and device alive.
        unsafe {
            self.buffer
                .0
                .device
                .device
                .unmap_memory(self.buffer.0.memory)
        };
        self.buffer
            .0
            .map_state
            .mapped
            .store(false, Ordering::Release);
    }
}
impl api::MappingBackend for Vulkan {
    type NativeMap = VulkanMap;
    type MapFuture<'a> = Ready<Result<VulkanMap>>;
    fn map<'a>(
        device: &'a VulkanDevice,
        buffer: &'a VulkanBuffer,
        range: Range<u64>,
        mode: api::MapMode,
    ) -> Self::MapFuture<'a> {
        ready(map(device, buffer, range, mode))
    }
    fn bytes(map: &VulkanMap) -> &[u8] {
        // SAFETY: the lease owns the allocation's mapping and offset/len were
        // checked against both its buffer and allocation bounds.
        unsafe { std::slice::from_raw_parts(map.ptr.add(map.offset), map.len) }
    }
    fn bytes_mut(map: &mut VulkanMap) -> &mut [u8] {
        assert!(map.mode == api::MapMode::Write);
        // SAFETY: the single allocation lease and mutable borrow exclude aliases.
        unsafe { std::slice::from_raw_parts_mut(map.ptr.add(map.offset), map.len) }
    }
    fn flush(map: &mut VulkanMap, bytes: Range<u64>) -> Result<()> {
        cache_op(map, bytes, true)
    }
    fn invalidate(map: &mut VulkanMap, bytes: Range<u64>) -> Result<()> {
        cache_op(map, bytes, false)
    }
    fn unmap(map: VulkanMap) {
        drop(map);
    }
}

pub(super) fn map(
    device: &VulkanDevice,
    buffer: &VulkanBuffer,
    range: Range<u64>,
    mode: api::MapMode,
) -> Result<VulkanMap> {
    if !Arc::ptr_eq(&device.0, &buffer.0.device) {
        return Err(Error::WrongDevice);
    }
    let state = &buffer.0.map_state;
    let required = match mode {
        api::MapMode::Read => api::BufferUsage::MAP_READ,
        api::MapMode::Write => api::BufferUsage::MAP_WRITE,
    };
    if !buffer.0.usage.contains(required)
        || !state.flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
    {
        return Err(Error::Unsupported("Vulkan buffer mapping mode"));
    }
    if range.start >= range.end || range.end > buffer.0.size {
        return Err(Error::InvalidRange);
    }
    let start = buffer
        .0
        .memory_offset
        .checked_add(range.start)
        .ok_or(Error::InvalidRange)?;
    let end = buffer
        .0
        .memory_offset
        .checked_add(range.end)
        .ok_or(Error::InvalidRange)?;
    if end > state.size || end > isize::MAX as u64 {
        return Err(Error::InvalidRange);
    }
    let offset = usize::try_from(start).map_err(|_| Error::InvalidRange)?;
    let len = usize::try_from(end - start).map_err(|_| Error::InvalidRange)?;
    if state.mapped.swap(true, Ordering::AcqRel) {
        return Err(Error::InvalidState);
    }
    // SAFETY: HOST_VISIBLE is checked, and the atomic lock owns the allocation's
    // sole mapping until VulkanMap::drop. Renderer supplies GPU completion.
    let result = unsafe {
        device.0.device.map_memory(
            buffer.0.memory,
            0,
            vk::WHOLE_SIZE,
            vk::MemoryMapFlags::empty(),
        )
    };
    let ptr = match result {
        Ok(ptr) if !ptr.is_null() => ptr.cast::<u8>(),
        Ok(_) => {
            // SAFETY: balances the successful map even if the driver returned null.
            unsafe { device.0.device.unmap_memory(buffer.0.memory) };
            state.mapped.store(false, Ordering::Release);
            return Err(Error::Backend("Vulkan map returned null".into()));
        }
        Err(error) => {
            state.mapped.store(false, Ordering::Release);
            return Err(vk_error(error));
        }
    };
    // SAFETY: the physical device belongs to this live instance.
    let atom = unsafe {
        device
            .0
            .instance
            .instance
            .get_physical_device_properties(device.0.physical)
    }
    .limits
    .non_coherent_atom_size;
    Ok(VulkanMap {
        buffer: buffer.clone(),
        ptr,
        offset,
        len,
        mode,
        atom,
    })
}

fn cache_op(map: &VulkanMap, bytes: Range<u64>, flush: bool) -> Result<()> {
    if bytes.start > bytes.end || bytes.end > map.len as u64 {
        return Err(Error::InvalidRange);
    }
    if bytes.is_empty()
        || map
            .buffer
            .0
            .map_state
            .flags
            .contains(vk::MemoryPropertyFlags::HOST_COHERENT)
    {
        return Ok(());
    }
    let (start, size) = atom_range(
        map.offset as u64 + bytes.start,
        map.offset as u64 + bytes.end,
        map.buffer.0.map_state.size,
        map.atom,
    );
    let range = vk::MappedMemoryRange::default()
        .memory(map.buffer.0.memory)
        .offset(start)
        .size(size);
    // SAFETY: allocation is mapped in full and range is atom aligned, or ends
    // at allocation size, as required by Vulkan for noncoherent memory.
    unsafe {
        if flush {
            map.buffer
                .0
                .device
                .device
                .flush_mapped_memory_ranges(&[range])
        } else {
            map.buffer
                .0
                .device
                .device
                .invalidate_mapped_memory_ranges(&[range])
        }
    }
    .map_err(vk_error)
}
fn atom_range(start: u64, end: u64, size: u64, atom: u64) -> (u64, u64) {
    let aligned_start = start / atom * atom;
    let aligned_end = end.saturating_add(atom - 1) / atom * atom;
    let aligned_end = aligned_end.min(size);
    (aligned_start, aligned_end - aligned_start)
}
#[cfg(test)]
mod tests {
    use super::atom_range;
    #[test]
    fn noncoherent_ranges_include_whole_atoms_and_allocation_end() {
        assert_eq!(atom_range(257, 300, 1024, 256), (256, 256));
        assert_eq!(atom_range(900, 999, 1000, 256), (768, 232));
    }
}
