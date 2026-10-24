//! Completion-gated mapping of Vulkan host-readable buffers.
//!
//! Core exposes a ticket only after the caller has supplied completion proof.
//! For a device-local source the encoder retains a host-visible staging buffer
//! and records the transfer plus host-read barrier.  Mapping borrows no native
//! pointer after unmap: bytes are copied into the portable result first.

#[cfg(test)]
use std::sync::atomic::{AtomicU64, Ordering};
use std::{
    future::{Ready, ready},
    ops::Range,
    sync::{Arc, Mutex, Weak},
};

use ash::vk;
use fluxel_rhi_core::api::{self, Error, MappingBackend, ReadbackData, Result};

use super::{DeviceState, Vulkan, VulkanBuffer, VulkanDevice, VulkanEncoder, mapping, vk_error};

const DEFAULT_PAGE_BYTES: u64 = 4 * 1024 * 1024;
const MAX_IDLE_PAGES: usize = 32;

/// Idle raw staging pages.  They cannot retain `DeviceState`, which owns this
/// pool; encoder/ticket leases keep busy pages alive through completion.
#[derive(Clone, Default)]
pub(super) struct ReadbackPool(Arc<PoolInner>);
#[derive(Default)]
struct PoolInner {
    idle: Mutex<Vec<RawPage>>,
    #[cfg(test)]
    creates: AtomicU64,
    #[cfg(test)]
    maps: AtomicU64,
}
#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct ReadbackDiagnostics {
    pub(super) creates: u64,
    pub(super) maps: u64,
    pub(super) idle: usize,
}
struct RawPage {
    device: ash::Device,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    mapped: *mut u8,
    size: u64,
    family: u32,
}
// SAFETY: pages are exclusively owned by one lease until it returns to the
// mutex-protected idle list; mapped bytes are read only after completion.
unsafe impl Send for RawPage {}
// SAFETY: shared access reaches a page only through the lease/page mutex.
unsafe impl Sync for RawPage {}
impl Drop for RawPage {
    fn drop(&mut self) {
        // SAFETY: this RawPage is the sole owner of its persistent map and
        // native handles; pool clear runs before device destruction.
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
}
impl Drop for Lease {
    fn drop(&mut self) {
        if let Some(page) = self.page.lock().unwrap_or_else(|p| p.into_inner()).take() {
            if let Some(pool) = self.pool.upgrade() {
                let mut idle = pool.idle.lock().unwrap_or_else(|p| p.into_inner());
                // Keep only standard pages. A one-off large readback should
                // return its memory to Vulkan rather than pinning a peak.
                if page.size <= DEFAULT_PAGE_BYTES && idle.len() < MAX_IDLE_PAGES {
                    idle.push(page);
                }
            }
        }
    }
}
#[derive(Clone)]
pub(crate) struct ReadbackSlice {
    lease: Arc<Lease>,
    // Field order matters: the raw-page lease returns/drops before this final
    // device owner, so a ticket may outlive every public Device handle.
    _device: Arc<DeviceState>,
    buffer: vk::Buffer,
}
impl ReadbackPool {
    #[cfg(test)]
    pub(super) fn diagnostics(&self) -> ReadbackDiagnostics {
        ReadbackDiagnostics {
            creates: self.0.creates.load(Ordering::Relaxed),
            maps: self.0.maps.load(Ordering::Relaxed),
            idle: self.0.idle.lock().unwrap_or_else(|p| p.into_inner()).len(),
        }
    }
    pub(super) fn clear(&self) {
        self.0
            .idle
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clear();
    }
    fn acquire(
        &self,
        device: &Arc<DeviceState>,
        family: u32,
        minimum: u64,
    ) -> Result<ReadbackSlice> {
        let mut idle = self.0.idle.lock().unwrap_or_else(|p| p.into_inner());
        let page = if let Some(index) = idle
            .iter()
            .position(|p| p.family == family && p.size >= minimum)
        {
            idle.swap_remove(index)
        } else {
            drop(idle);
            create_page(device, family, minimum, &self.0)?
        };
        Ok(ReadbackSlice {
            buffer: page.buffer,
            lease: Arc::new(Lease {
                pool: Arc::downgrade(&self.0),
                page: Mutex::new(Some(page)),
            }),
            _device: device.clone(),
        })
    }
}
fn create_page(
    device: &Arc<DeviceState>,
    family: u32,
    minimum: u64,
    _pool: &PoolInner,
) -> Result<RawPage> {
    let size = DEFAULT_PAGE_BYTES.max(minimum);
    let buffer = unsafe {
        device.device.create_buffer(
            &vk::BufferCreateInfo::default()
                .size(size)
                .usage(vk::BufferUsageFlags::TRANSFER_DST)
                .sharing_mode(vk::SharingMode::EXCLUSIVE),
            None,
        )
    }
    .map_err(vk_error)?;
    let requirements = unsafe { device.device.get_buffer_memory_requirements(buffer) };
    let properties = unsafe {
        device
            .instance
            .instance
            .get_physical_device_memory_properties(device.physical)
    };
    let required = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
    let Some(index) = (0..properties.memory_type_count).find(|&i| {
        requirements.memory_type_bits & (1 << i) != 0
            && properties.memory_types[i as usize]
                .property_flags
                .contains(required)
    }) else {
        unsafe { device.device.destroy_buffer(buffer, None) };
        return Err(Error::OutOfMemory);
    };
    let memory = match unsafe {
        device.device.allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(requirements.size)
                .memory_type_index(index),
            None,
        )
    }
    .map_err(vk_error)
    {
        Ok(memory) => memory,
        Err(error) => {
            unsafe { device.device.destroy_buffer(buffer, None) };
            return Err(error);
        }
    };
    if let Err(error) =
        unsafe { device.device.bind_buffer_memory(buffer, memory, 0) }.map_err(vk_error)
    {
        unsafe {
            device.device.free_memory(memory, None);
            device.device.destroy_buffer(buffer, None)
        };
        return Err(error);
    }
    let mapped = match unsafe {
        device
            .device
            .map_memory(memory, 0, requirements.size, vk::MemoryMapFlags::empty())
    }
    .map_err(vk_error)
    {
        Ok(mapped) if !mapped.is_null() => mapped.cast(),
        Ok(_) => {
            unsafe {
                device.device.unmap_memory(memory);
                device.device.destroy_buffer(buffer, None);
                device.device.free_memory(memory, None)
            };
            return Err(Error::Backend("Vulkan readback map returned null".into()));
        }
        Err(error) => {
            unsafe {
                device.device.destroy_buffer(buffer, None);
                device.device.free_memory(memory, None)
            };
            return Err(error);
        }
    };
    #[cfg(test)]
    {
        _pool.creates.fetch_add(1, Ordering::Relaxed);
        _pool.maps.fetch_add(1, Ordering::Relaxed);
    }
    Ok(RawPage {
        device: device.device.clone(),
        buffer,
        memory,
        mapped,
        // Buffer-copy bounds are limited by the declared buffer size, not
        // Vulkan's allocation-size padding.
        size,
        family,
    })
}

pub struct VulkanReadback {
    buffer: Option<VulkanBuffer>,
    staging: Option<ReadbackSlice>,
    range: Range<u64>,
}

impl api::ReadbackBackend for Vulkan {
    type NativeReadback = VulkanReadback;
    type ReadbackFuture<'a>
        = Ready<Result<ReadbackData>>
    where
        Self: 'a;

    fn encode_readback(
        encoder: &mut VulkanEncoder,
        source: &VulkanBuffer,
        range: Range<u64>,
    ) -> Result<VulkanReadback> {
        if range.start > range.end || range.end > source.0.size {
            return Err(Error::InvalidRange);
        }
        if !source.0.usage.contains(api::BufferUsage::MAP_READ) {
            if !source.0.usage.contains(api::BufferUsage::COPY_SRC) {
                return Err(Error::Unsupported("Vulkan readback requires COPY_SRC"));
            }
            let size = range.end - range.start;
            if size == 0 {
                return Err(Error::InvalidRange);
            }
            let family = if encoder.lane.ordinal == 1 {
                encoder.device.compute_family
            } else {
                encoder.device.graphics_family
            };
            let staging = encoder
                .device
                .readback_pool
                .acquire(&encoder.device, family, size)?;
            let ready = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(source.0.buffer)
                .offset(range.start)
                .size(size);
            // SAFETY: encoder owns a recording command buffer; source and
            // staging retain this device and the caller supplied transition
            // contract for source visibility before readback.
            unsafe {
                encoder.device.device.cmd_pipeline_barrier(
                    encoder.command,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[ready],
                    &[],
                );
                encoder.device.device.cmd_copy_buffer(
                    encoder.command,
                    source.0.buffer,
                    staging.buffer,
                    &[vk::BufferCopy {
                        src_offset: range.start,
                        dst_offset: 0,
                        size,
                    }],
                );
            }
            let host = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(staging.buffer)
                .offset(0)
                .size(size);
            // SAFETY: this records a host-read dependency for the staging
            // buffer in the same still-recording command buffer.
            unsafe {
                encoder.device.device.cmd_pipeline_barrier(
                    encoder.command,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::HOST,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[host],
                    &[],
                );
            }
            encoder.buffers.push(source.clone());
            encoder.readback_slices.push(staging.clone());
            return Ok(VulkanReadback {
                buffer: None,
                staging: Some(staging),
                range: 0..size,
            });
        }
        encoder.buffers.push(source.clone());
        Ok(VulkanReadback {
            buffer: Some(source.clone()),
            staging: None,
            range,
        })
    }

    fn map_readback<'a>(_: &'a VulkanDevice, ticket: VulkanReadback) -> Self::ReadbackFuture<'a> {
        ready(map(ticket))
    }
}

fn map(ticket: VulkanReadback) -> Result<ReadbackData> {
    let size = ticket.range.end - ticket.range.start;
    if size == 0 {
        return Ok(ReadbackData {
            bytes: Vec::new(),
            row_pitch: 0,
            slice_pitch: 0,
        });
    }
    let bytes = if let Some(buffer) = ticket.buffer {
        let device = VulkanDevice(buffer.0.device.clone());
        let mut map = mapping::map(&device, &buffer, ticket.range, api::MapMode::Read)?;
        Vulkan::invalidate(&mut map, 0..size)?;
        Vulkan::bytes(&map).to_vec()
    } else {
        let staging = ticket.staging.ok_or(Error::InvalidState)?;
        let page = staging.lease.page.lock().unwrap_or_else(|p| p.into_inner());
        let page = page.as_ref().ok_or(Error::InvalidState)?;
        // SAFETY: Core required completion proof before mapping; the coherent
        // page remains held by this ticket for the copy.
        unsafe {
            std::slice::from_raw_parts(page.mapped.add(ticket.range.start as usize), size as usize)
        }
        .to_vec()
    };
    Ok(ReadbackData {
        bytes,
        row_pitch: size,
        slice_pitch: size,
    })
}
