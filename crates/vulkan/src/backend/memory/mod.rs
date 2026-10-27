//! Native explicit memory placement; resources retain their heap until retirement.
use super::{
    DeviceState, Vulkan, VulkanBuffer, VulkanDevice, VulkanTexture,
    mapping::MappingState,
    resource::{self, BufferState, TextureState},
    vk_error,
};
use ash::vk;
use fluxel_rhi_core::api::{self, Error, Result};
use std::sync::Arc;

mod default_pool;
pub(super) use default_pool::{DefaultAllocationLease, DefaultResourcePool, ResourceClass};

/// Cold-path memory observation for the Vulkan default-resource pool.
///
/// Pool accounting is exact for its retained pages; native heap figures, when
/// the enabled `VK_EXT_memory_budget` extension supplies them, remain
/// implementation-defined process estimates.
pub struct MemoryObservation;

impl fluxel_rhi_core::backend::MemoryObservationOps<Vulkan> for MemoryObservation {
    const AVAILABLE: bool = true;

    fn snapshot(device: &VulkanDevice) -> Result<api::MemorySnapshot> {
        device.0.default_resource_pool.snapshot()
    }
}

#[derive(Clone)]
pub struct VulkanHeap(Arc<HeapState>);
struct HeapState {
    device: Arc<DeviceState>,
    memory: vk::DeviceMemory,
    size: u64,
    index: u32,
    mapping: Arc<MappingState>,
}
impl Drop for HeapState {
    fn drop(&mut self) {
        // SAFETY: resources and mappings retain this allocation until their last use.
        unsafe { self.device.device.free_memory(self.memory, None) }
    }
}

fn properties(device: &VulkanDevice) -> vk::PhysicalDeviceMemoryProperties {
    // SAFETY: instance and physical device are retained by the logical device.
    unsafe {
        device
            .0
            .instance
            .instance
            .get_physical_device_memory_properties(device.0.physical)
    }
}
fn granularity(device: &VulkanDevice) -> u64 {
    // SAFETY: the retained physical device is valid for this read-only query.
    let limits = unsafe {
        device
            .0
            .instance
            .instance
            .get_physical_device_properties(device.0.physical)
    }
    .limits;
    // A flush of one suballocation must not touch another suballocation's
    // noncoherent atom while that other range is in GPU use.
    limits
        .buffer_image_granularity
        .max(limits.non_coherent_atom_size)
        .max(1)
}
fn requirements(
    device: &VulkanDevice,
    req: vk::MemoryRequirements,
    dedicated: bool,
    flags: vk::MemoryPropertyFlags,
) -> Result<api::MemoryRequirements> {
    let props = properties(device);
    let mut bits = 0u64;
    for i in 0..props.memory_type_count {
        if req.memory_type_bits & (1 << i) != 0
            && props.memory_types[i as usize]
                .property_flags
                .contains(flags)
        {
            bits |= 1u64 << i;
        }
    }
    let alignment = req.alignment.max(granularity(device));
    let size = req
        .size
        .checked_add(alignment - 1)
        .ok_or(Error::InvalidRange)?
        / alignment
        * alignment;
    Ok(api::MemoryRequirements {
        size,
        alignment,
        native_type_bits: bits,
        dedicated_only: dedicated,
    })
}
fn buffer_req(
    device: &VulkanDevice,
    buffer: vk::Buffer,
    desc: &api::BufferDesc,
) -> Result<api::MemoryRequirements> {
    let mut dedicated = vk::MemoryDedicatedRequirements::default();
    let mut out = vk::MemoryRequirements2::default().push_next(&mut dedicated);
    // SAFETY: buffer belongs to the live device and output chain is valid.
    unsafe {
        device.0.device.get_buffer_memory_requirements2(
            &vk::BufferMemoryRequirementsInfo2::default().buffer(buffer),
            &mut out,
        )
    };
    let mut flags = match desc.memory {
        api::MemoryClass::Device => vk::MemoryPropertyFlags::DEVICE_LOCAL,
        _ => vk::MemoryPropertyFlags::HOST_VISIBLE,
    };
    if desc.usage.0 & (api::BufferUsage::MAP_READ.0 | api::BufferUsage::MAP_WRITE.0) != 0 {
        flags |= vk::MemoryPropertyFlags::HOST_VISIBLE;
    }
    requirements(
        device,
        out.memory_requirements,
        dedicated.requires_dedicated_allocation == vk::TRUE,
        flags,
    )
}
fn image_req(device: &VulkanDevice, image: vk::Image) -> Result<api::MemoryRequirements> {
    let mut dedicated = vk::MemoryDedicatedRequirements::default();
    let mut out = vk::MemoryRequirements2::default().push_next(&mut dedicated);
    // SAFETY: image belongs to the live device and output chain is valid.
    unsafe {
        device.0.device.get_image_memory_requirements2(
            &vk::ImageMemoryRequirementsInfo2::default().image(image),
            &mut out,
        )
    };
    requirements(
        device,
        out.memory_requirements,
        dedicated.requires_dedicated_allocation == vk::TRUE,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )
}
fn validate(
    device: &VulkanDevice,
    heap: &VulkanHeap,
    offset: u64,
    req: api::MemoryRequirements,
) -> Result<()> {
    if !Arc::ptr_eq(&device.0, &heap.0.device) {
        return Err(Error::WrongDevice);
    }
    if req.dedicated_only {
        return Err(Error::Unsupported(
            "Vulkan resource requires dedicated allocation",
        ));
    }
    if req.native_type_bits & (1u64 << heap.0.index) == 0 {
        return Err(Error::Unsupported("Vulkan heap memory type"));
    }
    if offset % req.alignment != 0
        || offset
            .checked_add(req.size)
            .is_none_or(|end| end > heap.0.size)
    {
        return Err(Error::InvalidRange);
    }
    Ok(())
}
impl fluxel_rhi_core::backend::MemoryBackend for Vulkan {
    type NativeHeap = VulkanHeap;
    fn buffer_requirements(
        device: &VulkanDevice,
        desc: &api::BufferDesc,
    ) -> Result<api::MemoryRequirements> {
        let buffer = resource::unbound_buffer(device, desc)?;
        let result = buffer_req(device, buffer, desc);
        // SAFETY: unpublished temporary buffer is no longer used.
        unsafe { device.0.device.destroy_buffer(buffer, None) };
        result
    }
    fn texture_requirements(
        device: &VulkanDevice,
        desc: &api::TextureDesc,
    ) -> Result<api::MemoryRequirements> {
        let image = resource::unbound_texture(device, desc, true)?;
        let result = image_req(device, image);
        // SAFETY: unpublished temporary image is no longer used.
        unsafe { device.0.device.destroy_image(image, None) };
        result
    }
    fn create_heap(device: &VulkanDevice, desc: &api::HeapDesc) -> Result<VulkanHeap> {
        let props = properties(device);
        if desc.size == 0 || desc.native_memory_class >= props.memory_type_count {
            return Err(Error::InvalidRange);
        }
        let ty = props.memory_types[desc.native_memory_class as usize];
        if desc.size > props.memory_heaps[ty.heap_index as usize].size {
            return Err(Error::OutOfMemory);
        }
        let mut address =
            vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        let mut info = vk::MemoryAllocateInfo::default()
            .allocation_size(desc.size)
            .memory_type_index(desc.native_memory_class);
        if device.0.ray_support.acceleration {
            info = info.push_next(&mut address);
        }
        // SAFETY: memory type index and size were validated against physical properties.
        let memory = unsafe { device.0.device.allocate_memory(&info, None) }.map_err(vk_error)?;
        Ok(VulkanHeap(Arc::new(HeapState {
            device: device.0.clone(),
            memory,
            size: desc.size,
            index: desc.native_memory_class,
            mapping: MappingState::new(desc.size, ty.property_flags),
        })))
    }
    fn place_buffer(
        device: &VulkanDevice,
        heap: &VulkanHeap,
        offset: u64,
        desc: &api::BufferDesc,
    ) -> Result<VulkanBuffer> {
        let buffer = resource::unbound_buffer(device, desc)?;
        let result = (|| {
            let req = buffer_req(device, buffer, desc)?;
            validate(device, heap, offset, req)?;
            // SAFETY: queried requirements, type, range, and alignment have been checked.
            unsafe {
                device
                    .0
                    .device
                    .bind_buffer_memory(buffer, heap.0.memory, offset)
            }
            .map_err(vk_error)?;
            Ok(VulkanBuffer(Arc::new(BufferState {
                device: device.0.clone(),
                buffer,
                memory: heap.0.memory,
                size: desc.size,
                usage: desc.usage,
                heap: Some(heap.clone()),
                default_allocation: None,
                memory_offset: offset,
                map_state: heap.0.mapping.clone(),
            })))
        })();
        if result.is_err() {
            // SAFETY: failure leaves the unpublished buffer owned here.
            unsafe { device.0.device.destroy_buffer(buffer, None) };
        }
        result
    }
    fn place_texture(
        device: &VulkanDevice,
        heap: &VulkanHeap,
        offset: u64,
        desc: &api::TextureDesc,
    ) -> Result<VulkanTexture> {
        let image = resource::unbound_texture(device, desc, true)?;
        let result = (|| {
            let req = image_req(device, image)?;
            validate(device, heap, offset, req)?;
            // SAFETY: queried requirements, type, range, and alignment have been checked.
            unsafe {
                device
                    .0
                    .device
                    .bind_image_memory(image, heap.0.memory, offset)
            }
            .map_err(vk_error)?;
            Ok(VulkanTexture(Arc::new(TextureState {
                device: device.0.clone(),
                image,
                memory: Some(heap.0.memory),
                heap: Some(heap.clone()),
                default_allocation: None,
                sparse: None,
                desc: desc.clone(),
            })))
        })();
        if result.is_err() {
            // SAFETY: failure leaves the unpublished image owned here.
            unsafe { device.0.device.destroy_image(image, None) };
        }
        result
    }
}
