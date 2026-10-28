//! Import platform-owned opaque external memory into native Vulkan resources.
use super::super::{
    VulkanBuffer, VulkanDevice, VulkanTexture,
    mapping::MappingState,
    resource::{self, BufferState, TextureState},
    vk_error,
};
use super::VulkanExternalMemoryHandle;
use ash::vk;
use fluxel_rhi_core::api::{self, Error, Result};
#[cfg(unix)]
use std::os::fd::{AsRawFd, IntoRawFd};
#[cfg(windows)]
use std::os::windows::io::AsRawHandle;
use std::sync::Arc;

#[cfg(windows)]
const HANDLE_TYPE: vk::ExternalMemoryHandleTypeFlags =
    vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32;
#[cfg(unix)]
const HANDLE_TYPE: vk::ExternalMemoryHandleTypeFlags = vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD;

fn check_features(features: vk::ExternalMemoryFeatureFlags, dedicated: bool) -> Result<()> {
    if !features.contains(vk::ExternalMemoryFeatureFlags::IMPORTABLE) {
        return Err(Error::Unsupported(
            "Vulkan external memory import for this resource",
        ));
    }
    if features.contains(vk::ExternalMemoryFeatureFlags::DEDICATED_ONLY) && !dedicated {
        return Err(Error::Unsupported(
            "Vulkan external memory requires dedicated allocation",
        ));
    }
    Ok(())
}
fn allocation(
    device: &VulkanDevice,
    handle: VulkanExternalMemoryHandle,
    req: vk::MemoryRequirements,
    offset: u64,
    dedicated_required: bool,
    buffer: vk::Buffer,
    image: vk::Image,
    address: bool,
) -> Result<(vk::DeviceMemory, vk::MemoryPropertyFlags)> {
    if !device.0.external_memory_handles {
        return Err(Error::Unsupported("Vulkan external memory handles"));
    }
    if handle.memory_type_index >= 32
        || req.memory_type_bits & (1u32 << handle.memory_type_index) == 0
    {
        return Err(Error::InvalidRange);
    }
    if offset % req.alignment != 0
        || offset
            .checked_add(req.size)
            .is_none_or(|end| end > handle.allocation_size)
        || (handle.dedicated && offset != 0)
    {
        return Err(Error::InvalidRange);
    }
    if dedicated_required && !handle.dedicated {
        return Err(Error::Unsupported(
            "Vulkan external memory requires dedicated allocation",
        ));
    }
    // SAFETY: the retained physical device is valid for capability queries.
    let properties = unsafe {
        device
            .0
            .instance
            .instance
            .get_physical_device_memory_properties(device.0.physical)
    };
    if handle.memory_type_index >= properties.memory_type_count {
        return Err(Error::InvalidRange);
    }
    let flags = properties.memory_types[handle.memory_type_index as usize].property_flags;
    let mut dedicated = vk::MemoryDedicatedAllocateInfo::default()
        .buffer(buffer)
        .image(image);
    let mut address_flags =
        vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
    #[cfg(windows)]
    let mut import = vk::ImportMemoryWin32HandleInfoKHR::default()
        .handle_type(HANDLE_TYPE)
        .handle(handle.handle.as_raw_handle() as isize);
    #[cfg(unix)]
    let mut import = vk::ImportMemoryFdInfoKHR::default()
        .handle_type(HANDLE_TYPE)
        .fd(handle.handle.as_raw_fd());
    let mut info = vk::MemoryAllocateInfo::default()
        .allocation_size(handle.allocation_size)
        .memory_type_index(handle.memory_type_index)
        .push_next(&mut import);
    if handle.dedicated {
        info = info.push_next(&mut dedicated);
    }
    if address {
        info = info.push_next(&mut address_flags);
    }
    // SAFETY: bridge guarantees the opaque native handle and allocation metadata;
    // queried native requirements and device memory type were validated above.
    let memory = unsafe { device.0.device.allocate_memory(&info, None) }.map_err(vk_error)?;
    #[cfg(unix)]
    {
        let _ = handle.handle.into_raw_fd();
    } // successful Vulkan import owns the fd
    Ok((memory, flags))
}

pub(super) fn import_buffer(
    device: &VulkanDevice,
    handle: VulkanExternalMemoryHandle,
    desc: &api::ExternalResourceDesc,
) -> Result<VulkanBuffer> {
    if !device.0.external_memory_handles {
        return Err(Error::Unsupported("Vulkan external memory handles"));
    }
    let buffer_desc = api::BufferDesc {
        label: None,
        size: desc.size,
        usage: desc.usage,
        memory: api::MemoryClass::Device,
    };
    if !matches!(
        resource::buffer_support(device, &buffer_desc),
        api::Support::Supported(_)
    ) {
        return Err(Error::Unsupported("Vulkan external buffer descriptor"));
    }
    let usage = resource::allocation::buffer_usage(desc.usage)?;
    let external_query = vk::PhysicalDeviceExternalBufferInfo::default()
        .usage(usage)
        .handle_type(HANDLE_TYPE);
    let mut external_props = vk::ExternalBufferProperties::default();
    // SAFETY: descriptor and output chain are initialized, physical device retained.
    unsafe {
        device
            .0
            .instance
            .instance
            .get_physical_device_external_buffer_properties(
                device.0.physical,
                &external_query,
                &mut external_props,
            )
    };
    check_features(
        external_props
            .external_memory_properties
            .external_memory_features,
        handle.dedicated,
    )?;
    let mut external = vk::ExternalMemoryBufferCreateInfo::default().handle_types(HANDLE_TYPE);
    let info = vk::BufferCreateInfo::default()
        .size(desc.size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .push_next(&mut external);
    // SAFETY: external memory features and descriptor were checked.
    let buffer = unsafe { device.0.device.create_buffer(&info, None) }.map_err(vk_error)?;
    let mut dedicated = vk::MemoryDedicatedRequirements::default();
    let mut requirements = vk::MemoryRequirements2::default().push_next(&mut dedicated);
    // SAFETY: newly created buffer and output chain are live.
    unsafe {
        device.0.device.get_buffer_memory_requirements2(
            &vk::BufferMemoryRequirementsInfo2::default().buffer(buffer),
            &mut requirements,
        )
    };
    let allocation_size = handle.allocation_size;
    let result = (|| {
        let (memory, flags) = allocation(
            device,
            handle,
            requirements.memory_requirements,
            desc.offset,
            dedicated.requires_dedicated_allocation == vk::TRUE,
            buffer,
            vk::Image::null(),
            usage.contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS),
        )?;
        // SAFETY: allocation was checked against requirements and remains owned here.
        if let Err(error) = unsafe {
            device
                .0
                .device
                .bind_buffer_memory(buffer, memory, desc.offset)
        } {
            // SAFETY: unpublished failed allocation is not in GPU use.
            unsafe { device.0.device.free_memory(memory, None) };
            return Err(vk_error(error));
        }
        Ok(VulkanBuffer(Arc::new(BufferState {
            device: device.0.clone(),
            buffer,
            memory,
            size: desc.size,
            usage: desc.usage,
            heap: None,
            default_allocation: None,
            memory_offset: desc.offset,
            map_state: MappingState::new(allocation_size, flags),
        })))
    })();
    if result.is_err() {
        // SAFETY: unpublished buffer may be destroyed after an error.
        unsafe { device.0.device.destroy_buffer(buffer, None) };
    }
    result
}

pub(super) fn import_texture(
    device: &VulkanDevice,
    handle: VulkanExternalMemoryHandle,
    desc: &api::TextureDesc,
) -> Result<VulkanTexture> {
    if !device.0.external_memory_handles {
        return Err(Error::Unsupported("Vulkan external memory handles"));
    }
    if !matches!(
        resource::texture_support(device, desc),
        api::Support::Supported(_)
    ) {
        return Err(Error::Unsupported("Vulkan external texture descriptor"));
    }
    let format = resource::native_format(&device.0, desc.format)
        .ok_or(Error::Unsupported("Vulkan external texture format"))?;
    let usage =
        resource::allocation::image_usage(desc.usage, resource::allocation::is_depth(desc.format))?;
    let flags = resource::image_flags(desc);
    let mut query_ext =
        vk::PhysicalDeviceExternalImageFormatInfo::default().handle_type(HANDLE_TYPE);
    let query = vk::PhysicalDeviceImageFormatInfo2::default()
        .format(format)
        .ty(resource::image_type(desc.dimension))
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(usage)
        .flags(flags)
        .push_next(&mut query_ext);
    let mut external_props = vk::ExternalImageFormatProperties::default();
    let mut props = vk::ImageFormatProperties2::default().push_next(&mut external_props);
    // SAFETY: query and its chains are valid, physical device is retained.
    unsafe {
        device
            .0
            .instance
            .instance
            .get_physical_device_image_format_properties2(device.0.physical, &query, &mut props)
    }
    .map_err(|error| {
        if error == vk::Result::ERROR_FORMAT_NOT_SUPPORTED {
            Error::Unsupported("Vulkan external texture format")
        } else {
            vk_error(error)
        }
    })?;
    let limits = props.image_format_properties;
    if desc.extent[0] > limits.max_extent.width
        || desc.extent[1] > limits.max_extent.height
        || desc.extent[2] > limits.max_extent.depth
        || desc.array_layers > limits.max_array_layers
        || desc.mip_levels > limits.max_mip_levels
        || !limits
            .sample_counts
            .contains(resource::allocation::sample_count(desc.samples)?)
    {
        return Err(Error::Unsupported("Vulkan external texture format limits"));
    }
    check_features(
        external_props
            .external_memory_properties
            .external_memory_features,
        handle.dedicated,
    )?;
    let mut ext = vk::ExternalMemoryImageCreateInfo::default().handle_types(HANDLE_TYPE);
    let info = vk::ImageCreateInfo::default()
        .flags(flags)
        .image_type(resource::image_type(desc.dimension))
        .format(format)
        .extent(vk::Extent3D {
            width: desc.extent[0],
            height: desc.extent[1],
            depth: desc.extent[2],
        })
        .mip_levels(desc.mip_levels)
        .array_layers(desc.array_layers)
        .samples(resource::allocation::sample_count(desc.samples)?)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .push_next(&mut ext);
    // SAFETY: queried descriptor and external handle type are supported.
    let image = unsafe { device.0.device.create_image(&info, None) }.map_err(vk_error)?;
    let mut dedicated = vk::MemoryDedicatedRequirements::default();
    let mut requirements = vk::MemoryRequirements2::default().push_next(&mut dedicated);
    // SAFETY: newly created image and output chain are live.
    unsafe {
        device.0.device.get_image_memory_requirements2(
            &vk::ImageMemoryRequirementsInfo2::default().image(image),
            &mut requirements,
        )
    };
    let offset = handle.image_offset;
    let result = (|| {
        let (memory, _) = allocation(
            device,
            handle,
            requirements.memory_requirements,
            offset,
            dedicated.requires_dedicated_allocation == vk::TRUE,
            vk::Buffer::null(),
            image,
            false,
        )?;
        // SAFETY: allocation was checked against image requirements.
        if let Err(error) = unsafe { device.0.device.bind_image_memory(image, memory, offset) } {
            // SAFETY: unpublished failed allocation is not in GPU use.
            unsafe { device.0.device.free_memory(memory, None) };
            return Err(vk_error(error));
        }
        Ok(VulkanTexture(Arc::new(TextureState {
            device: device.0.clone(),
            image,
            memory: Some(memory),
            heap: None,
            default_allocation: None,
            sparse: None,
            desc: desc.clone(),
            lazy_allocated: false,
        })))
    })();
    if result.is_err() {
        // SAFETY: unpublished image may be destroyed after an error.
        unsafe { device.0.device.destroy_image(image, None) };
    }
    result
}
