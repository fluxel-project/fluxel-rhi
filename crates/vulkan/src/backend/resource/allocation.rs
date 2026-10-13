//! Vulkan resource descriptor lowering and memory allocation.
//!
//! This module translates validated Core descriptors into Vulkan flags and
//! chooses a compatible memory type.  It does not own native resources or
//! track their lifetime; the resource wrappers in the parent module retain the
//! device and destroy the handles after GPU completion.

use super::*;

pub(super) fn allocate(
    device: &VulkanDevice,
    requirements: vk::MemoryRequirements,
    flags: vk::MemoryPropertyFlags,
) -> Result<vk::DeviceMemory> {
    // SAFETY: DeviceState keeps this physical device and its instance alive for
    // the read-only memory-property query.
    let properties = unsafe {
        device
            .0
            .instance
            .instance
            .get_physical_device_memory_properties(device.0.physical)
    };
    let index = (0..properties.memory_type_count)
        .find(|index| {
            requirements.memory_type_bits & (1 << index) != 0
                && properties.memory_types[*index as usize]
                    .property_flags
                    .contains(flags)
        })
        .ok_or(Error::OutOfMemory)?;
    let info = vk::MemoryAllocateInfo::default()
        .allocation_size(requirements.size)
        .memory_type_index(index);
    // SAFETY: requirements came directly from a live resource on this device;
    // the selected memory type satisfies its type mask and requested flags.
    unsafe { device.0.device.allocate_memory(&info, None) }.map_err(vk_error)
}

pub(super) fn memory_flags(memory: api::MemoryClass) -> vk::MemoryPropertyFlags {
    match memory {
        api::MemoryClass::Device => vk::MemoryPropertyFlags::DEVICE_LOCAL,
        api::MemoryClass::Upload => {
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
        }
        api::MemoryClass::Readback => {
            // Query/readback consumers map bytes after a completion point.
            // Request coherent memory so the narrow ReadbackBackend mapping
            // contract never exposes stale non-coherent cache lines.
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
        }
    }
}

pub(super) fn buffer_usage(usage: api::BufferUsage) -> Result<vk::BufferUsageFlags> {
    if usage.0 & !SUPPORTED_BUFFER_USAGE != 0 {
        return Err(Error::Unsupported("Vulkan buffer usage"));
    }
    let mut out = vk::BufferUsageFlags::empty();
    if usage.contains(api::BufferUsage::COPY_SRC) {
        out |= vk::BufferUsageFlags::TRANSFER_SRC;
    }
    if usage.contains(api::BufferUsage::COPY_DST) {
        out |= vk::BufferUsageFlags::TRANSFER_DST;
    }
    if usage.contains(api::BufferUsage::VERTEX) {
        out |= vk::BufferUsageFlags::VERTEX_BUFFER;
    }
    if usage.contains(api::BufferUsage::INDEX) {
        out |= vk::BufferUsageFlags::INDEX_BUFFER;
    }
    if usage.contains(api::BufferUsage::UNIFORM) {
        out |= vk::BufferUsageFlags::UNIFORM_BUFFER;
    }
    if usage.contains(api::BufferUsage::STORAGE) {
        out |= vk::BufferUsageFlags::STORAGE_BUFFER;
    }
    if usage.contains(api::BufferUsage::INDIRECT) {
        out |= vk::BufferUsageFlags::INDIRECT_BUFFER;
    }
    if usage.contains(api::BufferUsage::QUERY_RESOLVE) {
        out |= vk::BufferUsageFlags::TRANSFER_DST;
    }
    if out.is_empty() {
        return Err(Error::Unsupported("Vulkan buffer usage"));
    }
    Ok(out)
}

pub(super) fn image_usage(usage: api::TextureUsage, depth: bool) -> Result<vk::ImageUsageFlags> {
    if usage.0 & !SUPPORTED_TEXTURE_USAGE != 0 {
        return Err(Error::Unsupported("Vulkan texture usage"));
    }
    let mut out = vk::ImageUsageFlags::empty();
    if usage.contains(api::TextureUsage::COPY_SRC) {
        out |= vk::ImageUsageFlags::TRANSFER_SRC;
    }
    if usage.contains(api::TextureUsage::COPY_DST) {
        out |= vk::ImageUsageFlags::TRANSFER_DST;
    }
    if usage.contains(api::TextureUsage::TEXTURE_BINDING) {
        out |= vk::ImageUsageFlags::SAMPLED;
    }
    if usage.contains(api::TextureUsage::RENDER_ATTACHMENT) {
        out |= if depth {
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
        } else {
            vk::ImageUsageFlags::COLOR_ATTACHMENT
        };
    }
    if usage.contains(api::TextureUsage::STORAGE_BINDING) {
        out |= vk::ImageUsageFlags::STORAGE;
    }
    if out.is_empty() {
        return Err(Error::Unsupported("Vulkan texture usage"));
    }
    Ok(out)
}

pub(super) fn sample_count(count: u32) -> Result<vk::SampleCountFlags> {
    match count {
        1 => Ok(vk::SampleCountFlags::TYPE_1),
        2 => Ok(vk::SampleCountFlags::TYPE_2),
        4 => Ok(vk::SampleCountFlags::TYPE_4),
        8 => Ok(vk::SampleCountFlags::TYPE_8),
        _ => Err(Error::Unsupported("Vulkan sample count")),
    }
}

pub(super) fn sample_counts(flags: vk::SampleCountFlags) -> Vec<u32> {
    [
        (vk::SampleCountFlags::TYPE_1, 1),
        (vk::SampleCountFlags::TYPE_2, 2),
        (vk::SampleCountFlags::TYPE_4, 4),
        (vk::SampleCountFlags::TYPE_8, 8),
        (vk::SampleCountFlags::TYPE_16, 16),
        (vk::SampleCountFlags::TYPE_32, 32),
        (vk::SampleCountFlags::TYPE_64, 64),
    ]
    .into_iter()
    .filter_map(|(flag, count)| flags.contains(flag).then_some(count))
    .collect()
}

pub(super) fn max_mips(max_dimension: u32) -> u32 {
    max_dimension.ilog2() + 1
}
pub(super) fn filter(filter: api::FilterMode) -> vk::Filter {
    if filter == api::FilterMode::Linear {
        vk::Filter::LINEAR
    } else {
        vk::Filter::NEAREST
    }
}
pub(super) fn address(mode: api::AddressMode) -> Result<vk::SamplerAddressMode> {
    Ok(match mode {
        api::AddressMode::ClampToEdge => vk::SamplerAddressMode::CLAMP_TO_EDGE,
        api::AddressMode::Repeat => vk::SamplerAddressMode::REPEAT,
        api::AddressMode::MirrorRepeat => vk::SamplerAddressMode::MIRRORED_REPEAT,
        api::AddressMode::ClampToBorder => vk::SamplerAddressMode::CLAMP_TO_BORDER,
        api::AddressMode::ClampToZero => {
            return Err(Error::Unsupported("Vulkan clamp-to-zero sampler"));
        }
    })
}
pub(crate) fn format(format: api::TextureFormat) -> Option<vk::Format> {
    Some(match format {
        api::TextureFormat::Rgba8Unorm => vk::Format::R8G8B8A8_UNORM,
        api::TextureFormat::Rgba8UnormSrgb => vk::Format::R8G8B8A8_SRGB,
        api::TextureFormat::Bgra8Unorm => vk::Format::B8G8R8A8_UNORM,
        api::TextureFormat::Bgra8UnormSrgb => vk::Format::B8G8R8A8_SRGB,
        api::TextureFormat::Depth32Float => vk::Format::D32_SFLOAT,
        _ => return None,
    })
}
pub(super) fn is_depth(format: api::TextureFormat) -> bool {
    matches!(format, api::TextureFormat::Depth32Float)
}

pub(super) const SUPPORTED_BUFFER_USAGE: u64 = api::BufferUsage::COPY_SRC.0
    | api::BufferUsage::COPY_DST.0
    | api::BufferUsage::VERTEX.0
    | api::BufferUsage::INDEX.0
    | api::BufferUsage::UNIFORM.0
    | api::BufferUsage::STORAGE.0
    | api::BufferUsage::INDIRECT.0
    | api::BufferUsage::MAP_READ.0
    | api::BufferUsage::MAP_WRITE.0
    | api::BufferUsage::QUERY_RESOLVE.0;
pub(super) const SUPPORTED_TEXTURE_USAGE: u64 = api::TextureUsage::COPY_SRC.0
    | api::TextureUsage::COPY_DST.0
    | api::TextureUsage::TEXTURE_BINDING.0
    | api::TextureUsage::STORAGE_BINDING.0
    | api::TextureUsage::RENDER_ATTACHMENT.0;
