//! Vulkan resource descriptor lowering and memory allocation.
//!
//! This module translates validated Core descriptors into Vulkan flags and
//! chooses a compatible memory type.  It does not own native resources or
//! track their lifetime; the resource wrappers in the parent module retain the
//! device and destroy the handles after GPU completion.

use super::*;

pub(crate) fn allocate_with_address(
    device: &VulkanDevice,
    requirements: vk::MemoryRequirements,
    flags: vk::MemoryPropertyFlags,
    address: bool,
) -> Result<vk::DeviceMemory> {
    allocate_details(device, requirements, flags, address).map(|(memory, _)| memory)
}

pub(super) fn allocate_details(
    device: &VulkanDevice,
    requirements: vk::MemoryRequirements,
    flags: vk::MemoryPropertyFlags,
    address: bool,
) -> Result<(vk::DeviceMemory, vk::MemoryPropertyFlags)> {
    allocate_resource(device, requirements, flags, address, None)
}

pub(super) fn allocate_resource(
    device: &VulkanDevice,
    requirements: vk::MemoryRequirements,
    flags: vk::MemoryPropertyFlags,
    address: bool,
    dedicated: Option<(vk::Buffer, vk::Image)>,
) -> Result<(vk::DeviceMemory, vk::MemoryPropertyFlags)> {
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
    let mut address_flags =
        vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
    let mut info = vk::MemoryAllocateInfo::default()
        .allocation_size(requirements.size)
        .memory_type_index(index);
    if address {
        info = info.push_next(&mut address_flags);
    }
    let mut dedicated_info = vk::MemoryDedicatedAllocateInfo::default();
    if let Some((buffer, image)) = dedicated {
        dedicated_info = dedicated_info.buffer(buffer).image(image);
        info = info.push_next(&mut dedicated_info);
    }
    // SAFETY: requirements came directly from a live resource on this device;
    // the selected memory type satisfies its type mask and requested flags.
    unsafe { device.0.device.allocate_memory(&info, None) }
        .map(|memory| {
            (
                memory,
                properties.memory_types[index as usize].property_flags,
            )
        })
        .map_err(vk_error)
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

pub(in crate::backend) fn has_compatible_lazy_memory_type(
    device: &VulkanDevice,
    requirements: vk::MemoryRequirements,
) -> bool {
    // SAFETY: the retained physical device owns this immutable memory table.
    let properties = unsafe {
        device
            .0
            .instance
            .instance
            .get_physical_device_memory_properties(device.0.physical)
    };
    (0..properties.memory_type_count).any(|index| {
        (requirements.memory_type_bits & (1 << index)) != 0
            && properties.memory_types[index as usize]
                .property_flags
                .contains(vk::MemoryPropertyFlags::LAZILY_ALLOCATED)
    })
}

pub(in crate::backend) fn buffer_usage(usage: api::BufferUsage) -> Result<vk::BufferUsageFlags> {
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
    if usage.contains(api::BufferUsage::ACCEL_BUILD_INPUT) {
        out |= vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
    }
    if usage.contains(api::BufferUsage::ACCEL_SCRATCH) {
        out |= vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
    }
    if usage.contains(api::BufferUsage::SHADER_TABLE) {
        out |= vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
    }
    if out.is_empty() {
        return Err(Error::Unsupported("Vulkan buffer usage"));
    }
    Ok(out)
}

pub(in crate::backend) fn image_usage(
    usage: api::TextureUsage,
    depth: bool,
) -> Result<vk::ImageUsageFlags> {
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
        if !usage.contains(api::TextureUsage::TRANSIENT_ATTACHMENT) {
            // Persistent RHI resolve/copy operations require transfer usage.
            out |= vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST;
        }
        out |= if depth {
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
        } else {
            vk::ImageUsageFlags::COLOR_ATTACHMENT
        };
    }
    if usage.contains(api::TextureUsage::TRANSIENT_ATTACHMENT) {
        out |= vk::ImageUsageFlags::TRANSIENT_ATTACHMENT;
    }
    if usage.contains(api::TextureUsage::STORAGE_BINDING) {
        out |= vk::ImageUsageFlags::STORAGE;
    }
    if out.is_empty() {
        return Err(Error::Unsupported("Vulkan texture usage"));
    }
    Ok(out)
}

pub(in crate::backend) fn sample_count(count: u32) -> Result<vk::SampleCountFlags> {
    match count {
        1 => Ok(vk::SampleCountFlags::TYPE_1),
        2 => Ok(vk::SampleCountFlags::TYPE_2),
        4 => Ok(vk::SampleCountFlags::TYPE_4),
        8 => Ok(vk::SampleCountFlags::TYPE_8),
        16 => Ok(vk::SampleCountFlags::TYPE_16),
        32 => Ok(vk::SampleCountFlags::TYPE_32),
        64 => Ok(vk::SampleCountFlags::TYPE_64),
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
        api::AddressMode::ClampToZero => vk::SamplerAddressMode::CLAMP_TO_BORDER,
    })
}
pub(crate) fn format(format: api::TextureFormat) -> Option<vk::Format> {
    Some(match format {
        api::TextureFormat::R8Unorm => vk::Format::R8_UNORM,
        api::TextureFormat::R8Snorm => vk::Format::R8_SNORM,
        api::TextureFormat::R8Uint => vk::Format::R8_UINT,
        api::TextureFormat::R8Sint => vk::Format::R8_SINT,
        api::TextureFormat::Rg8Unorm => vk::Format::R8G8_UNORM,
        api::TextureFormat::Rg8Snorm => vk::Format::R8G8_SNORM,
        api::TextureFormat::Rg8Uint => vk::Format::R8G8_UINT,
        api::TextureFormat::Rg8Sint => vk::Format::R8G8_SINT,
        api::TextureFormat::Rgba8Unorm => vk::Format::R8G8B8A8_UNORM,
        api::TextureFormat::Rgba8UnormSrgb => vk::Format::R8G8B8A8_SRGB,
        api::TextureFormat::Rgba8Snorm => vk::Format::R8G8B8A8_SNORM,
        api::TextureFormat::Rgba8Uint => vk::Format::R8G8B8A8_UINT,
        api::TextureFormat::Rgba8Sint => vk::Format::R8G8B8A8_SINT,
        api::TextureFormat::Bgra8Unorm => vk::Format::B8G8R8A8_UNORM,
        api::TextureFormat::Bgra8UnormSrgb => vk::Format::B8G8R8A8_SRGB,
        api::TextureFormat::R16Unorm => vk::Format::R16_UNORM,
        api::TextureFormat::R16Snorm => vk::Format::R16_SNORM,
        api::TextureFormat::R16Uint => vk::Format::R16_UINT,
        api::TextureFormat::R16Sint => vk::Format::R16_SINT,
        api::TextureFormat::R16Float => vk::Format::R16_SFLOAT,
        api::TextureFormat::Rg16Unorm => vk::Format::R16G16_UNORM,
        api::TextureFormat::Rg16Snorm => vk::Format::R16G16_SNORM,
        api::TextureFormat::Rg16Uint => vk::Format::R16G16_UINT,
        api::TextureFormat::Rg16Sint => vk::Format::R16G16_SINT,
        api::TextureFormat::Rg16Float => vk::Format::R16G16_SFLOAT,
        api::TextureFormat::Rgba16Unorm => vk::Format::R16G16B16A16_UNORM,
        api::TextureFormat::Rgba16Snorm => vk::Format::R16G16B16A16_SNORM,
        api::TextureFormat::Rgba16Uint => vk::Format::R16G16B16A16_UINT,
        api::TextureFormat::Rgba16Sint => vk::Format::R16G16B16A16_SINT,
        api::TextureFormat::Rgba16Float => vk::Format::R16G16B16A16_SFLOAT,
        api::TextureFormat::R32Uint => vk::Format::R32_UINT,
        api::TextureFormat::R32Sint => vk::Format::R32_SINT,
        api::TextureFormat::R32Float => vk::Format::R32_SFLOAT,
        api::TextureFormat::Rg32Uint => vk::Format::R32G32_UINT,
        api::TextureFormat::Rg32Sint => vk::Format::R32G32_SINT,
        api::TextureFormat::Rg32Float => vk::Format::R32G32_SFLOAT,
        api::TextureFormat::Rgba32Uint => vk::Format::R32G32B32A32_UINT,
        api::TextureFormat::Rgba32Sint => vk::Format::R32G32B32A32_SINT,
        api::TextureFormat::Rgba32Float => vk::Format::R32G32B32A32_SFLOAT,
        api::TextureFormat::Rgb10a2Unorm => vk::Format::A2B10G10R10_UNORM_PACK32,
        api::TextureFormat::Rgb10a2Uint => vk::Format::A2B10G10R10_UINT_PACK32,
        api::TextureFormat::Rg11b10Ufloat => vk::Format::B10G11R11_UFLOAT_PACK32,
        api::TextureFormat::Rgb9e5Ufloat => vk::Format::E5B9G9R9_UFLOAT_PACK32,
        api::TextureFormat::Depth16Unorm => vk::Format::D16_UNORM,
        api::TextureFormat::Depth24Plus => vk::Format::X8_D24_UNORM_PACK32,
        api::TextureFormat::Depth24PlusStencil8 => vk::Format::D24_UNORM_S8_UINT,
        api::TextureFormat::Depth32Float => vk::Format::D32_SFLOAT,
        api::TextureFormat::Depth32FloatStencil8 => vk::Format::D32_SFLOAT_S8_UINT,
        api::TextureFormat::Stencil8 => vk::Format::S8_UINT,
        api::TextureFormat::Bc1RgbaUnorm => vk::Format::BC1_RGBA_UNORM_BLOCK,
        api::TextureFormat::Bc1RgbaUnormSrgb => vk::Format::BC1_RGBA_SRGB_BLOCK,
        api::TextureFormat::Bc2RgbaUnorm => vk::Format::BC2_UNORM_BLOCK,
        api::TextureFormat::Bc2RgbaUnormSrgb => vk::Format::BC2_SRGB_BLOCK,
        api::TextureFormat::Bc3RgbaUnorm => vk::Format::BC3_UNORM_BLOCK,
        api::TextureFormat::Bc3RgbaUnormSrgb => vk::Format::BC3_SRGB_BLOCK,
        api::TextureFormat::Bc4RUnorm => vk::Format::BC4_UNORM_BLOCK,
        api::TextureFormat::Bc4RSnorm => vk::Format::BC4_SNORM_BLOCK,
        api::TextureFormat::Bc5RgUnorm => vk::Format::BC5_UNORM_BLOCK,
        api::TextureFormat::Bc5RgSnorm => vk::Format::BC5_SNORM_BLOCK,
        api::TextureFormat::Bc6hRgbUfloat => vk::Format::BC6H_UFLOAT_BLOCK,
        api::TextureFormat::Bc6hRgbFloat => vk::Format::BC6H_SFLOAT_BLOCK,
        api::TextureFormat::Bc7RgbaUnorm => vk::Format::BC7_UNORM_BLOCK,
        api::TextureFormat::Bc7RgbaUnormSrgb => vk::Format::BC7_SRGB_BLOCK,
        api::TextureFormat::Etc2Rgb8Unorm => vk::Format::ETC2_R8G8B8_UNORM_BLOCK,
        api::TextureFormat::Etc2Rgb8UnormSrgb => vk::Format::ETC2_R8G8B8_SRGB_BLOCK,
        api::TextureFormat::Etc2Rgba8Unorm => vk::Format::ETC2_R8G8B8A8_UNORM_BLOCK,
        api::TextureFormat::Etc2Rgba8UnormSrgb => vk::Format::ETC2_R8G8B8A8_SRGB_BLOCK,
        api::TextureFormat::Etc2Rgb8A1Unorm => vk::Format::ETC2_R8G8B8A1_UNORM_BLOCK,
        api::TextureFormat::Etc2Rgb8A1UnormSrgb => vk::Format::ETC2_R8G8B8A1_SRGB_BLOCK,
        api::TextureFormat::EacR11Unorm => vk::Format::EAC_R11_UNORM_BLOCK,
        api::TextureFormat::EacR11Snorm => vk::Format::EAC_R11_SNORM_BLOCK,
        api::TextureFormat::EacRg11Unorm => vk::Format::EAC_R11G11_UNORM_BLOCK,
        api::TextureFormat::EacRg11Snorm => vk::Format::EAC_R11G11_SNORM_BLOCK,
        api::TextureFormat::R64Uint => vk::Format::R64_UINT,
        api::TextureFormat::Rg64Uint => vk::Format::R64G64_UINT,
        api::TextureFormat::Astc { block, srgb, hdr } => match (block, srgb, hdr) {
            ([4, 4], false, false) => vk::Format::ASTC_4X4_UNORM_BLOCK,
            ([4, 4], true, false) => vk::Format::ASTC_4X4_SRGB_BLOCK,
            ([4, 4], false, true) => vk::Format::ASTC_4X4_SFLOAT_BLOCK,
            ([5, 4], false, false) => vk::Format::ASTC_5X4_UNORM_BLOCK,
            ([5, 4], true, false) => vk::Format::ASTC_5X4_SRGB_BLOCK,
            ([5, 4], false, true) => vk::Format::ASTC_5X4_SFLOAT_BLOCK,
            ([5, 5], false, false) => vk::Format::ASTC_5X5_UNORM_BLOCK,
            ([5, 5], true, false) => vk::Format::ASTC_5X5_SRGB_BLOCK,
            ([5, 5], false, true) => vk::Format::ASTC_5X5_SFLOAT_BLOCK,
            ([6, 5], false, false) => vk::Format::ASTC_6X5_UNORM_BLOCK,
            ([6, 5], true, false) => vk::Format::ASTC_6X5_SRGB_BLOCK,
            ([6, 5], false, true) => vk::Format::ASTC_6X5_SFLOAT_BLOCK,
            ([6, 6], false, false) => vk::Format::ASTC_6X6_UNORM_BLOCK,
            ([6, 6], true, false) => vk::Format::ASTC_6X6_SRGB_BLOCK,
            ([6, 6], false, true) => vk::Format::ASTC_6X6_SFLOAT_BLOCK,
            ([8, 5], false, false) => vk::Format::ASTC_8X5_UNORM_BLOCK,
            ([8, 5], true, false) => vk::Format::ASTC_8X5_SRGB_BLOCK,
            ([8, 5], false, true) => vk::Format::ASTC_8X5_SFLOAT_BLOCK,
            ([8, 6], false, false) => vk::Format::ASTC_8X6_UNORM_BLOCK,
            ([8, 6], true, false) => vk::Format::ASTC_8X6_SRGB_BLOCK,
            ([8, 6], false, true) => vk::Format::ASTC_8X6_SFLOAT_BLOCK,
            ([8, 8], false, false) => vk::Format::ASTC_8X8_UNORM_BLOCK,
            ([8, 8], true, false) => vk::Format::ASTC_8X8_SRGB_BLOCK,
            ([8, 8], false, true) => vk::Format::ASTC_8X8_SFLOAT_BLOCK,
            ([10, 5], false, false) => vk::Format::ASTC_10X5_UNORM_BLOCK,
            ([10, 5], true, false) => vk::Format::ASTC_10X5_SRGB_BLOCK,
            ([10, 5], false, true) => vk::Format::ASTC_10X5_SFLOAT_BLOCK,
            ([10, 6], false, false) => vk::Format::ASTC_10X6_UNORM_BLOCK,
            ([10, 6], true, false) => vk::Format::ASTC_10X6_SRGB_BLOCK,
            ([10, 6], false, true) => vk::Format::ASTC_10X6_SFLOAT_BLOCK,
            ([10, 8], false, false) => vk::Format::ASTC_10X8_UNORM_BLOCK,
            ([10, 8], true, false) => vk::Format::ASTC_10X8_SRGB_BLOCK,
            ([10, 8], false, true) => vk::Format::ASTC_10X8_SFLOAT_BLOCK,
            ([10, 10], false, false) => vk::Format::ASTC_10X10_UNORM_BLOCK,
            ([10, 10], true, false) => vk::Format::ASTC_10X10_SRGB_BLOCK,
            ([10, 10], false, true) => vk::Format::ASTC_10X10_SFLOAT_BLOCK,
            ([12, 10], false, false) => vk::Format::ASTC_12X10_UNORM_BLOCK,
            ([12, 10], true, false) => vk::Format::ASTC_12X10_SRGB_BLOCK,
            ([12, 10], false, true) => vk::Format::ASTC_12X10_SFLOAT_BLOCK,
            ([12, 12], false, false) => vk::Format::ASTC_12X12_UNORM_BLOCK,
            ([12, 12], true, false) => vk::Format::ASTC_12X12_SRGB_BLOCK,
            ([12, 12], false, true) => vk::Format::ASTC_12X12_SFLOAT_BLOCK,
            _ => return None,
        },
    })
}
pub(in crate::backend) fn is_depth(format: api::TextureFormat) -> bool {
    matches!(
        format,
        api::TextureFormat::Depth16Unorm
            | api::TextureFormat::Depth24Plus
            | api::TextureFormat::Depth24PlusStencil8
            | api::TextureFormat::Depth32Float
            | api::TextureFormat::Depth32FloatStencil8
            | api::TextureFormat::Stencil8
    )
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
    | api::BufferUsage::QUERY_RESOLVE.0
    | api::BufferUsage::ACCEL_BUILD_INPUT.0
    | api::BufferUsage::ACCEL_SCRATCH.0
    | api::BufferUsage::SHADER_TABLE.0;
pub(super) const SUPPORTED_TEXTURE_USAGE: u64 = api::TextureUsage::COPY_SRC.0
    | api::TextureUsage::COPY_DST.0
    | api::TextureUsage::TEXTURE_BINDING.0
    | api::TextureUsage::STORAGE_BINDING.0
    | api::TextureUsage::RENDER_ATTACHMENT.0
    | api::TextureUsage::TRANSIENT_ATTACHMENT.0;
