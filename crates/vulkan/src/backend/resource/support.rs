//! Device resource capability queries and descriptor validation.
use super::*;

pub(in crate::backend) fn buffer_support(
    device: &VulkanDevice,
    desc: &api::BufferDesc,
) -> api::Support<api::BufferSupport> {
    if desc.size == 0 || desc.usage.0 == 0 {
        return api::Support::Unsupported("zero-sized Vulkan buffers");
    }
    if desc.usage.0 & !SUPPORTED_BUFFER_USAGE != 0 {
        return api::Support::Unsupported("Vulkan buffer usage");
    }
    let ray_usage = api::BufferUsage::ACCEL_BUILD_INPUT.0
        | api::BufferUsage::ACCEL_SCRATCH.0
        | api::BufferUsage::SHADER_TABLE.0;
    if desc.usage.0 & ray_usage != 0 && !device.0.ray_support.acceleration {
        return api::Support::Unsupported("Vulkan acceleration structure buffers");
    }
    if desc.usage.contains(api::BufferUsage::SHADER_TABLE) && !device.0.ray_support.pipeline {
        return api::Support::Unsupported("Vulkan ray pipeline buffers");
    }
    // SAFETY: DeviceState retains both instance and the selected physical
    // device for this read-only capability query.
    let limits = unsafe {
        device
            .0
            .instance
            .instance
            .get_physical_device_properties(device.0.physical)
    }
    .limits;
    // Descriptor range limits constrain individual bindings, not allocations.
    // Vulkan 1.3 maintenance4 exposes the actual maximum buffer size.
    let mut maintenance = vk::PhysicalDeviceMaintenance4Properties::default();
    let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut maintenance);
    // SAFETY: the physical device and chained property storage remain live.
    unsafe {
        device
            .0
            .instance
            .instance
            .get_physical_device_properties2(device.0.physical, &mut properties)
    };
    let max_size = maintenance.max_buffer_size;
    if desc.size > max_size {
        return api::Support::Unsupported("Vulkan buffer range limit");
    }
    api::Support::Supported(api::BufferSupport {
        alignment: if desc.usage.contains(api::BufferUsage::UNIFORM) {
            u64::from(limits.min_uniform_buffer_offset_alignment.max(1))
        } else {
            1
        },
        max_size,
    })
}

pub(in crate::backend) fn texture_support(
    device: &VulkanDevice,
    desc: &api::TextureDesc,
) -> api::Support<api::TextureSupport> {
    let valid_extent = match desc.dimension {
        api::TextureDimension::D1 => desc.extent[1] == 1 && desc.extent[2] == 1,
        api::TextureDimension::D2 => desc.extent[2] == 1,
        api::TextureDimension::D3 => desc.array_layers == 1,
    };
    if !valid_extent
        || desc.extent.contains(&0)
        || desc.array_layers == 0
        || desc.mip_levels == 0
        || desc.samples == 0
        || desc.usage.0 == 0
        || (desc.dimension != api::TextureDimension::D2 && desc.samples != 1)
        || (desc.samples != 1 && desc.mip_levels != 1)
    {
        return api::Support::Unsupported("Vulkan texture descriptor");
    }
    if desc.usage.0 & !SUPPORTED_TEXTURE_USAGE != 0 {
        return api::Support::Unsupported("Vulkan texture usage");
    }
    if desc
        .view_formats
        .iter()
        .any(|&format| !compatible_formats(desc.format, format))
    {
        return api::Support::Unsupported("Vulkan mutable texture view format");
    }
    use api::TextureFormat::*;
    let compression_enabled = match desc.format {
        Bc1RgbaUnorm | Bc1RgbaUnormSrgb | Bc2RgbaUnorm | Bc2RgbaUnormSrgb | Bc3RgbaUnorm
        | Bc3RgbaUnormSrgb | Bc4RUnorm | Bc4RSnorm | Bc5RgUnorm | Bc5RgSnorm | Bc6hRgbUfloat
        | Bc6hRgbFloat | Bc7RgbaUnorm | Bc7RgbaUnormSrgb => {
            device.0.features.texture_compression_bc == vk::TRUE
        }
        Etc2Rgb8Unorm | Etc2Rgb8UnormSrgb | Etc2Rgba8Unorm | Etc2Rgba8UnormSrgb
        | Etc2Rgb8A1Unorm | Etc2Rgb8A1UnormSrgb | EacR11Unorm | EacR11Snorm | EacRg11Unorm
        | EacRg11Snorm => device.0.features.texture_compression_etc2 == vk::TRUE,
        Astc { hdr: false, .. } => device.0.features.texture_compression_astc_ldr == vk::TRUE,
        Astc { hdr: true, .. } => device.0.texture_compression_astc_hdr,
        _ => true,
    };
    if !compression_enabled {
        return api::Support::Unsupported("Vulkan texture compression feature");
    }
    let Some(format) = native_format(&device.0, desc.format) else {
        return api::Support::Unsupported("Vulkan texture format");
    };
    // SAFETY: the retained instance and physical device outlive this read-only
    // query.
    let properties = unsafe {
        device
            .0
            .instance
            .instance
            .get_physical_device_properties(device.0.physical)
    };
    let limits = properties.limits;
    let dimension_limit = match desc.dimension {
        api::TextureDimension::D1 => limits.max_image_dimension1_d,
        api::TextureDimension::D2 => limits.max_image_dimension2_d,
        api::TextureDimension::D3 => limits.max_image_dimension3_d,
    };
    if desc.extent.iter().any(|&extent| extent > dimension_limit)
        || desc.array_layers > limits.max_image_array_layers
        || desc.mip_levels > max_mips(*desc.extent.iter().max().unwrap())
    {
        return api::Support::Unsupported("Vulkan texture dimension limit");
    }
    // SAFETY: as above, this only queries format properties of the retained
    // selected physical device.
    let features = unsafe {
        device
            .0
            .instance
            .instance
            .get_physical_device_format_properties(device.0.physical, format)
    }
    .optimal_tiling_features;
    // With EXTENDED_USAGE the required format features may be supplied by a
    // declared compatible view format rather than by the base image format.
    let mut usage_features = features;
    for &view in &desc.view_formats {
        if let Some(view_format) = native_format(&device.0, view) {
            // SAFETY: the physical device is retained by DeviceState.
            usage_features |= unsafe {
                device
                    .0
                    .instance
                    .instance
                    .get_physical_device_format_properties(device.0.physical, view_format)
            }
            .optimal_tiling_features;
        }
    }
    let depth = is_depth(desc.format);
    if (desc.usage.contains(api::TextureUsage::TEXTURE_BINDING)
        && !usage_features.contains(vk::FormatFeatureFlags::SAMPLED_IMAGE))
        || (desc.usage.contains(api::TextureUsage::RENDER_ATTACHMENT)
            && !(if depth {
                usage_features.contains(vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT)
            } else {
                usage_features.contains(vk::FormatFeatureFlags::COLOR_ATTACHMENT)
            }))
        || (desc.usage.contains(api::TextureUsage::STORAGE_BINDING)
            && !usage_features.contains(vk::FormatFeatureFlags::STORAGE_IMAGE))
    {
        return api::Support::Unsupported("Vulkan texture format usage");
    }
    let Ok(usage) = image_usage(desc.usage, depth) else {
        return api::Support::Unsupported("Vulkan texture usage");
    };
    // SAFETY: `format` and `usage` were derived from the validated descriptor;
    // the device retains the queried instance and physical device.
    let Ok(image_properties) = (unsafe {
        device
            .0
            .instance
            .instance
            .get_physical_device_image_format_properties(
                device.0.physical,
                format,
                image_type(desc.dimension),
                vk::ImageTiling::OPTIMAL,
                usage,
                image_flags(desc),
            )
    }) else {
        return api::Support::Unsupported("Vulkan texture format usage");
    };
    if desc.extent[0] > image_properties.max_extent.width
        || desc.extent[1] > image_properties.max_extent.height
        || desc.extent[2] > image_properties.max_extent.depth
        || desc.array_layers > image_properties.max_array_layers
        || desc.mip_levels > image_properties.max_mip_levels
    {
        return api::Support::Unsupported("Vulkan texture format limit");
    }
    let sample_counts = sample_counts(image_properties.sample_counts);
    if !sample_counts.contains(&desc.samples) {
        return api::Support::Unsupported("Vulkan texture sample count");
    }
    api::Support::Supported(api::TextureSupport {
        max_extent: [
            image_properties.max_extent.width,
            image_properties.max_extent.height,
            image_properties.max_extent.depth,
        ],
        max_array_layers: image_properties.max_array_layers,
        max_mips: image_properties.max_mip_levels,
        sample_counts,
        compatible_view_formats: std::iter::once(desc.format)
            .chain(
                desc.view_formats
                    .iter()
                    .copied()
                    .filter(|f| *f != desc.format),
            )
            .collect(),
        filterable: features.contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR),
        blendable: features.contains(vk::FormatFeatureFlags::COLOR_ATTACHMENT_BLEND),
        storage: features.contains(vk::FormatFeatureFlags::STORAGE_IMAGE),
    })
}

fn compatible_formats(a: api::TextureFormat, b: api::TextureFormat) -> bool {
    use api::TextureFormat::*;
    let color_class = |f| match f {
        R8Unorm | R8Snorm | R8Uint | R8Sint => 8,
        Rg8Unorm | Rg8Snorm | Rg8Uint | Rg8Sint | R16Unorm | R16Snorm | R16Uint | R16Sint
        | R16Float => 16,
        Rgba8Unorm | Rgba8UnormSrgb | Rgba8Snorm | Rgba8Uint | Rgba8Sint | Bgra8Unorm
        | Bgra8UnormSrgb | Rg16Unorm | Rg16Snorm | Rg16Uint | Rg16Sint | Rg16Float | R32Uint
        | R32Sint | R32Float | Rgb10a2Unorm | Rgb10a2Uint | Rg11b10Ufloat | Rgb9e5Ufloat => 32,
        Rgba16Unorm | Rgba16Snorm | Rgba16Uint | Rgba16Sint | Rgba16Float | Rg32Uint | Rg32Sint
        | Rg32Float | R64Uint => 64,
        Rgba32Uint | Rgba32Sint | Rgba32Float | Rg64Uint => 128,
        _ => 0,
    };
    a == b
        || (color_class(a) != 0 && color_class(a) == color_class(b))
        || matches!(
            (a, b),
            (Rgba8Unorm, Rgba8UnormSrgb)
                | (Rgba8UnormSrgb, Rgba8Unorm)
                | (Bgra8Unorm, Bgra8UnormSrgb)
                | (Bgra8UnormSrgb, Bgra8Unorm)
                | (Bc1RgbaUnorm, Bc1RgbaUnormSrgb)
                | (Bc1RgbaUnormSrgb, Bc1RgbaUnorm)
                | (Bc2RgbaUnorm, Bc2RgbaUnormSrgb)
                | (Bc2RgbaUnormSrgb, Bc2RgbaUnorm)
                | (Bc3RgbaUnorm, Bc3RgbaUnormSrgb)
                | (Bc3RgbaUnormSrgb, Bc3RgbaUnorm)
                | (Bc7RgbaUnorm, Bc7RgbaUnormSrgb)
                | (Bc7RgbaUnormSrgb, Bc7RgbaUnorm)
                | (Etc2Rgb8Unorm, Etc2Rgb8UnormSrgb)
                | (Etc2Rgb8UnormSrgb, Etc2Rgb8Unorm)
                | (Etc2Rgba8Unorm, Etc2Rgba8UnormSrgb)
                | (Etc2Rgba8UnormSrgb, Etc2Rgba8Unorm)
                | (Etc2Rgb8A1Unorm, Etc2Rgb8A1UnormSrgb)
                | (Etc2Rgb8A1UnormSrgb, Etc2Rgb8A1Unorm)
        )
        || matches!((a, b), (Astc {block: x, srgb: sx, hdr: false}, Astc {block: y, srgb: sy, hdr: false}) if x == y && sx != sy)
}
