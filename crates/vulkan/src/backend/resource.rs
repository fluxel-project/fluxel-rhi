//! Vulkan resource allocation and view creation.
//!
//! This module owns Vulkan allocation handles and tears them down before their
//! device.  It deliberately does not track resource state: Core transitions
//! carry the states selected by RenderGraph.
//!
//! Each native object retains its `DeviceState` through an `Arc`.  Its Drop
//! implementation therefore destroys child views, objects, and allocations
//! while the logical device is still alive.  The order is object before backing
//! allocation.  Resource state and cross-operation hazards remain Core input.

use std::sync::Arc;

mod allocation;
pub(super) use allocation::format;
use allocation::*;

use ash::vk;
use fluxel_rhi_core::api::{self, Error, Result};

use super::{DeviceState, VulkanDevice, vk_error};

#[derive(Clone)]
pub struct VulkanBuffer(pub(super) Arc<BufferState>);
pub(super) struct BufferState {
    pub(super) device: Arc<DeviceState>,
    pub(super) buffer: vk::Buffer,
    pub(super) memory: vk::DeviceMemory,
    pub(super) size: u64,
    pub(super) usage: api::BufferUsage,
}
impl Drop for BufferState {
    fn drop(&mut self) {
        // SAFETY: this Arc is the final owner of the buffer and its allocation;
        // all submitted encoders retain VulkanBuffer until their fence retires.
        unsafe {
            self.device.device.destroy_buffer(self.buffer, None);
            self.device.device.free_memory(self.memory, None);
        }
    }
}

#[derive(Clone)]
pub struct VulkanTexture(Arc<TextureState>);
impl VulkanTexture {
    pub(super) fn native(&self) -> vk::Image {
        self.0.image
    }
    pub(super) fn desc(&self) -> &api::TextureDesc {
        &self.0.desc
    }
}
struct TextureState {
    device: Arc<DeviceState>,
    pub(super) image: vk::Image,
    memory: vk::DeviceMemory,
    pub(super) desc: api::TextureDesc,
}
impl Drop for TextureState {
    fn drop(&mut self) {
        // SAFETY: this Arc is the final image owner; TextureViewState retains
        // VulkanTexture, so every view is destroyed before this allocation.
        unsafe {
            self.device.device.destroy_image(self.image, None);
            self.device.device.free_memory(self.memory, None);
        }
    }
}

#[derive(Clone)]
pub struct VulkanTextureView(Arc<TextureViewState>);
impl VulkanTextureView {
    pub(super) fn native(&self) -> vk::ImageView {
        self.0.view
    }
    pub(super) fn image(&self) -> vk::Image {
        self.0.texture.0.image
    }
    pub(super) fn format(&self) -> api::TextureFormat {
        self.0.desc.format
    }
    pub(super) fn dimension(&self) -> api::TextureViewDimension {
        self.0.desc.dimension
    }
    pub(super) fn samples(&self) -> u32 {
        self.0.texture.0.desc.samples
    }
    pub(super) fn usage(&self) -> api::TextureUsage {
        self.0.texture.0.desc.usage
    }
    /// Dynamic rendering maps multiview indices onto consecutive layers of a
    /// 2D-array attachment.  The public pass owns the selected mask; this
    /// helper only proves that the view can back every referenced layer.
    pub(super) fn attachment_extent_for_layers(
        &self,
        aspects: api::TextureAspects,
        layers: u32,
    ) -> Option<[u32; 2]> {
        let view = &self.0.desc;
        let texture = &self.0.texture.0.desc;
        if !(layers == 1 && view.dimension == api::TextureViewDimension::D2
            || layers > 1 && view.dimension == api::TextureViewDimension::D2Array)
            || view.aspects != aspects
            || view.mips.end != view.mips.start.checked_add(1)?
            || view.layers.start != 0
            || view.layers.end < layers
            || !texture.usage.contains(api::TextureUsage::RENDER_ATTACHMENT)
        {
            return None;
        }
        Some([
            (texture.extent[0] >> view.mips.start).max(1),
            (texture.extent[1] >> view.mips.start).max(1),
        ])
    }
}
struct TextureViewState {
    texture: VulkanTexture,
    pub(super) view: vk::ImageView,
    desc: api::TextureViewDesc,
}
impl Drop for TextureViewState {
    fn drop(&mut self) {
        // SAFETY: TextureViewState owns this image view and retains its image
        // and DeviceState, which outlive the destroy call.
        unsafe {
            self.texture
                .0
                .device
                .device
                .destroy_image_view(self.view, None)
        };
    }
}

#[derive(Clone)]
pub struct VulkanSampler(Arc<SamplerState>);
impl VulkanSampler {
    pub(super) fn native(&self) -> vk::Sampler {
        self.0.sampler
    }
}
struct SamplerState {
    device: Arc<DeviceState>,
    pub(super) sampler: vk::Sampler,
}
impl Drop for SamplerState {
    fn drop(&mut self) {
        // SAFETY: SamplerState is the final owner and retains the live device.
        unsafe { self.device.device.destroy_sampler(self.sampler, None) }
    }
}

pub(super) fn buffer_support(
    device: &VulkanDevice,
    desc: &api::BufferDesc,
) -> api::Support<api::BufferSupport> {
    if desc.size == 0 || desc.usage.0 == 0 {
        return api::Support::Unsupported("zero-sized Vulkan buffers");
    }
    if desc.usage.0 & !SUPPORTED_BUFFER_USAGE != 0 {
        return api::Support::Unsupported("Vulkan buffer usage");
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
    let max_size = if desc.usage.contains(api::BufferUsage::UNIFORM) {
        u64::from(limits.max_uniform_buffer_range)
    } else {
        // Vulkan has no single max-buffer-size limit.  Storage range is the
        // tightest range this backend can currently expose, so report it as a
        // conservative limit rather than inventing a larger value.
        u64::from(limits.max_storage_buffer_range)
    };
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

pub(super) fn texture_support(
    device: &VulkanDevice,
    desc: &api::TextureDesc,
) -> api::Support<api::TextureSupport> {
    if desc.dimension != api::TextureDimension::D2
        || desc.extent[0] == 0
        || desc.extent[1] == 0
        || desc.extent[2] != 1
        || desc.array_layers == 0
        || desc.mip_levels == 0
        || desc.samples == 0
        || desc.usage.0 == 0
    {
        return api::Support::Unsupported("Vulkan 2D texture descriptor");
    }
    if desc.usage.0 & !SUPPORTED_TEXTURE_USAGE != 0 {
        return api::Support::Unsupported("Vulkan texture usage");
    }
    if desc
        .view_formats
        .iter()
        .any(|&format| format != desc.format)
    {
        return api::Support::Unsupported("Vulkan mutable texture view format");
    }
    let Some(format) = format(desc.format) else {
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
    if desc.extent[0] > limits.max_image_dimension2_d
        || desc.extent[1] > limits.max_image_dimension2_d
        || desc.array_layers > limits.max_image_array_layers
        || desc.mip_levels > max_mips(desc.extent[0].max(desc.extent[1]))
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
    let depth = is_depth(desc.format);
    if (desc.usage.contains(api::TextureUsage::TEXTURE_BINDING)
        && !features.contains(vk::FormatFeatureFlags::SAMPLED_IMAGE))
        || (desc.usage.contains(api::TextureUsage::RENDER_ATTACHMENT)
            && !(if depth {
                features.contains(vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT)
            } else {
                features.contains(vk::FormatFeatureFlags::COLOR_ATTACHMENT)
            }))
        || (desc.usage.contains(api::TextureUsage::STORAGE_BINDING)
            && !features.contains(vk::FormatFeatureFlags::STORAGE_IMAGE))
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
                vk::ImageType::TYPE_2D,
                vk::ImageTiling::OPTIMAL,
                usage,
                vk::ImageCreateFlags::empty(),
            )
    }) else {
        return api::Support::Unsupported("Vulkan texture format usage");
    };
    if desc.extent[0] > image_properties.max_extent.width
        || desc.extent[1] > image_properties.max_extent.height
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
        compatible_view_formats: vec![desc.format],
        filterable: features.contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR),
        blendable: features.contains(vk::FormatFeatureFlags::COLOR_ATTACHMENT_BLEND),
        storage: features.contains(vk::FormatFeatureFlags::STORAGE_IMAGE),
    })
}

pub(super) fn create_buffer(device: &VulkanDevice, desc: &api::BufferDesc) -> Result<VulkanBuffer> {
    if !matches!(buffer_support(device, desc), api::Support::Supported(_)) {
        return Err(Error::Unsupported("Vulkan buffer descriptor"));
    }
    let usage = buffer_usage(desc.usage)?;
    let info = vk::BufferCreateInfo::default()
        .size(desc.size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    // SAFETY: `info` is fully initialized from a supported descriptor and the
    // logical device remains alive through DeviceState.
    let buffer = unsafe { device.0.device.create_buffer(&info, None) }.map_err(vk_error)?;
    // SAFETY: `buffer` was just created on this device and is unpublished.
    let requirements = unsafe { device.0.device.get_buffer_memory_requirements(buffer) };
    let memory = allocate(device, requirements, memory_flags(desc.memory))
        .inspect_err(|_| unsafe { device.0.device.destroy_buffer(buffer, None) })?;
    // SAFETY: allocation requirements selected this memory type and size; both
    // handles are unpublished, so the error path can destroy them immediately.
    unsafe { device.0.device.bind_buffer_memory(buffer, memory, 0) }.map_err(|error| {
        unsafe {
            device.0.device.destroy_buffer(buffer, None);
            device.0.device.free_memory(memory, None);
        }
        vk_error(error)
    })?;
    Ok(VulkanBuffer(Arc::new(BufferState {
        device: device.0.clone(),
        buffer,
        memory,
        size: desc.size,
        usage: desc.usage,
    })))
}

pub(super) fn create_texture(
    device: &VulkanDevice,
    desc: &api::TextureDesc,
) -> Result<VulkanTexture> {
    if !matches!(texture_support(device, desc), api::Support::Supported(_)) {
        return Err(Error::Unsupported("Vulkan texture descriptor"));
    }
    let format = format(desc.format).ok_or(Error::Unsupported("Vulkan texture format"))?;
    let usage = image_usage(desc.usage, is_depth(desc.format))?;
    let info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(vk::Extent3D {
            width: desc.extent[0],
            height: desc.extent[1],
            depth: 1,
        })
        .mip_levels(desc.mip_levels)
        .array_layers(desc.array_layers)
        .samples(sample_count(desc.samples)?)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    // SAFETY: `info` was built from a supported descriptor on the live device.
    let image = unsafe { device.0.device.create_image(&info, None) }.map_err(vk_error)?;
    // SAFETY: `image` is newly created and has not escaped this function.
    let requirements = unsafe { device.0.device.get_image_memory_requirements(image) };
    let memory = allocate(device, requirements, vk::MemoryPropertyFlags::DEVICE_LOCAL)
        .inspect_err(|_| unsafe { device.0.device.destroy_image(image, None) })?;
    // SAFETY: requirements guided allocation and neither handle is published
    // until this bind succeeds; failure cleanup therefore cannot race GPU use.
    unsafe { device.0.device.bind_image_memory(image, memory, 0) }.map_err(|error| {
        unsafe {
            device.0.device.destroy_image(image, None);
            device.0.device.free_memory(memory, None);
        }
        vk_error(error)
    })?;
    Ok(VulkanTexture(Arc::new(TextureState {
        device: device.0.clone(),
        image,
        memory,
        desc: desc.clone(),
    })))
}

pub(super) fn create_view(
    device: &VulkanDevice,
    texture: &VulkanTexture,
    desc: &api::TextureViewDesc,
) -> Result<VulkanTextureView> {
    if desc.mips.start >= desc.mips.end
        || desc.layers.start >= desc.layers.end
        || desc.mips.end > texture.0.desc.mip_levels
        || desc.layers.end > texture.0.desc.array_layers
    {
        return Err(Error::InvalidRange);
    }
    if desc.format != texture.0.desc.format {
        return Err(Error::Unsupported("Vulkan mutable texture view format"));
    }
    let format = format(desc.format).ok_or(Error::Unsupported("Vulkan texture view format"))?;
    let view_type = match desc.dimension {
        api::TextureViewDimension::D2 if desc.layers.end - desc.layers.start == 1 => {
            vk::ImageViewType::TYPE_2D
        }
        api::TextureViewDimension::D2Array => vk::ImageViewType::TYPE_2D_ARRAY,
        _ => return Err(Error::Unsupported("Vulkan texture view dimension")),
    };
    if is_depth(desc.format) != (desc.aspects == api::TextureAspects::DEPTH) {
        return Err(Error::Unsupported("Vulkan texture view aspect"));
    }
    let aspect = if desc.aspects == api::TextureAspects::COLOR {
        vk::ImageAspectFlags::COLOR
    } else if desc.aspects == api::TextureAspects::DEPTH {
        vk::ImageAspectFlags::DEPTH
    } else {
        return Err(Error::Unsupported("Vulkan texture aspect"));
    };
    let range = vk::ImageSubresourceRange {
        aspect_mask: aspect,
        base_mip_level: desc.mips.start,
        level_count: desc.mips.end - desc.mips.start,
        base_array_layer: desc.layers.start,
        layer_count: desc.layers.end - desc.layers.start,
    };
    let info = vk::ImageViewCreateInfo::default()
        .image(texture.0.image)
        .view_type(view_type)
        .format(format)
        .subresource_range(range);
    // SAFETY: checked ranges, format, and aspect describe the retained image;
    // the view will retain that image through TextureViewState.
    let view = unsafe { device.0.device.create_image_view(&info, None) }.map_err(vk_error)?;
    Ok(VulkanTextureView(Arc::new(TextureViewState {
        texture: texture.clone(),
        view,
        desc: desc.clone(),
    })))
}

pub(super) fn create_sampler(
    device: &VulkanDevice,
    desc: &api::SamplerDesc,
) -> Result<VulkanSampler> {
    if desc.anisotropy != 1 || desc.compare.is_some() || desc.border.is_some() {
        return Err(Error::Unsupported(
            "Vulkan anisotropic, comparison, or border sampler",
        ));
    }
    if desc
        .address
        .iter()
        .any(|mode| matches!(mode, api::AddressMode::ClampToBorder))
    {
        return Err(Error::Unsupported("Vulkan clamp-to-border sampler"));
    }
    let info = vk::SamplerCreateInfo::default()
        .mag_filter(filter(desc.mag_filter))
        .min_filter(filter(desc.min_filter))
        .mipmap_mode(if desc.mip_filter == api::FilterMode::Linear {
            vk::SamplerMipmapMode::LINEAR
        } else {
            vk::SamplerMipmapMode::NEAREST
        })
        .address_mode_u(address(desc.address[0])?)
        .address_mode_v(address(desc.address[1])?)
        .address_mode_w(address(desc.address[2])?)
        .min_lod(desc.lod.start)
        .max_lod(desc.lod.end)
        .max_anisotropy(1.0)
        .anisotropy_enable(false);
    // SAFETY: all fields come from the supported sampler descriptor and the
    // resulting handle is owned by the returned SamplerState.
    let sampler = unsafe { device.0.device.create_sampler(&info, None) }.map_err(vk_error)?;
    Ok(VulkanSampler(Arc::new(SamplerState {
        device: device.0.clone(),
        sampler,
    })))
}
