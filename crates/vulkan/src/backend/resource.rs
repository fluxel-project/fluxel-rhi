//! Vulkan resource allocation and view creation.
//!
//! This module owns Vulkan allocation handles and tears them down before their
//! device.  It deliberately does not track resource state: Core transitions
//! carry the states selected by RenderGraph.

use std::sync::{Arc, Mutex};

use ash::vk;
use fluxel_rhi_core::api::{self, Error, Result};

use super::{DeviceState, VulkanDevice, vk_error};

#[derive(Clone)]
pub struct VulkanBuffer(pub(super) Arc<BufferState>);
pub(super) struct BufferState {
    device: Arc<DeviceState>,
    pub(super) buffer: vk::Buffer,
    pub(super) memory: vk::DeviceMemory,
    pub(super) size: u64,
    pub(super) usage: api::BufferUsage,
}
impl Drop for BufferState {
    fn drop(&mut self) {
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
    /// Returns whether this is the first native use of one subresource.
    /// This is deliberately limited to Vulkan's `UNDEFINED` initial layout;
    /// all later layouts remain Core's explicit-transition responsibility.
    pub(super) fn take_initial_layout(&self, mip: u32, layer: u32) -> Result<bool> {
        let index = usize::try_from(mip)
            .ok()
            .and_then(|mip| mip.checked_mul(self.0.desc.array_layers as usize))
            .and_then(|base| base.checked_add(layer as usize))
            .ok_or(Error::InvalidRange)?;
        let mut initialized = self.0.initialized.lock().map_err(|_| Error::DeviceLost)?;
        let item = initialized.get_mut(index).ok_or(Error::InvalidRange)?;
        let was_initial = !*item;
        *item = true;
        Ok(was_initial)
    }
    pub(super) fn mark_initialized(
        &self,
        mips: std::ops::Range<u32>,
        layers: std::ops::Range<u32>,
    ) -> Result<()> {
        let mut initialized = self.0.initialized.lock().map_err(|_| Error::DeviceLost)?;
        for mip in mips {
            for layer in layers.clone() {
                let index = mip as usize * self.0.desc.array_layers as usize + layer as usize;
                *initialized.get_mut(index).ok_or(Error::InvalidRange)? = true;
            }
        }
        Ok(())
    }
}
struct TextureState {
    device: Arc<DeviceState>,
    pub(super) image: vk::Image,
    memory: vk::DeviceMemory,
    pub(super) desc: api::TextureDesc,
    initialized: Mutex<Vec<bool>>,
}
impl Drop for TextureState {
    fn drop(&mut self) {
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
    pub(super) fn attachment_extent(&self, aspects: api::TextureAspects) -> Option<[u32; 2]> {
        let view = &self.0.desc;
        let texture = &self.0.texture.0.desc;
        if view.dimension != api::TextureViewDimension::D2
            || view.aspects != aspects
            || view.mips.end != view.mips.start.checked_add(1)?
            || view.layers.end != view.layers.start.checked_add(1)?
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
    let buffer = unsafe { device.0.device.create_buffer(&info, None) }.map_err(vk_error)?;
    let requirements = unsafe { device.0.device.get_buffer_memory_requirements(buffer) };
    let memory = allocate(device, requirements, memory_flags(desc.memory))
        .inspect_err(|_| unsafe { device.0.device.destroy_buffer(buffer, None) })?;
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
    let image = unsafe { device.0.device.create_image(&info, None) }.map_err(vk_error)?;
    let requirements = unsafe { device.0.device.get_image_memory_requirements(image) };
    let memory = allocate(device, requirements, vk::MemoryPropertyFlags::DEVICE_LOCAL)
        .inspect_err(|_| unsafe { device.0.device.destroy_image(image, None) })?;
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
        initialized: Mutex::new(vec![
            false;
            desc.mip_levels as usize * desc.array_layers as usize
        ]),
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
    let sampler = unsafe { device.0.device.create_sampler(&info, None) }.map_err(vk_error)?;
    Ok(VulkanSampler(Arc::new(SamplerState {
        device: device.0.clone(),
        sampler,
    })))
}

fn allocate(
    device: &VulkanDevice,
    requirements: vk::MemoryRequirements,
    flags: vk::MemoryPropertyFlags,
) -> Result<vk::DeviceMemory> {
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
    unsafe { device.0.device.allocate_memory(&info, None) }.map_err(vk_error)
}

fn memory_flags(memory: api::MemoryClass) -> vk::MemoryPropertyFlags {
    match memory {
        api::MemoryClass::Device => vk::MemoryPropertyFlags::DEVICE_LOCAL,
        api::MemoryClass::Upload => {
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
        }
        api::MemoryClass::Readback => {
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_CACHED
        }
    }
}
fn buffer_usage(usage: api::BufferUsage) -> Result<vk::BufferUsageFlags> {
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
    if out.is_empty() {
        return Err(Error::Unsupported("Vulkan buffer usage"));
    }
    Ok(out)
}
fn image_usage(usage: api::TextureUsage, depth: bool) -> Result<vk::ImageUsageFlags> {
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
fn sample_count(count: u32) -> Result<vk::SampleCountFlags> {
    match count {
        1 => Ok(vk::SampleCountFlags::TYPE_1),
        2 => Ok(vk::SampleCountFlags::TYPE_2),
        4 => Ok(vk::SampleCountFlags::TYPE_4),
        8 => Ok(vk::SampleCountFlags::TYPE_8),
        _ => Err(Error::Unsupported("Vulkan sample count")),
    }
}
fn sample_counts(flags: vk::SampleCountFlags) -> Vec<u32> {
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
fn max_mips(max_dimension: u32) -> u32 {
    max_dimension.ilog2() + 1
}
fn filter(filter: api::FilterMode) -> vk::Filter {
    if filter == api::FilterMode::Linear {
        vk::Filter::LINEAR
    } else {
        vk::Filter::NEAREST
    }
}
fn address(mode: api::AddressMode) -> Result<vk::SamplerAddressMode> {
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
pub(super) fn format(format: api::TextureFormat) -> Option<vk::Format> {
    Some(match format {
        api::TextureFormat::Rgba8Unorm => vk::Format::R8G8B8A8_UNORM,
        api::TextureFormat::Rgba8UnormSrgb => vk::Format::R8G8B8A8_SRGB,
        api::TextureFormat::Bgra8Unorm => vk::Format::B8G8R8A8_UNORM,
        api::TextureFormat::Bgra8UnormSrgb => vk::Format::B8G8R8A8_SRGB,
        api::TextureFormat::Depth32Float => vk::Format::D32_SFLOAT,
        _ => return None,
    })
}
fn is_depth(format: api::TextureFormat) -> bool {
    matches!(format, api::TextureFormat::Depth32Float)
}

const SUPPORTED_BUFFER_USAGE: u64 = api::BufferUsage::COPY_SRC.0
    | api::BufferUsage::COPY_DST.0
    | api::BufferUsage::VERTEX.0
    | api::BufferUsage::INDEX.0
    | api::BufferUsage::UNIFORM.0
    | api::BufferUsage::STORAGE.0;
const SUPPORTED_TEXTURE_USAGE: u64 = api::TextureUsage::COPY_SRC.0
    | api::TextureUsage::COPY_DST.0
    | api::TextureUsage::TEXTURE_BINDING.0
    | api::TextureUsage::STORAGE_BINDING.0
    | api::TextureUsage::RENDER_ATTACHMENT.0;
