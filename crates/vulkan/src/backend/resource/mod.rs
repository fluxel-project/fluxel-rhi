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

pub(super) mod allocation;
mod support;
pub(super) use allocation::allocate_with_address;
pub(super) use allocation::format;
use allocation::*;
pub(super) use support::{buffer_support, texture_support};

pub(super) fn sparse_image_usage(
    usage: api::TextureUsage,
    format: api::TextureFormat,
) -> Result<vk::ImageUsageFlags> {
    image_usage(usage, is_depth(format))
}

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
    pub(super) heap: Option<super::memory::VulkanHeap>,
    pub(super) default_allocation: Option<super::memory::DefaultAllocationLease>,
    pub(super) memory_offset: u64,
    pub(super) map_state: Arc<super::mapping::MappingState>,
}
impl VulkanBuffer {
    pub(super) fn device_address(&self) -> Result<vk::DeviceAddress> {
        let address_usage = api::BufferUsage::ACCEL_BUILD_INPUT.0
            | api::BufferUsage::ACCEL_SCRATCH.0
            | api::BufferUsage::SHADER_TABLE.0;
        if self.0.usage.0 & address_usage == 0 {
            return Err(Error::Unsupported("Vulkan buffer has no device address"));
        }
        // SAFETY: address-capable usage and allocation were established when
        // this live device-owned buffer was created.
        let address = unsafe {
            self.0.device.device.get_buffer_device_address(
                &vk::BufferDeviceAddressInfo::default().buffer(self.0.buffer),
            )
        };
        if address == 0 {
            return Err(Error::InvalidState);
        }
        Ok(address)
    }
}
impl Drop for BufferState {
    fn drop(&mut self) {
        // SAFETY: this Arc is the final owner of the buffer and its allocation;
        // all submitted encoders retain VulkanBuffer until their fence retires.
        unsafe {
            self.device.device.destroy_buffer(self.buffer, None);
            if self.heap.is_none() && self.default_allocation.is_none() {
                self.device.device.free_memory(self.memory, None);
            }
        }
    }
}

#[derive(Clone)]
pub struct VulkanTexture(pub(super) Arc<TextureState>);
impl VulkanTexture {
    pub(super) fn native(&self) -> vk::Image {
        self.0.image
    }
    pub(super) fn desc(&self) -> &api::TextureDesc {
        &self.0.desc
    }
}
pub(super) struct TextureState {
    pub(super) device: Arc<DeviceState>,
    pub(super) image: vk::Image,
    pub(super) memory: Option<vk::DeviceMemory>,
    pub(super) heap: Option<super::memory::VulkanHeap>,
    pub(super) default_allocation: Option<super::memory::DefaultAllocationLease>,
    pub(super) sparse: Option<super::sparse::SparseTextureState>,
    pub(super) desc: api::TextureDesc,
}
impl Drop for TextureState {
    fn drop(&mut self) {
        // SAFETY: this Arc is the final image owner; TextureViewState retains
        // VulkanTexture, so every view is destroyed before this allocation.
        unsafe {
            self.device.device.destroy_image(self.image, None);
            if let Some(memory) = self
                .memory
                .filter(|_| self.heap.is_none() && self.default_allocation.is_none())
            {
                self.device.device.free_memory(memory, None);
            }
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
    pub(super) fn aspects(&self) -> api::TextureAspects {
        self.0.desc.aspects
    }
    pub(super) fn mip_range(&self) -> std::ops::Range<u32> {
        self.0.desc.mips.clone()
    }
    pub(super) fn layer_range(&self) -> std::ops::Range<u32> {
        self.0.desc.layers.clone()
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
        if !(layers == 1
            && (view.dimension == api::TextureViewDimension::D2
                || view.dimension == api::TextureViewDimension::D2Array)
            || layers > 1 && view.dimension == api::TextureViewDimension::D2Array)
            || view.aspects.0 & aspects.0 != aspects.0
            || view.mips.end != view.mips.start.checked_add(1)?
            || view.layers.end.checked_sub(view.layers.start)? < layers
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
    pub(super) fn integer_native(&self) -> vk::Sampler {
        self.0.integer_sampler
    }
}
struct SamplerState {
    device: Arc<DeviceState>,
    pub(super) sampler: vk::Sampler,
    integer_sampler: vk::Sampler,
}
impl Drop for SamplerState {
    fn drop(&mut self) {
        // SAFETY: SamplerState is the final owner and retains the live device.
        unsafe {
            self.device.device.destroy_sampler(self.sampler, None);
            self.device
                .device
                .destroy_sampler(self.integer_sampler, None);
        }
    }
}

pub(super) fn unbound_buffer(device: &VulkanDevice, desc: &api::BufferDesc) -> Result<vk::Buffer> {
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
    unsafe { device.0.device.create_buffer(&info, None) }.map_err(vk_error)
}

pub(super) fn create_buffer(device: &VulkanDevice, desc: &api::BufferDesc) -> Result<VulkanBuffer> {
    let buffer = unbound_buffer(device, desc)?;
    // SAFETY: `buffer` was just created on this device and is unpublished.
    let mut dedicated = vk::MemoryDedicatedRequirements::default();
    let mut requirements2 = vk::MemoryRequirements2::default().push_next(&mut dedicated);
    // SAFETY: the unpublished buffer and output chain belong to this device.
    unsafe {
        device.0.device.get_buffer_memory_requirements2(
            &vk::BufferMemoryRequirementsInfo2::default().buffer(buffer),
            &mut requirements2,
        )
    };
    let requirements = requirements2.memory_requirements;
    let address = buffer_usage(desc.usage)?.contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS);
    let dedicated_buffer = (dedicated.requires_dedicated_allocation == vk::TRUE
        || dedicated.prefers_dedicated_allocation == vk::TRUE)
        .then_some((buffer, vk::Image::null()));
    let mut memory_properties = memory_flags(desc.memory);
    if desc.usage.0 & (api::BufferUsage::MAP_READ.0 | api::BufferUsage::MAP_WRITE.0) != 0 {
        memory_properties |= vk::MemoryPropertyFlags::HOST_VISIBLE;
    }
    let pooled = if dedicated_buffer.is_none() {
        match device.0.default_resource_pool.allocate(
            requirements,
            memory_properties,
            address,
            u64::from(device.0.ray_support.scratch_alignment)
                .max(u64::from(device.0.ray_support.shader_group_base_alignment)),
            super::memory::ResourceClass::Buffer,
            desc.usage.0 & (api::BufferUsage::MAP_READ.0 | api::BufferUsage::MAP_WRITE.0) != 0,
        ) {
            Ok(pooled) => pooled,
            Err(error) => {
                unsafe { device.0.device.destroy_buffer(buffer, None) };
                return Err(error);
            }
        }
    } else {
        None
    };
    let (memory, offset, mapping, lease) = if let Some((allocation, mapping)) = pooled {
        // SAFETY: the page type and aligned free range satisfy the queried
        // requirements; unpublished failure cleanup drops only this lease.
        if let Err(error) = unsafe {
            device
                .0
                .device
                .bind_buffer_memory(buffer, allocation.memory, allocation.offset)
        } {
            unsafe { device.0.device.destroy_buffer(buffer, None) };
            return Err(vk_error(error));
        }
        (
            allocation.memory,
            allocation.offset,
            mapping,
            Some(allocation.lease),
        )
    } else {
        let (memory, flags) = match allocate_resource(
            device,
            requirements,
            memory_properties,
            address,
            dedicated_buffer,
        ) {
            Ok(allocated) => allocated,
            Err(error) => {
                unsafe { device.0.device.destroy_buffer(buffer, None) };
                return Err(error);
            }
        };
        // SAFETY: the committed allocation satisfies the queried requirements.
        if let Err(error) = unsafe { device.0.device.bind_buffer_memory(buffer, memory, 0) } {
            unsafe {
                device.0.device.destroy_buffer(buffer, None);
                device.0.device.free_memory(memory, None);
            }
            return Err(vk_error(error));
        }
        (
            memory,
            0,
            Some(super::mapping::MappingState::new(requirements.size, flags)),
            None,
        )
    };
    Ok(VulkanBuffer(Arc::new(BufferState {
        device: device.0.clone(),
        buffer,
        memory,
        size: desc.size,
        usage: desc.usage,
        heap: None,
        default_allocation: lease,
        memory_offset: offset,
        map_state: mapping.expect("buffer mapping state is initialized on every path"),
    })))
}

pub(super) fn unbound_texture(
    device: &VulkanDevice,
    desc: &api::TextureDesc,
    alias: bool,
) -> Result<vk::Image> {
    if !matches!(texture_support(device, desc), api::Support::Supported(_)) {
        return Err(Error::Unsupported("Vulkan texture descriptor"));
    }
    let format =
        native_format(&device.0, desc.format).ok_or(Error::Unsupported("Vulkan texture format"))?;
    let usage = image_usage(desc.usage, is_depth(desc.format))?;
    let info = vk::ImageCreateInfo::default()
        .flags(
            image_flags(desc)
                | if alias {
                    vk::ImageCreateFlags::ALIAS
                } else {
                    vk::ImageCreateFlags::empty()
                },
        )
        .image_type(image_type(desc.dimension))
        .format(format)
        .extent(vk::Extent3D {
            width: desc.extent[0],
            height: desc.extent[1],
            depth: desc.extent[2],
        })
        .mip_levels(desc.mip_levels)
        .array_layers(desc.array_layers)
        .samples(sample_count(desc.samples)?)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    // SAFETY: `info` was built from a supported descriptor on the live device.
    unsafe { device.0.device.create_image(&info, None) }.map_err(vk_error)
}

pub(super) fn create_texture(
    device: &VulkanDevice,
    desc: &api::TextureDesc,
) -> Result<VulkanTexture> {
    let image = unbound_texture(device, desc, false)?;
    // SAFETY: `image` is newly created and has not escaped this function.
    let mut dedicated = vk::MemoryDedicatedRequirements::default();
    let mut requirements2 = vk::MemoryRequirements2::default().push_next(&mut dedicated);
    // SAFETY: the unpublished image and output chain belong to this device.
    unsafe {
        device.0.device.get_image_memory_requirements2(
            &vk::ImageMemoryRequirementsInfo2::default().image(image),
            &mut requirements2,
        )
    };
    let requirements = requirements2.memory_requirements;
    let dedicated_image = (dedicated.requires_dedicated_allocation == vk::TRUE
        || dedicated.prefers_dedicated_allocation == vk::TRUE)
        .then_some((vk::Buffer::null(), image));
    let pooled = if dedicated_image.is_none() {
        match device.0.default_resource_pool.allocate(
            requirements,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            false,
            1,
            super::memory::ResourceClass::OptimalImage,
            false,
        ) {
            Ok(pooled) => pooled,
            Err(error) => {
                unsafe { device.0.device.destroy_image(image, None) };
                return Err(error);
            }
        }
    } else {
        None
    };
    let (memory, lease) = if let Some((allocation, _mapping)) = pooled {
        // SAFETY: optimal images use a disjoint pool class, and this page range
        // satisfies the native memory type, size, and alignment requirements.
        if let Err(error) = unsafe {
            device
                .0
                .device
                .bind_image_memory(image, allocation.memory, allocation.offset)
        } {
            unsafe { device.0.device.destroy_image(image, None) };
            return Err(vk_error(error));
        }
        (None, Some(allocation.lease))
    } else {
        let (memory, _) = match allocate_resource(
            device,
            requirements,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            false,
            dedicated_image,
        ) {
            Ok(allocated) => allocated,
            Err(error) => {
                unsafe { device.0.device.destroy_image(image, None) };
                return Err(error);
            }
        };
        // SAFETY: requirements guided allocation and neither handle is published
        // until this bind succeeds; failure cleanup cannot race GPU use.
        if let Err(error) = unsafe { device.0.device.bind_image_memory(image, memory, 0) } {
            unsafe {
                device.0.device.destroy_image(image, None);
                device.0.device.free_memory(memory, None);
            }
            return Err(vk_error(error));
        }
        (Some(memory), None)
    };
    Ok(VulkanTexture(Arc::new(TextureState {
        device: device.0.clone(),
        image,
        memory,
        heap: None,
        default_allocation: lease,
        sparse: None,
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
    if desc.format != texture.0.desc.format && !texture.0.desc.view_formats.contains(&desc.format) {
        return Err(Error::Unsupported("Vulkan mutable texture view format"));
    }
    let format = native_format(&device.0, desc.format)
        .ok_or(Error::Unsupported("Vulkan texture view format"))?;
    let count = desc.layers.end - desc.layers.start;
    let view_type = match (texture.0.desc.dimension, desc.dimension) {
        (api::TextureDimension::D1, api::TextureViewDimension::D1) if count == 1 => {
            vk::ImageViewType::TYPE_1D
        }
        (api::TextureDimension::D2, api::TextureViewDimension::D2) if count == 1 => {
            vk::ImageViewType::TYPE_2D
        }
        (api::TextureDimension::D2, api::TextureViewDimension::D2Array) => {
            vk::ImageViewType::TYPE_2D_ARRAY
        }
        (api::TextureDimension::D2, api::TextureViewDimension::Cube)
            if count == 6
                && image_flags(&texture.0.desc).contains(vk::ImageCreateFlags::CUBE_COMPATIBLE) =>
        {
            vk::ImageViewType::CUBE
        }
        (api::TextureDimension::D2, api::TextureViewDimension::CubeArray)
            if count % 6 == 0
                && device.0.features.image_cube_array == vk::TRUE
                && image_flags(&texture.0.desc).contains(vk::ImageCreateFlags::CUBE_COMPATIBLE) =>
        {
            vk::ImageViewType::CUBE_ARRAY
        }
        (api::TextureDimension::D3, api::TextureViewDimension::D3) if count == 1 => {
            vk::ImageViewType::TYPE_3D
        }
        _ => return Err(Error::Unsupported("Vulkan texture view dimension")),
    };
    let allowed = match desc.format {
        api::TextureFormat::Depth16Unorm
        | api::TextureFormat::Depth24Plus
        | api::TextureFormat::Depth32Float => api::TextureAspects::DEPTH.0,
        api::TextureFormat::Depth24PlusStencil8 | api::TextureFormat::Depth32FloatStencil8 => {
            api::TextureAspects::DEPTH.0 | api::TextureAspects::STENCIL.0
        }
        api::TextureFormat::Stencil8 => api::TextureAspects::STENCIL.0,
        _ => api::TextureAspects::COLOR.0,
    };
    if desc.aspects.0 == 0 || desc.aspects.0 & !allowed != 0 {
        return Err(Error::InvalidRange);
    }
    let mut aspect = vk::ImageAspectFlags::empty();
    if desc.aspects.0 & api::TextureAspects::COLOR.0 != 0 {
        aspect |= vk::ImageAspectFlags::COLOR;
    }
    if desc.aspects.0 & api::TextureAspects::DEPTH.0 != 0 {
        aspect |= vk::ImageAspectFlags::DEPTH;
    }
    if desc.aspects.0 & api::TextureAspects::STENCIL.0 != 0 {
        aspect |= vk::ImageAspectFlags::STENCIL;
    }
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

/// Create an internal 2D attachment view of one 3D image slice.
pub(super) fn create_slice_view(
    device: &VulkanDevice,
    texture: &VulkanTexture,
    mip: u32,
    slice: u32,
    aspects: api::TextureAspects,
) -> Result<VulkanTextureView> {
    let desc = &texture.0.desc;
    if desc.dimension != api::TextureDimension::D3
        || mip >= desc.mip_levels
        || slice >= (desc.extent[2] >> mip).max(1)
        || !desc.usage.contains(api::TextureUsage::RENDER_ATTACHMENT)
    {
        return Err(Error::InvalidRange);
    }
    let mut aspect = vk::ImageAspectFlags::empty();
    if aspects.0 & api::TextureAspects::COLOR.0 != 0 {
        aspect |= vk::ImageAspectFlags::COLOR;
    }
    if aspects.0 & api::TextureAspects::DEPTH.0 != 0 {
        aspect |= vk::ImageAspectFlags::DEPTH;
    }
    if aspects.0 & api::TextureAspects::STENCIL.0 != 0 {
        aspect |= vk::ImageAspectFlags::STENCIL;
    }
    if aspect.is_empty() {
        return Err(Error::InvalidRange);
    }
    let range = vk::ImageSubresourceRange {
        aspect_mask: aspect,
        base_mip_level: mip,
        level_count: 1,
        base_array_layer: slice,
        layer_count: 1,
    };
    let info = vk::ImageViewCreateInfo::default()
        .image(texture.0.image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(
            native_format(&device.0, desc.format)
                .ok_or(Error::Unsupported("Vulkan texture format"))?,
        )
        .subresource_range(range);
    // SAFETY: the image has 2D-array compatibility and the selected mip and
    // depth slice are in bounds; the returned view retains the image.
    let view = unsafe { device.0.device.create_image_view(&info, None) }.map_err(vk_error)?;
    Ok(VulkanTextureView(Arc::new(TextureViewState {
        texture: texture.clone(),
        view,
        desc: api::TextureViewDesc {
            format: desc.format,
            dimension: api::TextureViewDimension::D2,
            aspects,
            mips: mip..mip + 1,
            layers: slice..slice + 1,
        },
    })))
}

pub(super) fn create_sampler(
    device: &VulkanDevice,
    desc: &api::SamplerDesc,
) -> Result<VulkanSampler> {
    if desc.anisotropy == 0 {
        return Err(Error::InvalidRange);
    }
    if desc.anisotropy > 1 {
        if !device.0.sampler_anisotropy
            || f32::from(desc.anisotropy) > device.0.max_sampler_anisotropy
        {
            return Err(Error::Unsupported(
                "Vulkan sampler anisotropy exceeds device support",
            ));
        }
    }
    let clamp_to_border = desc
        .address
        .iter()
        .any(|mode| matches!(mode, api::AddressMode::ClampToBorder));
    let clamp_to_zero = desc.address.contains(&api::AddressMode::ClampToZero);
    if clamp_to_zero
        && clamp_to_border
        && !matches!(
            desc.border,
            Some(api::BorderColor::TransparentBlack | api::BorderColor::ZeroInteger)
        )
    {
        return Err(Error::Unsupported(
            "Vulkan sampler has incompatible border colors",
        ));
    }
    let border = match (clamp_to_border, desc.border) {
        (true, Some(api::BorderColor::TransparentBlack)) => {
            vk::BorderColor::FLOAT_TRANSPARENT_BLACK
        }
        (true, Some(api::BorderColor::OpaqueBlack)) => vk::BorderColor::FLOAT_OPAQUE_BLACK,
        (true, Some(api::BorderColor::OpaqueWhite)) => vk::BorderColor::FLOAT_OPAQUE_WHITE,
        (true, Some(api::BorderColor::ZeroInteger)) => vk::BorderColor::FLOAT_TRANSPARENT_BLACK,
        (true, None) => return Err(Error::InvalidState),
        (false, Some(_)) => return Err(Error::InvalidState),
        (false, None) => vk::BorderColor::FLOAT_TRANSPARENT_BLACK,
    };
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
        .max_anisotropy(f32::from(desc.anisotropy))
        .anisotropy_enable(desc.anisotropy > 1)
        .compare_enable(desc.compare.is_some())
        .compare_op(match desc.compare.unwrap_or(api::CompareFunction::Always) {
            api::CompareFunction::Never => vk::CompareOp::NEVER,
            api::CompareFunction::Less => vk::CompareOp::LESS,
            api::CompareFunction::Equal => vk::CompareOp::EQUAL,
            api::CompareFunction::LessEqual => vk::CompareOp::LESS_OR_EQUAL,
            api::CompareFunction::Greater => vk::CompareOp::GREATER,
            api::CompareFunction::NotEqual => vk::CompareOp::NOT_EQUAL,
            api::CompareFunction::GreaterEqual => vk::CompareOp::GREATER_OR_EQUAL,
            api::CompareFunction::Always => vk::CompareOp::ALWAYS,
        })
        .border_color(border);
    // SAFETY: all fields come from the supported sampler descriptor and the
    // resulting handle is owned by the returned SamplerState.
    let sampler = unsafe { device.0.device.create_sampler(&info, None) }.map_err(vk_error)?;
    let int_border = match border {
        vk::BorderColor::FLOAT_OPAQUE_BLACK => vk::BorderColor::INT_OPAQUE_BLACK,
        vk::BorderColor::FLOAT_OPAQUE_WHITE => vk::BorderColor::INT_OPAQUE_WHITE,
        _ => vk::BorderColor::INT_TRANSPARENT_BLACK,
    };
    // SAFETY: this is the same valid descriptor with the matching integer border color.
    let integer_sampler = unsafe {
        device
            .0
            .device
            .create_sampler(&info.border_color(int_border), None)
    }
    .map_err(|error| {
        // SAFETY: sampler has not escaped and has no references.
        unsafe { device.0.device.destroy_sampler(sampler, None) };
        vk_error(error)
    })?;
    Ok(VulkanSampler(Arc::new(SamplerState {
        device: device.0.clone(),
        sampler,
        integer_sampler,
    })))
}

pub(super) fn image_type(dimension: api::TextureDimension) -> vk::ImageType {
    match dimension {
        api::TextureDimension::D1 => vk::ImageType::TYPE_1D,
        api::TextureDimension::D2 => vk::ImageType::TYPE_2D,
        api::TextureDimension::D3 => vk::ImageType::TYPE_3D,
    }
}

pub(super) fn image_flags(desc: &api::TextureDesc) -> vk::ImageCreateFlags {
    let mut flags = vk::ImageCreateFlags::empty();
    if desc.dimension == api::TextureDimension::D2
        && desc.extent[0] == desc.extent[1]
        && desc.array_layers >= 6
    {
        flags |= vk::ImageCreateFlags::CUBE_COMPATIBLE;
    }
    if desc.dimension == api::TextureDimension::D3
        && desc.usage.contains(api::TextureUsage::RENDER_ATTACHMENT)
    {
        flags |= vk::ImageCreateFlags::TYPE_2D_ARRAY_COMPATIBLE;
    }
    if desc.view_formats.iter().any(|&f| f != desc.format) {
        flags |= vk::ImageCreateFlags::MUTABLE_FORMAT | vk::ImageCreateFlags::EXTENDED_USAGE;
    }
    flags
}

/// Select one device-wide Vulkan representation for abstract Plus formats so
/// texture views and pipeline attachment formats always agree.
pub(super) fn native_format(device: &DeviceState, value: api::TextureFormat) -> Option<vk::Format> {
    let (preferred, fallback) = match value {
        api::TextureFormat::Depth24Plus => {
            (vk::Format::X8_D24_UNORM_PACK32, vk::Format::D32_SFLOAT)
        }
        api::TextureFormat::Depth24PlusStencil8 => (
            vk::Format::D24_UNORM_S8_UINT,
            vk::Format::D32_SFLOAT_S8_UINT,
        ),
        _ => return format(value),
    };
    // SAFETY: the physical device is retained by DeviceState.
    let properties = unsafe {
        device
            .instance
            .instance
            .get_physical_device_format_properties(device.physical, preferred)
    };
    if properties
        .optimal_tiling_features
        .contains(vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT)
    {
        Some(preferred)
    } else {
        Some(fallback)
    }
}
