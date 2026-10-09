use super::memory::memory_type;
use crate::api::resource::{
    backend::TextureBackend,
    subresource::TextureAspects,
    texture::{TextureDescriptor, TextureDimension, TextureUsage, TextureViewCompatibility},
};
use crate::backend::vulkan::format::vk_format;
use crate::backend::vulkan::platform::device::VulkanShared;
use ash::vk;
use std::{any::Any, sync::Arc};

pub(crate) struct VulkanTexture {
    shared: Arc<VulkanShared>,
    image: vk::Image,
    memory: vk::DeviceMemory,
    format: vk::Format,
}
impl VulkanTexture {
    pub(crate) fn image(&self) -> vk::Image {
        self.image
    }
    pub(crate) fn format(&self) -> vk::Format {
        self.format
    }
}
impl TextureBackend for VulkanTexture {
    fn as_any(&self) -> &dyn Any {
        self
    }
}
impl Drop for VulkanTexture {
    fn drop(&mut self) {
        unsafe {
            self.shared.device.destroy_image(self.image, None);
            self.shared.device.free_memory(self.memory, None);
        }
    }
}

pub(crate) fn create_texture(
    shared: Arc<VulkanShared>,
    desc: &TextureDescriptor,
) -> Result<VulkanTexture, vk::Result> {
    let Some(format) = vk_format(desc.format) else {
        return Err(vk::Result::ERROR_FORMAT_NOT_SUPPORTED);
    };
    let mut flags = vk::ImageCreateFlags::empty();
    if desc
        .view_compatibility
        .contains(TextureViewCompatibility::CUBE)
    {
        flags |= vk::ImageCreateFlags::CUBE_COMPATIBLE;
    }
    // Vulkan requires MUTABLE_FORMAT at image creation before any view may use
    // a format other than the image's own. Portable capability validation has
    // already proved every declared pair view-compatible; this flag grants the
    // native permission and does not broaden that public set.
    if !desc.view_formats.is_empty() {
        flags |= vk::ImageCreateFlags::MUTABLE_FORMAT;
    }
    let info = vk::ImageCreateInfo::default()
        .flags(flags)
        .image_type(match desc.dimension {
            TextureDimension::D1 => vk::ImageType::TYPE_1D,
            TextureDimension::D2 => vk::ImageType::TYPE_2D,
            TextureDimension::D3 => vk::ImageType::TYPE_3D,
            _ => crate::unknown_portable_variant(),
        })
        .format(format)
        .extent(vk::Extent3D {
            width: desc.extent.width,
            height: desc.extent.height,
            depth: desc.extent.depth,
        })
        .mip_levels(desc.mip_levels)
        .array_layers(desc.array_layers)
        .samples(vk::SampleCountFlags::from_raw(desc.sample_count))
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(usage(desc.usage))
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    let image = unsafe { shared.device.create_image(&info, None) }?;
    let requirements = unsafe { shared.device.get_image_memory_requirements(image) };
    let Some(memory_type_index) = memory_type(&shared, requirements.memory_type_bits, desc.memory)
    else {
        unsafe { shared.device.destroy_image(image, None) };
        return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
    };
    let allocation = vk::MemoryAllocateInfo::default()
        .allocation_size(requirements.size)
        .memory_type_index(memory_type_index);
    let memory = match unsafe { shared.device.allocate_memory(&allocation, None) } {
        Ok(value) => value,
        Err(error) => {
            unsafe { shared.device.destroy_image(image, None) };
            return Err(error);
        }
    };
    if let Err(error) = unsafe { shared.device.bind_image_memory(image, memory, 0) } {
        unsafe {
            shared.device.free_memory(memory, None);
            shared.device.destroy_image(image, None);
        }
        return Err(error);
    }
    // Direct encoders must be freely recordable before their eventual queue
    // order is known.  Vulkan otherwise requires each encoder to predict a
    // predecessor's optimal layout. Establish GENERAL once, synchronously,
    // before the image escapes; all normal texture commands then use GENERAL
    // and only carry access/stage dependencies. This is intentionally a
    // correctness baseline and can cost optimal-layout performance.
    if let Err(error) = initialize_general_layout(&shared, image, desc) {
        unsafe {
            shared.device.free_memory(memory, None);
            shared.device.destroy_image(image, None);
        }
        return Err(error);
    }
    Ok(VulkanTexture {
        shared,
        image,
        memory,
        format,
    })
}

fn initialize_general_layout(
    shared: &VulkanShared,
    image: vk::Image,
    desc: &TextureDescriptor,
) -> Result<(), vk::Result> {
    let pool_info = vk::CommandPoolCreateInfo::default()
        .queue_family_index(shared.graphics_family)
        .flags(vk::CommandPoolCreateFlags::TRANSIENT);
    let pool = unsafe { shared.device.create_command_pool(&pool_info, None) }?;
    let allocation = vk::CommandBufferAllocateInfo::default()
        .command_pool(pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    let buffer = match unsafe { shared.device.allocate_command_buffers(&allocation) } {
        Ok(mut values) => values.pop().expect("one initialization command buffer"),
        Err(error) => {
            unsafe { shared.device.destroy_command_pool(pool, None) };
            return Err(error);
        }
    };
    let begin =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    if let Err(error) = unsafe { shared.device.begin_command_buffer(buffer, &begin) } {
        unsafe { shared.device.destroy_command_pool(pool, None) };
        return Err(error);
    }
    let barrier = vk::ImageMemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::empty())
        .dst_access_mask(vk::AccessFlags::empty())
        .old_layout(vk::ImageLayout::UNDEFINED)
        .new_layout(vk::ImageLayout::GENERAL)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(aspects(desc))
                .base_mip_level(0)
                .level_count(desc.mip_levels)
                .base_array_layer(0)
                .layer_count(desc.array_layers),
        );
    unsafe {
        shared.device.cmd_pipeline_barrier(
            buffer,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        );
    }
    if let Err(error) = unsafe { shared.device.end_command_buffer(buffer) } {
        unsafe { shared.device.destroy_command_pool(pool, None) };
        return Err(error);
    }
    let fence_info = vk::FenceCreateInfo::default();
    let fence = match unsafe { shared.device.create_fence(&fence_info, None) } {
        Ok(fence) => fence,
        Err(error) => {
            unsafe { shared.device.destroy_command_pool(pool, None) };
            return Err(error);
        }
    };
    let submit = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&buffer));
    let result = {
        let _queue = shared.queue_guard();
        unsafe {
            shared
                .device
                .queue_submit(shared.graphics_queue, &[submit], fence)
        }
    }
    .and_then(|()| unsafe { shared.device.wait_for_fences(&[fence], true, u64::MAX) });
    unsafe {
        shared.device.destroy_fence(fence, None);
        shared.device.destroy_command_pool(pool, None);
    }
    result
}

fn aspects(desc: &TextureDescriptor) -> vk::ImageAspectFlags {
    let aspects = crate::api::format::format_aspects(desc.format);
    let mut result = vk::ImageAspectFlags::empty();
    if aspects.contains(TextureAspects::COLOR) {
        result |= vk::ImageAspectFlags::COLOR;
    }
    if aspects.contains(TextureAspects::DEPTH) {
        result |= vk::ImageAspectFlags::DEPTH;
    }
    if aspects.contains(TextureAspects::STENCIL) {
        result |= vk::ImageAspectFlags::STENCIL;
    }
    result
}
fn usage(value: TextureUsage) -> vk::ImageUsageFlags {
    let mut flags = vk::ImageUsageFlags::empty();
    if value.contains(TextureUsage::COPY_SRC) {
        flags |= vk::ImageUsageFlags::TRANSFER_SRC;
    }
    if value.contains(TextureUsage::COPY_DST) {
        flags |= vk::ImageUsageFlags::TRANSFER_DST;
    }
    if value.contains(TextureUsage::SAMPLED) {
        flags |= vk::ImageUsageFlags::SAMPLED;
    }
    if value.contains(TextureUsage::STORAGE) {
        flags |= vk::ImageUsageFlags::STORAGE;
    }
    if value.contains(TextureUsage::COLOR_ATTACHMENT) {
        flags |= vk::ImageUsageFlags::COLOR_ATTACHMENT;
    }
    if value.contains(TextureUsage::DEPTH_STENCIL_ATTACHMENT) {
        flags |= vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT;
    }
    flags
}
