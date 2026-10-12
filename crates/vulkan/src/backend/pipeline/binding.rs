//! Vulkan descriptor-set layouts, pools, and bind groups.

use super::super::{
    DeviceState, VulkanBuffer, VulkanDevice, VulkanSampler, VulkanTextureView, vk_error,
};
use ash::vk;
use fluxel_rhi_core::api::{self, Error, Result};
use std::{ops::Range, sync::Arc};

#[derive(Clone)]
pub struct VulkanBindGroupLayout(pub(crate) Arc<GroupLayout>);
impl VulkanBindGroupLayout {
    pub(crate) fn native(&self) -> vk::DescriptorSetLayout {
        self.0.layout
    }
}
#[derive(Clone)]
pub struct VulkanBindGroup(Arc<Group>);
impl VulkanBindGroup {
    pub(crate) fn native(&self) -> vk::DescriptorSet {
        self.0.set
    }
    pub(crate) fn layout(&self) -> vk::DescriptorSetLayout {
        self.0._layout.native()
    }

    /// Validates Vulkan dynamic uniform offsets in the API-defined binding
    /// order.  The encoder passes the same offsets to `vkCmdBindDescriptorSets`.
    pub(crate) fn validate_dynamic_offsets(&self, offsets: &[u32]) -> Result<()> {
        if offsets.len() != self.0.dynamic_uniforms.len() {
            return Err(Error::InvalidRange);
        }
        for (slot, &offset) in self.0.dynamic_uniforms.iter().zip(offsets) {
            let offset = u64::from(offset);
            if offset % slot.alignment != 0 {
                return Err(Error::InvalidRange);
            }
            let start = slot
                .range
                .start
                .checked_add(offset)
                .ok_or(Error::InvalidRange)?;
            let end = start.checked_add(slot.size).ok_or(Error::InvalidRange)?;
            if end > slot.buffer.0.size {
                return Err(Error::InvalidRange);
            }
        }
        Ok(())
    }
}
struct Group {
    device: Arc<DeviceState>,
    pool: vk::DescriptorPool,
    pub(crate) set: vk::DescriptorSet,
    // Vulkan requires this layout to outlive descriptor sets allocated from it.
    _layout: VulkanBindGroupLayout,
    _retained: Vec<Retained>,
    // Sorted by binding, as required by the RHI dynamic-offset contract.
    dynamic_uniforms: Vec<DynamicUniform>,
}
struct DynamicUniform {
    binding: u32,
    buffer: VulkanBuffer,
    range: Range<u64>,
    size: u64,
    alignment: u64,
}
enum Retained {
    Buffer(VulkanBuffer),
    View(VulkanTextureView),
    Sampler(VulkanSampler),
}
impl Drop for Group {
    fn drop(&mut self) {
        unsafe { self.device.device.destroy_descriptor_pool(self.pool, None) }
    }
}
pub(crate) struct GroupLayout {
    device: Arc<DeviceState>,
    pub(crate) layout: vk::DescriptorSetLayout,
    entries: Arc<[api::BindGroupLayoutEntry]>,
}
impl Drop for GroupLayout {
    fn drop(&mut self) {
        unsafe {
            self.device
                .device
                .destroy_descriptor_set_layout(self.layout, None)
        }
    }
}

pub(crate) fn create_group_layout(
    device: &VulkanDevice,
    desc: &api::BindGroupLayoutDesc,
) -> Result<VulkanBindGroupLayout> {
    let bindings: Result<Vec<_>> = desc
        .entries
        .iter()
        .map(|e| {
            if !matches!(e.count, api::BindingCount::One)
                || e.partially_bound
                || e.nonuniform_indexing
            {
                return Err(Error::Unsupported("Vulkan descriptor array layout"));
            }
            Ok(vk::DescriptorSetLayoutBinding::default()
                .binding(e.binding)
                .descriptor_type(dtype(&e.ty)?)
                .descriptor_count(1)
                .stage_flags(stages(&e.visibility)?))
        })
        .collect();
    let bindings = bindings?;
    let info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    let layout =
        unsafe { device.0.device.create_descriptor_set_layout(&info, None) }.map_err(vk_error)?;
    Ok(VulkanBindGroupLayout(Arc::new(GroupLayout {
        device: device.0.clone(),
        layout,
        entries: Arc::from(desc.entries.clone()),
    })))
}
pub(crate) fn create_group(
    device: &VulkanDevice,
    desc: &api::BindGroupDesc<'_, super::super::Vulkan>,
) -> Result<VulkanBindGroup> {
    let layout = unsafe { desc.layout.native_ref() };
    let entries = &layout.0.entries;
    let mut sizes = Vec::new();
    for entry in entries.as_ref() {
        sizes.push(vk::DescriptorPoolSize {
            ty: dtype(&entry.ty)?,
            descriptor_count: 1,
        });
    }
    let pool = unsafe {
        device.0.device.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::default()
                .max_sets(1)
                .pool_sizes(&sizes),
            None,
        )
    }
    .map_err(vk_error)?;
    let layouts = [layout.0.layout];
    let set = match unsafe {
        device.0.device.allocate_descriptor_sets(
            &vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(pool)
                .set_layouts(&layouts),
        )
    } {
        Ok(v) => v[0],
        Err(e) => {
            unsafe { device.0.device.destroy_descriptor_pool(pool, None) };
            return Err(vk_error(e));
        }
    };
    // Put the pool and the layout under RAII before validating/writing entries.
    // Any later error must destroy the pool and keep its layout alive until then.
    let mut group = Group {
        device: device.0.clone(),
        pool,
        set,
        _layout: layout.clone(),
        _retained: Vec::with_capacity(desc.entries.len()),
        dynamic_uniforms: Vec::new(),
    };
    for entry in desc.entries {
        let ty = &entries
            .iter()
            .find(|x| x.binding == entry.binding)
            .ok_or(Error::InvalidState)?
            .ty;
        match (&entry.resource, ty) {
            (
                api::BindingResource::Buffer { buffer, range },
                api::BindingType::UniformBuffer {
                    dynamic_offset,
                    min_size,
                },
            ) => {
                let b = unsafe { buffer.native_ref() };
                if !b.0.usage.contains(api::BufferUsage::UNIFORM)
                    || range.start >= range.end
                    || range.end > b.0.size
                    || range.end - range.start < *min_size
                {
                    return Err(Error::InvalidRange);
                }
                let alignment = u64::from(
                    unsafe {
                        device
                            .0
                            .instance
                            .instance
                            .get_physical_device_properties(device.0.physical)
                    }
                    .limits
                    .min_uniform_buffer_offset_alignment
                    .max(1),
                );
                if *dynamic_offset && range.start % alignment != 0 {
                    return Err(Error::InvalidRange);
                }
                // Vulkan applies a dynamic offset in addition to this range's
                // base.  Describing only the minimum binding size leaves every
                // valid dynamic element addressable within a larger buffer.
                let descriptor_range = if *dynamic_offset {
                    if *min_size == 0 {
                        return Err(Error::Unsupported(
                            "Vulkan dynamic uniform binding requires min_size",
                        ));
                    }
                    *min_size
                } else {
                    range.end - range.start
                };
                let info = vk::DescriptorBufferInfo {
                    buffer: b.0.buffer,
                    offset: range.start,
                    range: descriptor_range,
                };
                let write = vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(entry.binding)
                    .descriptor_type(dtype(ty)?)
                    .buffer_info(std::slice::from_ref(&info));
                unsafe {
                    device
                        .0
                        .device
                        .update_descriptor_sets(std::slice::from_ref(&write), &[])
                };
                group._retained.push(Retained::Buffer(b.clone()));
                if *dynamic_offset {
                    group.dynamic_uniforms.push(DynamicUniform {
                        binding: entry.binding,
                        buffer: b.clone(),
                        range: range.clone(),
                        size: *min_size,
                        alignment,
                    });
                }
            }
            (
                api::BindingResource::TextureView(view),
                api::BindingType::SampledTexture {
                    dimension,
                    multisampled,
                    ..
                },
            ) => {
                let native = unsafe { view.native_ref() };
                if !native.usage().contains(api::TextureUsage::TEXTURE_BINDING)
                    || native.dimension() != *dimension
                    || native.samples() != if *multisampled { 2 } else { 1 }
                {
                    return Err(Error::InvalidRange);
                }
                let info = vk::DescriptorImageInfo::default()
                    .image_view(native.native())
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
                let write = vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(entry.binding)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .image_info(std::slice::from_ref(&info));
                unsafe {
                    device
                        .0
                        .device
                        .update_descriptor_sets(std::slice::from_ref(&write), &[])
                };
                group._retained.push(Retained::View(native.clone()));
            }
            (
                api::BindingResource::TextureView(view),
                api::BindingType::StorageTexture {
                    format,
                    dimension,
                    access,
                },
            ) => {
                if !matches!(access, api::StorageAccess::Write) {
                    return Err(Error::Unsupported("Vulkan storage texture access"));
                }
                let native = unsafe { view.native_ref() };
                if !native.usage().contains(api::TextureUsage::STORAGE_BINDING)
                    || native.format() != *format
                    || native.dimension() != *dimension
                    || native.samples() != 1
                {
                    return Err(Error::InvalidRange);
                }
                let info = vk::DescriptorImageInfo::default()
                    .image_view(native.native())
                    .image_layout(vk::ImageLayout::GENERAL);
                let write = vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(entry.binding)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(std::slice::from_ref(&info));
                unsafe {
                    device
                        .0
                        .device
                        .update_descriptor_sets(std::slice::from_ref(&write), &[])
                };
                group._retained.push(Retained::View(native.clone()));
            }
            (api::BindingResource::Sampler(sampler), api::BindingType::Sampler { .. }) => {
                let info = vk::DescriptorImageInfo::default()
                    .sampler(unsafe { sampler.native_ref() }.native());
                let write = vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(entry.binding)
                    .descriptor_type(vk::DescriptorType::SAMPLER)
                    .image_info(std::slice::from_ref(&info));
                unsafe {
                    device
                        .0
                        .device
                        .update_descriptor_sets(std::slice::from_ref(&write), &[])
                };
                group
                    ._retained
                    .push(Retained::Sampler(unsafe { sampler.native_ref() }.clone()));
            }
            _ => return Err(Error::Unsupported("Vulkan descriptor resource")),
        }
    }
    group
        .dynamic_uniforms
        .sort_unstable_by_key(|slot| slot.binding);
    Ok(VulkanBindGroup(Arc::new(group)))
}

fn dtype(t: &api::BindingType) -> Result<vk::DescriptorType> {
    Ok(match t {
        api::BindingType::UniformBuffer { dynamic_offset, .. } => {
            if *dynamic_offset {
                vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC
            } else {
                vk::DescriptorType::UNIFORM_BUFFER
            }
        }
        api::BindingType::SampledTexture { .. } => vk::DescriptorType::SAMPLED_IMAGE,
        api::BindingType::StorageTexture { access, .. }
            if matches!(access, api::StorageAccess::Write) =>
        {
            vk::DescriptorType::STORAGE_IMAGE
        }
        api::BindingType::Sampler { .. } => vk::DescriptorType::SAMPLER,
        _ => return Err(Error::Unsupported("Vulkan binding type")),
    })
}
fn stages(s: &[api::ShaderStage]) -> Result<vk::ShaderStageFlags> {
    let mut flags = vk::ShaderStageFlags::empty();
    for stage in s {
        flags |= match stage {
            api::ShaderStage::Vertex => vk::ShaderStageFlags::VERTEX,
            api::ShaderStage::Fragment => vk::ShaderStageFlags::FRAGMENT,
            api::ShaderStage::Compute => vk::ShaderStageFlags::COMPUTE,
            _ => return Err(Error::Unsupported("Vulkan descriptor shader stage")),
        };
    }
    if flags.is_empty() {
        return Err(Error::InvalidRange);
    }
    Ok(flags)
}
