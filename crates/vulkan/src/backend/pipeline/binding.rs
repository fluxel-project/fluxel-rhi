//! Vulkan descriptor-set layouts, pools, and bind groups.
//!
//! A bind group owns the descriptor pool that allocated its sole set and retains
//! every referenced resource. Encoders retain the bind group through GPU
//! completion, so destroying the pool cannot invalidate an in-flight set.

use super::super::{
    DeviceState, VulkanBuffer, VulkanDevice, VulkanSampler, VulkanTextureView, vk_error,
};
use ash::vk;
use fluxel_rhi_core::api::{self, Error, Result};
use std::{ops::Range, sync::Arc};

mod layout;

pub use layout::VulkanBindGroupLayout;
pub(crate) use layout::create_group_layout;

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
        // SAFETY: Group owns the descriptor pool and retains its set/layout
        // inputs; submitted encoders retain the Group through fence retirement.
        unsafe { self.device.device.destroy_descriptor_pool(self.pool, None) }
    }
}
pub(crate) fn create_group(
    device: &VulkanDevice,
    desc: &api::BindGroupDesc<'_, super::super::Vulkan>,
) -> Result<VulkanBindGroup> {
    // SAFETY: `Device::create_bind_group` checked the layout and every resource
    // DeviceId before this backend entry point; this borrow is used only here.
    let layout = unsafe { desc.layout.native_ref() };
    let entries = &layout.0.entries;
    let mut sizes = Vec::new();
    let mut variable_count = None;
    for (layout_entry, entry) in entries.as_ref().iter().zip(desc.entries) {
        let descriptor_count = resource_count(&layout_entry.count, &entry.resource)?;
        if matches!(layout_entry.count, api::BindingCount::RuntimeSized { .. }) {
            if variable_count.replace(descriptor_count).is_some() {
                return Err(Error::InvalidState);
            }
            if descriptor_count
                > layout
                    .0
                    .runtime_descriptor_count
                    .ok_or(Error::InvalidState)?
            {
                return Err(Error::InvalidRange);
            }
        }
        let ty = dtype(&layout_entry.ty)?;
        if let Some(size) = sizes
            .iter_mut()
            .find(|size: &&mut vk::DescriptorPoolSize| size.ty == ty)
        {
            size.descriptor_count = size
                .descriptor_count
                .checked_add(descriptor_count)
                .ok_or(Error::InvalidRange)?;
        } else {
            sizes.push(vk::DescriptorPoolSize {
                ty,
                descriptor_count,
            });
        }
    }
    // Each bind group owns a small pool for its one set. This keeps allocation
    // lifetime identical to the group and avoids a backend-wide descriptor-set
    // allocator whose retirement policy would duplicate Renderer ownership.
    // SAFETY: pool sizes are derived from the validated layout and one set is
    // requested. DeviceState keeps the logical device alive for the result.
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
    let variable_counts = variable_count.into_iter().collect::<Vec<_>>();
    let mut variable_allocate = vk::DescriptorSetVariableDescriptorCountAllocateInfo::default()
        .descriptor_counts(&variable_counts);
    let mut allocate = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(pool)
        .set_layouts(&layouts);
    if !variable_counts.is_empty() {
        allocate = allocate.push_next(&mut variable_allocate);
    }
    // SAFETY: `pool`, `layouts`, and optional variable count info all remain
    // live for allocation, and the descriptor-count chain matches this layout.
    let set = match unsafe { device.0.device.allocate_descriptor_sets(&allocate) } {
        Ok(v) => v[0],
        Err(e) => {
            // SAFETY: allocation failed before a Group took ownership, so this
            // function is the unique owner of `pool` and the device is live.
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
                // SAFETY: Core checked the buffer DeviceId before calling this
                // backend; the Group clones it before returning or submission.
                let b = unsafe { buffer.native_ref() };
                if !b.0.usage.contains(api::BufferUsage::UNIFORM)
                    || range.start >= range.end
                    || range.end > b.0.size
                    || range.end - range.start < *min_size
                {
                    return Err(Error::InvalidRange);
                }
                // SAFETY: DeviceState retains the VkInstance that owns this
                // physical device, which remains valid for this property query.
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
                // SAFETY: this newly allocated set is private to `group`; the
                // write references local info valid for the call and a live buffer.
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
                // SAFETY: Core checked the view DeviceId before this backend
                // call; `group` retains the cloned view after this write.
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
                // SAFETY: this private descriptor set and its local image info
                // remain valid for the synchronous update call.
                unsafe {
                    device
                        .0
                        .device
                        .update_descriptor_sets(std::slice::from_ref(&write), &[])
                };
                group._retained.push(Retained::View(native.clone()));
            }
            (
                api::BindingResource::Array(values),
                api::BindingType::SampledTexture {
                    dimension,
                    multisampled,
                    ..
                },
            ) => {
                let mut infos = Vec::with_capacity(values.len());
                for value in values {
                    let api::BindingResource::TextureView(view) = value.as_ref().ok_or(
                        Error::Unsupported("Vulkan partially-bound descriptor array"),
                    )?
                    else {
                        return Err(Error::InvalidState);
                    };
                    // SAFETY: Core validated every array element's DeviceId;
                    // `group` retains each view before this function returns.
                    let native = unsafe { view.native_ref() };
                    if !native.usage().contains(api::TextureUsage::TEXTURE_BINDING)
                        || native.dimension() != *dimension
                        || native.samples() != if *multisampled { 2 } else { 1 }
                    {
                        return Err(Error::InvalidRange);
                    }
                    infos.push(
                        vk::DescriptorImageInfo::default()
                            .image_view(native.native())
                            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
                    );
                    group._retained.push(Retained::View(native.clone()));
                }
                let write = vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(entry.binding)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .image_info(&infos);
                // SAFETY: `infos` and the private descriptor set remain valid
                // throughout this synchronous descriptor update.
                unsafe {
                    device
                        .0
                        .device
                        .update_descriptor_sets(std::slice::from_ref(&write), &[])
                };
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
                // SAFETY: Core checked the view DeviceId; the Group retains the
                // cloned view, keeping its native image view alive for the set.
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
                // SAFETY: the set is private until Group construction finishes;
                // the descriptor image info is valid for this synchronous call.
                unsafe {
                    device
                        .0
                        .device
                        .update_descriptor_sets(std::slice::from_ref(&write), &[])
                };
                group._retained.push(Retained::View(native.clone()));
            }
            (api::BindingResource::Sampler(sampler), api::BindingType::Sampler { .. }) => {
                // SAFETY: Core checked the sampler DeviceId before this call;
                // Group clones it immediately after updating the private set.
                let native = unsafe { sampler.native_ref() };
                let info = vk::DescriptorImageInfo::default().sampler(native.native());
                let write = vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(entry.binding)
                    .descriptor_type(vk::DescriptorType::SAMPLER)
                    .image_info(std::slice::from_ref(&info));
                // SAFETY: the private set and stack descriptor info remain valid
                // for this synchronous update on the live logical device.
                unsafe {
                    device
                        .0
                        .device
                        .update_descriptor_sets(std::slice::from_ref(&write), &[])
                };
                group._retained.push(Retained::Sampler(native.clone()));
            }
            _ => return Err(Error::Unsupported("Vulkan descriptor resource")),
        }
    }
    group
        .dynamic_uniforms
        .sort_unstable_by_key(|slot| slot.binding);
    Ok(VulkanBindGroup(Arc::new(group)))
}

fn resource_count<B: api::Backend>(
    count: &api::BindingCount,
    resource: &api::BindingResource<'_, B>,
) -> Result<u32> {
    match (count, resource) {
        (api::BindingCount::One, _) => Ok(1),
        (api::BindingCount::Fixed(expected), api::BindingResource::Array(values))
            if values.len() == *expected as usize =>
        {
            Ok(*expected)
        }
        (api::BindingCount::RuntimeSized { .. }, api::BindingResource::Array(values))
            if !values.is_empty() =>
        {
            u32::try_from(values.len()).map_err(|_| Error::InvalidRange)
        }
        _ => Err(Error::InvalidState),
    }
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
