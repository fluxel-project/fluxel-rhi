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
use std::{
    ops::Range,
    sync::{Arc, Mutex},
};

mod layout;
mod variant;

pub use layout::VulkanBindGroupLayout;
pub(crate) use layout::{create_group_layout, create_group_layout_variant};

#[derive(Clone)]
pub struct VulkanBindGroup(Arc<Group>);
impl VulkanBindGroup {
    pub(crate) fn native(&self) -> vk::DescriptorSet {
        self.0.set
    }
    pub(crate) fn layout(&self) -> vk::DescriptorSetLayout {
        self.0._layout.native()
    }

    /// Validates one offset per Core binding and expands array offsets to the
    /// per-element sequence required by `vkCmdBindDescriptorSets`.
    pub(crate) fn validate_dynamic_offsets(&self, offsets: &[u32]) -> Result<Vec<u32>> {
        let count = self
            .0
            .dynamic_uniforms
            .iter()
            .enumerate()
            .filter(|(i, slot)| *i == 0 || self.0.dynamic_uniforms[*i - 1].binding != slot.binding)
            .count();
        if offsets.len() != count {
            return Err(Error::InvalidRange);
        }
        let mut expanded = Vec::with_capacity(self.0.dynamic_uniforms.len());
        let mut index = 0;
        let mut previous = None;
        for slot in &self.0.dynamic_uniforms {
            if previous.is_some_and(|binding| binding != slot.binding) {
                index += 1;
            }
            previous = Some(slot.binding);
            let offset = offsets[index];
            expanded.push(offset);
            let offset = u64::from(offset);
            if offset % slot.alignment != 0 {
                return Err(Error::InvalidRange);
            }
            let Some(buffer) = &slot.buffer else { continue };
            let start = slot
                .range
                .start
                .checked_add(offset)
                .ok_or(Error::InvalidRange)?;
            let end = start.checked_add(slot.size).ok_or(Error::InvalidRange)?;
            if end > buffer.0.size {
                return Err(Error::InvalidRange);
            }
        }
        Ok(expanded)
    }
}
struct Group {
    device: Arc<DeviceState>,
    pool: vk::DescriptorPool,
    pub(crate) set: vk::DescriptorSet,
    // Vulkan requires this layout to outlive descriptor sets allocated from it.
    _layout: VulkanBindGroupLayout,
    _retained: Vec<Retained>,
    snapshots: Vec<variant::Snapshot>,
    pool_sizes: Vec<vk::DescriptorPoolSize>,
    variable_count: Option<u32>,
    variants: Mutex<Vec<variant::Variant>>,
    // Sorted by binding, as required by the RHI dynamic-offset contract.
    dynamic_uniforms: Vec<DynamicUniform>,
}
struct DynamicUniform {
    binding: u32,
    buffer: Option<VulkanBuffer>,
    range: Range<u64>,
    size: u64,
    alignment: u64,
}
enum Retained {
    Buffer(VulkanBuffer),
    View(VulkanTextureView),
    Sampler(VulkanSampler),
    Acceleration(super::super::VulkanAcceleration),
}
impl Drop for Group {
    fn drop(&mut self) {
        // SAFETY: Group owns the descriptor pool and retains its set/layout
        // inputs; submitted encoders retain the Group through fence retirement.
        unsafe {
            self.device.device.destroy_descriptor_pool(self.pool, None);
        }
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
        let mut descriptor_count = resource_count(&layout_entry.count, &entry.resource)?;
        if matches!(layout_entry.count, api::BindingCount::RuntimeSized { .. }) {
            if descriptor_count
                > layout
                    .0
                    .runtime_descriptor_count
                    .ok_or(Error::InvalidState)?
            {
                return Err(Error::InvalidRange);
            }
            if layout.0.runtime_dynamic_binding == Some(layout_entry.binding) {
                descriptor_count = layout
                    .0
                    .runtime_descriptor_count
                    .ok_or(Error::InvalidState)?;
            } else if variable_count.replace(descriptor_count).is_some() {
                return Err(Error::InvalidState);
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
        snapshots: Vec::new(),
        pool_sizes: sizes,
        variable_count,
        variants: Mutex::new(Vec::new()),
        dynamic_uniforms: Vec::new(),
    };
    for entry in desc.entries {
        let native_binding = entry.binding;
        let ty = &entries
            .iter()
            .find(|x| x.binding == entry.binding)
            .ok_or(Error::InvalidState)?
            .ty;
        let elements: Vec<(u32, &api::BindingResource<'_, super::super::Vulkan>)> =
            match &entry.resource {
                api::BindingResource::Array(values) => {
                    if matches!(
                        ty,
                        api::BindingType::UniformBuffer {
                            dynamic_offset: true,
                            ..
                        } | api::BindingType::StorageBuffer {
                            dynamic_offset: true,
                            ..
                        }
                    ) {
                        let limits = unsafe {
                            device
                                .0
                                .instance
                                .instance
                                .get_physical_device_properties(device.0.physical)
                        }
                        .limits;
                        let alignment = match ty {
                            api::BindingType::UniformBuffer { .. } => {
                                limits.min_uniform_buffer_offset_alignment
                            }
                            _ => limits.min_storage_buffer_offset_alignment,
                        }
                        .max(1);
                        for value in values {
                            if value.is_none() {
                                group.dynamic_uniforms.push(DynamicUniform {
                                    binding: entry.binding,
                                    buffer: None,
                                    range: 0..0,
                                    size: 0,
                                    alignment,
                                });
                            }
                        }
                    }
                    values
                        .iter()
                        .enumerate()
                        .filter_map(|(i, value)| value.as_ref().map(|value| (i as u32, value)))
                        .collect()
                }
                value => vec![(0, value)],
            };
        for (element, resource) in elements {
            match (resource, ty) {
                (
                    api::BindingResource::Acceleration(acceleration),
                    api::BindingType::AccelerationStructure,
                ) => {
                    // SAFETY: Core checked the acceleration structure's device identity.
                    let native = unsafe { acceleration.native_ref() };
                    if !native.0.top {
                        return Err(Error::InvalidState);
                    }
                    let handles = [native.0.native];
                    let mut acceleration_info =
                        vk::WriteDescriptorSetAccelerationStructureKHR::default()
                            .acceleration_structures(&handles);
                    let write = vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(native_binding)
                        .dst_array_element(element)
                        .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                        .descriptor_count(1)
                        .push_next(&mut acceleration_info);
                    // SAFETY: the private set, pNext and structure stay live through this call.
                    unsafe {
                        device
                            .0
                            .device
                            .update_descriptor_sets(std::slice::from_ref(&write), &[])
                    };
                    group.snapshots.push(variant::Snapshot::Acceleration {
                        binding: entry.binding,
                        element,
                        handle: native.0.native,
                    });
                    group._retained.push(Retained::Acceleration(native.clone()));
                }
                (
                    api::BindingResource::StorageBufferView { buffer, range, .. },
                    api::BindingType::StorageBuffer {
                        dynamic_offset,
                        min_size,
                        ..
                    },
                ) => {
                    // SAFETY: Core checked this buffer's device identity.
                    let native = unsafe { buffer.native_ref() };
                    let limits = unsafe {
                        device
                            .0
                            .instance
                            .instance
                            .get_physical_device_properties(device.0.physical)
                    }
                    .limits;
                    if !native.0.usage.contains(api::BufferUsage::STORAGE)
                        || range.start >= range.end
                        || range.end > native.0.size
                        || range.end - range.start < *min_size
                        || range.start % limits.min_storage_buffer_offset_alignment.max(1) != 0
                        || range.end - range.start > u64::from(limits.max_storage_buffer_range)
                    {
                        return Err(Error::InvalidRange);
                    }
                    let size = range.end - range.start;
                    let info = vk::DescriptorBufferInfo {
                        buffer: native.0.buffer,
                        offset: range.start,
                        range: size,
                    };
                    let write = vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(native_binding)
                        .dst_array_element(element)
                        .descriptor_type(dtype(ty)?)
                        .buffer_info(std::slice::from_ref(&info));
                    // SAFETY: descriptor data and private set are valid through this call;
                    // the retained buffer owns the native allocation thereafter.
                    unsafe {
                        device
                            .0
                            .device
                            .update_descriptor_sets(std::slice::from_ref(&write), &[])
                    };
                    group.snapshots.push(variant::Snapshot::Buffer {
                        binding: entry.binding,
                        element,
                        ty: dtype(ty)?,
                        info,
                    });
                    group._retained.push(Retained::Buffer(native.clone()));
                    if *dynamic_offset {
                        group.dynamic_uniforms.push(DynamicUniform {
                            binding: entry.binding,
                            buffer: Some(native.clone()),
                            range: range.clone(),
                            size,
                            alignment: limits.min_storage_buffer_offset_alignment.max(1),
                        });
                    }
                }
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
                    let limits = unsafe {
                        device
                            .0
                            .instance
                            .instance
                            .get_physical_device_properties(device.0.physical)
                    }
                    .limits;
                    let alignment = limits.min_uniform_buffer_offset_alignment.max(1);
                    if range.start % alignment != 0
                        || range.end - range.start > u64::from(limits.max_uniform_buffer_range)
                    {
                        return Err(Error::InvalidRange);
                    }
                    // Dynamic offsets shift the entire public binding range.
                    let descriptor_range = range.end - range.start;
                    let info = vk::DescriptorBufferInfo {
                        buffer: b.0.buffer,
                        offset: range.start,
                        range: descriptor_range,
                    };
                    let write = vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(native_binding)
                        .dst_array_element(element)
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
                    group.snapshots.push(variant::Snapshot::Buffer {
                        binding: entry.binding,
                        element,
                        ty: dtype(ty)?,
                        info,
                    });
                    group._retained.push(Retained::Buffer(b.clone()));
                    if *dynamic_offset {
                        group.dynamic_uniforms.push(DynamicUniform {
                            binding: entry.binding,
                            buffer: Some(b.clone()),
                            range: range.clone(),
                            size: descriptor_range,
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
                        || (native.samples() > 1) != *multisampled
                    {
                        return Err(Error::InvalidRange);
                    }
                    let info = vk::DescriptorImageInfo::default()
                        .image_view(native.native())
                        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
                    let write = vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(native_binding)
                        .dst_array_element(element)
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
                    group.snapshots.push(variant::Snapshot::Image {
                        binding: entry.binding,
                        element,
                        ty: dtype(ty)?,
                        info,
                    });
                    group._retained.push(Retained::View(native.clone()));
                }
                (
                    api::BindingResource::TextureView(view),
                    api::BindingType::StorageTexture {
                        format,
                        dimension,
                        access: _,
                    },
                ) => {
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
                        .dst_binding(native_binding)
                        .dst_array_element(element)
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
                    group.snapshots.push(variant::Snapshot::Image {
                        binding: entry.binding,
                        element,
                        ty: dtype(ty)?,
                        info,
                    });
                    group._retained.push(Retained::View(native.clone()));
                }
                (api::BindingResource::Sampler(sampler), api::BindingType::Sampler { .. }) => {
                    // SAFETY: Core checked the sampler DeviceId before this call;
                    // Group clones it immediately after updating the private set.
                    let native = unsafe { sampler.native_ref() };
                    let info = vk::DescriptorImageInfo::default().sampler(native.native());
                    let write = vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(native_binding)
                        .dst_array_element(element)
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
                    group.snapshots.push(variant::Snapshot::Sampler {
                        binding: entry.binding,
                        element,
                        float: native.native(),
                        integer: native.integer_native(),
                    });
                    group._retained.push(Retained::Sampler(native.clone()));
                }
                _ => return Err(Error::Unsupported("Vulkan descriptor resource")),
            }
        }
    }
    if let Some(binding) = layout.0.runtime_dynamic_binding {
        let wanted = layout
            .0
            .runtime_descriptor_count
            .ok_or(Error::InvalidState)? as usize;
        let present = group
            .dynamic_uniforms
            .iter()
            .filter(|slot| slot.binding == binding)
            .count();
        let alignment = group
            .dynamic_uniforms
            .iter()
            .find(|slot| slot.binding == binding)
            .map(|slot| slot.alignment)
            .ok_or(Error::InvalidState)?;
        for _ in present..wanted {
            group.dynamic_uniforms.push(DynamicUniform {
                binding,
                buffer: None,
                range: 0..0,
                size: 0,
                alignment,
            });
        }
    }
    group.dynamic_uniforms.sort_by_key(|slot| slot.binding);
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
        api::BindingType::StorageTexture { .. } => vk::DescriptorType::STORAGE_IMAGE,
        api::BindingType::Sampler { .. } => vk::DescriptorType::SAMPLER,
        api::BindingType::StorageBuffer { dynamic_offset, .. } => {
            if *dynamic_offset {
                vk::DescriptorType::STORAGE_BUFFER_DYNAMIC
            } else {
                vk::DescriptorType::STORAGE_BUFFER
            }
        }
        api::BindingType::AccelerationStructure => vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
        _ => return Err(Error::Unsupported("Vulkan binding type")),
    })
}
