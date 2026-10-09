//! One immutable portable bind group lowered to Vulkan descriptors.

use std::any::Any;
use std::sync::Arc;

use ash::vk;

use crate::api::binding::backend::BindGroupBackend;
use crate::api::binding::{BindGroupDescriptor, BindingCount, BindingKind, BindingResource};
use crate::api::resource::buffer::BufferBinding;
use crate::api::resource::subresource::TextureAspects;
use crate::backend::vulkan::failure::VulkanFailure;
use crate::backend::vulkan::ffi;
use crate::backend::vulkan::platform::device::VulkanShared;
use crate::backend::vulkan::resource::{VulkanBuffer, VulkanSampler, VulkanTextureView};

use super::layout::{
    descriptor_type, descriptor_type_for_slot, layout_binding_flags, layout_bindings,
};

/// Native immutable descriptor packet behind one portable bind group.
///
/// The pool owns the set and must be destroyed before the layout.  `shared` is
/// the sole native-device ownership domain; it makes that destruction order safe
/// even when the portable handle is retired after `VulkanDevice` itself.
pub(crate) struct VulkanBindGroup {
    shared: Arc<VulkanShared>,
    pool: vk::DescriptorPool,
    layout: vk::DescriptorSetLayout,
    set: vk::DescriptorSet,
    dynamic_alignments: Vec<u64>,
}

impl VulkanBindGroup {
    /// Descriptor set needed by future compute/raster bind commands.
    pub(crate) fn set(&self) -> vk::DescriptorSet {
        self.set
    }

    pub(crate) fn validate_dynamic_offsets(&self, offsets: &[u32]) -> Result<(), VulkanFailure> {
        if offsets.len() != self.dynamic_alignments.len() {
            return Err(VulkanFailure::Unsupported {
                what: "Vulkan bind-group dynamic offsets",
                why: "the portable offset count differs from the native dynamic descriptor count",
            });
        }
        if offsets
            .iter()
            .zip(&self.dynamic_alignments)
            .any(|(offset, alignment)| u64::from(*offset) % (*alignment).max(1) != 0)
        {
            return Err(VulkanFailure::Unsupported {
                what: "Vulkan bind-group dynamic offsets",
                why: "an offset does not meet the device's buffer binding alignment",
            });
        }
        Ok(())
    }
}

impl BindGroupBackend for VulkanBindGroup {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl Drop for VulkanBindGroup {
    fn drop(&mut self) {
        unsafe {
            // A descriptor set is pool-owned.  Destroy it first, then release
            // the layout it was allocated against; this remains valid without
            // FREE_DESCRIPTOR_SET because individual set destruction is absent.
            self.shared.device.destroy_descriptor_pool(self.pool, None);
            self.shared
                .device
                .destroy_descriptor_set_layout(self.layout, None);
        }
    }
}

/// Creates one dedicated descriptor pool/layout/set and writes its immutable
/// resource references.  Any native failure retains its `VkResult` until the
/// owning device's loss authority observes it.
pub(in crate::backend::vulkan) fn create_bind_group(
    shared: Arc<VulkanShared>,
    descriptor: &BindGroupDescriptor,
) -> Result<VulkanBindGroup, VulkanFailure> {
    let runtime_descriptor_count = runtime_descriptor_count(descriptor)?;
    if runtime_descriptor_count.is_some_and(|count| count > shared.max_runtime_sampled_descriptors)
    {
        return Err(VulkanFailure::Unsupported {
            what: "a Vulkan runtime sampled-texture descriptor array above the device limit",
            why: "the packet's active count exceeds the descriptor-set layout's declared native maximum",
        });
    }
    let pool_sizes = pool_sizes(descriptor)?;
    let bindings = layout_bindings(
        descriptor.layout.descriptor(),
        shared.max_runtime_sampled_descriptors,
    )?;
    let mut layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    let binding_flags = layout_binding_flags(descriptor.layout.descriptor())?;
    let mut binding_flags_info = binding_flags
        .as_ref()
        .map(|flags| vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(flags));
    if let Some(binding_flags_info) = binding_flags_info.as_mut() {
        layout_info = layout_info.push_next(binding_flags_info);
    }
    let layout = unsafe {
        shared
            .device
            .create_descriptor_set_layout(&layout_info, None)
    }
    .map_err(|result| native(result, "vkCreateDescriptorSetLayout"))?;

    let pool_info = vk::DescriptorPoolCreateInfo::default()
        .max_sets(1)
        .pool_sizes(&pool_sizes);
    let pool = match unsafe { shared.device.create_descriptor_pool(&pool_info, None) } {
        Ok(pool) => pool,
        Err(result) => {
            unsafe { shared.device.destroy_descriptor_set_layout(layout, None) };
            return Err(native(result, "vkCreateDescriptorPool"));
        }
    };

    let layouts = [layout];
    let mut allocate = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(pool)
        .set_layouts(&layouts);
    let runtime_counts = runtime_descriptor_count.map(|count| [count]);
    let mut variable_counts = runtime_counts.as_ref().map(|counts| {
        vk::DescriptorSetVariableDescriptorCountAllocateInfo::default().descriptor_counts(counts)
    });
    if let Some(variable_counts) = variable_counts.as_mut() {
        allocate = allocate.push_next(variable_counts);
    }
    let set = match unsafe { shared.device.allocate_descriptor_sets(&allocate) } {
        Ok(mut sets) => match sets.pop() {
            Some(set) => set,
            None => {
                unsafe {
                    shared.device.destroy_descriptor_pool(pool, None);
                    shared.device.destroy_descriptor_set_layout(layout, None);
                }
                return Err(VulkanFailure::Unsupported {
                    what: "Vulkan descriptor-set allocation",
                    why: "the driver returned no descriptor set for one requested layout",
                });
            }
        },
        Err(result) => {
            unsafe {
                shared.device.destroy_descriptor_pool(pool, None);
                shared.device.destroy_descriptor_set_layout(layout, None);
            }
            return Err(native(result, "vkAllocateDescriptorSets"));
        }
    };

    if let Err(error) = write_descriptor_set(&shared, set, descriptor) {
        unsafe {
            shared.device.destroy_descriptor_pool(pool, None);
            shared.device.destroy_descriptor_set_layout(layout, None);
        }
        return Err(error);
    }

    let dynamic_alignments = descriptor
        .layout
        .descriptor()
        .entries
        .iter()
        .filter(|entry| entry.dynamic_offset)
        .flat_map(|entry| {
            let alignment = match entry.kind {
                BindingKind::UniformBuffer { .. } => shared.min_uniform_buffer_offset_alignment,
                BindingKind::StorageBuffer { .. } => shared.min_storage_buffer_offset_alignment,
                _ => 1,
            };
            std::iter::repeat_n(alignment.max(1), entry.count.elements() as usize)
        })
        .collect();
    Ok(VulkanBindGroup {
        shared,
        pool,
        layout,
        set,
        dynamic_alignments,
    })
}

fn pool_sizes(
    descriptor: &BindGroupDescriptor,
) -> Result<Vec<vk::DescriptorPoolSize>, VulkanFailure> {
    // Five portable kinds map one-to-one to five descriptor types. Keeping the
    // accumulation explicit avoids relying on enum integer values as map keys.
    let mut uniform = 0;
    let mut dynamic_uniform = 0;
    let mut storage_buffer = 0;
    let mut dynamic_storage_buffer = 0;
    let mut sampled_image = 0;
    let mut storage_image = 0;
    let mut sampler = 0;
    for entry in &descriptor.layout.descriptor().entries {
        let count = entry_descriptor_count(descriptor, entry)?;
        match descriptor_type_for_slot(&entry.kind, entry.dynamic_offset) {
            vk::DescriptorType::UNIFORM_BUFFER => uniform += count,
            vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC => dynamic_uniform += count,
            vk::DescriptorType::STORAGE_BUFFER => storage_buffer += count,
            vk::DescriptorType::STORAGE_BUFFER_DYNAMIC => dynamic_storage_buffer += count,
            vk::DescriptorType::SAMPLED_IMAGE => sampled_image += count,
            vk::DescriptorType::STORAGE_IMAGE => storage_image += count,
            vk::DescriptorType::SAMPLER => sampler += count,
            _ => unreachable!("portable binding kinds have closed Vulkan descriptor mapping"),
        }
    }
    Ok([
        (vk::DescriptorType::UNIFORM_BUFFER, uniform),
        (vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC, dynamic_uniform),
        (vk::DescriptorType::STORAGE_BUFFER, storage_buffer),
        (
            vk::DescriptorType::STORAGE_BUFFER_DYNAMIC,
            dynamic_storage_buffer,
        ),
        (vk::DescriptorType::SAMPLED_IMAGE, sampled_image),
        (vk::DescriptorType::STORAGE_IMAGE, storage_image),
        (vk::DescriptorType::SAMPLER, sampler),
    ]
    .into_iter()
    .filter(|(_, count)| *count != 0)
    .map(|(ty, descriptor_count)| vk::DescriptorPoolSize {
        ty,
        descriptor_count,
    })
    .collect())
}

/// Returns the active packet count for the one Vulkan variable-count binding.
/// Its layout declares the device maximum, while allocation receives this exact
/// positive packet length (32 for the descriptor-indexing example).
fn runtime_descriptor_count(
    descriptor: &BindGroupDescriptor,
) -> Result<Option<u32>, VulkanFailure> {
    let mut result = None;
    for entry in &descriptor.layout.descriptor().entries {
        if entry.count != BindingCount::RuntimeSized {
            continue;
        }
        if result.is_some() {
            return Err(VulkanFailure::Unsupported {
                what: "several runtime-sized Vulkan descriptor bindings in one group",
                why: "Vulkan permits one variable descriptor count per descriptor set",
            });
        }
        if !matches!(entry.kind, BindingKind::SampledTexture { .. }) {
            return Err(VulkanFailure::Unsupported {
                what: "a non-sampled-texture runtime descriptor array on Vulkan",
                why: "this descriptor-indexing slice enables runtime arrays only for sampled textures",
            });
        }
        result = Some(entry_descriptor_count(descriptor, entry)?);
    }
    Ok(result)
}

fn entry_descriptor_count(
    descriptor: &BindGroupDescriptor,
    layout_entry: &crate::api::binding::BindingSlot,
) -> Result<u32, VulkanFailure> {
    if layout_entry.count != BindingCount::RuntimeSized {
        return Ok(layout_entry.count.elements());
    }
    let entry = descriptor
        .entries
        .iter()
        .find(|entry| entry.slot == layout_entry.slot)
        .ok_or(VulkanFailure::Unsupported {
            what: "a Vulkan runtime descriptor packet without its layout slot",
            why: "portable bind-group validation should provide one entry for every layout slot",
        })?;
    let length = match &entry.resource {
        BindingResource::TextureArray(values) => values.len(),
        _ => {
            return Err(VulkanFailure::Unsupported {
                what: "a Vulkan runtime sampled-texture descriptor packet",
                why: "runtime descriptor arrays must carry a texture-array resource",
            });
        }
    };
    u32::try_from(length).map_err(|_| VulkanFailure::Unsupported {
        what: "a Vulkan runtime descriptor array larger than u32",
        why: "VkDescriptorSetVariableDescriptorCountAllocateInfo encodes counts as u32",
    })
}

fn write_descriptor_set(
    shared: &VulkanShared,
    set: vk::DescriptorSet,
    descriptor: &BindGroupDescriptor,
) -> Result<(), VulkanFailure> {
    // `VkWriteDescriptorSet` borrows its info arrays, so construct and submit
    // each entry independently. A group is immutable, and update calls are
    // void in Vulkan; all fallible work occurred before this point.
    for entry in &descriptor.entries {
        let Some(layout_entry) = descriptor.layout.slot(entry.slot) else {
            return Err(VulkanFailure::Unsupported {
                what: "Vulkan bind-group packet",
                why: "portable validation did not leave a layout slot for an entry",
            });
        };
        write_entry(
            shared,
            set,
            entry.slot.get(),
            &layout_entry.kind,
            layout_entry.dynamic_offset,
            &entry.resource,
        )?;
    }
    Ok(())
}

fn write_entry(
    shared: &VulkanShared,
    set: vk::DescriptorSet,
    binding: u32,
    kind: &BindingKind,
    dynamic_offset: bool,
    resource: &BindingResource,
) -> Result<(), VulkanFailure> {
    match kind {
        BindingKind::UniformBuffer { .. } | BindingKind::StorageBuffer { .. } => write_buffers(
            shared,
            set,
            binding,
            descriptor_type_for_slot(kind, dynamic_offset),
            resource,
        ),
        BindingKind::SampledTexture { .. } | BindingKind::StorageTexture { .. } => {
            write_images(shared, set, binding, descriptor_type(kind), resource)
        }
        BindingKind::Sampler { .. } => write_samplers(shared, set, binding, resource),
        BindingKind::AccelerationStructure | BindingKind::ExternalTexture => {
            Err(VulkanFailure::Unsupported {
                what: "a Vulkan acceleration-structure or external-texture descriptor",
                why: "the descriptor layout gate must reject this unavailable Vulkan extension path",
            })
        }
        _ => crate::unknown_portable_variant(),
    }
}

fn write_buffers(
    shared: &VulkanShared,
    set: vk::DescriptorSet,
    binding: u32,
    descriptor_type: vk::DescriptorType,
    resource: &BindingResource,
) -> Result<(), VulkanFailure> {
    let one;
    let bindings: &[BufferBinding] = match resource {
        BindingResource::Buffer(value) => {
            one = [value.clone()];
            &one
        }
        BindingResource::BufferArray(values) => values,
        _ => return packet_mismatch("buffer", "a non-buffer resource"),
    };
    let mut infos = Vec::with_capacity(bindings.len());
    for value in bindings {
        let Some(buffer) = value
            .buffer
            .native()
            .as_any()
            .downcast_ref::<VulkanBuffer>()
        else {
            return packet_mismatch("Vulkan buffer", "a resource from another backend");
        };
        infos.push(
            vk::DescriptorBufferInfo::default()
                .buffer(buffer.buffer())
                .offset(value.range.offset)
                .range(value.range.size),
        );
    }
    let write = vk::WriteDescriptorSet::default()
        .dst_set(set)
        .dst_binding(binding)
        .descriptor_type(descriptor_type)
        .buffer_info(&infos);
    unsafe { shared.device.update_descriptor_sets(&[write], &[]) };
    Ok(())
}

fn write_images(
    shared: &VulkanShared,
    set: vk::DescriptorSet,
    binding: u32,
    descriptor_type: vk::DescriptorType,
    resource: &BindingResource,
) -> Result<(), VulkanFailure> {
    let one;
    let views: &[crate::api::resource::TextureView] = match resource {
        BindingResource::Texture(value) => {
            one = [value.clone()];
            &one
        }
        BindingResource::TextureArray(values) => values,
        _ => return packet_mismatch("texture", "a non-texture resource"),
    };
    let mut infos = Vec::with_capacity(views.len());
    for value in views {
        let Some(view) = value.native().as_any().downcast_ref::<VulkanTextureView>() else {
            return packet_mismatch("Vulkan texture view", "a resource from another backend");
        };
        infos.push(
            vk::DescriptorImageInfo::default()
                .image_view(view.view())
                .image_layout(descriptor_image_layout(
                    descriptor_type,
                    value.descriptor().aspects,
                )),
        );
    }
    let write = vk::WriteDescriptorSet::default()
        .dst_set(set)
        .dst_binding(binding)
        .descriptor_type(descriptor_type)
        .image_info(&infos);
    unsafe { shared.device.update_descriptor_sets(&[write], &[]) };
    Ok(())
}

/// The layout recorded in an immutable image descriptor.
///
/// Command lowering must use this exact rule before binding the descriptor.
/// In particular, depth/stencil sampled views need Vulkan's depth/stencil
/// read-only layout rather than the color-image shader-read layout.  Keeping
/// the mapping here prevents descriptor metadata and image barriers from
/// quietly drifting apart as more command paths are added.
pub(crate) fn descriptor_image_layout(
    descriptor_type: vk::DescriptorType,
    aspects: TextureAspects,
) -> vk::ImageLayout {
    match descriptor_type {
        vk::DescriptorType::SAMPLED_IMAGE => {
            if aspects.contains(TextureAspects::DEPTH) || aspects.contains(TextureAspects::STENCIL)
            {
                vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL
            } else {
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
            }
        }
        vk::DescriptorType::STORAGE_IMAGE => vk::ImageLayout::GENERAL,
        _ => unreachable!("image-layout mapping only receives portable image descriptor types"),
    }
}

fn write_samplers(
    shared: &VulkanShared,
    set: vk::DescriptorSet,
    binding: u32,
    resource: &BindingResource,
) -> Result<(), VulkanFailure> {
    let one;
    let samplers: &[crate::api::resource::Sampler] = match resource {
        BindingResource::Sampler(value) => {
            one = [value.clone()];
            &one
        }
        BindingResource::SamplerArray(values) => values,
        _ => return packet_mismatch("sampler", "a non-sampler resource"),
    };
    let mut infos = Vec::with_capacity(samplers.len());
    for value in samplers {
        let Some(sampler) = value.native().as_any().downcast_ref::<VulkanSampler>() else {
            return packet_mismatch("Vulkan sampler", "a resource from another backend");
        };
        infos.push(vk::DescriptorImageInfo::default().sampler(sampler.sampler()));
    }
    let write = vk::WriteDescriptorSet::default()
        .dst_set(set)
        .dst_binding(binding)
        .descriptor_type(vk::DescriptorType::SAMPLER)
        .image_info(&infos);
    unsafe { shared.device.update_descriptor_sets(&[write], &[]) };
    Ok(())
}

fn packet_mismatch<T>(expected: &'static str, actual: &'static str) -> Result<T, VulkanFailure> {
    Err(VulkanFailure::Unsupported {
        what: expected,
        why: actual,
    })
}

fn native(result: vk::Result, operation: &'static str) -> VulkanFailure {
    VulkanFailure::Native(ffi::NativeError::new(result, operation))
}
