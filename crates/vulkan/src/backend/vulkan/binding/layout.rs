//! The single Vulkan mapping authority for a portable bind-group layout.
//!
//! A Vulkan descriptor-set layout has a native object identity, while the
//! portable `BindGroupLayout` intentionally does not.  The first baseline gives
//! every immutable group its own `VkDescriptorSetLayout`; a future pipeline
//! layout must call this function again (or cache its result) rather than grow a
//! second, subtly different mapping.

use ash::vk;

use crate::api::binding::{BindGroupLayoutDescriptor, BindingCount, BindingKind};
use crate::api::shader::ShaderStages;
use crate::backend::vulkan::failure::VulkanFailure;

/// Maps one validated portable layout to Vulkan layout bindings in canonical
/// slot order.
pub(in crate::backend::vulkan) fn layout_bindings(
    descriptor: &BindGroupLayoutDescriptor,
    runtime_sampled_descriptor_count: u32,
) -> Result<Vec<vk::DescriptorSetLayoutBinding<'static>>, VulkanFailure> {
    let mut bindings = Vec::with_capacity(descriptor.entries.len());
    for entry in &descriptor.entries {
        if matches!(
            entry.kind,
            BindingKind::AccelerationStructure | BindingKind::ExternalTexture
        ) {
            return Err(VulkanFailure::Unsupported {
                what: "Vulkan acceleration-structure or external-texture binding",
                why: "this device slice does not enable the required extension and descriptor lowering",
            });
        }
        bindings.push(
            vk::DescriptorSetLayoutBinding::default()
                .binding(entry.slot.get())
                .descriptor_type(descriptor_type_for_slot(&entry.kind, entry.dynamic_offset))
                .descriptor_count(descriptor_count(entry, runtime_sampled_descriptor_count)?)
                .stage_flags(shader_stages(entry.visibility)),
        );
    }
    Ok(bindings)
}

/// Binding flags for a descriptor-indexed variable-count layout. Vulkan permits
/// one such binding and requires it to have the largest binding number.
pub(in crate::backend::vulkan) fn layout_binding_flags(
    descriptor: &BindGroupLayoutDescriptor,
) -> Result<Option<Vec<vk::DescriptorBindingFlags>>, VulkanFailure> {
    let Some(runtime_index) = descriptor
        .entries
        .iter()
        .position(|entry| entry.count == BindingCount::RuntimeSized)
    else {
        return Ok(None);
    };
    if descriptor
        .entries
        .iter()
        .skip(runtime_index + 1)
        .any(|entry| entry.count == BindingCount::RuntimeSized)
    {
        return Err(VulkanFailure::Unsupported {
            what: "several runtime-sized Vulkan descriptor bindings in one group",
            why: "VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT permits at most one binding per set",
        });
    }
    if runtime_index + 1 != descriptor.entries.len() {
        return Err(VulkanFailure::Unsupported {
            what: "a runtime-sized Vulkan descriptor binding that is not last",
            why: "VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT requires the highest binding number",
        });
    }
    Ok(Some(
        descriptor
            .entries
            .iter()
            .map(|entry| {
                (entry.count == BindingCount::RuntimeSized)
                    .then_some(vk::DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT)
                    .unwrap_or_else(vk::DescriptorBindingFlags::empty)
            })
            .collect(),
    ))
}

fn descriptor_count(
    entry: &crate::api::binding::BindingSlot,
    runtime_sampled_descriptor_count: u32,
) -> Result<u32, VulkanFailure> {
    match entry.count {
        BindingCount::One | BindingCount::Fixed(_) => Ok(entry.count.elements()),
        BindingCount::RuntimeSized => {
            if !matches!(entry.kind, BindingKind::SampledTexture { .. })
                || runtime_sampled_descriptor_count == 0
            {
                return Err(VulkanFailure::Unsupported {
                    what: "a Vulkan runtime-sized descriptor binding",
                    why: "this backend enables variable descriptor count only for sampled textures with a non-zero native limit",
                });
            }
            Ok(runtime_sampled_descriptor_count)
        }
        _ => crate::unknown_portable_variant(),
    }
}

pub(crate) fn descriptor_type(kind: &BindingKind) -> vk::DescriptorType {
    match kind {
        BindingKind::UniformBuffer { .. } => vk::DescriptorType::UNIFORM_BUFFER,
        BindingKind::StorageBuffer { .. } => vk::DescriptorType::STORAGE_BUFFER,
        BindingKind::SampledTexture { .. } => vk::DescriptorType::SAMPLED_IMAGE,
        BindingKind::StorageTexture { .. } => vk::DescriptorType::STORAGE_IMAGE,
        BindingKind::Sampler { .. } => vk::DescriptorType::SAMPLER,
        // `layout_bindings` rejects these before this mapping is reached. Keep
        // this match exhaustive so new public binding vocabulary cannot turn
        // into an accidental native descriptor declaration.
        BindingKind::AccelerationStructure | BindingKind::ExternalTexture => {
            // Unreachable: `layout_bindings` rejects this before asking for a
            // descriptor type.  Use a valid sentinel rather than an invalid
            // Vulkan enum in case a future refactor accidentally evaluates it.
            vk::DescriptorType::SAMPLER
        }
        _ => crate::unknown_portable_variant(),
    }
}

pub(crate) fn descriptor_type_for_slot(
    kind: &BindingKind,
    dynamic_offset: bool,
) -> vk::DescriptorType {
    if dynamic_offset {
        match kind {
            BindingKind::UniformBuffer { .. } => vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
            BindingKind::StorageBuffer { .. } => vk::DescriptorType::STORAGE_BUFFER_DYNAMIC,
            _ => descriptor_type(kind),
        }
    } else {
        descriptor_type(kind)
    }
}

fn shader_stages(stages: ShaderStages) -> vk::ShaderStageFlags {
    let mut flags = vk::ShaderStageFlags::empty();
    if stages.contains(ShaderStages::VERTEX) {
        flags |= vk::ShaderStageFlags::VERTEX;
    }
    if stages.contains(ShaderStages::FRAGMENT) {
        flags |= vk::ShaderStageFlags::FRAGMENT;
    }
    if stages.contains(ShaderStages::COMPUTE) {
        flags |= vk::ShaderStageFlags::COMPUTE;
    }
    flags
}
