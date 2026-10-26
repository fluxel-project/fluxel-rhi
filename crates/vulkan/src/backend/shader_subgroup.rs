//! Vulkan subgroup feature admission for direct and Naga-generated SPIR-V.
//!
//! Vulkan reports both the stages that admit subgroup operations and the
//! operation families they support. Core's backend-specific `subgroups`
//! requirement therefore cannot be answered by a generic device boolean:
//! trusted direct artifacts additionally declare their actual subgroup
//! capability families. This module keeps that native capability inspection
//! narrow; it is not a SPIR-V validator.

use super::{VulkanDevice, api};
use ash::vk;
use fluxel_rhi_core::api::{Error, Result};

const OP_CAPABILITY: u32 = 17;

pub(super) fn requested(desc: &api::ShaderDesc) -> bool {
    desc.required_features
        .iter()
        .any(|feature| feature == "subgroups")
}

fn stage(stage: api::ShaderStage) -> vk::ShaderStageFlags {
    match stage {
        api::ShaderStage::Vertex => vk::ShaderStageFlags::VERTEX,
        api::ShaderStage::Fragment => vk::ShaderStageFlags::FRAGMENT,
        api::ShaderStage::Compute => vk::ShaderStageFlags::COMPUTE,
        api::ShaderStage::Task => vk::ShaderStageFlags::TASK_EXT,
        api::ShaderStage::Mesh => vk::ShaderStageFlags::MESH_EXT,
        api::ShaderStage::RayGeneration => vk::ShaderStageFlags::RAYGEN_KHR,
        api::ShaderStage::AnyHit => vk::ShaderStageFlags::ANY_HIT_KHR,
        api::ShaderStage::ClosestHit => vk::ShaderStageFlags::CLOSEST_HIT_KHR,
        api::ShaderStage::Miss => vk::ShaderStageFlags::MISS_KHR,
        api::ShaderStage::Intersection => vk::ShaderStageFlags::INTERSECTION_KHR,
        api::ShaderStage::Callable => vk::ShaderStageFlags::CALLABLE_KHR,
    }
}

fn stages(entries: &[api::ShaderEntryPoint]) -> vk::ShaderStageFlags {
    entries
        .iter()
        .fold(vk::ShaderStageFlags::empty(), |flags, entry| {
            flags | stage(entry.stage)
        })
}

fn validate(
    entries: &[api::ShaderEntryPoint],
    supported_stages: vk::ShaderStageFlags,
    supported_operations: vk::SubgroupFeatureFlags,
    quad_operations_in_all_stages: bool,
    required_operations: vk::SubgroupFeatureFlags,
) -> Result<()> {
    let required_stages = stages(entries);
    if required_stages.is_empty()
        || !supported_operations.contains(required_operations)
        || !supported_stages.contains(required_stages)
    {
        return Err(Error::Unsupported("Vulkan subgroups"));
    }
    if required_operations.contains(vk::SubgroupFeatureFlags::QUAD)
        && !quad_operations_in_all_stages
        && !(vk::ShaderStageFlags::FRAGMENT | vk::ShaderStageFlags::COMPUTE)
            .contains(required_stages)
    {
        return Err(Error::Unsupported("Vulkan subgroup quad operations"));
    }
    Ok(())
}

pub(super) fn validate_declared(device: &VulkanDevice, desc: &api::ShaderDesc) -> Result<()> {
    if !requested(desc) {
        return Ok(());
    }
    validate(
        &desc.entries,
        device.0.subgroup_stages,
        device.0.subgroup_operations,
        device.0.subgroup_quad_operations_in_all_stages,
        vk::SubgroupFeatureFlags::BASIC,
    )
}

fn required_operations(words: &[u32]) -> Result<vk::SubgroupFeatureFlags> {
    if words.len() < 5 {
        return Err(Error::Shader("invalid SPIR-V header".into()));
    }
    let mut required = vk::SubgroupFeatureFlags::BASIC;
    let mut cursor = 5;
    while cursor < words.len() {
        let header = words[cursor];
        let count = (header >> 16) as usize;
        if count == 0
            || cursor
                .checked_add(count)
                .is_none_or(|end| end > words.len())
        {
            return Err(Error::Shader("invalid SPIR-V instruction".into()));
        }
        if header & 0xffff == OP_CAPABILITY && count >= 2 {
            let capability = words[cursor + 1];
            let operation = match capability {
                61 => Some(vk::SubgroupFeatureFlags::BASIC),
                62 => Some(vk::SubgroupFeatureFlags::VOTE),
                63 => Some(vk::SubgroupFeatureFlags::ARITHMETIC),
                64 => Some(vk::SubgroupFeatureFlags::BALLOT),
                65 => Some(vk::SubgroupFeatureFlags::SHUFFLE),
                66 => Some(vk::SubgroupFeatureFlags::SHUFFLE_RELATIVE),
                67 => Some(vk::SubgroupFeatureFlags::CLUSTERED),
                68 => Some(vk::SubgroupFeatureFlags::QUAD),
                // These pre-Vulkan-1.1 capability spellings need extensions
                // this backend does not enable. Do not silently equate them
                // with the core capability family.
                4423 | 4431 | 5297 => {
                    return Err(Error::Unsupported("Vulkan SPIR-V subgroup capability"));
                }
                _ => None,
            };
            if let Some(operation) = operation {
                required |= operation;
            }
        }
        cursor += count;
    }
    Ok(required)
}

pub(super) fn validate_spirv(
    device: &VulkanDevice,
    desc: &api::ShaderDesc,
    words: &[u32],
) -> Result<()> {
    if !requested(desc) {
        return Ok(());
    }
    validate(
        &desc.entries,
        device.0.subgroup_stages,
        device.0.subgroup_operations,
        device.0.subgroup_quad_operations_in_all_stages,
        required_operations(words)?,
    )
}

pub(super) fn supports_stage(device: &VulkanDevice, shader_stage: api::ShaderStage) -> bool {
    device.0.subgroup_stages.contains(stage(shader_stage))
}

#[cfg(test)]
#[path = "../tests/backend/shader_subgroup.rs"]
mod tests;
