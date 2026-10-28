//! Vulkan descriptor-set layout ownership and runtime-array support checks.
//!
//! Vulkan requires a runtime-sized binding's upper bound when the layout is
//! created, before any descriptor set supplies its actual element count. Core
//! carries that upper bound in `BindingCount::RuntimeSized`; this module lowers
//! it faithfully and asks the driver whether the complete layout is supported.
//! It does not allocate descriptor pools or write descriptors; those operations
//! belong to the sibling bind-group owner.

use super::super::super::{DeviceState, VulkanDevice, vk_error};
use ash::vk;
use fluxel_rhi_core::api::{self, Error, Result};
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Clone)]
pub struct VulkanBindGroupLayout(pub(crate) Arc<GroupLayout>);
impl VulkanBindGroupLayout {
    pub(crate) fn native(&self) -> vk::DescriptorSetLayout {
        self.0.layout
    }
}

pub(crate) struct GroupLayout {
    pub(super) device: Arc<DeviceState>,
    pub(crate) layout: vk::DescriptorSetLayout,
    pub(crate) entries: Arc<[api::BindGroupLayoutEntry]>,
    pub(crate) native_bindings: HashMap<u32, u32>,
    pub(crate) sampler_mirrors: HashMap<u32, u32>,
    /// The public contract's layout-time upper bound for the sole runtime
    /// binding. Bind-group allocation validates its actual count against this.
    pub(super) runtime_descriptor_count: Option<u32>,
    pub(super) runtime_dynamic_binding: Option<u32>,
}
impl Drop for GroupLayout {
    fn drop(&mut self) {
        // SAFETY: every Group retains this layout until its descriptor set and
        // pool are retired, and this Arc retains the live logical device.
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
    create_group_layout_variant(device, desc, &HashMap::new())
}

pub(crate) fn create_group_layout_variant(
    device: &VulkanDevice,
    desc: &api::BindGroupLayoutDesc,
    modes: &HashMap<u32, u8>,
) -> Result<VulkanBindGroupLayout> {
    let runtime_descriptor_count = desc.entries.iter().find_map(|entry| match entry.count {
        api::BindingCount::RuntimeSized { max_count } => Some(max_count),
        _ => None,
    });
    let runtime_dynamic_binding = desc.entries.iter().find_map(|entry| {
        (matches!(entry.count, api::BindingCount::RuntimeSized { .. })
            && matches!(
                entry.ty,
                api::BindingType::UniformBuffer {
                    dynamic_offset: true,
                    ..
                } | api::BindingType::StorageBuffer {
                    dynamic_offset: true,
                    ..
                }
            ))
        .then_some(entry.binding)
    });
    if runtime_dynamic_binding.is_some() && !device.0.descriptor_partially_bound {
        return Err(Error::Unsupported("Vulkan partially-bound descriptors"));
    }
    let mut native_bindings: HashMap<u32, u32> = desc
        .entries
        .iter()
        .map(|e| (e.binding, e.binding))
        .collect();
    let mut sampler_mirrors = HashMap::new();
    let mut next = desc.entries.iter().map(|e| e.binding).max().unwrap_or(0);
    for entry in &desc.entries {
        if matches!(entry.ty, api::BindingType::Sampler { .. })
            && modes.get(&entry.binding) == Some(&2)
        {
            next = next.checked_add(1).ok_or(Error::InvalidRange)?;
            sampler_mirrors.insert(entry.binding, next);
        }
    }
    if let Some(entry) = desc.entries.iter().find(|e| {
        matches!(e.count, api::BindingCount::RuntimeSized { .. })
            && runtime_dynamic_binding != Some(e.binding)
    }) {
        if !sampler_mirrors.is_empty() {
            next = next.checked_add(1).ok_or(Error::InvalidRange)?;
            native_bindings.insert(entry.binding, next);
        }
    }
    let mut bindings: Vec<_> = desc
        .entries
        .iter()
        .map(|entry| {
            create_layout_binding(device, entry).map(|v| v.binding(native_bindings[&entry.binding]))
        })
        .collect::<Result<_>>()?;
    let mut binding_flags: Vec<_> = desc
        .entries
        .iter()
        .map(|entry| {
            let mut flags = vk::DescriptorBindingFlags::empty();
            if matches!(entry.count, api::BindingCount::RuntimeSized { .. })
                && runtime_dynamic_binding != Some(entry.binding)
            {
                flags |= vk::DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT;
            }
            if entry.partially_bound || runtime_dynamic_binding == Some(entry.binding) {
                flags |= vk::DescriptorBindingFlags::PARTIALLY_BOUND;
            }
            flags
        })
        .collect();
    for entry in &desc.entries {
        if let Some(&binding) = sampler_mirrors.get(&entry.binding) {
            bindings.push(create_layout_binding(device, entry)?.binding(binding));
            let mut flag = vk::DescriptorBindingFlags::empty();
            if entry.partially_bound {
                if !device.0.descriptor_partially_bound {
                    return Err(Error::Unsupported("Vulkan partially-bound sampler mirror"));
                }
                flag |= vk::DescriptorBindingFlags::PARTIALLY_BOUND;
            }
            binding_flags.push(flag);
        }
    }
    let mut ordered: Vec<_> = bindings.into_iter().zip(binding_flags).collect();
    ordered.sort_by_key(|(binding, _)| binding.binding);
    let (bindings, binding_flags): (Vec<_>, Vec<_>) = ordered.into_iter().unzip();
    let mut flags =
        vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&binding_flags);
    if runtime_descriptor_count.is_some() && runtime_dynamic_binding.is_none() {
        ensure_runtime_array_layout_supported(device, &bindings, &mut flags)?;
    }
    let info = vk::DescriptorSetLayoutCreateInfo::default()
        .bindings(&bindings)
        .push_next(&mut flags);
    let mut support = vk::DescriptorSetLayoutSupport::default();
    // SAFETY: the complete create-info and output are live for the query.
    unsafe {
        device
            .0
            .device
            .get_descriptor_set_layout_support(&info, &mut support)
    };
    if support.supported != vk::TRUE {
        return Err(Error::Unsupported("Vulkan descriptor set layout limits"));
    }
    // SAFETY: `bindings` and the matching binding-flag chain remain alive for
    // this synchronous call. DeviceState retains the native logical device.
    let layout =
        unsafe { device.0.device.create_descriptor_set_layout(&info, None) }.map_err(vk_error)?;
    Ok(VulkanBindGroupLayout(Arc::new(GroupLayout {
        device: device.0.clone(),
        layout,
        entries: Arc::from(desc.entries.clone()),
        native_bindings,
        sampler_mirrors,
        runtime_descriptor_count,
        runtime_dynamic_binding,
    })))
}

fn create_layout_binding(
    device: &VulkanDevice,
    entry: &api::BindGroupLayoutEntry,
) -> Result<vk::DescriptorSetLayoutBinding<'static>> {
    if entry
        .visibility
        .iter()
        .any(|s| matches!(s, api::ShaderStage::Mesh))
        && device.0.mesh.is_none()
    {
        return Err(Error::Unsupported("Vulkan mesh shader"));
    }
    if entry
        .visibility
        .iter()
        .any(|s| matches!(s, api::ShaderStage::Task))
        && !device.0.mesh.as_ref().is_some_and(|m| m.task)
    {
        return Err(Error::Unsupported("Vulkan task shader"));
    }
    if matches!(entry.ty, api::BindingType::AccelerationStructure)
        && !device.0.ray_support.acceleration
    {
        return Err(Error::Unsupported("Vulkan acceleration structure"));
    }
    if entry.partially_bound && !device.0.descriptor_partially_bound {
        return Err(Error::Unsupported("Vulkan partially-bound descriptors"));
    }
    if entry.nonuniform_indexing {
        let enabled = match &entry.ty {
            api::BindingType::UniformBuffer { .. } => device.0.nonuniform_uniform_buffer,
            api::BindingType::StorageBuffer { .. } => device.0.nonuniform_storage_buffer,
            api::BindingType::StorageTexture { .. } => device.0.nonuniform_storage_image,
            api::BindingType::SampledTexture { .. }
            | api::BindingType::Sampler { .. }
            | api::BindingType::AccelerationStructure => device.0.nonuniform_sampled_image,
            api::BindingType::VideoFrame => false,
        };
        if !enabled {
            return Err(Error::Unsupported("Vulkan nonuniform descriptor indexing"));
        }
    }
    let descriptor_count = match entry.count {
        api::BindingCount::One => 1,
        api::BindingCount::Fixed(count) => count,
        api::BindingCount::RuntimeSized { max_count } => {
            let dynamic = matches!(
                entry.ty,
                api::BindingType::UniformBuffer {
                    dynamic_offset: true,
                    ..
                } | api::BindingType::StorageBuffer {
                    dynamic_offset: true,
                    ..
                }
            );
            if !device.0.runtime_descriptor_array
                || (!dynamic && !device.0.variable_descriptor_count)
            {
                return Err(Error::Unsupported("Vulkan descriptor indexing"));
            }
            max_count
        }
    };
    if descriptor_count == 0 {
        return Err(Error::InvalidRange);
    }
    Ok(vk::DescriptorSetLayoutBinding::default()
        .binding(entry.binding)
        .descriptor_type(dtype(&entry.ty)?)
        .descriptor_count(descriptor_count)
        .stage_flags(stages(&entry.visibility)?))
}

fn ensure_runtime_array_layout_supported(
    device: &VulkanDevice,
    bindings: &[vk::DescriptorSetLayoutBinding<'_>],
    binding_flags: &mut vk::DescriptorSetLayoutBindingFlagsCreateInfo<'_>,
) -> Result<()> {
    let requested = bindings
        .last()
        .map(|binding| binding.descriptor_count)
        .ok_or(Error::InvalidState)?;
    let info = vk::DescriptorSetLayoutCreateInfo::default()
        .bindings(bindings)
        .push_next(binding_flags);
    let mut variable_support = vk::DescriptorSetVariableDescriptorCountLayoutSupport::default();
    let supported = {
        let mut support =
            vk::DescriptorSetLayoutSupport::default().push_next(&mut variable_support);
        // SAFETY: the create-info borrows live binding arrays and the support
        // output chain remains valid through this synchronous device query.
        unsafe {
            device
                .0
                .device
                .get_descriptor_set_layout_support(&info, &mut support)
        };
        support.supported == vk::TRUE
    };
    if !supported || variable_support.max_variable_descriptor_count < requested {
        return Err(Error::Unsupported("Vulkan runtime descriptor-array layout"));
    }
    Ok(())
}

fn dtype(ty: &api::BindingType) -> Result<vk::DescriptorType> {
    Ok(match ty {
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

fn stages(stages: &[api::ShaderStage]) -> Result<vk::ShaderStageFlags> {
    let mut flags = vk::ShaderStageFlags::empty();
    for stage in stages {
        flags |= match stage {
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
        };
    }
    if flags.is_empty() {
        return Err(Error::InvalidRange);
    }
    Ok(flags)
}
