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
    pub(super) entries: Arc<[api::BindGroupLayoutEntry]>,
    /// The public contract's layout-time upper bound for the sole runtime
    /// binding. Bind-group allocation validates its actual count against this.
    pub(super) runtime_descriptor_count: Option<u32>,
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
    let runtime_descriptor_count = desc.entries.iter().find_map(|entry| match entry.count {
        api::BindingCount::RuntimeSized { max_count } => Some(max_count),
        _ => None,
    });
    let bindings: Vec<_> = desc
        .entries
        .iter()
        .map(|entry| create_layout_binding(device, entry))
        .collect::<Result<_>>()?;
    let binding_flags: Vec<_> = desc
        .entries
        .iter()
        .map(|entry| match entry.count {
            api::BindingCount::RuntimeSized { .. } => {
                vk::DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT
            }
            _ => vk::DescriptorBindingFlags::empty(),
        })
        .collect();
    let mut flags =
        vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&binding_flags);
    if runtime_descriptor_count.is_some() {
        ensure_runtime_array_layout_supported(device, &bindings, &mut flags)?;
    }
    let info = vk::DescriptorSetLayoutCreateInfo::default()
        .bindings(&bindings)
        .push_next(&mut flags);
    // SAFETY: `bindings` and the matching binding-flag chain remain alive for
    // this synchronous call. DeviceState retains the native logical device.
    let layout =
        unsafe { device.0.device.create_descriptor_set_layout(&info, None) }.map_err(vk_error)?;
    Ok(VulkanBindGroupLayout(Arc::new(GroupLayout {
        device: device.0.clone(),
        layout,
        entries: Arc::from(desc.entries.clone()),
        runtime_descriptor_count,
    })))
}

fn create_layout_binding(
    device: &VulkanDevice,
    entry: &api::BindGroupLayoutEntry,
) -> Result<vk::DescriptorSetLayoutBinding<'static>> {
    if entry.partially_bound {
        return Err(Error::Unsupported("Vulkan partially-bound descriptors"));
    }
    if !matches!(entry.count, api::BindingCount::One)
        && !matches!(&entry.ty, api::BindingType::SampledTexture { .. })
    {
        return Err(Error::Unsupported("Vulkan descriptor array binding type"));
    }
    if entry.nonuniform_indexing && !matches!(&entry.ty, api::BindingType::SampledTexture { .. }) {
        return Err(Error::Unsupported("Vulkan nonuniform binding type"));
    }
    let descriptor_count = match entry.count {
        api::BindingCount::One => 1,
        api::BindingCount::Fixed(count) => count,
        api::BindingCount::RuntimeSized { max_count } => {
            if !device.0.descriptor_indexing {
                return Err(Error::Unsupported("Vulkan descriptor indexing"));
            }
            max_count
        }
    };
    if descriptor_count == 0 || descriptor_count > device.0.max_sampled_images {
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
        api::BindingType::StorageTexture { access, .. }
            if matches!(access, api::StorageAccess::Write) =>
        {
            vk::DescriptorType::STORAGE_IMAGE
        }
        api::BindingType::Sampler { .. } => vk::DescriptorType::SAMPLER,
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
            _ => return Err(Error::Unsupported("Vulkan descriptor shader stage")),
        };
    }
    if flags.is_empty() {
        return Err(Error::InvalidRange);
    }
    Ok(flags)
}
