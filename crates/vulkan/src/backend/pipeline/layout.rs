//! Vulkan pipeline layouts and push-constant range validation.
//!
//! A pipeline layout retains its descriptor-set layouts and validated push-range
//! contract. It can therefore validate later bind/push commands without a
//! global descriptor or resource-state tracker.

use super::super::{DeviceState, VulkanDevice, vk_error};
use super::binding::{VulkanBindGroup, VulkanBindGroupLayout};
use ash::vk;
use fluxel_rhi_core::api::{self, Error, Result};
use std::{ops::Range, sync::Arc};

#[derive(Clone)]
pub(crate) struct PushConstants(Arc<[PushRange]>);
#[derive(Clone)]
struct PushRange {
    stages: api::StageMask,
    range: Range<u32>,
}
impl PushConstants {
    pub(crate) fn validate(
        &self,
        stages: api::StageMask,
        offset: u32,
        bytes: &[u8],
    ) -> Result<vk::ShaderStageFlags> {
        let size = u32::try_from(bytes.len()).map_err(|_| Error::InvalidRange)?;
        let end = offset.checked_add(size).ok_or(Error::InvalidRange)?;
        if stages.0 == 0 || bytes.is_empty() || offset % 4 != 0 || size % 4 != 0 {
            return Err(Error::InvalidRange);
        }
        let native = push_stages(stages)?;
        if self.0.iter().any(|declared| {
            stages.0 & !declared.stages.0 == 0
                && offset >= declared.range.start
                && end <= declared.range.end
        }) {
            Ok(native)
        } else {
            Err(Error::InvalidRange)
        }
    }
}

#[derive(Clone)]
pub struct VulkanPipelineLayout(pub(crate) Arc<Layout>);
pub(crate) struct Layout {
    device: Arc<DeviceState>,
    pub(crate) layout: vk::PipelineLayout,
    // A pipeline layout references every descriptor-set layout used to create it.
    _groups: Arc<[VulkanBindGroupLayout]>,
    push_constants: PushConstants,
}
impl VulkanPipelineLayout {
    pub(crate) fn native(&self) -> vk::PipelineLayout {
        self.0.layout
    }
    pub(crate) fn push_constants(&self) -> PushConstants {
        self.0.push_constants.clone()
    }
    pub(crate) fn validate_group(&self, slot: u32, group: &VulkanBindGroup) -> Result<()> {
        if self
            .0
            ._groups
            .get(slot as usize)
            .is_none_or(|layout| layout.native() != group.layout())
        {
            return Err(Error::InvalidState);
        }
        Ok(())
    }
}
impl Drop for Layout {
    fn drop(&mut self) {
        // SAFETY: recorded pipelines retain this layout until fence retirement;
        // this final Arc still owns the live DeviceState used for destruction.
        unsafe {
            self.device
                .device
                .destroy_pipeline_layout(self.layout, None)
        }
    }
}

pub(crate) fn create_layout(
    device: &VulkanDevice,
    desc: &api::PipelineLayoutDesc<'_, super::super::Vulkan>,
) -> Result<VulkanPipelineLayout> {
    // SAFETY: DeviceState retains the VkInstance owning `physical`; querying
    // immutable physical-device properties has no resource lifetime effect.
    let limits = unsafe {
        device
            .0
            .instance
            .instance
            .get_physical_device_properties(device.0.physical)
    }
    .limits;
    let native_ranges: Result<Vec<_>> = desc
        .push_constants
        .iter()
        .map(|range| {
            let size = range
                .range
                .end
                .checked_sub(range.range.start)
                .ok_or(Error::InvalidRange)?;
            if size == 0
                || range.range.start % 4 != 0
                || size % 4 != 0
                || range.range.end > limits.max_push_constants_size
            {
                return Err(Error::InvalidRange);
            }
            Ok(vk::PushConstantRange::default()
                .stage_flags(push_stages(shader_stages(&range.stages)?)?)
                .offset(range.range.start)
                .size(size))
        })
        .collect();
    let native_ranges = native_ranges?;
    let push_ranges: Arc<[_]> = desc
        .push_constants
        .iter()
        .map(|range| {
            Ok(PushRange {
                stages: shader_stages(&range.stages)?,
                range: range.range.clone(),
            })
        })
        .collect::<Result<_>>()?;
    let groups: Arc<[_]> = desc
        .groups
        .iter()
        .map(|group| {
            // SAFETY: `Device::create_pipeline_layout` checked every group
            // DeviceId against `device` before entering this backend function.
            unsafe { group.native_ref() }.clone()
        })
        .collect();
    let sets: Vec<_> = groups.iter().map(VulkanBindGroupLayout::native).collect();
    let info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(&sets)
        .push_constant_ranges(&native_ranges);
    // SAFETY: the set-layout and push-range slices outlive this call, and all
    // handles belong to the live DeviceState retained by the resulting Layout.
    let layout =
        unsafe { device.0.device.create_pipeline_layout(&info, None) }.map_err(vk_error)?;
    Ok(VulkanPipelineLayout(Arc::new(Layout {
        device: device.0.clone(),
        layout,
        _groups: groups,
        push_constants: PushConstants(push_ranges),
    })))
}

fn shader_stages(stages: &[api::ShaderStage]) -> Result<api::StageMask> {
    let mut mask = 0;
    for stage in stages {
        mask |= match stage {
            api::ShaderStage::Vertex => api::StageMask::VERTEX.0,
            api::ShaderStage::Fragment => api::StageMask::FRAGMENT.0,
            api::ShaderStage::Compute => api::StageMask::COMPUTE.0,
            _ => return Err(Error::Unsupported("Vulkan push-constant shader stage")),
        };
    }
    if mask == 0 {
        return Err(Error::InvalidRange);
    }
    Ok(api::StageMask(mask))
}
fn push_stages(stages: api::StageMask) -> Result<vk::ShaderStageFlags> {
    let supported =
        api::StageMask::VERTEX.0 | api::StageMask::FRAGMENT.0 | api::StageMask::COMPUTE.0;
    if stages.0 == 0 || stages.0 & !supported != 0 {
        return Err(Error::Unsupported("Vulkan push-constant stage mask"));
    }
    let mut native = vk::ShaderStageFlags::empty();
    if stages.0 & api::StageMask::VERTEX.0 != 0 {
        native |= vk::ShaderStageFlags::VERTEX;
    }
    if stages.0 & api::StageMask::FRAGMENT.0 != 0 {
        native |= vk::ShaderStageFlags::FRAGMENT;
    }
    if stages.0 & api::StageMask::COMPUTE.0 != 0 {
        native |= vk::ShaderStageFlags::COMPUTE;
    }
    Ok(native)
}
