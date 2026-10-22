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
pub(crate) struct PushConstants {
    ranges: Arc<[PushRange]>,
    stages: Arc<[PushStage]>,
}
#[derive(Clone)]
struct PushRange {
    stages: api::StageMask,
    range: Range<u32>,
}
#[derive(Clone)]
struct PushStage {
    mask: api::StageMask,
    native: vk::ShaderStageFlags,
    base: u32,
}
impl PushConstants {
    pub(crate) fn base(&self, stage: api::ShaderStage) -> u32 {
        let mask = shader_stages(&[stage]).expect("single shader stage");
        self.stages
            .iter()
            .find(|v| v.mask.0 == mask.0)
            .map_or(0, |v| v.base)
    }
    pub(crate) fn validate(
        &self,
        stages: api::StageMask,
        offset: u32,
        bytes: &[u8],
    ) -> Result<Vec<(vk::ShaderStageFlags, u32)>> {
        let size = u32::try_from(bytes.len()).map_err(|_| Error::InvalidRange)?;
        let end = offset.checked_add(size).ok_or(Error::InvalidRange)?;
        if stages.0 == 0 || bytes.is_empty() || offset % 4 != 0 || size % 4 != 0 {
            return Err(Error::InvalidRange);
        }
        push_stages(stages)?;
        let mut out = Vec::new();
        let mut matched = 0;
        for stage in self
            .stages
            .iter()
            .filter(|stage| stages.0 & stage.mask.0 != 0)
        {
            let mut cursor = offset;
            while cursor < end {
                let next = self
                    .ranges
                    .iter()
                    .filter(|r| {
                        r.stages.0 & stage.mask.0 != 0
                            && r.range.start <= cursor
                            && r.range.end > cursor
                    })
                    .map(|r| r.range.end)
                    .max()
                    .ok_or(Error::InvalidRange)?;
                cursor = next.min(end);
            }
            out.push((
                stage.native,
                stage.base.checked_add(offset).ok_or(Error::InvalidRange)?,
            ));
            matched |= stage.mask.0;
        }
        if matched != stages.0 {
            return Err(Error::InvalidRange);
        }
        Ok(out)
    }
}

#[derive(Clone)]
pub struct VulkanPipelineLayout(pub(crate) Arc<Layout>);
pub(crate) struct Layout {
    device: Arc<DeviceState>,
    pub(crate) layout: vk::PipelineLayout,
    // A pipeline layout references every descriptor-set layout used to create it.
    _groups: Arc<[VulkanBindGroupLayout]>,
    public_groups: Arc<[VulkanBindGroupLayout]>,
    sampler_modes: Vec<std::collections::HashMap<u32, u8>>,
    native_push_ranges: Arc<[vk::PushConstantRange]>,
    push_constants: PushConstants,
}
impl VulkanPipelineLayout {
    pub(crate) fn for_shaders(
        &self,
        shaders: &[(&super::super::VulkanShader, &str, api::ShaderStage)],
    ) -> Result<Self> {
        let mut modes = vec![std::collections::HashMap::new(); self.0.public_groups.len()];
        for (shader, entry, stage) in shaders {
            super::super::shader::sampler_usage(shader, entry, *stage, &mut modes)?;
        }
        for group in &mut modes {
            for mode in group.values_mut() {
                *mode = match *mode {
                    1 => 0,
                    2 => 1,
                    3 => 2,
                    _ => return Err(Error::InvalidState),
                };
            }
        }
        self.sampler_variant(modes)
    }
    pub(crate) fn descriptor_binding_maps(
        &self,
    ) -> Vec<(
        std::collections::HashMap<u32, u32>,
        std::collections::HashMap<u32, u32>,
    )> {
        self.0
            ._groups
            .iter()
            .map(|group| {
                (
                    group.0.native_bindings.clone(),
                    group.0.sampler_mirrors.clone(),
                )
            })
            .collect()
    }
    pub(crate) fn native(&self) -> vk::PipelineLayout {
        self.0.layout
    }
    pub(crate) fn push_constants(&self) -> PushConstants {
        self.0.push_constants.clone()
    }
    pub(crate) fn native_group(
        &self,
        slot: u32,
        group: &VulkanBindGroup,
    ) -> Result<vk::DescriptorSet> {
        self.validate_group(slot, group)?;
        let index = slot as usize;
        group.native_variant(&self.0._groups[index], &self.0.sampler_modes[index])
    }
    pub(crate) fn sampler_variant(
        &self,
        modes: Vec<std::collections::HashMap<u32, u8>>,
    ) -> Result<Self> {
        if modes.iter().all(|m| m.values().all(|v| *v == 0)) {
            return Ok(self.clone());
        }
        if modes.len() != self.0.public_groups.len() {
            return Err(Error::InvalidRange);
        }
        let device = VulkanDevice(self.0.device.clone());
        let mut groups = Vec::with_capacity(modes.len());
        for (group, mode) in self.0.public_groups.iter().zip(&modes) {
            if mode.values().any(|v| *v == 2) {
                groups.push(super::binding::create_group_layout_variant(
                    &device,
                    &api::BindGroupLayoutDesc {
                        label: None,
                        entries: group.0.entries.to_vec(),
                    },
                    mode,
                )?);
            } else {
                groups.push(group.clone());
            }
        }
        let limits = unsafe {
            device
                .0
                .instance
                .instance
                .get_physical_device_properties(device.0.physical)
        }
        .limits;
        validate_sampler_limits(&groups, &limits)?;
        let sets: Vec<_> = groups.iter().map(VulkanBindGroupLayout::native).collect();
        let info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&sets)
            .push_constant_ranges(&self.0.native_push_ranges);
        // SAFETY: descriptor layouts and push ranges remain alive for creation and ownership.
        let layout =
            unsafe { device.0.device.create_pipeline_layout(&info, None) }.map_err(vk_error)?;
        Ok(Self(Arc::new(Layout {
            device: device.0.clone(),
            layout,
            _groups: Arc::from(groups),
            public_groups: self.0.public_groups.clone(),
            sampler_modes: modes,
            native_push_ranges: self.0.native_push_ranges.clone(),
            push_constants: self.0.push_constants.clone(),
        })))
    }
    pub(crate) fn validate_group(&self, slot: u32, group: &VulkanBindGroup) -> Result<()> {
        if self
            .0
            .public_groups
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
    if desc
        .push_constants
        .iter()
        .any(|r| r.stages.iter().any(|s| matches!(s, api::ShaderStage::Mesh)))
        && device.0.mesh.is_none()
    {
        return Err(Error::Unsupported("Vulkan mesh shader"));
    }
    if desc
        .push_constants
        .iter()
        .any(|r| r.stages.iter().any(|s| matches!(s, api::ShaderStage::Task)))
        && !device.0.mesh.as_ref().is_some_and(|m| m.task)
    {
        return Err(Error::Unsupported("Vulkan task shader"));
    }
    if desc.push_constants.iter().any(|r| {
        r.stages.iter().any(|s| {
            matches!(
                s,
                api::ShaderStage::RayGeneration
                    | api::ShaderStage::AnyHit
                    | api::ShaderStage::ClosestHit
                    | api::ShaderStage::Miss
                    | api::ShaderStage::Intersection
                    | api::ShaderStage::Callable
            )
        })
    }) && !device.0.ray_support.pipeline
    {
        return Err(Error::Unsupported("Vulkan ray pipeline"));
    }
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
    let push_ranges: Arc<[_]> = desc
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
            Ok(PushRange {
                stages: shader_stages(&range.stages)?,
                range: range.range.clone(),
            })
        })
        .collect::<Result<_>>()?;
    let masks = [
        api::StageMask::VERTEX,
        api::StageMask::FRAGMENT,
        api::StageMask::COMPUTE,
        api::StageMask::TASK,
        api::StageMask::MESH,
        api::StageMask::RAY,
    ];
    let mut merged = Vec::new();
    let mut stages = Vec::new();
    let mut cursor = 0u32;
    for mask in masks {
        let visible: Vec<_> = push_ranges
            .iter()
            .filter(|r| r.stages.0 & mask.0 != 0)
            .collect();
        if visible.is_empty() {
            continue;
        }
        let start = visible.iter().map(|r| r.range.start).min().unwrap();
        let end = visible.iter().map(|r| r.range.end).max().unwrap();
        let base = cursor.saturating_sub(start);
        let native_end = base.checked_add(end).ok_or(Error::InvalidRange)?;
        if native_end > limits.max_push_constants_size {
            return Err(Error::Unsupported("Vulkan push constant size limit"));
        }
        let native = push_stages(mask)?;
        merged.push(
            vk::PushConstantRange::default()
                .stage_flags(native)
                .offset(base + start)
                .size(end - start),
        );
        stages.push(PushStage { mask, native, base });
        cursor = native_end;
    }
    let push_constants = PushConstants {
        ranges: push_ranges,
        stages: Arc::from(stages),
    };
    let groups: Arc<[_]> = desc
        .groups
        .iter()
        .map(|group| {
            // SAFETY: `Device::create_pipeline_layout` checked every group
            // DeviceId against `device` before entering this backend function.
            unsafe { group.native_ref() }.clone()
        })
        .collect();
    validate_sampler_limits(&groups, &limits)?;
    let sets: Vec<_> = groups.iter().map(VulkanBindGroupLayout::native).collect();
    let info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(&sets)
        .push_constant_ranges(&merged);
    // SAFETY: the set-layout and push-range slices outlive this call, and all
    // handles belong to the live DeviceState retained by the resulting Layout.
    let layout =
        unsafe { device.0.device.create_pipeline_layout(&info, None) }.map_err(vk_error)?;
    Ok(VulkanPipelineLayout(Arc::new(Layout {
        device: device.0.clone(),
        layout,
        public_groups: groups.clone(),
        sampler_modes: vec![std::collections::HashMap::new(); groups.len()],
        native_push_ranges: Arc::from(merged),
        _groups: groups,
        push_constants,
    })))
}

fn validate_sampler_limits(
    groups: &[VulkanBindGroupLayout],
    limits: &vk::PhysicalDeviceLimits,
) -> Result<()> {
    // The set-layout support query does not check aggregate pipeline limits.
    // Only sampler bindings with a private integer mirror count twice.
    let stages = [
        api::ShaderStage::Vertex,
        api::ShaderStage::Fragment,
        api::ShaderStage::Compute,
        api::ShaderStage::Task,
        api::ShaderStage::Mesh,
        api::ShaderStage::RayGeneration,
        api::ShaderStage::AnyHit,
        api::ShaderStage::ClosestHit,
        api::ShaderStage::Miss,
        api::ShaderStage::Intersection,
        api::ShaderStage::Callable,
    ];
    let mut total_samplers = 0u64;
    let mut per_stage_samplers = [0u64; 11];
    for group in groups.iter() {
        for entry in group.0.entries.iter() {
            if !matches!(entry.ty, api::BindingType::Sampler { .. }) {
                continue;
            }
            let count = sampler_descriptor_count(
                entry.count,
                group.0.sampler_mirrors.contains_key(&entry.binding),
            );
            total_samplers += count;
            for (index, stage) in stages.iter().enumerate() {
                if entry.visibility.contains(stage) {
                    per_stage_samplers[index] += count;
                }
            }
        }
    }
    check_sampler_limits(total_samplers, &per_stage_samplers, limits)
}

fn sampler_descriptor_count(count: api::BindingCount, mirrored: bool) -> u64 {
    let count = match count {
        api::BindingCount::One => 1,
        api::BindingCount::Fixed(n) | api::BindingCount::RuntimeSized { max_count: n } => {
            u64::from(n)
        }
    };
    count * if mirrored { 2 } else { 1 }
}

fn check_sampler_limits(
    total: u64,
    stages: &[u64],
    limits: &vk::PhysicalDeviceLimits,
) -> Result<()> {
    if total > u64::from(limits.max_descriptor_set_samplers)
        || stages
            .iter()
            .any(|n| *n > u64::from(limits.max_per_stage_descriptor_samplers))
    {
        return Err(Error::Unsupported(
            "Vulkan pipeline sampler descriptor limit",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod sampler_budget_tests {
    use super::*;
    #[test]
    fn ordinary_and_integer_bindings_reach_native_limit() {
        let limits = vk::PhysicalDeviceLimits {
            max_descriptor_set_samplers: 8,
            max_per_stage_descriptor_samplers: 8,
            ..Default::default()
        };
        let ordinary = sampler_descriptor_count(api::BindingCount::Fixed(8), false);
        assert!(check_sampler_limits(ordinary, &[ordinary], &limits).is_ok());
        let mixed = sampler_descriptor_count(api::BindingCount::Fixed(8), true);
        assert!(check_sampler_limits(mixed, &[mixed], &limits).is_err());
    }
}

fn shader_stages(stages: &[api::ShaderStage]) -> Result<api::StageMask> {
    let mut mask = 0;
    for stage in stages {
        mask |= match stage {
            api::ShaderStage::Vertex => api::StageMask::VERTEX.0,
            api::ShaderStage::Fragment => api::StageMask::FRAGMENT.0,
            api::ShaderStage::Compute => api::StageMask::COMPUTE.0,
            api::ShaderStage::Task => api::StageMask::TASK.0,
            api::ShaderStage::Mesh => api::StageMask::MESH.0,
            api::ShaderStage::RayGeneration
            | api::ShaderStage::AnyHit
            | api::ShaderStage::ClosestHit
            | api::ShaderStage::Miss
            | api::ShaderStage::Intersection
            | api::ShaderStage::Callable => api::StageMask::RAY.0,
        };
    }
    if mask == 0 {
        return Err(Error::InvalidRange);
    }
    Ok(api::StageMask(mask))
}
fn push_stages(stages: api::StageMask) -> Result<vk::ShaderStageFlags> {
    let supported = api::StageMask::VERTEX.0
        | api::StageMask::FRAGMENT.0
        | api::StageMask::COMPUTE.0
        | api::StageMask::TASK.0
        | api::StageMask::MESH.0
        | api::StageMask::RAY.0;
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
    if stages.0 & api::StageMask::TASK.0 != 0 {
        native |= vk::ShaderStageFlags::TASK_EXT;
    }
    if stages.0 & api::StageMask::MESH.0 != 0 {
        native |= vk::ShaderStageFlags::MESH_EXT;
    }
    if stages.0 & api::StageMask::RAY.0 != 0 {
        native |= vk::ShaderStageFlags::RAYGEN_KHR
            | vk::ShaderStageFlags::ANY_HIT_KHR
            | vk::ShaderStageFlags::CLOSEST_HIT_KHR
            | vk::ShaderStageFlags::MISS_KHR
            | vk::ShaderStageFlags::INTERSECTION_KHR
            | vk::ShaderStageFlags::CALLABLE_KHR;
    }
    Ok(native)
}
