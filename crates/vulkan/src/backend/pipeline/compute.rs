//! Native Vulkan compute-pipeline creation.
//!
//! The created pipeline retains its layout. Shader modules need only survive
//! `vkCreateComputePipelines`, while an encoder retains the resulting pipeline
//! through submission completion.

use super::super::{DeviceState, VulkanDevice, VulkanShader, vk_error};
use super::layout::VulkanPipelineLayout;
use ash::vk;
use fluxel_rhi_core::api::{self, Error, Result};
use std::{ffi::CString, sync::Arc};

/// A compute pipeline and the layout whose native handles it uses.
#[derive(Clone)]
pub struct VulkanComputePipeline(Arc<Compute>);
impl VulkanComputePipeline {
    pub(crate) fn native(&self) -> vk::Pipeline {
        self.0.pipeline
    }
    pub(crate) fn layout(&self) -> vk::PipelineLayout {
        self.0.layout.native()
    }
    pub(crate) fn layout_state(&self) -> VulkanPipelineLayout {
        self.0.layout.clone()
    }
}

struct Compute {
    device: Arc<DeviceState>,
    pipeline: vk::Pipeline,
    // The pipeline layout owns descriptor-set layouts and push ranges.
    layout: VulkanPipelineLayout,
}
impl Drop for Compute {
    fn drop(&mut self) {
        // SAFETY: encoders retain the Compute Arc through fence retirement, so
        // no in-flight command buffer can still reference this pipeline.
        unsafe { self.device.device.destroy_pipeline(self.pipeline, None) }
    }
}

pub(crate) fn create_compute(
    device: &VulkanDevice,
    desc: &api::ComputePipelineDesc<super::super::Vulkan>,
) -> Result<VulkanComputePipeline> {
    if let Some(size) = desc.required_subgroup_size {
        if !device.0.subgroup_size_control
            || !device
                .0
                .required_subgroup_size_stages
                .contains(vk::ShaderStageFlags::COMPUTE)
        {
            return Err(Error::Unsupported("Vulkan compute subgroup size control"));
        }
        if !size.is_power_of_two()
            || size < device.0.min_subgroup_size
            || size > device.0.max_subgroup_size
        {
            return Err(Error::InvalidRange);
        }
    }
    // SAFETY: `ComputeDomain::create_pipeline` checked the shader and layout
    // DeviceIds against `device`; this borrow survives pipeline creation only.
    let source: &VulkanShader = unsafe { desc.compute.module.native_ref() };
    let (shader, _consumed) = super::super::shader::compile_variant(
        device,
        source,
        unsafe { desc.layout.native_ref() },
        &desc.compute.entry,
        api::ShaderStage::Compute,
        &desc.compute.constants,
        unsafe { desc.layout.native_ref() }
            .push_constants()
            .base(api::ShaderStage::Compute),
    )?;
    let layout = unsafe { desc.layout.native_ref() }.for_shaders(&[(
        &shader,
        &desc.compute.entry,
        api::ShaderStage::Compute,
    )])?;
    let (shader, consumed) = super::super::shader::compile_variant(
        device,
        source,
        &layout,
        &desc.compute.entry,
        api::ShaderStage::Compute,
        &desc.compute.constants,
        layout.push_constants().base(api::ShaderStage::Compute),
    )?;
    if consumed {
        validate_generated_local_size(device, shader.words(), &desc.compute.entry)?;
    }
    let entry = CString::new(desc.compute.entry.as_str())
        .map_err(|_| Error::Shader("shader entry contains NUL".into()))?;
    let specialization = super::specialization::Specialization::new(
        &shader,
        if consumed {
            &[]
        } else {
            &desc.compute.constants
        },
    )?;
    let specialization_info = specialization.info();
    // SAFETY: the same Core validation checked this layout's DeviceId. Cloning
    // it retains descriptor layouts for the lifetime of the native pipeline.
    let mut subgroup = vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default()
        .required_subgroup_size(desc.required_subgroup_size.unwrap_or(0));
    let mut stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(shader.module())
        .name(&entry)
        .specialization_info(&specialization_info);
    if desc.required_subgroup_size.is_some() {
        stage = stage.push_next(&mut subgroup);
    }
    let info = vk::ComputePipelineCreateInfo::default()
        .stage(stage)
        .layout(layout.native());
    // SAFETY: `stage`, `entry`, and `layout` remain live throughout the call;
    // all native handles belong to the live DeviceState.
    let created = super::cache::with_cache(desc.cache.as_ref(), |cache| unsafe {
        device
            .0
            .device
            .create_compute_pipelines(cache, &[info], None)
    });
    let pipelines = match created {
        Ok(pipelines) => pipelines,
        Err((pipelines, error)) => {
            for pipeline in pipelines {
                if pipeline != vk::Pipeline::null() {
                    // SAFETY: these partial results belong to this device and
                    // never escaped the failed creation call.
                    unsafe { device.0.device.destroy_pipeline(pipeline, None) };
                }
            }
            return Err(vk_error(error));
        }
    };
    let pipeline = pipelines
        .into_iter()
        .next()
        .ok_or_else(|| Error::Backend("Vulkan returned no compute pipeline".into()))?;
    Ok(VulkanComputePipeline(Arc::new(Compute {
        device: device.0.clone(),
        pipeline,
        layout,
    })))
}

// Naga has already applied WGSL pipeline overrides and emits literal LocalSize.
// Native SPIR-V can use LocalSizeId and arbitrary specialization expressions,
// so it cannot be checked by this narrowly scoped parser.
fn validate_generated_local_size(device: &VulkanDevice, words: &[u32], entry: &str) -> Result<()> {
    let mut selected = None;
    let mut modes = Vec::new();
    let mut offset = 5;
    while offset < words.len() {
        let count = (words[offset] >> 16) as usize;
        if count == 0 || offset + count > words.len() {
            return Err(Error::Shader("invalid SPIR-V instruction".into()));
        }
        let inst = &words[offset..offset + count];
        match inst[0] & 0xffff {
            15 if count >= 4 && inst[1] == 5 => {
                let name = inst[3..]
                    .iter()
                    .flat_map(|w| w.to_le_bytes())
                    .take_while(|&b| b != 0)
                    .collect::<Vec<_>>();
                if name == entry.as_bytes() {
                    selected = Some(inst[2]);
                }
            }
            16 if count >= 6 && inst[2] == 17 => modes.push((inst[1], [inst[3], inst[4], inst[5]])),
            _ => {}
        }
        offset += count;
    }
    let size = modes
        .into_iter()
        .find_map(|(id, size)| (Some(id) == selected).then_some(size))
        .ok_or_else(|| Error::Shader("missing generated compute local size".into()))?;
    let invocations = size
        .iter()
        .try_fold(1u64, |n, &v| n.checked_mul(u64::from(v)))
        .ok_or(Error::InvalidRange)?;
    if size.contains(&0)
        || size
            .iter()
            .zip(device.0.max_compute_workgroup_size)
            .any(|(&v, max)| v > max)
        || invocations > u64::from(device.0.max_compute_workgroup_invocations)
    {
        return Err(Error::InvalidRange);
    }
    Ok(())
}
