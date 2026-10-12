//! Native Vulkan compute-pipeline creation.

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
        unsafe { self.device.device.destroy_pipeline(self.pipeline, None) }
    }
}

pub(crate) fn create_compute(
    device: &VulkanDevice,
    desc: &api::ComputePipelineDesc<super::super::Vulkan>,
) -> Result<VulkanComputePipeline> {
    if !desc.compute.constants.is_empty()
        || desc.required_subgroup_size.is_some()
        || desc.cache.is_some()
    {
        return Err(Error::Unsupported("Vulkan compute pipeline option"));
    }
    let shader: &VulkanShader = unsafe { desc.compute.module.native_ref() };
    let entry = CString::new(desc.compute.entry.as_str())
        .map_err(|_| Error::Shader("shader entry contains NUL".into()))?;
    let layout = unsafe { desc.layout.native_ref() }.clone();
    let stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(shader.module())
        .name(&entry);
    let info = vk::ComputePipelineCreateInfo::default()
        .stage(stage)
        .layout(layout.native());
    let pipeline = unsafe {
        device
            .0
            .device
            .create_compute_pipelines(vk::PipelineCache::null(), &[info], None)
    }
    .map_err(|(_, error)| vk_error(error))?
    .into_iter()
    .next()
    .ok_or_else(|| Error::Backend("Vulkan returned no compute pipeline".into()))?;
    Ok(VulkanComputePipeline(Arc::new(Compute {
        device: device.0.clone(),
        pipeline,
        layout,
    })))
}
