//! Vulkan ray pipeline and shader group handles.
use super::super::{
    DeviceState, Vulkan, VulkanBindGroup, VulkanDevice, VulkanEncoder,
    pipeline::VulkanPipelineLayout, vk_error,
};
use ash::vk;
use fluxel_rhi_core::api::{self, Error, Result};
use std::{ffi::CString, sync::Arc};
#[derive(Clone)]
pub struct VulkanRayPipeline(pub(crate) Arc<PipelineState>);
pub(crate) struct PipelineState {
    device: Arc<DeviceState>,
    native: vk::Pipeline,
    layout: VulkanPipelineLayout,
    identifiers: Vec<Vec<u8>>,
}
impl Drop for PipelineState {
    fn drop(&mut self) {
        // SAFETY: final owner and submitted encoders retain the pipeline through completion.
        unsafe { self.device.device.destroy_pipeline(self.native, None) }
    }
}
fn stage(stage: api::ShaderStage) -> Result<vk::ShaderStageFlags> {
    Ok(match stage {
        api::ShaderStage::RayGeneration => vk::ShaderStageFlags::RAYGEN_KHR,
        api::ShaderStage::Miss => vk::ShaderStageFlags::MISS_KHR,
        api::ShaderStage::ClosestHit => vk::ShaderStageFlags::CLOSEST_HIT_KHR,
        api::ShaderStage::AnyHit => vk::ShaderStageFlags::ANY_HIT_KHR,
        api::ShaderStage::Intersection => vk::ShaderStageFlags::INTERSECTION_KHR,
        api::ShaderStage::Callable => vk::ShaderStageFlags::CALLABLE_KHR,
        _ => return Err(Error::Unsupported("Vulkan ray shader stage")),
    })
}
pub(super) fn create(
    device: &VulkanDevice,
    desc: &api::RayPipelineDesc<Vulkan>,
) -> Result<VulkanRayPipeline> {
    let loader = device
        .0
        .ray_loaders
        .as_ref()
        .and_then(|x| x.pipeline.as_ref())
        .ok_or(Error::Unsupported("Vulkan ray pipeline"))?;
    if desc.shaders.is_empty()
        || desc.groups.is_empty()
        || desc.max_recursion == 0
        || desc.max_recursion > device.0.ray_support.max_recursion_depth
        || desc.hit_attribute_bytes > device.0.ray_support.max_hit_attribute_size
        || desc.cache.is_some()
    {
        return Err(Error::InvalidRange);
    }
    let mut names = Vec::new();
    let mut shaders = Vec::new();
    let mut kinds = Vec::new();
    for shader in &desc.shaders {
        if !shader.constants.is_empty() {
            return Err(Error::Unsupported("Vulkan ray specialization"));
        }
        // SAFETY: Core validated the module's device identity.
        let module = unsafe { shader.module.native_ref() };
        let kind = module
            .entries()
            .iter()
            .find(|entry| entry.name == shader.entry)
            .ok_or(Error::InvalidState)?
            .stage;
        kinds.push(kind);
        shaders.push(module.clone());
        names.push(CString::new(shader.entry.as_str()).map_err(|_| Error::InvalidRange)?);
    }
    let stages: Vec<_> = shaders
        .iter()
        .zip(&names)
        .zip(&kinds)
        .map(|((shader, name), &kind)| {
            Ok(vk::PipelineShaderStageCreateInfo::default()
                .stage(stage(kind)?)
                .module(shader.module())
                .name(name))
        })
        .collect::<Result<_>>()?;
    let mut groups = Vec::new();
    for group in &desc.groups {
        let index = |v: Option<u32>, expected: &[api::ShaderStage]| -> Result<u32> {
            if let Some(i) = v {
                if !expected.contains(kinds.get(i as usize).ok_or(Error::InvalidRange)?) {
                    return Err(Error::InvalidState);
                }
                Ok(i)
            } else {
                Ok(vk::SHADER_UNUSED_KHR)
            }
        };
        if group
            .local_constants
            .is_some_and(|c| c.size_bytes == 0 || c.size_bytes % 4 != 0)
        {
            return Err(Error::InvalidRange);
        }
        let general = index(
            group.general,
            &[
                api::ShaderStage::RayGeneration,
                api::ShaderStage::Miss,
                api::ShaderStage::Callable,
            ],
        )?;
        let closest = index(group.closest_hit, &[api::ShaderStage::ClosestHit])?;
        let any = index(group.any_hit, &[api::ShaderStage::AnyHit])?;
        let intersection = index(group.intersection, &[api::ShaderStage::Intersection])?;
        let general_group = general != vk::SHADER_UNUSED_KHR;
        if general_group
            && (closest != vk::SHADER_UNUSED_KHR
                || any != vk::SHADER_UNUSED_KHR
                || intersection != vk::SHADER_UNUSED_KHR)
            || !general_group
                && closest == vk::SHADER_UNUSED_KHR
                && any == vk::SHADER_UNUSED_KHR
                && intersection == vk::SHADER_UNUSED_KHR
        {
            return Err(Error::InvalidState);
        }
        let ty = if general_group {
            vk::RayTracingShaderGroupTypeKHR::GENERAL
        } else if intersection != vk::SHADER_UNUSED_KHR {
            vk::RayTracingShaderGroupTypeKHR::PROCEDURAL_HIT_GROUP
        } else {
            vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP
        };
        groups.push(
            vk::RayTracingShaderGroupCreateInfoKHR::default()
                .ty(ty)
                .general_shader(general)
                .closest_hit_shader(closest)
                .any_hit_shader(any)
                .intersection_shader(intersection),
        );
    }
    // SAFETY: Core validated layout device identity.
    let layout = unsafe { desc.layout.native_ref() }.clone();
    let info = vk::RayTracingPipelineCreateInfoKHR::default()
        .stages(&stages)
        .groups(&groups)
        .max_pipeline_ray_recursion_depth(desc.max_recursion)
        .layout(layout.native());
    // SAFETY: all descriptor slices and shader modules remain live during synchronous creation.
    let native = match unsafe {
        loader.create_ray_tracing_pipelines(
            vk::DeferredOperationKHR::null(),
            vk::PipelineCache::null(),
            &[info],
            None,
        )
    } {
        Ok(p) => p[0],
        Err((pipelines, error)) => {
            for p in pipelines {
                if p != vk::Pipeline::null() {
                    unsafe { device.0.device.destroy_pipeline(p, None) };
                }
            }
            return Err(vk_error(error));
        }
    };
    let handle_size = device.0.ray_support.shader_group_handle_size as usize;
    let identifiers = match unsafe {
        loader.get_ray_tracing_shader_group_handles(
            native,
            0,
            groups.len() as u32,
            groups.len() * handle_size,
        )
    } {
        Ok(bytes) => bytes
            .chunks_exact(handle_size)
            .map(|x| x.to_vec())
            .collect(),
        Err(error) => {
            unsafe { device.0.device.destroy_pipeline(native, None) };
            return Err(vk_error(error));
        }
    };
    Ok(VulkanRayPipeline(Arc::new(PipelineState {
        device: device.0.clone(),
        native,
        layout,
        identifiers,
    })))
}
pub(super) fn identifier(p: &VulkanRayPipeline, index: u32) -> Result<Vec<u8>> {
    p.0.identifiers
        .get(index as usize)
        .cloned()
        .ok_or(Error::InvalidRange)
}
pub(super) fn bind(e: &mut VulkanEncoder, p: &VulkanRayPipeline) -> Result<()> {
    if e.raster || e.compute {
        return Err(Error::InvalidState);
    }
    // SAFETY: recording command buffer and same-device live pipeline.
    unsafe {
        e.device.device.cmd_bind_pipeline(
            e.command,
            vk::PipelineBindPoint::RAY_TRACING_KHR,
            p.0.native,
        )
    };
    e.ray_layout_state = Some(p.0.layout.clone());
    e.ray_pipelines.push(p.clone());
    Ok(())
}
pub(super) fn bind_group(
    e: &mut VulkanEncoder,
    slot: u32,
    group: &VulkanBindGroup,
    offsets: &[u32],
) -> Result<()> {
    if e.raster || e.compute {
        return Err(Error::InvalidState);
    }
    group.validate_dynamic_offsets(offsets)?;
    let layout = e.ray_layout_state.as_ref().ok_or(Error::InvalidState)?;
    layout.validate_group(slot, group)?;
    // SAFETY: group and layout compatibility and offsets are validated above.
    unsafe {
        e.device.device.cmd_bind_descriptor_sets(
            e.command,
            vk::PipelineBindPoint::RAY_TRACING_KHR,
            layout.native(),
            slot,
            &[group.native()],
            offsets,
        )
    };
    e.groups.push(group.clone());
    Ok(())
}
pub(super) fn trace(
    e: &mut VulkanEncoder,
    pipeline: &VulkanRayPipeline,
    table: &api::ShaderTable<Vulkan>,
    dims: [u32; 3],
) -> Result<()> {
    if e.raster
        || e.compute
        || e.ray_layout_state.is_none()
        || dims.contains(&0)
        || !e
            .ray_pipelines
            .last()
            .is_some_and(|bound| Arc::ptr_eq(&bound.0, &pipeline.0))
    {
        return Err(Error::InvalidState);
    }
    let count = u64::from(dims[0])
        .checked_mul(u64::from(dims[1]))
        .and_then(|n| n.checked_mul(u64::from(dims[2])))
        .ok_or(Error::InvalidRange)?;
    if count > u64::from(e.device.ray_support.max_dispatch_invocations) {
        return Err(Error::InvalidRange);
    }
    let properties = e.device.ray_support;
    if properties.shader_group_handle_size == 0
        || properties.shader_group_handle_alignment == 0
        || properties.shader_group_base_alignment == 0
    {
        return Err(Error::InvalidState);
    }
    let region = |region: Option<&api::ShaderTableRegion<Vulkan>>| -> Result<(
        vk::StridedDeviceAddressRegionKHR,
        Option<super::super::VulkanBuffer>,
    )> {
        let Some(region) = region else {
            return Ok((vk::StridedDeviceAddressRegionKHR::default(), None));
        };
        // SAFETY: Core checked the table buffer's device identity.
        let buffer = unsafe { region.buffer.native_ref() };
        if !buffer.0.usage.contains(api::BufferUsage::SHADER_TABLE)
            || region
                .offset
                .checked_add(region.size)
                .is_none_or(|end| end > buffer.0.size)
            || region.stride < u64::from(properties.shader_group_handle_size)
            || region.stride > u64::from(properties.max_shader_group_stride)
            || region.stride % u64::from(properties.shader_group_handle_alignment) != 0
        {
            return Err(Error::InvalidRange);
        }
        let address = buffer
            .device_address()?
            .checked_add(region.offset)
            .ok_or(Error::InvalidRange)?;
        if address % u64::from(properties.shader_group_base_alignment) != 0 {
            return Err(Error::InvalidRange);
        }
        Ok((
            vk::StridedDeviceAddressRegionKHR {
                device_address: address,
                size: region.size,
                stride: region.stride,
            },
            Some(buffer.clone()),
        ))
    };
    let (raygen, a) = region(Some(&table.raygen))?;
    let (miss, b) = region(table.miss.as_ref())?;
    let (hit, c) = region(table.hit.as_ref())?;
    let (callable, d) = region(table.callable.as_ref())?;
    let loader = e
        .device
        .ray_loaders
        .as_ref()
        .and_then(|x| x.pipeline.as_ref())
        .ok_or(Error::Unsupported("Vulkan ray pipeline"))?;
    // SAFETY: every address, size, stride, and alignment was checked and buffers
    // are retained below until the command buffer's completion.
    unsafe {
        loader.cmd_trace_rays(
            e.command, &raygen, &miss, &hit, &callable, dims[0], dims[1], dims[2],
        )
    };
    e.buffers.extend([a, b, c, d].into_iter().flatten());
    e.ray_pipelines.push(pipeline.clone());
    Ok(())
}
