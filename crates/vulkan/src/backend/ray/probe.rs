//! Device feature negotiation for the optional Vulkan ray routes.
//!
//! Native availability is kept separate from Core capability exposure: the
//! latter is enabled only when every required command and lifetime path exists.
use ash::{Device, Instance, vk};
use fluxel_rhi_core::api::{Error, Result};
use std::ffi::{CStr, c_char};

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct RaySupport {
    pub acceleration: bool,
    pub pipeline: bool,
    pub query: bool,
    pub shader_group_handle_size: u32,
    pub shader_group_handle_alignment: u32,
    pub shader_group_base_alignment: u32,
    pub max_shader_group_stride: u32,
    pub max_recursion_depth: u32,
    pub max_hit_attribute_size: u32,
    pub max_dispatch_invocations: u32,
    pub scratch_alignment: u32,
    pub max_geometry_count: u64,
    pub max_primitive_count: u64,
    pub max_instance_count: u64,
}

impl RaySupport {
    /// Extension names for a Vulkan 1.3 device; buffer device address and
    /// SPIR-V 1.4 dependencies are core since Vulkan 1.2.
    pub(crate) fn extensions(&self) -> Vec<*const c_char> {
        let mut names = Vec::new();
        if self.acceleration {
            names.push(ash::khr::deferred_host_operations::NAME.as_ptr());
            names.push(ash::khr::acceleration_structure::NAME.as_ptr());
        }
        if self.pipeline {
            names.push(ash::khr::ray_tracing_pipeline::NAME.as_ptr());
        }
        if self.query {
            names.push(ash::khr::ray_query::NAME.as_ptr());
        }
        names
    }
}

pub(crate) fn probe(instance: &Instance, physical: vk::PhysicalDevice) -> Result<RaySupport> {
    // SAFETY: physical originates from this live instance and Vulkan returns
    // a copied extension-properties vector.
    let extensions = unsafe { instance.enumerate_device_extension_properties(physical) }
        .map_err(|e| Error::Backend(format!("Vulkan extension query: {e:?}")))?;
    let has = |wanted: &CStr| {
        extensions.iter().any(|extension| {
            // SAFETY: Vulkan guarantees extension_name is NUL terminated.
            unsafe { CStr::from_ptr(extension.extension_name.as_ptr()) == wanted }
        })
    };
    let has_acceleration = has(ash::khr::acceleration_structure::NAME)
        && has(ash::khr::deferred_host_operations::NAME);
    if !has_acceleration {
        return Ok(RaySupport::default());
    }
    let has_pipeline = has(ash::khr::ray_tracing_pipeline::NAME);
    let has_query = has(ash::khr::ray_query::NAME);
    let mut core12 = vk::PhysicalDeviceVulkan12Features::default();
    let mut accel = vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default();
    let mut pipeline = vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::default();
    let mut query = vk::PhysicalDeviceRayQueryFeaturesKHR::default();
    let mut features = vk::PhysicalDeviceFeatures2::default()
        .push_next(&mut core12)
        .push_next(&mut accel);
    if has_pipeline {
        features = features.push_next(&mut pipeline);
    }
    if has_query {
        features = features.push_next(&mut query);
    }
    // SAFETY: every chained output node remains valid throughout this call.
    unsafe { instance.get_physical_device_features2(physical, &mut features) };
    let acceleration =
        core12.buffer_device_address == vk::TRUE && accel.acceleration_structure == vk::TRUE;
    if !acceleration {
        return Ok(RaySupport::default());
    }
    let mut accel_properties = vk::PhysicalDeviceAccelerationStructurePropertiesKHR::default();
    let mut pipeline_properties = vk::PhysicalDeviceRayTracingPipelinePropertiesKHR::default();
    let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut accel_properties);
    if has_pipeline {
        properties = properties.push_next(&mut pipeline_properties);
    }
    // SAFETY: each output node belongs to the supported extension and remains live.
    unsafe { instance.get_physical_device_properties2(physical, &mut properties) };
    Ok(RaySupport {
        acceleration,
        pipeline: has_pipeline && pipeline.ray_tracing_pipeline == vk::TRUE,
        query: has_query && query.ray_query == vk::TRUE,
        shader_group_handle_size: pipeline_properties.shader_group_handle_size,
        shader_group_handle_alignment: pipeline_properties.shader_group_handle_alignment,
        shader_group_base_alignment: pipeline_properties.shader_group_base_alignment,
        max_shader_group_stride: pipeline_properties.max_shader_group_stride,
        max_recursion_depth: pipeline_properties.max_ray_recursion_depth,
        max_hit_attribute_size: pipeline_properties.max_ray_hit_attribute_size,
        max_dispatch_invocations: pipeline_properties.max_ray_dispatch_invocation_count,
        scratch_alignment: accel_properties.min_acceleration_structure_scratch_offset_alignment,
        max_geometry_count: accel_properties.max_geometry_count,
        max_primitive_count: accel_properties.max_primitive_count,
        max_instance_count: accel_properties.max_instance_count,
    })
}

pub(crate) struct RayLoaders {
    pub acceleration: ash::khr::acceleration_structure::Device,
    pub pipeline: Option<ash::khr::ray_tracing_pipeline::Device>,
}
impl RayLoaders {
    pub(crate) fn new(instance: &Instance, device: &Device, support: RaySupport) -> Option<Self> {
        support.acceleration.then(|| Self {
            acceleration: ash::khr::acceleration_structure::Device::new(instance, device),
            pipeline: support
                .pipeline
                .then(|| ash::khr::ray_tracing_pipeline::Device::new(instance, device)),
        })
    }
}
