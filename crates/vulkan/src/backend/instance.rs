//! Vulkan adapter discovery and logical-device opening.
//!
//! This module owns feature negotiation and creates the single graphics queue.
//! It reports facts to Core but does not schedule work; command submission and
//! resource lifetime remain in their dedicated backend modules.

use super::*;
use fluxel_rhi_core::api::WaitRoute;

impl InstanceBackend for Vulkan {
    type NativeInstance = VulkanInstance;
    type OpenFuture<'a> = Ready<Result<(VulkanDevice, DeviceCapabilities)>>;

    fn enumerate(instance: &VulkanInstance, scope: InstanceId) -> Result<Vec<AdapterInfo>> {
        // SAFETY: InstanceState retains the live Vulkan instance for the whole
        // call; returned physical-device handles are consumed immediately.
        let physical = unsafe { instance.0.instance.enumerate_physical_devices() }
            .map_err(|error| Error::Backend(format!("Vulkan adapter enumeration: {error:?}")))?;
        let adapters = physical
            .into_iter()
            .enumerate()
            .filter_map(|(ordinal, physical)| {
                // SAFETY: `physical` came from this still-live instance and
                // the driver writes an ash-owned result vector.
                let families = unsafe {
                    instance
                        .0
                        .instance
                        .get_physical_device_queue_family_properties(physical)
                };
                families
                    .iter()
                    .any(|family| family.queue_flags.contains(vk::QueueFlags::GRAPHICS))
                    .then(|| {
                        // SAFETY: `physical` belongs to this live instance.
                        let properties =
                            unsafe { instance.0.instance.get_physical_device_properties(physical) };
                        // SAFETY: Vulkan guarantees a NUL-terminated device
                        // name inside `PhysicalDeviceProperties`.
                        let name =
                            unsafe { std::ffi::CStr::from_ptr(properties.device_name.as_ptr()) }
                                .to_string_lossy()
                                .into_owned();
                        AdapterInfo {
                            id: AdapterId {
                                instance: scope,
                                ordinal: ordinal as u64,
                            },
                            name,
                        }
                    })
            })
            .collect();
        Ok(adapters)
    }

    fn open<'a>(
        instance: &'a VulkanInstance,
        device: DeviceId,
        request: &'a DeviceRequest,
    ) -> Self::OpenFuture<'a> {
        ready((|| {
            if !request.required_features.is_empty() {
                return Err(Error::Unsupported("requested Vulkan device features"));
            }
            // SAFETY: the instance remains retained throughout device opening.
            let physical =
                unsafe { instance.0.instance.enumerate_physical_devices() }.map_err(|error| {
                    Error::Backend(format!(
                        "Vulkan device-open physical enumeration: {error:?}"
                    ))
                })?;
            let physical = physical
                .get(request.adapter.ordinal as usize)
                .copied()
                .ok_or(Error::Unsupported("the requested Vulkan adapter"))?;
            // SAFETY: the selected handle originated from this live instance;
            // queue-family properties are copied out by the driver.
            let family = unsafe {
                instance
                    .0
                    .instance
                    .get_physical_device_queue_family_properties(physical)
            }
            .iter()
            .position(|f| f.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .ok_or(Error::Unsupported("a Vulkan graphics queue"))? as u32;
            let ray_support = super::ray::probe::probe(&instance.0.instance, physical)?;
            // SAFETY: extension enumeration and feature/property queries use the selected physical device.
            let device_extensions = unsafe {
                instance
                    .0
                    .instance
                    .enumerate_device_extension_properties(physical)
            }
            .map_err(|error| {
                Error::Backend(format!("Vulkan device extension enumeration: {error:?}"))
            })?;
            let mesh_extension = device_extensions.iter().any(|e| unsafe { std::ffi::CStr::from_ptr(e.extension_name.as_ptr()) } == ash::ext::mesh_shader::NAME);
            let conservative_rasterization = device_extensions.iter().any(|e| {
                // SAFETY: Vulkan extension names are fixed NUL-terminated arrays.
                (unsafe { std::ffi::CStr::from_ptr(e.extension_name.as_ptr()) })
                    == ash::ext::conservative_rasterization::NAME
            });
            let has_maintenance5 = device_extensions.iter().any(|extension| {
                // SAFETY: the driver supplies NUL-terminated extension names.
                (unsafe { std::ffi::CStr::from_ptr(extension.extension_name.as_ptr()) })
                    == ash::khr::maintenance5::NAME
            });
            let mut maintenance5_features = vk::PhysicalDeviceMaintenance5FeaturesKHR::default();
            if has_maintenance5 {
                let mut features =
                    vk::PhysicalDeviceFeatures2::default().push_next(&mut maintenance5_features);
                // SAFETY: all query nodes remain live for this synchronous call.
                unsafe {
                    instance
                        .0
                        .instance
                        .get_physical_device_features2(physical, &mut features)
                };
            }
            let maintenance5 = has_maintenance5 && maintenance5_features.maintenance5 == vk::TRUE;
            #[cfg(windows)]
            let external_names = (
                ash::khr::external_memory_win32::NAME,
                ash::khr::external_semaphore_win32::NAME,
            );
            #[cfg(not(windows))]
            let external_names = (
                ash::khr::external_memory_fd::NAME,
                ash::khr::external_semaphore_fd::NAME,
            );
            let has_extension = |name: &std::ffi::CStr| {
                device_extensions.iter().any(|extension| {
                    // SAFETY: Vulkan guarantees NUL termination for extension names.
                    (unsafe { std::ffi::CStr::from_ptr(extension.extension_name.as_ptr()) }) == name
                })
            };
            let external_memory_handles = has_extension(external_names.0);
            let external_semaphore_handles = has_extension(external_names.1);
            #[cfg(windows)]
            let semaphore_handle = vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_WIN32;
            #[cfg(not(windows))]
            let semaphore_handle = vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD;
            let mut external_semaphore_properties = vk::ExternalSemaphoreProperties::default();
            if external_semaphore_handles {
                // SAFETY: the physical device and output storage are live; the
                // selected handle type belongs to this platform's extension.
                unsafe {
                    instance
                        .0
                        .instance
                        .get_physical_device_external_semaphore_properties(
                            physical,
                            &vk::PhysicalDeviceExternalSemaphoreInfo::default()
                                .handle_type(semaphore_handle),
                            &mut external_semaphore_properties,
                        );
                }
            }
            let external_interop = external_memory_handles
                && external_semaphore_handles
                && external_semaphore_properties
                    .external_semaphore_features
                    .contains(
                        vk::ExternalSemaphoreFeatureFlags::IMPORTABLE
                            | vk::ExternalSemaphoreFeatureFlags::EXPORTABLE,
                    );
            let mut mesh_features = vk::PhysicalDeviceMeshShaderFeaturesEXT::default();
            if mesh_extension {
                let mut features =
                    vk::PhysicalDeviceFeatures2::default().push_next(&mut mesh_features);
                unsafe {
                    instance
                        .0
                        .instance
                        .get_physical_device_features2(physical, &mut features);
                }
            }
            let mesh_available = mesh_extension && mesh_features.mesh_shader == vk::TRUE;
            let task_available = mesh_available && mesh_features.task_shader == vk::TRUE;
            let mesh_multiview_available =
                mesh_available && mesh_features.multiview_mesh_shader == vk::TRUE;
            let mut mesh_properties = vk::PhysicalDeviceMeshShaderPropertiesEXT::default();
            if mesh_available {
                let mut properties =
                    vk::PhysicalDeviceProperties2::default().push_next(&mut mesh_properties);
                unsafe {
                    instance
                        .0
                        .instance
                        .get_physical_device_properties2(physical, &mut properties);
                }
            }
            let (
                dynamic_rendering,
                shader_clip_distance,
                multi_draw_indirect,
                sampler_anisotropy,
                multiview,
                max_multiview_views,
                pipeline_statistics,
                draw_indirect_count,
                timeline_semaphore,
                sparse_features,
            ) = {
                let mut available13 = vk::PhysicalDeviceVulkan13Features::default();
                let mut available12 = vk::PhysicalDeviceVulkan12Features::default();
                let mut available_multiview = vk::PhysicalDeviceMultiviewFeatures::default();
                let mut available = vk::PhysicalDeviceFeatures2::default()
                    .push_next(&mut available13)
                    .push_next(&mut available12)
                    .push_next(&mut available_multiview);
                // SAFETY: all pNext nodes are stack-owned for this call and
                // chained by ash with the correct Vulkan structure types.
                unsafe {
                    instance
                        .0
                        .instance
                        .get_physical_device_features2(physical, &mut available)
                };
                let core_features = available.features;
                let shader_clip_distance = core_features.shader_clip_distance == vk::TRUE;
                let multi_draw_indirect = core_features.multi_draw_indirect == vk::TRUE;
                let sampler_anisotropy = core_features.sampler_anisotropy == vk::TRUE;
                let dynamic_rendering = available13.dynamic_rendering == vk::TRUE;
                let multiview = available_multiview.multiview == vk::TRUE;
                let mut multiview_properties = vk::PhysicalDeviceMultiviewProperties::default();
                let mut properties =
                    vk::PhysicalDeviceProperties2::default().push_next(&mut multiview_properties);
                // SAFETY: `properties` and its pNext output node are valid,
                // mutable storage until the query returns.
                unsafe {
                    instance
                        .0
                        .instance
                        .get_physical_device_properties2(physical, &mut properties)
                };
                let max_multiview_views = multiview_properties.max_multiview_view_count;
                let pipeline_statistics = core_features.pipeline_statistics_query == vk::TRUE;
                let draw_indirect_count = available12.draw_indirect_count == vk::TRUE;
                let timeline_semaphore = available12.timeline_semaphore == vk::TRUE;
                (
                    dynamic_rendering,
                    shader_clip_distance,
                    multi_draw_indirect,
                    sampler_anisotropy,
                    multiview,
                    max_multiview_views,
                    pipeline_statistics,
                    draw_indirect_count,
                    timeline_semaphore,
                    core_features.sparse_binding == vk::TRUE
                        && (core_features.sparse_residency_image2_d == vk::TRUE
                            || core_features.sparse_residency_image3_d == vk::TRUE),
                )
            };
            let mut enabled12 = vk::PhysicalDeviceVulkan12Features::default();
            let mut enabled13 = vk::PhysicalDeviceVulkan13Features::default();
            let mut supported = vk::PhysicalDeviceFeatures2::default()
                .push_next(&mut enabled12)
                .push_next(&mut enabled13);
            // SAFETY: each output node is live until the synchronous query returns.
            unsafe {
                instance
                    .0
                    .instance
                    .get_physical_device_features2(physical, &mut supported)
            };
            // Enable the supported base feature set so resource and shader
            // admission checks reflect the actual logical-device capabilities.
            let enabled_features = supported.features;
            let mut subgroup = vk::PhysicalDeviceSubgroupSizeControlProperties::default();
            let mut maintenance4_properties = vk::PhysicalDeviceMaintenance4Properties::default();
            let mut core11_properties = vk::PhysicalDeviceVulkan11Properties::default();
            let mut subgroup_properties = vk::PhysicalDeviceProperties2::default()
                .push_next(&mut subgroup)
                .push_next(&mut maintenance4_properties)
                .push_next(&mut core11_properties);
            // SAFETY: physical belongs to this instance and the output chain is valid.
            unsafe {
                instance
                    .0
                    .instance
                    .get_physical_device_properties2(physical, &mut subgroup_properties)
            };
            if !dynamic_rendering {
                return Err(Error::Unsupported("Vulkan dynamic rendering"));
            }
            // SAFETY: this physical device belongs to the retained instance.
            let family_properties = unsafe {
                instance
                    .0
                    .instance
                    .get_physical_device_queue_family_properties(physical)
            };
            let family_info = family_properties[family as usize];
            let separate_compute_family = family_properties
                .iter()
                .enumerate()
                .filter(|(index, properties)| {
                    *index != family as usize
                        && properties.queue_count > 0
                        && properties.queue_flags.contains(vk::QueueFlags::COMPUTE)
                })
                .min_by_key(|(_, properties)| {
                    properties.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                })
                .map(|(index, _)| index as u32);
            let compute_family = separate_compute_family.unwrap_or(family);
            let compute_info = family_properties[compute_family as usize];
            let sparse_image = sparse_features
                && family_info
                    .queue_flags
                    .contains(vk::QueueFlags::SPARSE_BINDING);
            let has_compute_queue = (separate_compute_family.is_some()
                || (family_info.queue_count >= 2
                    && family_info.queue_flags.contains(vk::QueueFlags::COMPUTE)))
                && timeline_semaphore;
            let priorities = if has_compute_queue && compute_family == family {
                vec![1.0_f32, 1.0]
            } else {
                vec![1.0_f32]
            };
            let queue = vk::DeviceQueueCreateInfo::default()
                .queue_family_index(family)
                .queue_priorities(&priorities);
            let compute_priorities = [1.0_f32];
            let mut queues = vec![queue];
            if has_compute_queue && compute_family != family {
                queues.push(
                    vk::DeviceQueueCreateInfo::default()
                        .queue_family_index(compute_family)
                        .queue_priorities(&compute_priorities),
                );
            }
            let mut extensions = vec![ash::khr::swapchain::NAME.as_ptr()];
            extensions.extend(ray_support.extensions());
            if mesh_available {
                extensions.push(ash::ext::mesh_shader::NAME.as_ptr());
            }
            if conservative_rasterization {
                extensions.push(ash::ext::conservative_rasterization::NAME.as_ptr());
            }
            if maintenance5 {
                extensions.push(ash::khr::maintenance5::NAME.as_ptr());
            }
            if external_memory_handles {
                extensions.push(external_names.0.as_ptr());
            }
            if external_semaphore_handles {
                extensions.push(external_names.1.as_ptr());
            }
            let mut vulkan13 = vk::PhysicalDeviceVulkan13Features::default()
                .dynamic_rendering(true)
                .maintenance4(enabled13.maintenance4 == vk::TRUE)
                .texture_compression_astc_hdr(enabled13.texture_compression_astc_hdr == vk::TRUE)
                .subgroup_size_control(enabled13.subgroup_size_control == vk::TRUE)
                .compute_full_subgroups(enabled13.compute_full_subgroups == vk::TRUE);
            let mut vulkan12 = vk::PhysicalDeviceVulkan12Features::default()
                .runtime_descriptor_array(enabled12.runtime_descriptor_array == vk::TRUE)
                .descriptor_binding_variable_descriptor_count(
                    enabled12.descriptor_binding_variable_descriptor_count == vk::TRUE,
                )
                .descriptor_binding_partially_bound(
                    enabled12.descriptor_binding_partially_bound == vk::TRUE,
                )
                .shader_uniform_buffer_array_non_uniform_indexing(
                    enabled12.shader_uniform_buffer_array_non_uniform_indexing == vk::TRUE,
                )
                .shader_sampled_image_array_non_uniform_indexing(
                    enabled12.shader_sampled_image_array_non_uniform_indexing == vk::TRUE,
                )
                .shader_storage_buffer_array_non_uniform_indexing(
                    enabled12.shader_storage_buffer_array_non_uniform_indexing == vk::TRUE,
                )
                .shader_storage_image_array_non_uniform_indexing(
                    enabled12.shader_storage_image_array_non_uniform_indexing == vk::TRUE,
                )
                .separate_depth_stencil_layouts(
                    enabled12.separate_depth_stencil_layouts == vk::TRUE,
                )
                .draw_indirect_count(draw_indirect_count)
                .timeline_semaphore(timeline_semaphore)
                .buffer_device_address(ray_support.acceleration);
            let mut multiview_features =
                vk::PhysicalDeviceMultiviewFeatures::default().multiview(multiview);
            let mut mesh_enable = vk::PhysicalDeviceMeshShaderFeaturesEXT::default()
                .mesh_shader(mesh_available)
                .task_shader(task_available)
                .multiview_mesh_shader(mesh_multiview_available && multiview);
            let mut accel_enable = vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default()
                .acceleration_structure(ray_support.acceleration);
            let mut ray_pipeline_enable =
                vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::default()
                    .ray_tracing_pipeline(ray_support.pipeline);
            let mut ray_query_enable =
                vk::PhysicalDeviceRayQueryFeaturesKHR::default().ray_query(ray_support.query);
            let mut create = vk::DeviceCreateInfo::default()
                .queue_create_infos(&queues)
                .enabled_extension_names(&extensions)
                .enabled_features(&enabled_features)
                .push_next(&mut vulkan13)
                .push_next(&mut vulkan12)
                .push_next(&mut multiview_features);
            if mesh_available {
                create = create.push_next(&mut mesh_enable);
            }
            if maintenance5 {
                create = create.push_next(&mut maintenance5_features);
            }
            if ray_support.acceleration {
                create = create.push_next(&mut accel_enable);
            }
            if ray_support.pipeline {
                create = create.push_next(&mut ray_pipeline_enable);
            }
            if ray_support.query {
                create = create.push_next(&mut ray_query_enable);
            }
            // SAFETY: every pNext node and pointer in `create` is retained on
            // this stack for the call; `physical` belongs to the instance.
            let native = unsafe { instance.0.instance.create_device(physical, &create, None) }
                .map_err(|error| Error::Backend(format!("Vulkan create_device: {error:?}")))?;
            // SAFETY: device creation requested queue 0 from this family.
            let mesh = mesh_available.then(|| {
                super::mesh::MeshDevice::new(
                    &instance.0.instance,
                    &native,
                    task_available,
                    &mesh_properties,
                )
            });
            let ray_loaders =
                super::ray::probe::RayLoaders::new(&instance.0.instance, &native, ray_support);
            let graphics_queue = unsafe { native.get_device_queue(family, 0) };
            // SAFETY: queue 1 was requested exactly when this branch is taken.
            let compute_queue = has_compute_queue.then(|| unsafe {
                native.get_device_queue(compute_family, u32::from(compute_family == family))
            });
            // SAFETY: physical device and instance are still live.
            let properties =
                unsafe { instance.0.instance.get_physical_device_properties(physical) };
            let limits = properties.limits;
            // SAFETY: the queried format and physical device belong to this
            // still-live instance; the driver returns plain copied properties.
            let rgba8_features = unsafe {
                instance
                    .0
                    .instance
                    .get_physical_device_format_properties(physical, vk::Format::R8G8B8A8_UNORM)
            }
            .optimal_tiling_features;
            Ok((
                VulkanDevice(Arc::new(DeviceState {
                    features: enabled_features,
                    conservative_rasterization,
                    maintenance5,
                    external_memory_handles,
                    external_semaphore_handles,
                    compute_family,
                    compute_timestamp_bits: compute_info.timestamp_valid_bits,
                    compute_sparse_binding: compute_info
                        .queue_flags
                        .contains(vk::QueueFlags::SPARSE_BINDING),
                    graphics_compute: family_info.queue_flags.contains(vk::QueueFlags::COMPUTE),
                    max_compute_workgroups: limits.max_compute_work_group_count,
                    max_compute_workgroup_size: limits.max_compute_work_group_size,
                    max_compute_workgroup_invocations: limits.max_compute_work_group_invocations,
                    texture_compression_astc_hdr: enabled13.texture_compression_astc_hdr
                        == vk::TRUE,
                    descriptor_partially_bound: enabled12.descriptor_binding_partially_bound
                        == vk::TRUE,
                    nonuniform_uniform_buffer: enabled12
                        .shader_uniform_buffer_array_non_uniform_indexing
                        == vk::TRUE,
                    nonuniform_sampled_image: enabled12
                        .shader_sampled_image_array_non_uniform_indexing
                        == vk::TRUE,
                    nonuniform_storage_buffer: enabled12
                        .shader_storage_buffer_array_non_uniform_indexing
                        == vk::TRUE,
                    nonuniform_storage_image: enabled12
                        .shader_storage_image_array_non_uniform_indexing
                        == vk::TRUE,
                    runtime_descriptor_array: enabled12.runtime_descriptor_array == vk::TRUE,
                    variable_descriptor_count: enabled12
                        .descriptor_binding_variable_descriptor_count
                        == vk::TRUE,
                    separate_depth_stencil_layouts: enabled12.separate_depth_stencil_layouts
                        == vk::TRUE,
                    subgroup_size_control: enabled13.subgroup_size_control == vk::TRUE,
                    min_subgroup_size: subgroup.min_subgroup_size,
                    max_subgroup_size: subgroup.max_subgroup_size,
                    required_subgroup_size_stages: subgroup.required_subgroup_size_stages,
                    lane_frontier: Mutex::new([None, None]),
                    device: native,
                    instance: instance.0.clone(),
                    physical,
                    graphics_queue,
                    graphics_family: family,
                    compute_queue,
                    sparse_image,
                    timeline_semaphore,
                    mesh,
                    ray_support,
                    ray_loaders,
                    shader_clip_distance,
                    multi_draw_indirect,
                    draw_indirect_count,
                    pipeline_statistics,
                    timestamp_period: f64::from(limits.timestamp_period),
                    timestamp_bits: family_info.timestamp_valid_bits,
                    sampler_anisotropy,
                    max_sampler_anisotropy: limits.max_sampler_anisotropy,
                    multiview,
                    max_multiview_views,
                    multiview_mesh_shader: mesh_multiview_available && multiview,
                    max_draw_indirect_count: limits.max_draw_indirect_count,
                    queue_gate: Mutex::new(()),
                    submit_gate: Mutex::new(()),
                })),
                DeviceCapabilities {
                    compute: family_info.queue_flags.contains(vk::QueueFlags::COMPUTE)
                        || has_compute_queue,
                    placed_memory: true,
                    native_interop: true,
                    external_memory: external_interop,
                    indirect_draw: true,
                    indirect_first_instance: enabled_features.draw_indirect_first_instance
                        == vk::TRUE,
                    multi_draw: multi_draw_indirect,
                    multi_draw_count: draw_indirect_count,
                    mesh: mesh_available,
                    timestamp: family_info.timestamp_valid_bits != 0
                        || (has_compute_queue && compute_info.timestamp_valid_bits != 0),
                    occlusion: true,
                    storage_buffers: true,
                    pipeline_statistics,
                    multiview,
                    multi_queue: has_compute_queue,
                    sparse: sparse_image && timeline_semaphore,
                    ray: ray_support.pipeline && ray_support.acceleration,
                    acceleration_structure: ray_support.acceleration,
                    ray_query: ray_support.query && ray_support.acceleration,
                    binding_arrays: true,
                    runtime_sized_binding_arrays: enabled12.runtime_descriptor_array == vk::TRUE
                        && enabled12.descriptor_binding_variable_descriptor_count == vk::TRUE,
                    nonuniform_sampled_texture_and_storage_buffer_indexing: enabled12
                        .shader_sampled_image_array_non_uniform_indexing
                        == vk::TRUE
                        && enabled12.shader_storage_buffer_array_non_uniform_indexing == vk::TRUE,
                    storage_textures: rgba8_features
                        .contains(vk::FormatFeatureFlags::STORAGE_IMAGE),
                    push_constants: limits.max_push_constants_size != 0,
                    // Vulkan 1.3 dynamic rendering supplies the inherited
                    // rendering-info path used by `secondary.rs`.
                    secondary_commands: true,
                    lanes: if has_compute_queue {
                        let graphics = LaneId { device, ordinal: 0 };
                        let compute = LaneId { device, ordinal: 1 };
                        Arc::from([
                            LaneInfo {
                                id: graphics,
                                class: LaneClass::Graphics,
                                routes: vec![(compute, WaitRoute::NativeGpuWait)],
                            },
                            LaneInfo {
                                id: compute,
                                class: LaneClass::Compute,
                                routes: vec![(graphics, WaitRoute::NativeGpuWait)],
                            },
                        ])
                    } else {
                        Arc::from([LaneInfo {
                            id: LaneId { device, ordinal: 0 },
                            class: LaneClass::Graphics,
                            routes: Vec::new(),
                        }])
                    },
                    limits: DeviceLimits {
                        max_bind_groups: limits.max_bound_descriptor_sets,
                        max_bindings_per_group: core11_properties.max_per_set_descriptors,
                        max_color_attachments: limits.max_color_attachments,
                        max_texture_dimension_2d: limits.max_image_dimension2_d,
                        max_texture_array_layers: limits.max_image_array_layers,
                        max_buffer_bytes: maintenance4_properties.max_buffer_size,
                        max_vertex_buffers: limits.max_vertex_input_bindings,
                        max_compute_workgroups: limits.max_compute_work_group_count,
                        max_compute_workgroup_size: limits.max_compute_work_group_size,
                        max_vertex_attributes: limits.max_vertex_input_attributes,
                        max_push_constant_bytes: limits.max_push_constants_size,
                        max_multiview_views,
                        uniform_offset_alignment: limits.min_uniform_buffer_offset_alignment,
                        storage_offset_alignment: limits.min_storage_buffer_offset_alignment,
                        ..Default::default()
                    },
                    ..Default::default()
                },
            ))
        })())
    }
}
