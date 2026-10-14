//! Vulkan adapter discovery and logical-device opening.
//!
//! This module owns feature negotiation and creates the single graphics queue.
//! It reports facts to Core but does not schedule work; command submission and
//! resource lifetime remain in their dedicated backend modules.

use super::*;

impl InstanceBackend for Vulkan {
    type NativeInstance = VulkanInstance;
    type OpenFuture<'a> = Ready<Result<(VulkanDevice, DeviceCapabilities)>>;

    fn enumerate(instance: &VulkanInstance, scope: InstanceId) -> Result<Vec<AdapterInfo>> {
        // SAFETY: InstanceState retains the live Vulkan instance for the whole
        // call; returned physical-device handles are consumed immediately.
        let physical =
            unsafe { instance.0.instance.enumerate_physical_devices() }.map_err(vk_error)?;
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
                unsafe { instance.0.instance.enumerate_physical_devices() }.map_err(vk_error)?;
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
            let (
                dynamic_rendering,
                shader_clip_distance,
                multi_draw_indirect,
                sampler_anisotropy,
                descriptor_indexing,
                multiview,
                max_multiview_views,
            ) = {
                let mut available13 = vk::PhysicalDeviceVulkan13Features::default();
                let mut available_descriptor_indexing =
                    vk::PhysicalDeviceDescriptorIndexingFeatures::default();
                let mut available_multiview = vk::PhysicalDeviceMultiviewFeatures::default();
                let mut available = vk::PhysicalDeviceFeatures2::default()
                    .push_next(&mut available13)
                    .push_next(&mut available_descriptor_indexing)
                    .push_next(&mut available_multiview);
                // SAFETY: all pNext nodes are stack-owned for this call and
                // chained by ash with the correct Vulkan structure types.
                unsafe {
                    instance
                        .0
                        .instance
                        .get_physical_device_features2(physical, &mut available)
                };
                let shader_clip_distance = available.features.shader_clip_distance == vk::TRUE;
                let multi_draw_indirect = available.features.multi_draw_indirect == vk::TRUE;
                let sampler_anisotropy = available.features.sampler_anisotropy == vk::TRUE;
                let dynamic_rendering = available13.dynamic_rendering == vk::TRUE;
                let descriptor_indexing = available_descriptor_indexing.runtime_descriptor_array
                    == vk::TRUE
                    && available_descriptor_indexing.descriptor_binding_variable_descriptor_count
                        == vk::TRUE
                    && available_descriptor_indexing
                        .shader_sampled_image_array_non_uniform_indexing
                        == vk::TRUE;
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
                (
                    dynamic_rendering,
                    shader_clip_distance,
                    multi_draw_indirect,
                    sampler_anisotropy,
                    descriptor_indexing,
                    multiview,
                    max_multiview_views,
                )
            };
            if !dynamic_rendering {
                return Err(Error::Unsupported("Vulkan dynamic rendering"));
            }
            let enabled_features = vk::PhysicalDeviceFeatures::default()
                .shader_clip_distance(shader_clip_distance)
                .multi_draw_indirect(multi_draw_indirect)
                .sampler_anisotropy(sampler_anisotropy);
            let priorities = [1.0_f32];
            let queue = vk::DeviceQueueCreateInfo::default()
                .queue_family_index(family)
                .queue_priorities(&priorities);
            let extensions = [ash::khr::swapchain::NAME.as_ptr()];
            let mut vulkan13 =
                vk::PhysicalDeviceVulkan13Features::default().dynamic_rendering(true);
            let mut descriptor_indexing_features =
                vk::PhysicalDeviceDescriptorIndexingFeatures::default()
                    .runtime_descriptor_array(descriptor_indexing)
                    .descriptor_binding_variable_descriptor_count(descriptor_indexing)
                    .shader_sampled_image_array_non_uniform_indexing(descriptor_indexing);
            let mut multiview_features =
                vk::PhysicalDeviceMultiviewFeatures::default().multiview(multiview);
            let create = vk::DeviceCreateInfo::default()
                .queue_create_infos(std::slice::from_ref(&queue))
                .enabled_extension_names(&extensions)
                .enabled_features(&enabled_features)
                .push_next(&mut vulkan13)
                .push_next(&mut descriptor_indexing_features)
                .push_next(&mut multiview_features);
            // SAFETY: every pNext node and pointer in `create` is retained on
            // this stack for the call; `physical` belongs to the instance.
            let native = unsafe { instance.0.instance.create_device(physical, &create, None) }
                .map_err(vk_error)?;
            // SAFETY: device creation requested queue 0 from this family.
            let graphics_queue = unsafe { native.get_device_queue(family, 0) };
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
                    device: native,
                    instance: instance.0.clone(),
                    physical,
                    graphics_queue,
                    graphics_family: family,
                    shader_clip_distance,
                    multi_draw_indirect,
                    sampler_anisotropy,
                    max_sampler_anisotropy: limits.max_sampler_anisotropy,
                    multiview,
                    descriptor_indexing,
                    max_sampled_images: limits
                        .max_per_stage_descriptor_sampled_images
                        .min(limits.max_descriptor_set_sampled_images),
                    max_draw_indirect_count: limits.max_draw_indirect_count,
                    queue_gate: Mutex::new(()),
                    submit_gate: Mutex::new(()),
                })),
                DeviceCapabilities {
                    compute: true,
                    indirect_draw: true,
                    multi_draw: multi_draw_indirect,
                    multiview,
                    binding_arrays: descriptor_indexing,
                    runtime_sized_binding_arrays: descriptor_indexing,
                    nonuniform_sampled_texture_and_storage_buffer_indexing: descriptor_indexing,
                    storage_textures: rgba8_features
                        .contains(vk::FormatFeatureFlags::STORAGE_IMAGE),
                    push_constants: limits.max_push_constants_size != 0,
                    // Vulkan 1.3 dynamic rendering supplies the inherited
                    // rendering-info path used by `secondary.rs`.
                    secondary_commands: true,
                    lanes: Arc::from([LaneInfo {
                        id: LaneId { device, ordinal: 0 },
                        class: LaneClass::Graphics,
                        routes: Vec::new(),
                    }]),
                    limits: DeviceLimits {
                        max_bind_groups: 4,
                        max_bindings_per_group: 1024,
                        max_color_attachments: limits.max_color_attachments,
                        max_texture_dimension_2d: limits.max_image_dimension2_d,
                        max_texture_array_layers: limits.max_image_array_layers,
                        max_buffer_bytes: limits.max_storage_buffer_range as u64,
                        max_vertex_buffers: limits.max_vertex_input_bindings,
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
