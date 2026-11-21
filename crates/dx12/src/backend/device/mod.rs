//! Native adapter discovery, device creation and runtime capability queries.
//!
//! Every optional path is selected from native device facts. The device owns
//! shared queues, descriptor heaps and immutable native caches; it does not
//! infer resource uses or build an execution schedule.

mod adapter;
mod descriptor_limits;
mod shader_model;
mod surface_probe;

const MAX_BIND_GROUPS: u32 = 8;
const MAX_BINDINGS_PER_GROUP: u32 = 64;

use super::*;
#[cfg(test)]
use adapter::REQUESTED_ADAPTER_UNAVAILABLE;
use adapter::adapter_at;
#[cfg(all(test, feature = "hardware-tests"))]
use adapter::adapter_name;
use windows::Win32::Graphics::Direct3D12::{
    D3D12_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT, D3D12_FEATURE_ARCHITECTURE,
    D3D12_FEATURE_D3D12_OPTIONS4, D3D12_FEATURE_DATA_ARCHITECTURE,
    D3D12_FEATURE_DATA_D3D12_OPTIONS4, D3D12_FORMAT_SUPPORT2_UAV_ATOMIC_ADD,
    D3D12_FORMAT_SUPPORT2_UAV_ATOMIC_BITWISE_OPS,
    D3D12_FORMAT_SUPPORT2_UAV_ATOMIC_SIGNED_MIN_OR_MAX,
    D3D12_FORMAT_SUPPORT2_UAV_ATOMIC_UNSIGNED_MIN_OR_MAX,
    D3D12_IA_VERTEX_INPUT_RESOURCE_SLOT_COUNT, D3D12_IA_VERTEX_INPUT_STRUCTURE_ELEMENT_COUNT,
    D3D12_RAYTRACING_MAX_DECLARABLE_TRACE_RECURSION_DEPTH,
    D3D12_RAYTRACING_MAX_GEOMETRIES_PER_BOTTOM_LEVEL_ACCELERATION_STRUCTURE,
    D3D12_RAYTRACING_MAX_INSTANCES_PER_TOP_LEVEL_ACCELERATION_STRUCTURE,
    D3D12_RAYTRACING_MAX_PRIMITIVES_PER_BOTTOM_LEVEL_ACCELERATION_STRUCTURE,
    D3D12_RAYTRACING_MAX_RAY_GENERATION_SHADER_THREADS,
    D3D12_REQ_MULTI_ELEMENT_STRUCTURE_SIZE_IN_BYTES, D3D12_REQ_TEXTURE1D_U_DIMENSION,
    D3D12_REQ_TEXTURE2D_ARRAY_AXIS_DIMENSION, D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION,
    D3D12_REQ_TEXTURE3D_U_V_OR_W_DIMENSION, D3D12_RESOURCE_BINDING_TIER_3,
    D3D12_SIMULTANEOUS_RENDER_TARGET_COUNT,
};
use windows::Win32::Graphics::Dxgi::DXGI_ERROR_UNSUPPORTED;

impl InstanceBackend for Dx12 {
    type NativeInstance = Dx12Instance;
    type OpenFuture<'a> = Ready<Result<(Dx12Device, DeviceCapabilities)>>;
    fn enumerate(instance: &Dx12Instance, scope: InstanceId) -> Result<Vec<AdapterInfo>> {
        adapter::enumerate(instance, scope)
    }
    fn request_adapter(
        instance: &Dx12Instance,
        scope: InstanceId,
        preference: api::AdapterPreference,
    ) -> Result<AdapterInfo> {
        adapter::select_adapter(instance, scope, preference)
    }
    fn adapter_properties(
        instance: &Dx12Instance,
        id: AdapterId,
    ) -> Result<api::AdapterProperties> {
        adapter::properties(instance, id)
    }
    fn open<'a>(
        instance: &'a Dx12Instance,
        device: DeviceId,
        request: &'a DeviceRequest,
    ) -> Self::OpenFuture<'a> {
        Self::open_device(instance, device, request, &[])
    }
    fn open_with_capacities<'a>(
        instance: &'a Dx12Instance,
        device: DeviceId,
        request: &'a DeviceRequest,
        minimum: &'a [api::DeviceCapacity],
    ) -> impl Future<Output = Result<(Dx12Device, DeviceCapabilities)>> + 'a {
        Self::open_device(instance, device, request, minimum)
    }
}

impl Dx12 {
    fn open_device(
        instance: &Dx12Instance,
        device: DeviceId,
        request: &DeviceRequest,
        minimum: &[api::DeviceCapacity],
    ) -> Ready<Result<(Dx12Device, DeviceCapabilities)>> {
        let result = (|| {
            let diagnostics = std::env::var_os("FLUXEL_DX12_DEBUG").is_some();
            if diagnostics {
                let mut debug: Option<ID3D12Debug> = None;
                // SAFETY: `debug` is a valid local output slot for this COM query.
                unsafe { D3D12GetDebugInterface(&mut debug) }.map_err(native_error)?;
                // SAFETY: the returned interface remains owned by `debug` for
                // this call and EnableDebugLayer has no borrowed Rust inputs.
                unsafe {
                    debug
                        .ok_or_else(|| Error::Backend("D3D12 debug interface unavailable".into()))?
                        .EnableDebugLayer();
                }
            }
            let adapter = adapter_at(instance, request.adapter.ordinal)?;
            let mut native: Option<ID3D12Device> = None;
            // SAFETY: adapter is live and `native` is the output slot for the
            // newly created device interface.
            unsafe { D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_11_0, &mut native) }.map_err(
                |error| {
                    if error.code() == DXGI_ERROR_UNSUPPORTED {
                        Error::Unsupported("D3D12 on the requested DXGI adapter")
                    } else {
                        native_error(error)
                    }
                },
            )?;
            let native = native
                .ok_or_else(|| Error::Backend("D3D12CreateDevice returned no device".into()))?;
            let shader_model = shader_model::highest(&native);
            let mut architecture = D3D12_FEATURE_DATA_ARCHITECTURE::default();
            // SAFETY: the initialized structure requests the architecture of
            // node zero, used by the ordinary queues and heaps of this device.
            let memory_topology = unsafe {
                native.CheckFeatureSupport(
                    D3D12_FEATURE_ARCHITECTURE,
                    (&mut architecture as *mut D3D12_FEATURE_DATA_ARCHITECTURE).cast(),
                    std::mem::size_of_val(&architecture) as u32,
                )
            }
            .ok()
            .map(|()| {
                if architecture.UMA.as_bool() {
                    api::DeviceMemoryTopology::Unified {
                        cache_coherent: architecture.CacheCoherentUMA.as_bool(),
                    }
                } else {
                    api::DeviceMemoryTopology::Separate
                }
            });
            let mut wave =
                windows::Win32::Graphics::Direct3D12::D3D12_FEATURE_DATA_D3D12_OPTIONS1::default();
            // SAFETY: the native device receives the exact initialized
            // OPTIONS1 structure for the duration of this synchronous call.
            let options1_available = unsafe {
                native.CheckFeatureSupport(
                    windows::Win32::Graphics::Direct3D12::D3D12_FEATURE_D3D12_OPTIONS1,
                    (&mut wave as *mut _) as *mut _,
                    std::mem::size_of_val(&wave) as u32,
                )
            }
            .is_ok();
            let shader_int64 =
                shader_model >= 60 && options1_available && wave.Int64ShaderOps.as_bool();
            let subgroup_widths = (shader_model >= 60
                && options1_available
                && wave.WaveOps.as_bool()
                && wave.WaveLaneCountMin > 0
                && wave.WaveLaneCountMin <= wave.WaveLaneCountMax)
                .then_some(api::SubgroupWidths {
                    min: wave.WaveLaneCountMin,
                    max: wave.WaveLaneCountMax,
                });
            let mut options4 = D3D12_FEATURE_DATA_D3D12_OPTIONS4::default();
            // SAFETY: the output is the exact native type for OPTIONS4.
            let native_16bit_shader_ops = shader_model >= 62
                && unsafe {
                    native.CheckFeatureSupport(
                        D3D12_FEATURE_D3D12_OPTIONS4,
                        (&mut options4 as *mut D3D12_FEATURE_DATA_D3D12_OPTIONS4).cast(),
                        std::mem::size_of_val(&options4) as u32,
                    )
                }
                .is_ok()
                && options4.Native16BitShaderOpsSupported.as_bool();
            let mut gpu_va = windows::Win32::Graphics::Direct3D12::
                D3D12_FEATURE_DATA_GPU_VIRTUAL_ADDRESS_SUPPORT::default();
            // SAFETY: the initialized structure has the exact type and size
            // requested by this synchronous D3D12 feature query.
            unsafe {
                native.CheckFeatureSupport(
                    windows::Win32::Graphics::Direct3D12::D3D12_FEATURE_GPU_VIRTUAL_ADDRESS_SUPPORT,
                    (&mut gpu_va as *mut _) as *mut _,
                    std::mem::size_of_val(&gpu_va) as u32,
                )
            }
            .map_err(native_error)?;
            let max_buffer_bytes =
                max_committed_buffer_bytes(gpu_va.MaxGPUVirtualAddressBitsPerProcess)?;
            let (ray_device, dxr_1_1) = ray::probe(&native);
            let ray_supported = ray_device.is_some();
            let ray_query_supported = dxr_1_1 && shader_model >= 65;
            let mesh_supported = mesh::probe(&native, shader_model);
            let mesh_shader_limits = mesh_supported.then(|| mesh::shader_limits(&native));
            let uint_atomics =
                texture::texture_atomic_support(&native, api::TextureFormat::R32Uint);
            let sint_atomics =
                texture::texture_atomic_support(&native, api::TextureFormat::R32Sint);
            let texture_atomic_r32uint_compute =
                uint_atomics.contains(D3D12_FORMAT_SUPPORT2_UAV_ATOMIC_ADD);
            let texture_atomic_r32sint_compute =
                sint_atomics.contains(D3D12_FORMAT_SUPPORT2_UAV_ATOMIC_ADD);
            let texture_atomic_r32sint_min_compute =
                sint_atomics.contains(D3D12_FORMAT_SUPPORT2_UAV_ATOMIC_SIGNED_MIN_OR_MAX);
            let texture_atomic_r32sint_max_compute = texture_atomic_r32sint_min_compute;
            // The native bits group min/max and bitwise operations, whereas
            // Core reports the separately supported shader operations.
            let texture_atomic_r32uint_min_compute =
                uint_atomics.contains(D3D12_FORMAT_SUPPORT2_UAV_ATOMIC_UNSIGNED_MIN_OR_MAX);
            let texture_atomic_r32uint_max_compute = texture_atomic_r32uint_min_compute;
            let texture_atomic_r32uint_bitwise_compute =
                uint_atomics.contains(D3D12_FORMAT_SUPPORT2_UAV_ATOMIC_BITWISE_OPS);
            let texture_atomic_r32uint_xor_compute = texture_atomic_r32uint_bitwise_compute;
            let texture_atomic_r32uint_and_compute = texture_atomic_r32uint_bitwise_compute;
            let texture_atomic_r32uint_or_compute = texture_atomic_r32uint_bitwise_compute;
            let mut options =
                windows::Win32::Graphics::Direct3D12::D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
            // SAFETY: this structure matches the feature query selector.
            let options_available = unsafe {
                native.CheckFeatureSupport(
                    windows::Win32::Graphics::Direct3D12::D3D12_FEATURE_D3D12_OPTIONS,
                    &mut options as *mut _ as *mut _,
                    std::mem::size_of_val(&options) as u32,
                )
            }
            .is_ok();
            let shader_f64 = shader_model >= 60
                && options_available
                && options.DoublePrecisionFloatShaderOps.as_bool();
            use windows::Win32::Graphics::Direct3D12::D3D12_TILED_RESOURCES_TIER_NOT_SUPPORTED;
            let tiled_resources_tier = if options_available {
                options.TiledResourcesTier
            } else {
                D3D12_TILED_RESOURCES_TIER_NOT_SUPPORTED
            };
            let residency_device = native
                .cast::<windows::Win32::Graphics::Direct3D12::ID3D12Device3>()
                .ok();
            let conservative_raster = options_available && options.ConservativeRasterizationTier
                != windows::Win32::Graphics::Direct3D12::D3D12_CONSERVATIVE_RASTERIZATION_TIER_NOT_SUPPORTED;
            let resource_heap_tier2 = options.ResourceHeapTier.0
                >= windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_HEAP_TIER_2.0;
            let descriptor_table_limits =
                descriptor_limits::query(&native, options.ResourceBindingTier);
            let mut options12 =
                windows::Win32::Graphics::Direct3D12::D3D12_FEATURE_DATA_D3D12_OPTIONS12::default();
            // SAFETY: as above, the feature structure exactly matches the
            // synchronous D3D12 OPTIONS12 query and is not retained by D3D12.
            let enhanced_barriers_supported = unsafe {
                native.CheckFeatureSupport(
                    windows::Win32::Graphics::Direct3D12::D3D12_FEATURE_D3D12_OPTIONS12,
                    (&mut options12 as *mut _) as *mut _,
                    std::mem::size_of_val(&options12) as u32,
                )
            }
            .is_ok()
                && options12.EnhancedBarriersSupported.as_bool();
            let mut options13 =
                windows::Win32::Graphics::Direct3D12::D3D12_FEATURE_DATA_D3D12_OPTIONS13::default();
            // SAFETY: the initialized OPTIONS13 structure remains live for this
            // synchronous D3D12 feature query.
            let texel_aligned_color_copies = unsafe {
                native.CheckFeatureSupport(
                    windows::Win32::Graphics::Direct3D12::D3D12_FEATURE_D3D12_OPTIONS13,
                    (&mut options13 as *mut _) as *mut _,
                    std::mem::size_of_val(&options13) as u32,
                )
            }
            .is_ok()
                && options13
                    .UnrestrictedBufferTextureCopyPitchSupported
                    .as_bool();
            let planar_nv12 = texture::probe_planar_texture(&native, api::TextureFormat::Nv12);
            let planar_p010 = texture::probe_planar_texture(&native, api::TextureFormat::P010);
            let mut options14 =
                windows::Win32::Graphics::Direct3D12::D3D12_FEATURE_DATA_D3D12_OPTIONS14::default();
            // SAFETY: the initialized OPTIONS14 structure matches the query
            // and is exclusively borrowed for this call.
            let stencil_reference_per_face = unsafe {
                native.CheckFeatureSupport(
                    windows::Win32::Graphics::Direct3D12::D3D12_FEATURE_D3D12_OPTIONS14,
                    (&mut options14 as *mut _) as *mut _,
                    std::mem::size_of_val(&options14) as u32,
                )
            }
            .is_ok()
                && options14
                    .IndependentFrontAndBackStencilRefMaskSupported
                    .as_bool();
            let mut options16 =
                windows::Win32::Graphics::Direct3D12::D3D12_FEATURE_DATA_D3D12_OPTIONS16::default();
            // SAFETY: the feature structure exactly matches the synchronous
            // OPTIONS16 query and remains live for its whole call.
            let options16_available = unsafe {
                native.CheckFeatureSupport(
                    windows::Win32::Graphics::Direct3D12::D3D12_FEATURE_D3D12_OPTIONS16,
                    (&mut options16 as *mut _) as *mut _,
                    std::mem::size_of_val(&options16) as u32,
                )
            }
            .is_ok();
            let dynamic_depth_bias_supported =
                options16_available && options16.DynamicDepthBiasSupported.as_bool();
            let gpu_upload_buffers =
                options16_available && options16.GPUUploadHeapSupported.as_bool();
            let cpu_visible_heaps = memory::cpu_visible::CpuVisibleHeapProfiles::for_device(
                &native,
                memory_topology,
                gpu_upload_buffers,
            );
            let descriptor_indexing = shader_model >= 60;
            let mut options3 =
                windows::Win32::Graphics::Direct3D12::D3D12_FEATURE_DATA_D3D12_OPTIONS3::default();
            // SAFETY: as above, this query receives the exact initialized
            // feature structure and does not retain its pointer.
            let options3_available = unsafe {
                native.CheckFeatureSupport(
                    windows::Win32::Graphics::Direct3D12::D3D12_FEATURE_D3D12_OPTIONS3,
                    (&mut options3 as *mut _) as *mut _,
                    std::mem::size_of_val(&options3) as u32,
                )
            }
            .is_ok();
            let barycentrics_supported = shader_model >= 61
                && options3_available
                && options3.BarycentricsSupported.as_bool();
            let multiview = options3_available
                && options3.ViewInstancingTier.0
                    >= windows::Win32::Graphics::Direct3D12::D3D12_VIEW_INSTANCING_TIER_1.0
                && native
                    .cast::<windows::Win32::Graphics::Direct3D12::ID3D12Device2>()
                    .is_ok();
            let shader_view_index = multiview && shader_model >= 61;
            let mesh_multiview = multiview && mesh_supported;
            let lane = LaneId { device, ordinal: 0 };
            let compute_lane = LaneId { device, ordinal: 1 };
            let caps = DeviceCapabilities {
                shader_f16: native_16bit_shader_ops,
                shader_i16: native_16bit_shader_ops,
                shader_f64,
                shader_int64,
                memory_topology,
                subgroup_widths,
                lanes: Arc::from([
                    LaneInfo {
                        id: lane,
                        class: LaneClass::Graphics,
                        routes: vec![
                            (lane, api::WaitRoute::SameQueue),
                            (compute_lane, api::WaitRoute::NativeGpuWait),
                        ],
                    },
                    LaneInfo {
                        id: compute_lane,
                        class: LaneClass::Compute,
                        routes: vec![
                            (compute_lane, api::WaitRoute::SameQueue),
                            (lane, api::WaitRoute::NativeGpuWait),
                        ],
                    },
                ]),
                limits: DeviceLimits {
                    max_bind_groups: MAX_BIND_GROUPS,
                    descriptor_table_limits: Some(descriptor_table_limits),
                    max_bindings_per_group: MAX_BINDINGS_PER_GROUP,
                    // Each dynamic storage slot uses one SRV or UAV table entry.
                    max_dynamic_storage_buffers_per_pipeline_layout: (MAX_BIND_GROUPS
                        * MAX_BINDINGS_PER_GROUP)
                        .min(
                            descriptor_table_limits.read_only_resources
                                + descriptor_table_limits.writable_resources,
                        )
                        .min(descriptor_table_limits.total_resource_descriptors),
                    max_dynamic_uniform_buffers_per_pipeline_layout: if options.ResourceBindingTier
                        == D3D12_RESOURCE_BINDING_TIER_3
                    {
                        pipeline::MAX_DYNAMIC_UNIFORM_BUFFERS
                    } else {
                        pipeline::MAX_DYNAMIC_UNIFORM_BUFFERS
                            .min(D3D12_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT)
                    },
                    max_color_attachments: D3D12_SIMULTANEOUS_RENDER_TARGET_COUNT,
                    max_texture_dimension_1d: D3D12_REQ_TEXTURE1D_U_DIMENSION,
                    max_texture_dimension_2d: D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION,
                    max_texture_dimension_3d: D3D12_REQ_TEXTURE3D_U_V_OR_W_DIMENSION,
                    max_texture_array_layers: D3D12_REQ_TEXTURE2D_ARRAY_AXIS_DIMENSION,
                    max_multiview_views: if multiview {
                        windows::Win32::Graphics::Direct3D12::D3D12_MAX_VIEW_INSTANCE_COUNT
                    } else {
                        0
                    },
                    max_mesh_indirect_count: if mesh_supported { u32::MAX } else { 0 },
                    max_mesh_workgroup_total_count: if mesh_supported {
                        mesh::MAX_DISPATCH_WORKGROUPS
                    } else {
                        0
                    },
                    max_mesh_workgroups_per_dimension: if mesh_supported {
                        mesh::MAX_DISPATCH_WORKGROUPS_PER_DIMENSION
                    } else {
                        0
                    },
                    mesh_shader_limits,
                    max_ray_dispatch_count: if ray_supported {
                        D3D12_RAYTRACING_MAX_RAY_GENERATION_SHADER_THREADS
                    } else {
                        0
                    },
                    max_ray_recursion_depth: if ray_supported {
                        D3D12_RAYTRACING_MAX_DECLARABLE_TRACE_RECURSION_DEPTH
                    } else {
                        0
                    },
                    max_blas_primitive_count: if ray_supported {
                        D3D12_RAYTRACING_MAX_PRIMITIVES_PER_BOTTOM_LEVEL_ACCELERATION_STRUCTURE
                    } else {
                        0
                    },
                    max_blas_geometry_count: if ray_supported {
                        D3D12_RAYTRACING_MAX_GEOMETRIES_PER_BOTTOM_LEVEL_ACCELERATION_STRUCTURE
                    } else {
                        0
                    },
                    max_tlas_instance_count: if ray_supported {
                        D3D12_RAYTRACING_MAX_INSTANCES_PER_TOP_LEVEL_ACCELERATION_STRUCTURE
                    } else {
                        0
                    },
                    max_buffer_bytes,
                    max_vertex_buffers: D3D12_IA_VERTEX_INPUT_RESOURCE_SLOT_COUNT,
                    max_vertex_attributes: D3D12_IA_VERTEX_INPUT_STRUCTURE_ELEMENT_COUNT,
                    max_vertex_buffer_array_stride: D3D12_REQ_MULTI_ELEMENT_STRUCTURE_SIZE_IN_BYTES,
                    max_compute_workgroups: compute::MAX_DISPATCH_GROUPS,
                    max_compute_workgroup_size: compute::MAX_WORKGROUP_SIZE,
                    max_compute_invocations_per_workgroup: compute::MAX_WORKGROUP_INVOCATIONS,
                    max_compute_workgroup_storage_size: compute::MAX_WORKGROUP_STORAGE_SIZE,
                    uniform_offset_alignment: descriptor::CBV_ALIGNMENT,
                    max_uniform_buffer_binding_size: descriptor::MAX_UNIFORM_BUFFER_BINDING_SIZE,
                    max_push_constant_bytes: pipeline::MAX_IMMEDIATE_BYTES,
                    ..Default::default()
                },
                compute: true,
                multi_queue: true,
                mesh: mesh_supported,
                mesh_multi_indirect: mesh_supported,
                mesh_indirect_count: mesh_supported,
                sparse: tiled_resources_tier != D3D12_TILED_RESOURCES_TIER_NOT_SUPPORTED,
                ray: ray_supported,
                acceleration_structure: ray_supported,
                acceleration_structure_update: ray_supported,
                acceleration_geometry_transform: ray_supported,
                acceleration_build_size_output: ray_supported,
                shader_layout_inference: true,
                acceleration_compaction_size: ray_supported,
                acceleration_compaction: ray_supported,
                acceleration_structure_binding_arrays: ray_supported && descriptor_indexing,
                runtime_sized_acceleration_structure_binding_arrays: ray_supported
                    && descriptor_indexing,
                partially_bound_acceleration_structure_binding_arrays: ray_supported
                    && descriptor_indexing,
                ray_query: ray_query_supported,
                shader_primitive_index: shader_model >= 60,
                shader_early_depth_test: shader_model >= 60,
                shader_barycentrics: barycentrics_supported,
                storage_textures: true,
                texture_atomic_r32uint_compute,
                texture_atomic_r32sint_compute,
                texture_atomic_r32sint_min_compute,
                texture_atomic_r32sint_max_compute,
                texture_atomic_r32uint_min_compute,
                texture_atomic_r32uint_max_compute,
                texture_atomic_r32uint_xor_compute,
                texture_atomic_r32uint_and_compute,
                texture_atomic_r32uint_or_compute,
                storage_buffers: true,
                timestamp: true,
                timestamp_raster_pass: true,
                timestamp_compute_pass: true,
                occlusion: true,
                pipeline_statistics: true,
                debug_markers: true,
                debug_markers_encoder: true,
                debug_markers_raster_pass: true,
                debug_markers_compute_pass: true,
                memory_observation: true,
                dynamic_depth_bias: dynamic_depth_bias_supported,
                dynamic_depth_bias_clamp: dynamic_depth_bias_supported,
                stencil_reference_per_face,
                stencil_masks_per_face: stencil_reference_per_face,
                rect_attachment_clear: true,
                conservative_raster: Some(conservative_raster),
                direct_buffer_upload: true,
                gpu_upload_buffers,
                direct_texture_upload: true,
                completion_timeout: true,
                sampled_texture_component_mapping: true,
                texture_view_usage: true,
                texture_3d_attachment_slice: true,
                texel_aligned_color_copies,
                planar_nv12,
                planar_p010,
                push_constants: true,
                indirect_draw: true,
                indirect_first_instance: true,
                multi_draw: true,
                multi_draw_count: true,
                multi_draw_count_profile: Some(encoder::MULTI_DRAW_COUNT_PROFILE),
                binding_arrays: descriptor_indexing,
                runtime_sized_binding_arrays: descriptor_indexing,
                // The DX12-local Naga patch separates descriptor selection
                // from raw-buffer addressing. Layout/shader admission bounds
                // storage arrays to fixed, fully-bound compute bindings.
                nonuniform_sampled_texture_and_storage_buffer_indexing: descriptor_indexing,
                // Fixed CBV arrays use ConstantBuffer<T> declarations and
                // explicit non-uniform selectors. Shared buffer-array
                // admission enforces the same fully-bound compute scope.
                nonuniform_uniform_buffer_indexing: descriptor_indexing,
                nonuniform_storage_buffer_indexing: descriptor_indexing,
                nonuniform_sampled_texture_indexing: descriptor_indexing,
                nonuniform_storage_texture_indexing: descriptor_indexing,
                nonuniform_sampler_indexing: descriptor_indexing,
                nonuniform_acceleration_structure_indexing: ray_supported && descriptor_indexing,
                secondary_commands: true,
                native_interop: true,
                external_memory: true,
                placed_memory: true,
                explicit_heap_residency: true,
                cpu_visible_heap_residency: true,
                committed_resource_residency: true,
                sparse_allocation_residency: tiled_resources_tier
                    != D3D12_TILED_RESOURCES_TIER_NOT_SUPPORTED,
                async_heap_residency: residency_device.is_some(),
                residency_queue_waits: residency_device.is_some(),
                multiview,
                mesh_multiview,
                shader_view_index,
                ..Default::default()
            };
            caps.require_named_features(&request.required_features)?;
            caps.limits.require_capacities(minimum)?;
            let queue_desc = D3D12_COMMAND_QUEUE_DESC {
                Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                Priority: D3D12_COMMAND_QUEUE_PRIORITY_NORMAL.0,
                Flags: D3D12_COMMAND_QUEUE_FLAG_NONE,
                NodeMask: 0,
            };
            // SAFETY: queue descriptor is fully initialized and native owns
            // the returned queue COM interface.
            let queue = unsafe { native.CreateCommandQueue::<ID3D12CommandQueue>(&queue_desc) }
                .map_err(native_error)?;
            // SAFETY: the device creates and returns an owned fence with the
            // supplied initial value and no borrowed pointer arguments.
            let fence = unsafe { native.CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_NONE) }
                .map_err(native_error)?;
            let descriptors = descriptor::create_heaps(&native)?;
            let cpu_descriptors = descriptor::CpuDescriptorPool::new(&native);
            let debug_info = if diagnostics {
                let info: ID3D12InfoQueue1 = native.cast().map_err(native_error)?;
                let mut cookie = 0;
                // SAFETY: info owns the callback registration until device
                // destruction; the callback context is null and its ABI matches.
                unsafe {
                    info.RegisterMessageCallback(
                        Some(debug_message),
                        D3D12_MESSAGE_CALLBACK_FLAG_NONE,
                        std::ptr::null_mut(),
                        &mut cookie,
                    )
                }
                .map_err(native_error)?;
                Some(info)
            } else {
                None
            };
            Ok((
                Dx12Device(Arc::new(Dx12DeviceState {
                    native: native.clone(),
                    descriptor_table_limits,
                    texel_aligned_color_copies,
                    planar_nv12,
                    planar_p010,
                    shader_model,
                    shader_view_index,
                    native_16bit_shader_ops,
                    shader_f64,
                    shader_int64,
                    barycentrics_supported,
                    subgroup_widths,
                    max_buffer_bytes,
                    ray_device,
                    dxr_1_1,
                    ray_query_supported,
                    mesh_supported,
                    mesh_multiview,
                    tiled_resources_tier,
                    residency_device,
                    resource_heap_tier2,
                    enhanced_barriers_supported,
                    dynamic_depth_bias_supported,
                    gpu_upload_buffers,
                    cpu_visible_heaps,
                    stencil_reference_per_face,
                    sparse_in_flight: Mutex::new(Vec::new()),
                    multi_queue: queue::MultiQueueState::new(&native)?,
                    queue,
                    fence,
                    next_fence: AtomicU64::new(1),
                    queue_lock: Mutex::new(()),
                    query_submission_gate: Mutex::new(()),
                    in_flight: Mutex::new(Vec::new()),
                    direct_commands: Arc::default(),
                    upload_pool: Arc::default(),
                    blit_compute_cache: Arc::default(),
                    mip_blit_cache: Arc::default(),
                    integer_clear_cache: Arc::default(),
                    indirect_pack_cache: Arc::default(),
                    indirect_signatures: Arc::default(),
                    mesh_indirect_signature: Mutex::new(None),
                    abandoned_fence_events: Mutex::new(Vec::new()),
                    descriptors,
                    cpu_descriptors,
                    default_resources: memory::DefaultResourcePool::new(
                        native.clone(),
                        adapter.cast().ok(),
                    ),
                    _debug_info: debug_info,
                })),
                caps,
            ))
        })();
        ready(result)
    }
}

#[cfg(test)]
#[path = "../../tests/backend/device.rs"]
mod tests;

#[cfg(all(test, feature = "hardware-tests"))]
#[path = "../../tests/backend/device/adapter_gpu.rs"]
mod adapter_gpu;
