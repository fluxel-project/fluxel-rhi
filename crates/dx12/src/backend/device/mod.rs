//! Native adapter discovery, device creation and runtime capability queries.
//!
//! Every optional path is selected from native device facts. The device owns
//! shared queues, descriptor heaps and immutable native caches; it does not
//! infer resource uses or build an execution schedule.

use super::*;
mod shader_model;
use windows::Win32::Graphics::Direct3D12::{
    D3D12_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT, D3D12_FEATURE_ARCHITECTURE,
    D3D12_FEATURE_DATA_ARCHITECTURE, D3D12_RAYTRACING_MAX_DECLARABLE_TRACE_RECURSION_DEPTH,
    D3D12_RAYTRACING_MAX_GEOMETRIES_PER_BOTTOM_LEVEL_ACCELERATION_STRUCTURE,
    D3D12_RAYTRACING_MAX_INSTANCES_PER_TOP_LEVEL_ACCELERATION_STRUCTURE,
    D3D12_RAYTRACING_MAX_PRIMITIVES_PER_BOTTOM_LEVEL_ACCELERATION_STRUCTURE,
    D3D12_RAYTRACING_MAX_RAY_GENERATION_SHADER_THREADS,
    D3D12_REQ_MULTI_ELEMENT_STRUCTURE_SIZE_IN_BYTES, D3D12_REQ_TEXTURE1D_U_DIMENSION,
    D3D12_REQ_TEXTURE2D_ARRAY_AXIS_DIMENSION, D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION,
    D3D12_REQ_TEXTURE3D_U_V_OR_W_DIMENSION, D3D12_RESOURCE_BINDING_TIER_3,
};
use windows::Win32::Graphics::Dxgi::{
    DXGI_ADAPTER_DESC1, DXGI_ADAPTER_FLAG_SOFTWARE, DXGI_ERROR_NOT_FOUND, DXGI_ERROR_UNSUPPORTED,
    DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, DXGI_GPU_PREFERENCE_MINIMUM_POWER, IDXGIAdapter1,
    IDXGIDevice, IDXGIFactory4, IDXGIFactory6,
};

const REQUESTED_ADAPTER_UNAVAILABLE: &str = "the requested DXGI adapter";
// Software selection must not reuse the ordinal of a physical DXGI adapter.
const WARP_ORDINAL: u64 = u32::MAX as u64 + 1;

/// Resolves a public DXGI ordinal without narrowing an untrusted `u64`.
///
/// DXGI ordinals are dense `u32` values and end with `DXGI_ERROR_NOT_FOUND`.
/// Keeping that translation here makes `enumerate` and `open` agree on both
/// the ordinal namespace and the structured error returned for stale IDs.
fn adapter_at(instance: &Dx12Instance, ordinal: u64) -> Result<IDXGIAdapter1> {
    if ordinal == WARP_ORDINAL {
        let factory = instance
            .factory
            .cast::<IDXGIFactory4>()
            .map_err(|_| Error::Unsupported("DXGI WARP selection"))?;
        // SAFETY: the factory is live and DXGI returns an owned adapter reference.
        return unsafe { factory.EnumWarpAdapter() }.map_err(native_error);
    }
    let ordinal =
        u32::try_from(ordinal).map_err(|_| Error::Unsupported(REQUESTED_ADAPTER_UNAVAILABLE))?;
    // SAFETY: the factory is owned by `instance`; the returned COM reference
    // is transferred to the caller on success.
    match unsafe { instance.factory.EnumAdapters1(ordinal) } {
        Ok(adapter) => Ok(adapter),
        Err(error) if error.code() == DXGI_ERROR_NOT_FOUND => {
            Err(Error::Unsupported(REQUESTED_ADAPTER_UNAVAILABLE))
        }
        Err(error) => Err(native_error(error)),
    }
}

fn adapter_desc(adapter: &IDXGIAdapter1) -> Result<DXGI_ADAPTER_DESC1> {
    // SAFETY: `adapter` is a live COM interface and GetDesc1 only writes an
    // initialized descriptor returned by the Windows binding.
    unsafe { adapter.GetDesc1() }.map_err(native_error)
}

fn desc_name(desc: &DXGI_ADAPTER_DESC1) -> String {
    let end = desc
        .Description
        .iter()
        .position(|x| *x == 0)
        .unwrap_or(desc.Description.len());
    String::from_utf16_lossy(&desc.Description[..end])
}

fn adapter_name(adapter: &IDXGIAdapter1) -> Result<String> {
    Ok(desc_name(&adapter_desc(adapter)?))
}

fn is_software(desc: &DXGI_ADAPTER_DESC1) -> bool {
    desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0
}

fn supports_d3d12(adapter: &IDXGIAdapter1) -> bool {
    // SAFETY: a null output asks D3D12 to check the requested minimum feature
    // level without creating a device. The input adapter remains live.
    unsafe {
        D3D12CreateDevice::<_, ID3D12Device>(adapter, D3D_FEATURE_LEVEL_11_0, std::ptr::null_mut())
    }
    .is_ok()
}

fn select_adapter(
    instance: &Dx12Instance,
    scope: InstanceId,
    preference: api::AdapterPreference,
) -> Result<AdapterInfo> {
    use api::AdapterPreference;

    if preference == AdapterPreference::Software {
        let adapter = adapter_at(instance, WARP_ORDINAL)?;
        if !supports_d3d12(&adapter) {
            return Err(Error::Unsupported("D3D12 on the WARP adapter"));
        }
        return Ok(AdapterInfo {
            id: AdapterId {
                instance: scope,
                ordinal: WARP_ORDINAL,
            },
            name: adapter_name(&adapter)?,
        });
    }

    if preference == AdapterPreference::Default {
        return Dx12::enumerate(instance, scope)?
            .into_iter()
            .next()
            .ok_or(Error::Unsupported("a Direct3D 12 adapter"));
    }

    let preference = match preference {
        AdapterPreference::LowPower => DXGI_GPU_PREFERENCE_MINIMUM_POWER,
        AdapterPreference::HighPerformance => DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
        _ => unreachable!("default and software selection are handled above"),
    };
    let factory = instance
        .factory
        .cast::<IDXGIFactory6>()
        .map_err(|_| Error::Unsupported("DXGI GPU preference selection"))?;
    for rank in 0..=u32::MAX {
        // SAFETY: this is a live factory and DXGI returns an owned adapter.
        let adapter: IDXGIAdapter1 =
            match unsafe { factory.EnumAdapterByGpuPreference(rank, preference) } {
                Ok(adapter) => adapter,
                Err(error) if error.code() == DXGI_ERROR_NOT_FOUND => break,
                Err(error) => return Err(native_error(error)),
            };
        let desc = adapter_desc(&adapter)?;
        if is_software(&desc) || !supports_d3d12(&adapter) {
            continue;
        }
        // Preference order is different from enumeration order. Match native
        // identities before returning an ID that Instance::open can resolve.
        for ordinal in 0..=u32::MAX {
            match adapter_at(instance, u64::from(ordinal)) {
                Ok(candidate) if adapter_desc(&candidate)?.AdapterLuid == desc.AdapterLuid => {
                    return Ok(AdapterInfo {
                        id: AdapterId {
                            instance: scope,
                            ordinal: u64::from(ordinal),
                        },
                        name: desc_name(&desc),
                    });
                }
                Ok(_) => {}
                Err(Error::Unsupported(REQUESTED_ADAPTER_UNAVAILABLE)) => break,
                Err(error) => return Err(error),
            }
        }
    }
    Err(Error::Unsupported(
        "a Direct3D 12 adapter for the requested preference",
    ))
}

impl InstanceBackend for Dx12 {
    type NativeInstance = Dx12Instance;
    type OpenFuture<'a> = Ready<Result<(Dx12Device, DeviceCapabilities)>>;
    fn enumerate(instance: &Dx12Instance, scope: InstanceId) -> Result<Vec<AdapterInfo>> {
        let mut adapters = Vec::new();
        for ordinal in 0..=u32::MAX {
            match adapter_at(instance, u64::from(ordinal)) {
                Ok(adapter) => adapters.push(AdapterInfo {
                    id: AdapterId {
                        instance: scope,
                        ordinal: u64::from(ordinal),
                    },
                    name: adapter_name(&adapter)?,
                }),
                Err(Error::Unsupported(REQUESTED_ADAPTER_UNAVAILABLE)) => break,
                Err(error) => return Err(error),
            }
        }
        Ok(adapters)
    }
    fn request_adapter(
        instance: &Dx12Instance,
        scope: InstanceId,
        preference: api::AdapterPreference,
    ) -> Result<AdapterInfo> {
        select_adapter(instance, scope, preference)
    }
    fn adapter_properties(
        instance: &Dx12Instance,
        id: AdapterId,
    ) -> Result<api::AdapterProperties> {
        let adapter = adapter_at(instance, id.ordinal)?;
        let desc = adapter_desc(&adapter)?;
        // SAFETY: the adapter and IID remain live. The optional DXGI query
        // returns a version by value.
        let driver_version = unsafe { adapter.CheckInterfaceSupport(&IDXGIDevice::IID) }
            .ok()
            .map(|version| {
                let version = version as u64;
                [48, 32, 16, 0].map(|shift| (version >> shift) as u16)
            });
        Ok(api::AdapterProperties {
            backend: "dx12",
            vendor_id: Some(desc.VendorId),
            device_id: Some(desc.DeviceId),
            driver_version,
            is_software: Some(is_software(&desc)),
            dedicated_video_memory: Some(desc.DedicatedVideoMemory as u64),
            shared_system_memory: Some(desc.SharedSystemMemory as u64),
        })
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
            let wave_ops = unsafe {
                native.CheckFeatureSupport(
                    windows::Win32::Graphics::Direct3D12::D3D12_FEATURE_D3D12_OPTIONS1,
                    (&mut wave as *mut _) as *mut _,
                    std::mem::size_of_val(&wave) as u32,
                )
            }
            .is_ok()
                && wave.WaveOps.as_bool();
            let subgroup_widths = (wave_ops
                && wave.WaveLaneCountMin > 0
                && wave.WaveLaneCountMin <= wave.WaveLaneCountMax)
                .then_some(api::SubgroupWidths {
                    min: wave.WaveLaneCountMin,
                    max: wave.WaveLaneCountMax,
                });
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
            let ray_device = ray::probe(&native);
            let ray_supported = ray_device.is_some();
            let ray_query_supported = ray::probe_inline(&native, shader_model);
            let mesh_supported = mesh::probe(&native, shader_model);
            let texture_atomic_r32uint_compute = texture::format_supports_texture_atomic_add_native(
                &native,
                api::TextureFormat::R32Uint,
            );
            let texture_atomic_r32sint_compute = texture::format_supports_texture_atomic_add_native(
                &native,
                api::TextureFormat::R32Sint,
            );
            let texture_atomic_r32sint_min_compute =
                texture::format_supports_texture_atomic_signed_min_native(
                    &native,
                    api::TextureFormat::R32Sint,
                );
            let texture_atomic_r32sint_max_compute = texture_atomic_r32sint_min_compute;
            let texture_atomic_r32uint_min_compute =
                texture::format_supports_texture_atomic_unsigned_min_native(
                    &native,
                    api::TextureFormat::R32Uint,
                );
            // D3D12 exposes one format-support bit for unsigned texture min
            // and max; keep the public operation capabilities separate while
            // deriving both from that native fact.
            let texture_atomic_r32uint_max_compute = texture_atomic_r32uint_min_compute;
            let texture_atomic_r32uint_bitwise_compute =
                texture::format_supports_texture_atomic_bitwise_native(
                    &native,
                    api::TextureFormat::R32Uint,
                );
            let texture_atomic_r32uint_xor_compute = texture_atomic_r32uint_bitwise_compute;
            let texture_atomic_r32uint_and_compute = texture_atomic_r32uint_bitwise_compute;
            let texture_atomic_r32uint_or_compute = texture_atomic_r32uint_bitwise_compute;
            let mut options =
                windows::Win32::Graphics::Direct3D12::D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
            // SAFETY: this structure matches the feature query selector.
            let sparse_supported = unsafe {
            native.CheckFeatureSupport(
                windows::Win32::Graphics::Direct3D12::D3D12_FEATURE_D3D12_OPTIONS,
                &mut options as *mut _ as *mut _,
                std::mem::size_of_val(&options) as u32,
            )
        }
        .is_ok()
            && options.TiledResourcesTier
                != windows::Win32::Graphics::Direct3D12::D3D12_TILED_RESOURCES_TIER_NOT_SUPPORTED;
            let resource_heap_tier2 = options.ResourceHeapTier.0
                >= windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_HEAP_TIER_2.0;
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
            let mut options16 =
                windows::Win32::Graphics::Direct3D12::D3D12_FEATURE_DATA_D3D12_OPTIONS16::default();
            // SAFETY: the feature structure exactly matches the synchronous
            // OPTIONS16 query and remains live for its whole call.
            let dynamic_depth_bias_supported = unsafe {
                native.CheckFeatureSupport(
                    windows::Win32::Graphics::Direct3D12::D3D12_FEATURE_D3D12_OPTIONS16,
                    (&mut options16 as *mut _) as *mut _,
                    std::mem::size_of_val(&options16) as u32,
                )
            }
            .is_ok()
                && options16.DynamicDepthBiasSupported.as_bool();
            let descriptor_indexing = shader_model >= 60;
            let mut options3 =
                windows::Win32::Graphics::Direct3D12::D3D12_FEATURE_DATA_D3D12_OPTIONS3::default();
            // SAFETY: as above, this query receives the exact initialized
            // feature structure and does not retain its pointer.
            let multiview = unsafe {
                native.CheckFeatureSupport(
                    windows::Win32::Graphics::Direct3D12::D3D12_FEATURE_D3D12_OPTIONS3,
                    (&mut options3 as *mut _) as *mut _,
                    std::mem::size_of_val(&options3) as u32,
                )
            }
            .is_ok()
                && options3.ViewInstancingTier.0
                    >= windows::Win32::Graphics::Direct3D12::D3D12_VIEW_INSTANCING_TIER_1.0
                && native
                    .cast::<windows::Win32::Graphics::Direct3D12::ID3D12Device2>()
                    .is_ok();
            let lane = LaneId { device, ordinal: 0 };
            let compute_lane = LaneId { device, ordinal: 1 };
            let caps = DeviceCapabilities {
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
                    max_bind_groups: 8,
                    max_bindings_per_group: 64,
                    max_dynamic_uniform_buffers_per_pipeline_layout: if options.ResourceBindingTier
                        == D3D12_RESOURCE_BINDING_TIER_3
                    {
                        pipeline::MAX_DYNAMIC_UNIFORM_BUFFERS
                    } else {
                        pipeline::MAX_DYNAMIC_UNIFORM_BUFFERS
                            .min(D3D12_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT)
                    },
                    max_color_attachments: 8,
                    max_texture_dimension_1d: D3D12_REQ_TEXTURE1D_U_DIMENSION,
                    max_texture_dimension_2d: D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION,
                    max_texture_dimension_3d: D3D12_REQ_TEXTURE3D_U_V_OR_W_DIMENSION,
                    max_texture_array_layers: D3D12_REQ_TEXTURE2D_ARRAY_AXIS_DIMENSION,
                    max_multiview_views: if multiview { 4 } else { 0 },
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
                    max_vertex_buffers: 32,
                    max_vertex_attributes: 32,
                    max_vertex_buffer_array_stride: D3D12_REQ_MULTI_ELEMENT_STRUCTURE_SIZE_IN_BYTES,
                    max_compute_workgroups: compute::MAX_DISPATCH_GROUPS,
                    max_compute_workgroup_size: compute::MAX_WORKGROUP_SIZE,
                    max_compute_invocations_per_workgroup: compute::MAX_WORKGROUP_INVOCATIONS,
                    max_compute_workgroup_storage_size: compute::MAX_WORKGROUP_STORAGE_SIZE,
                    uniform_offset_alignment: descriptor::CBV_ALIGNMENT,
                    max_uniform_buffer_binding_size: descriptor::MAX_UNIFORM_BUFFER_BINDING_SIZE,
                    max_push_constant_bytes: 128,
                    ..Default::default()
                },
                compute: true,
                multi_queue: true,
                mesh: mesh_supported,
                mesh_multi_indirect: mesh_supported,
                mesh_indirect_count: mesh_supported,
                sparse: sparse_supported,
                ray: ray_supported,
                acceleration_structure: ray_supported,
                acceleration_structure_binding_arrays: ray_supported && descriptor_indexing,
                ray_query: ray_query_supported,
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
                occlusion: true,
                pipeline_statistics: true,
                debug_markers: true,
                debug_markers_encoder: true,
                debug_markers_raster_pass: true,
                debug_markers_compute_pass: true,
                memory_observation: true,
                dynamic_depth_bias: dynamic_depth_bias_supported,
                dynamic_depth_bias_clamp: dynamic_depth_bias_supported,
                rect_attachment_clear: true,
                push_constants: true,
                indirect_draw: true,
                indirect_first_instance: true,
                multi_draw: true,
                multi_draw_count: true,
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
                multiview,
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
                    shader_model,
                    subgroup_widths,
                    max_buffer_bytes,
                    ray_device,
                    ray_query_supported,
                    mesh_supported,
                    sparse_supported,
                    resource_heap_tier2,
                    enhanced_barriers_supported,
                    dynamic_depth_bias_supported,
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
