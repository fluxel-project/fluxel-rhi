//! Native adapter discovery, device creation and runtime capability queries.
//!
//! Every optional path is selected from native device facts. The device owns
//! shared queues, descriptor heaps and immutable native caches; it does not
//! infer resource uses or build an execution schedule.

use super::*;
use windows::Win32::Graphics::Dxgi::{DXGI_ERROR_NOT_FOUND, DXGI_ERROR_UNSUPPORTED, IDXGIAdapter1};

const REQUESTED_ADAPTER_UNAVAILABLE: &str = "the requested DXGI adapter";

/// Resolves a public DXGI ordinal without narrowing an untrusted `u64`.
///
/// DXGI ordinals are dense `u32` values and end with `DXGI_ERROR_NOT_FOUND`.
/// Keeping that translation here makes `enumerate` and `open` agree on both
/// the ordinal namespace and the structured error returned for stale IDs.
fn adapter_at(instance: &Dx12Instance, ordinal: u64) -> Result<IDXGIAdapter1> {
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

fn adapter_name(adapter: &IDXGIAdapter1) -> Result<String> {
    // SAFETY: `adapter` is a live COM interface and GetDesc1 only writes an
    // initialized descriptor returned by the Windows binding.
    let desc = unsafe { adapter.GetDesc1() }.map_err(native_error)?;
    let end = desc
        .Description
        .iter()
        .position(|x| *x == 0)
        .unwrap_or(desc.Description.len());
    Ok(String::from_utf16_lossy(&desc.Description[..end]))
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
    fn open<'a>(
        instance: &'a Dx12Instance,
        device: DeviceId,
        request: &'a DeviceRequest,
    ) -> Self::OpenFuture<'a> {
        let result = (|| {
            if !request.required_features.is_empty() {
                return Err(Error::Unsupported("requested DX12 device features"));
            }
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
            let ray_query_supported = ray::probe_inline(&native);
            let mesh_supported = mesh::probe(&native);
            let mut options =
                windows::Win32::Graphics::Direct3D12::D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
            // SAFETY: this structure matches the feature query selector.
            let sparse_supported = unsafe { native.CheckFeatureSupport(windows::Win32::Graphics::Direct3D12::D3D12_FEATURE_D3D12_OPTIONS, &mut options as *mut _ as *mut _, std::mem::size_of_val(&options) as u32) }.is_ok()
                && options.TiledResourcesTier != windows::Win32::Graphics::Direct3D12::D3D12_TILED_RESOURCES_TIER_NOT_SUPPORTED;
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
            let mut shader_model =
                windows::Win32::Graphics::Direct3D12::D3D12_FEATURE_DATA_SHADER_MODEL {
                    HighestShaderModel: windows::Win32::Graphics::Direct3D12::D3D_SHADER_MODEL_6_0,
                };
            // SAFETY: the feature struct is the exact size/type requested and
            // stays valid for D3D12's synchronous feature query.
            let descriptor_indexing = unsafe {
                native.CheckFeatureSupport(
                    windows::Win32::Graphics::Direct3D12::D3D12_FEATURE_SHADER_MODEL,
                    (&mut shader_model as *mut _) as *mut _,
                    std::mem::size_of_val(&shader_model) as u32,
                )
            }
            .is_ok()
                && shader_model.HighestShaderModel.0
                    >= windows::Win32::Graphics::Direct3D12::D3D_SHADER_MODEL_6_0.0;
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
            let lane = LaneId { device, ordinal: 0 };
            let compute_lane = LaneId { device, ordinal: 1 };
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
                    abandoned_fence_events: Mutex::new(Vec::new()),
                    descriptors,
                    cpu_descriptors,
                    default_resources: memory::DefaultResourcePool::new(
                        native.clone(),
                        adapter.cast().ok(),
                    ),
                    _debug_info: debug_info,
                })),
                DeviceCapabilities {
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
                        max_color_attachments: 8,
                        max_texture_dimension_2d: 16_384,
                        max_texture_array_layers: 2_048,
                        max_multiview_views: if multiview { 4 } else { 0 },
                        max_buffer_bytes,
                        max_vertex_buffers: 32,
                        max_vertex_attributes: 32,
                        // D3D12 guarantees these dispatch dimensions at feature
                        // level 11; this path exposes one direct queue.
                        max_compute_workgroups: [65_535; 3],
                        max_compute_workgroup_size: [1_024, 1_024, 64],
                        uniform_offset_alignment: 256,
                        max_push_constant_bytes: 128,
                        ..Default::default()
                    },
                    compute: true,
                    multi_queue: true,
                    mesh: mesh_supported,
                    sparse: sparse_supported,
                    ray: ray_supported,
                    acceleration_structure: ray_supported,
                    acceleration_structure_binding_arrays: ray_supported && descriptor_indexing,
                    ray_query: ray_query_supported,
                    storage_textures: true,
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
                    nonuniform_sampled_texture_and_storage_buffer_indexing: descriptor_indexing,
                    nonuniform_uniform_buffer_indexing: descriptor_indexing,
                    nonuniform_storage_buffer_indexing: descriptor_indexing,
                    nonuniform_sampled_texture_indexing: descriptor_indexing,
                    nonuniform_sampler_indexing: descriptor_indexing,
                    nonuniform_acceleration_structure_indexing: ray_supported
                        && descriptor_indexing,
                    secondary_commands: true,
                    native_interop: true,
                    external_memory: true,
                    placed_memory: true,
                    multiview,
                    ..Default::default()
                },
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
