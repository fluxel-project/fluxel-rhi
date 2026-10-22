//! Statically typed DX12 instance, device, resource, and native-boundary entry points.
//!
//! Its native associated types remain concrete through the public API. This
//! module creates the device-wide D3D12 objects and delegates each resource or
//! command domain to a focused child module. `Dx12DeviceState` owns the direct
//! queue, fence, descriptor stores and native-object caches; recorded work
//! retains every command-list dependency until fence retirement in
//! `encoder::submit`. It does not perform render-graph scheduling, hazard
//! planning, or CPU command replay.

use std::{
    collections::HashMap,
    ffi::CStr,
    future::{Ready, ready},
    sync::{Arc, Mutex, atomic::AtomicU64},
};

use fluxel_rhi_core::api::{
    self, AdapterId, AdapterInfo, Backend, DeviceCapabilities, DeviceId, DeviceLimits,
    DeviceRequest, Error, InstanceBackend, InstanceId, LaneClass, LaneId, LaneInfo, Result,
};
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_UNKNOWN, DXGI_SAMPLE_DESC};
use windows::Win32::Graphics::{
    Direct3D::D3D_FEATURE_LEVEL_11_0,
    Direct3D12::{
        D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_COMMAND_QUEUE_DESC, D3D12_COMMAND_QUEUE_FLAG_NONE,
        D3D12_COMMAND_QUEUE_PRIORITY_NORMAL, D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
        D3D12_FENCE_FLAG_NONE, D3D12_HEAP_FLAG_NONE, D3D12_HEAP_PROPERTIES,
        D3D12_HEAP_TYPE_DEFAULT, D3D12_MEMORY_POOL_UNKNOWN, D3D12_MESSAGE_CALLBACK_FLAG_NONE,
        D3D12_MESSAGE_CATEGORY, D3D12_MESSAGE_ID, D3D12_MESSAGE_SEVERITY, D3D12_RESOURCE_DESC,
        D3D12_RESOURCE_DIMENSION_BUFFER, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON,
        D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12CreateDevice, D3D12GetDebugInterface,
        ID3D12CommandQueue, ID3D12CommandSignature, ID3D12Debug, ID3D12Device, ID3D12Fence,
        ID3D12InfoQueue1, ID3D12Resource,
    },
    Dxgi::{CreateDXGIFactory2, DXGI_CREATE_FACTORY_FLAGS, IDXGIFactory1},
};
use windows::core::Interface;

mod compute;
mod descriptor;
mod encoder;
mod mesh;
mod pipeline;
mod query;
mod queue;
mod ray;
mod readback;
mod sampler;
mod secondary;
mod sparse;
mod surface;
mod texture;

/// Direct3D 12 selected once by the host.  Resource and command associated
/// types stay concrete throughout the typed execution API.
pub struct Dx12;
impl Dx12 {
    /// Creates a typed surface for a live Win32 window owned by the caller.
    /// # Safety
    /// The raw window handle must remain valid until the returned surface and
    /// every image lease from it have been retired.
    pub unsafe fn surface_from_raw_window_handle(
        device: &api::Device<Self>,
        handle: raw_window_handle::RawWindowHandle,
    ) -> Result<api::Surface<Self>> {
        // SAFETY: this public method carries the host-window lifetime contract
        // required by the narrower DXGI construction helper.
        unsafe { surface::create_surface(device, handle) }
    }
}

#[derive(Clone)]
pub struct Dx12Device(Arc<Dx12DeviceState>);
struct Dx12DeviceState {
    pub(super) native: ID3D12Device,
    /// Native interface is retained only on devices that report DXR tier 1.0.
    /// Public capability remains disabled until the complete ray path exists.
    pub(super) ray_device: Option<windows::Win32::Graphics::Direct3D12::ID3D12Device5>,
    pub(super) ray_query_supported: bool,
    pub(super) mesh_supported: bool,
    pub(super) sparse_supported: bool,
    pub(super) sparse_in_flight: Mutex<Vec<(u64, Vec<sparse::SparseRetained>)>>,
    pub(super) queue: ID3D12CommandQueue,
    pub(super) multi_queue: queue::MultiQueueState,
    pub(super) fence: ID3D12Fence,
    pub(super) next_fence: AtomicU64,
    pub(super) queue_lock: Mutex<()>,
    pub(super) query_submission_gate: Mutex<()>,
    /// Work remains here until the corresponding native fence proves GPU
    /// completion. A `u64::MAX` value is quarantined indeterminate work.
    pub(super) in_flight: Mutex<Vec<(u64, Vec<encoder::Dx12RecordedWork>)>>,
    pub(super) direct_commands: Arc<encoder::DirectCommandPool>,
    pub(super) mip_blit_cache: Arc<encoder::MipBlitCache>,
    /// Dropped futures cannot unregister SetEventOnCompletion, so their HANDLE
    /// ownership stays here until the associated fence has passed it.
    pub(super) abandoned_fence_events: Mutex<Vec<encoder::fence_wait::AbandonedFenceEvent>>,
    /// Native ExecuteIndirect signatures are immutable and device-owned.
    /// Sharing this cache through each buffer avoids per-recording creation
    /// without introducing a command scheduler or retaining command streams.
    pub(super) indirect_signatures: Arc<Mutex<HashMap<(bool, u32), ID3D12CommandSignature>>>,
    pub(super) descriptors: descriptor::DescriptorHeaps,
    // Retain the registration for the device lifetime when diagnostics are
    // explicitly requested.  Production devices do not enable the debug layer.
    _debug_info: Option<ID3D12InfoQueue1>,
}

unsafe extern "system" fn debug_message(
    _: D3D12_MESSAGE_CATEGORY,
    severity: D3D12_MESSAGE_SEVERITY,
    id: D3D12_MESSAGE_ID,
    description: windows::core::PCSTR,
    _: *mut std::ffi::c_void,
) {
    let text = if description.is_null() {
        "<no message>".into()
    } else {
        // SAFETY: D3D12 invokes this callback with a valid NUL-terminated
        // diagnostic string for the duration of this synchronous callback.
        unsafe { CStr::from_ptr(description.as_ptr().cast()) }
            .to_string_lossy()
            .into_owned()
    };
    eprintln!("D3D12 [{severity:?}/{id:?}]: {text}");
}

/// DXGI factory used by [`api::Instance`] when opening a `Dx12` device.
#[derive(Clone)]
pub struct Dx12Instance {
    factory: IDXGIFactory1,
}

impl Dx12Instance {
    /// Opens the process-local DXGI factory. No adapter or GPU device is
    /// created until `Instance::<Dx12>::open` is called.
    pub fn new() -> Result<Self> {
        // SAFETY: factory creation has no borrowed inputs and returns an owned
        // COM interface on success.
        let factory = unsafe { CreateDXGIFactory2::<IDXGIFactory1>(DXGI_CREATE_FACTORY_FLAGS(0)) }
            .map_err(native_error)?;
        Ok(Self { factory })
    }

    /// Creates an instance with a fresh identity.
    pub fn instance(&self) -> Result<api::Instance<Dx12>> {
        // SAFETY: the fresh identity is unique and this clone retains the
        // DXGI factory for every instance operation.
        Ok(unsafe { api::Instance::from_native(InstanceId::allocate()?, self.clone()) })
    }

    /// Opens the default DXGI adapter through the static API.
    pub async fn open_default(&self) -> Result<api::Device<Dx12>> {
        let instance = self.instance()?;
        let adapter = instance
            .enumerate()?
            .into_iter()
            .next()
            .ok_or(Error::Unsupported("a Direct3D 12 adapter"))?;
        instance
            .open(DeviceRequest {
                adapter: adapter.id,
                required_features: Vec::new(),
            })
            .await
    }
}

pub struct Dx12SurfaceImage(pub(crate) surface::ImageLease);
#[derive(Clone)]
pub struct Dx12Surface(pub(crate) Arc<surface::SurfaceState>);

#[derive(Clone)]
pub struct Dx12Buffer {
    pub(super) resource: ID3D12Resource,
    pub(super) size: u64,
    pub(super) usage: api::BufferUsage,
    pub(super) memory: api::MemoryClass,
    pub(super) indirect_signatures: Arc<Mutex<HashMap<(bool, u32), ID3D12CommandSignature>>>,
}
macro_rules! native_arc { ($($name:ident),* $(,)?) => { $(#[derive(Clone)] pub struct $name(Arc<()>);)* }; }
native_arc!(Dx12PipelineCache);
#[derive(Clone)]
pub struct Dx12Sparse(Arc<sparse::SparseAllocationState>);
#[derive(Clone)]
pub struct Dx12MeshPipeline(pub(crate) Arc<mesh::MeshPipelineState>);
#[derive(Clone)]
pub struct Dx12Acceleration(pub(crate) Arc<ray::AccelerationState>);
#[derive(Clone)]
pub struct Dx12RayPipeline(pub(crate) Arc<ray::pipeline::RayPipelineState>);
#[derive(Clone)]
pub struct Dx12QuerySet(pub(crate) Arc<query::QuerySetState>);
#[derive(Clone)]
pub struct Dx12PipelineLayout(pub(crate) Arc<pipeline::PipelineLayoutState>);
#[derive(Clone)]
pub struct Dx12RasterPipeline(pub(crate) Arc<pipeline::RasterPipelineState>);
#[derive(Clone)]
pub struct Dx12Texture(pub(crate) Arc<texture::TextureState>);
#[derive(Clone)]
pub struct Dx12TextureView(pub(crate) Arc<texture::TextureViewState>);
#[derive(Clone)]
pub struct Dx12Sampler(pub(crate) Arc<sampler::SamplerState>);
#[derive(Clone)]
pub struct Dx12BindGroupLayout(pub(crate) Arc<descriptor::BindGroupLayoutState>);
#[derive(Clone)]
pub struct Dx12BindGroup(pub(crate) Arc<descriptor::BindGroupState>);
#[derive(Clone)]
pub struct Dx12Shader {
    pub(crate) bytes: Arc<[u8]>,
    pub(crate) entries: Vec<api::ShaderEntryPoint>,
}
#[derive(Clone)]
pub struct Dx12ComputePipeline(pub(crate) Arc<compute::ComputePipelineState>);

impl Backend for Dx12 {
    type NativeDevice = Dx12Device;
    type NativeBuffer = Dx12Buffer;
    type NativeTexture = Dx12Texture;
    type NativeTextureView = Dx12TextureView;
    type NativeSampler = Dx12Sampler;
    type NativeBindGroupLayout = Dx12BindGroupLayout;
    type NativeBindGroup = Dx12BindGroup;
    type NativePipelineLayout = Dx12PipelineLayout;
    type NativeShader = Dx12Shader;
    type NativeRasterPipeline = Dx12RasterPipeline;
    type NativeComputePipeline = Dx12ComputePipeline;
    type NativeMeshPipeline = Dx12MeshPipeline;
    type NativeRayPipeline = Dx12RayPipeline;
    type NativePipelineCache = Dx12PipelineCache;
    type NativeAcceleration = Dx12Acceleration;
    type NativeSparseAllocation = Dx12Sparse;
    type NativeQuerySet = Dx12QuerySet;
    type NativeEncoder = encoder::Dx12Encoder;
    type NativeCompletion = encoder::Dx12Completion;
    type NativeSurface = Dx12Surface;
    type NativeSurfaceImage = Dx12SurfaceImage;
    type NativeUncertainty = ();
    type Compute = compute::Compute;
    type Mesh = mesh::Mesh;
    type Ray = ray::Ray;
    type Extras = encoder::RasterExtras;
}

impl InstanceBackend for Dx12 {
    type NativeInstance = Dx12Instance;
    type OpenFuture<'a> = Ready<Result<(Dx12Device, DeviceCapabilities)>>;
    fn enumerate(instance: &Dx12Instance, scope: InstanceId) -> Result<Vec<AdapterInfo>> {
        // SAFETY: the instance owns the factory; ordinal zero is the only
        // adapter this backend advertises and the returned COM object is owned.
        let adapter = unsafe { instance.factory.EnumAdapters1(0) }.map_err(native_error)?;
        // SAFETY: `adapter` is a live COM interface and GetDesc1 only writes
        // an initialized value returned by the Windows binding.
        let desc = unsafe { adapter.GetDesc1() }.map_err(native_error)?;
        let end = desc
            .Description
            .iter()
            .position(|x| *x == 0)
            .unwrap_or(desc.Description.len());
        Ok(vec![AdapterInfo {
            id: AdapterId {
                instance: scope,
                ordinal: 0,
            },
            name: String::from_utf16_lossy(&desc.Description[..end]),
        }])
    }
    fn open<'a>(
        instance: &'a Dx12Instance,
        device: DeviceId,
        request: &'a DeviceRequest,
    ) -> Self::OpenFuture<'a> {
        let result = (|| {
            if request.adapter.ordinal != 0 {
                return Err(Error::Unsupported("the requested DXGI adapter"));
            }
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
            // SAFETY: factory ownership and the advertised ordinal match
            // enumerate above; the returned adapter COM reference is retained locally.
            let adapter = unsafe { instance.factory.EnumAdapters1(0) }.map_err(native_error)?;
            let mut native: Option<ID3D12Device> = None;
            // SAFETY: adapter is live and `native` is the output slot for the
            // newly created device interface.
            unsafe { D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_11_0, &mut native) }
                .map_err(native_error)?;
            let native = native
                .ok_or_else(|| Error::Backend("D3D12CreateDevice returned no device".into()))?;
            let ray_device = ray::probe(&native);
            let ray_supported = ray_device.is_some();
            let ray_query_supported = ray::probe_inline(&native);
            let mesh_supported = mesh::probe(&native);
            let mut options =
                windows::Win32::Graphics::Direct3D12::D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
            // SAFETY: this structure matches the feature query selector.
            let sparse_supported = unsafe { native.CheckFeatureSupport(windows::Win32::Graphics::Direct3D12::D3D12_FEATURE_D3D12_OPTIONS, &mut options as *mut _ as *mut _, std::mem::size_of_val(&options) as u32) }.is_ok()
                && options.TiledResourcesTier != windows::Win32::Graphics::Direct3D12::D3D12_TILED_RESOURCES_TIER_NOT_SUPPORTED;
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
                    ray_device,
                    ray_query_supported,
                    mesh_supported,
                    sparse_supported,
                    sparse_in_flight: Mutex::new(Vec::new()),
                    multi_queue: queue::MultiQueueState::new(&native)?,
                    queue,
                    fence,
                    next_fence: AtomicU64::new(1),
                    queue_lock: Mutex::new(()),
                    query_submission_gate: Mutex::new(()),
                    in_flight: Mutex::new(Vec::new()),
                    direct_commands: Arc::default(),
                    mip_blit_cache: Arc::default(),
                    indirect_signatures: Arc::default(),
                    abandoned_fence_events: Mutex::new(Vec::new()),
                    descriptors,
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
                        max_color_attachments: 1,
                        max_texture_dimension_2d: 16_384,
                        max_texture_array_layers: 2_048,
                        max_multiview_views: if multiview { 2 } else { 0 },
                        max_buffer_bytes: 1 << 30,
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
                    ray_query: ray_query_supported,
                    storage_textures: true,
                    storage_buffers: true,
                    timestamp: true,
                    occlusion: true,
                    pipeline_statistics: true,
                    push_constants: true,
                    indirect_draw: true,
                    multi_draw: true,
                    multi_draw_count: true,
                    binding_arrays: descriptor_indexing,
                    runtime_sized_binding_arrays: descriptor_indexing,
                    nonuniform_sampled_texture_and_storage_buffer_indexing: descriptor_indexing,
                    secondary_commands: true,
                    multiview,
                    ..Default::default()
                },
            ))
        })();
        ready(result)
    }
}

impl api::ResourceBackend for Dx12 {
    fn buffer_support(
        device: &Dx12Device,
        desc: &api::BufferDesc,
    ) -> api::Support<api::BufferSupport> {
        if desc.size == 0 {
            api::Support::Unsupported("zero-sized buffers")
        } else if desc.size > (1 << 30) {
            api::Support::Unsupported("DX12 buffer size limit")
        } else if matches!(desc.memory, api::MemoryClass::Readback) {
            let readback_writes = api::BufferUsage::COPY_DST.0 | api::BufferUsage::QUERY_RESOLVE.0;
            if desc.usage.contains(api::BufferUsage::MAP_READ)
                && desc.usage.0 & readback_writes != 0
                && desc.usage.0 & !(readback_writes | api::BufferUsage::MAP_READ.0) == 0
            {
                api::Support::Supported(api::BufferSupport {
                    alignment: 1,
                    max_size: 1 << 30,
                })
            } else {
                api::Support::Unsupported("DX12 readback buffer usage")
            }
        } else if !matches!(desc.memory, api::MemoryClass::Device) {
            api::Support::Unsupported("DX12 upload buffers")
        } else if desc.usage.0
            & (api::BufferUsage::ACCEL_BUILD_INPUT.0
                | api::BufferUsage::ACCEL_SCRATCH.0
                | api::BufferUsage::SHADER_TABLE.0)
            != 0
            && device.0.ray_device.is_none()
        {
            api::Support::Unsupported("DX12 ray tracing tier 1.0")
        } else if desc.usage.0
            & !(api::BufferUsage::VERTEX.0
                | api::BufferUsage::INDEX.0
                | api::BufferUsage::UNIFORM.0
                | api::BufferUsage::STORAGE.0
                | api::BufferUsage::COPY_DST.0
                | api::BufferUsage::COPY_SRC.0
                | api::BufferUsage::INDIRECT.0
                | api::BufferUsage::QUERY_RESOLVE.0
                | api::BufferUsage::ACCEL_BUILD_INPUT.0
                | api::BufferUsage::ACCEL_SCRATCH.0
                | api::BufferUsage::SHADER_TABLE.0)
            != 0
        {
            api::Support::Unsupported("DX12 buffer usage")
        } else if desc.usage.contains(api::BufferUsage::UNIFORM) && desc.size % 256 != 0 {
            api::Support::Unsupported("DX12 uniform buffer size alignment")
        } else {
            api::Support::Supported(api::BufferSupport {
                alignment: if desc.usage.contains(api::BufferUsage::UNIFORM) {
                    256
                } else {
                    1
                },
                max_size: 1 << 30,
            })
        }
    }
    fn texture_support(
        device: &Dx12Device,
        desc: &api::TextureDesc,
    ) -> api::Support<api::TextureSupport> {
        texture::support(device, desc)
    }
    fn create_buffer(device: &Dx12Device, desc: &api::BufferDesc) -> Result<Dx12Buffer> {
        if let api::Support::Unsupported(reason) =
            <Self as api::ResourceBackend>::buffer_support(device, desc)
        {
            return Err(Error::Unsupported(reason));
        }
        let readback = desc.memory == api::MemoryClass::Readback;
        let heap = D3D12_HEAP_PROPERTIES {
            Type: if readback {
                windows::Win32::Graphics::Direct3D12::D3D12_HEAP_TYPE_READBACK
            } else {
                D3D12_HEAP_TYPE_DEFAULT
            },
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
            MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
            CreationNodeMask: 1,
            VisibleNodeMask: 1,
        };
        let native = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Alignment: 0,
            Width: desc.size,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_UNKNOWN,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: if desc.usage.contains(api::BufferUsage::ACCEL_SCRATCH)
                || desc.usage.contains(api::BufferUsage::STORAGE)
            {
                windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
            } else {
                D3D12_RESOURCE_FLAG_NONE
            },
        };
        let mut resource = None;
        // SAFETY: heap and buffer descriptors are initialized local values;
        // D3D12 writes the owned resource COM interface into `resource` only.
        unsafe {
            device.0.native.CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &native,
                if readback {
                    windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATE_COPY_DEST
                } else {
                    D3D12_RESOURCE_STATE_COMMON
                },
                None,
                &mut resource,
            )
        }
        .map_err(native_error)?;
        Ok(Dx12Buffer {
            resource: resource.ok_or_else(|| {
                Error::Backend("CreateCommittedResource returned no buffer".into())
            })?,
            size: desc.size,
            usage: desc.usage,
            memory: desc.memory,
            indirect_signatures: device.0.indirect_signatures.clone(),
        })
    }
    fn create_texture(device: &Dx12Device, desc: &api::TextureDesc) -> Result<Dx12Texture> {
        texture::create(device, desc)
    }
    fn create_view(
        device: &Dx12Device,
        texture: &Dx12Texture,
        desc: &api::TextureViewDesc,
    ) -> Result<Dx12TextureView> {
        texture::create_view(device, texture, desc)
    }
    fn create_sampler(device: &Dx12Device, desc: &api::SamplerDesc) -> Result<Dx12Sampler> {
        sampler::create_sampler(device, desc)
    }
    fn create_bind_group_layout(
        device: &Dx12Device,
        desc: &api::BindGroupLayoutDesc,
    ) -> Result<Dx12BindGroupLayout> {
        descriptor::create_bind_group_layout(device, desc)
    }
    fn create_bind_group(
        device: &Dx12Device,
        desc: &api::BindGroupDesc<'_, Self>,
    ) -> Result<Dx12BindGroup> {
        descriptor::create_bind_group(device, desc)
    }
    fn create_pipeline_layout(
        device: &Dx12Device,
        desc: &api::PipelineLayoutDesc<'_, Self>,
    ) -> Result<Dx12PipelineLayout> {
        pipeline::create_pipeline_layout(device, desc)
    }
}

impl api::ShaderBackend for Dx12 {
    type DirectFuture<'a> = Ready<Result<Dx12Shader>>;
    type NagaFuture<'a> = Ready<Result<Dx12Shader>>;
    fn shader_route(device: &Dx12Device, desc: &api::ShaderDesc) -> api::ShaderRoute {
        if !shader_features_supported(&desc.required_features, device.0.ray_query_supported) {
            return api::ShaderRoute::Unsupported;
        }
        match desc.format {
            api::ShaderFormat::Dxil => api::ShaderRoute::Direct(api::ShaderTarget {
                format: api::ShaderFormat::Dxil,
                environment: "d3d12".into(),
                abi_key: "dxil".into(),
            }),
            _ => api::ShaderRoute::Unsupported,
        }
    }
    fn compile_direct<'a>(
        _: &'a Dx12Device,
        desc: &'a api::ShaderDesc,
        _: &'a api::ShaderTarget,
    ) -> Self::DirectFuture<'a> {
        ready(if desc.bytes.is_empty() {
            Err(Error::Shader("empty DXIL artifact".into()))
        } else {
            Ok(Dx12Shader {
                bytes: desc.bytes.clone(),
                entries: desc.entries.clone(),
            })
        })
    }
    fn compile_naga<'a>(
        _: &'a Dx12Device,
        _: &'a api::ShaderDesc,
        _: &'a api::ShaderTarget,
    ) -> Self::NagaFuture<'a> {
        ready(Err(Error::Unsupported("DX12 Naga shader route")))
    }
}

fn shader_features_supported(features: &[String], ray_query_supported: bool) -> bool {
    features.iter().all(|feature| match feature.as_str() {
        api::SHADER_FEATURE_RAY_QUERY => ray_query_supported,
        "shaderClipDistance" => true,
        _ => false,
    })
}

#[cfg(test)]
mod shader_feature_tests {
    use super::*;
    #[test]
    fn validates_ray_query_and_known_legacy_feature() {
        let ray = vec![api::SHADER_FEATURE_RAY_QUERY.to_owned()];
        assert!(!shader_features_supported(&ray, false));
        assert!(shader_features_supported(&ray, true));
        assert!(shader_features_supported(
            &["shaderClipDistance".into()],
            false
        ));
        assert!(!shader_features_supported(
            &["unknown-feature".into()],
            true
        ));
    }
}

fn native_error(error: windows::core::Error) -> Error {
    Error::Backend(format!("Direct3D 12: {error}"))
}

impl fluxel_rhi_core::api::ExecutionBackend for Dx12 {
    type Mode = fluxel_rhi_core::api::DeferredExecution;
}
