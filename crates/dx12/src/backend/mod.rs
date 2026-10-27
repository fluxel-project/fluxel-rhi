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
    self, AdapterId, AdapterInfo, DeviceCapabilities, DeviceId, DeviceLimits, DeviceRequest, Error,
    InstanceId, LaneClass, LaneId, LaneInfo, Result,
};
use fluxel_rhi_core::backend::{Backend, InstanceBackend};
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
use windows::Win32::Graphics::{
    Direct3D::D3D_FEATURE_LEVEL_11_0,
    Direct3D12::{
        D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_COMMAND_QUEUE_DESC, D3D12_COMMAND_QUEUE_FLAG_NONE,
        D3D12_COMMAND_QUEUE_PRIORITY_NORMAL, D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
        D3D12_FENCE_FLAG_NONE, D3D12_HEAP_FLAG_NONE, D3D12_HEAP_PROPERTIES,
        D3D12_HEAP_TYPE_DEFAULT, D3D12_MEMORY_POOL_UNKNOWN, D3D12_MESSAGE_CALLBACK_FLAG_NONE,
        D3D12_MESSAGE_CATEGORY, D3D12_MESSAGE_ID, D3D12_MESSAGE_SEVERITY, D3D12_RESOURCE_DESC,
        D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON, D3D12CreateDevice,
        D3D12GetDebugInterface, ID3D12CommandQueue, ID3D12CommandSignature, ID3D12Debug,
        ID3D12Device, ID3D12Fence, ID3D12InfoQueue1, ID3D12Resource,
    },
    Dxgi::{CreateDXGIFactory2, DXGI_CREATE_FACTORY_FLAGS, IDXGIFactory1},
};
use windows::core::Interface;

mod compute;
mod descriptor;
mod device;
mod encoder;
mod external;
mod interop;
mod mapping;
mod memory;
mod mesh;
mod pipeline;
mod query;
mod queue;
mod ray;
mod readback;
mod resource;
mod sampler;
mod secondary;
mod shader_naga;
mod sparse;
mod surface;
mod texture;

pub use external::{Dx12ExternalHandle, Dx12ExternalSync};
pub use memory::Dx12Heap;

/// Direct3D 12 selected once by the host.  Resource and command associated
/// types stay concrete throughout the typed execution API.
#[derive(Clone)]
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
    /// Committed buffers consume the process GPU virtual-address range.
    pub(super) max_buffer_bytes: u64,
    /// Native interface is retained only on devices that report DXR tier 1.0;
    /// the opened device reports that fact through the ray capability.
    pub(super) ray_device: Option<windows::Win32::Graphics::Direct3D12::ID3D12Device5>,
    pub(super) ray_query_supported: bool,
    pub(super) mesh_supported: bool,
    pub(super) sparse_supported: bool,
    /// Resource Heap Tier 2 permits a heap that mixes buffers and texture
    /// classes. Tier 1 requires the narrow heap flags selected by memory.rs.
    pub(super) resource_heap_tier2: bool,
    /// Immutable OPTIONS12 fact queried once when the native device opens.
    pub(super) enhanced_barriers_supported: bool,
    /// Immutable OPTIONS16 fact required by dynamic-depth-bias PSOs and
    /// `ID3D12GraphicsCommandList9::RSSetDepthBias`.
    pub(super) dynamic_depth_bias_supported: bool,
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
    pub(super) upload_pool: Arc<encoder::upload::UploadPool>,
    pub(super) mip_blit_cache: Arc<encoder::MipBlitCache>,
    pub(super) blit_compute_cache: Arc<encoder::BlitComputeCache>,
    pub(super) integer_clear_cache: Arc<encoder::integer_clear::IntegerClearCache>,
    /// Immutable private compute PSO used only to pack WGSL indirect records.
    pub(super) indirect_pack_cache: Arc<encoder::indirect_pack::IndirectPackCache>,
    /// Dropped futures cannot unregister SetEventOnCompletion, so their HANDLE
    /// ownership stays here until the associated fence has passed it.
    pub(super) abandoned_fence_events: Mutex<Vec<encoder::fence_wait::AbandonedFenceEvent>>,
    /// Native ExecuteIndirect signatures are immutable and device-owned.
    /// Sharing this cache through each buffer avoids per-recording creation
    /// without introducing a command scheduler or retaining command streams.
    pub(super) indirect_signatures: Arc<Mutex<HashMap<(bool, u32), ID3D12CommandSignature>>>,
    /// Mesh indirect records always use the 12-byte DISPATCH_MESH signature.
    pub(super) mesh_indirect_signature: Mutex<Option<ID3D12CommandSignature>>,
    pub(super) descriptors: descriptor::DescriptorHeaps,
    /// Paged non-shader-visible allocations for texture-view CPU descriptors.
    pub(super) cpu_descriptors: descriptor::CpuDescriptorPool,
    /// Ordinary resource ranges retained by their final native owners.
    pub(super) default_resources: memory::DefaultResourcePool,
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
    pub(super) allocation_size: u64,
    pub(super) mapped: Arc<std::sync::atomic::AtomicBool>,
    pub(super) usage: api::BufferUsage,
    pub(super) memory: api::MemoryClass,
    pub(super) indirect_signatures: Arc<Mutex<HashMap<(bool, u32), ID3D12CommandSignature>>>,
    /// A placed resource keeps its native heap alive independently of the
    /// caller's `MemoryHeap` wrapper. Committed resources leave this empty.
    #[allow(dead_code)] // Retains the native page independently of pool teardown.
    pub(super) backing_heap: Option<windows::Win32::Graphics::Direct3D12::ID3D12Heap>,
    /// Returned only after this resource COM reference has been released.
    #[allow(dead_code)] // Drop ownership returns the range; it is not otherwise queried.
    pub(super) default_allocation: Option<memory::DefaultAllocationLease>,
}
#[derive(Clone)]
pub struct Dx12PipelineCache(pub(crate) Arc<pipeline::cache::PipelineCacheState>);
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
    /// Source needs entry-specific DXC compilation only for dialects whose
    /// artifact depends on pipeline specialization or stage selection.
    pub(crate) source: Option<Dx12ShaderSource>,
}

#[derive(Clone)]
pub(crate) enum Dx12ShaderSource {
    Wgsl(api::ShaderDesc),
    Hlsl(api::ShaderDesc),
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
    type MemoryObservation = memory::MemoryObservation;
}

impl fluxel_rhi_core::backend::ShaderBackend for Dx12 {
    type DirectFuture<'a> = pipeline::worker::ColdFuture<Dx12Shader>;
    type NagaFuture<'a> = pipeline::worker::ColdFuture<Dx12Shader>;
    fn shader_route(device: &Dx12Device, desc: &api::ShaderDesc) -> api::ShaderRoute {
        if !shader_features_supported(
            &desc.required_features,
            device.0.ray_query_supported,
            shader_naga::wave_ops_supported(&device.0.native),
        ) {
            return api::ShaderRoute::Unsupported;
        }
        match desc.format {
            api::ShaderFormat::Dxil => api::ShaderRoute::Direct(api::ShaderTarget {
                format: api::ShaderFormat::Dxil,
                environment: "d3d12".into(),
                abi_key: "dxil".into(),
            }),
            api::ShaderFormat::Hlsl(model)
                if direct_hlsl_entries_supported(desc)
                    && shader_naga::hlsl_shader_model_supported(&device.0.native, model) =>
            {
                api::ShaderRoute::Direct(api::ShaderTarget {
                    format: api::ShaderFormat::Dxil,
                    environment: "d3d12".into(),
                    abi_key: format!("hlsl-sm{model}"),
                })
            }
            api::ShaderFormat::Wgsl
                if !desc.entries.is_empty()
                    && desc.entries.iter().all(|e| {
                        matches!(
                            e.stage,
                            api::ShaderStage::Compute
                                | api::ShaderStage::Vertex
                                | api::ShaderStage::Fragment
                        )
                    })
                    && shader_naga::shader_model_supported(&device.0.native) =>
            {
                api::ShaderRoute::Naga(api::ShaderTarget {
                    format: api::ShaderFormat::Dxil,
                    environment: "d3d12".into(),
                    abi_key: "naga27-sm6-compute".into(),
                })
            }
            _ => api::ShaderRoute::Unsupported,
        }
    }
    fn compile_direct<'a>(
        _: &'a Dx12Device,
        desc: &'a api::ShaderDesc,
        _: &'a api::ShaderTarget,
    ) -> Self::DirectFuture<'a> {
        let desc = desc.clone();
        pipeline::worker::submit(move || match desc.format {
            api::ShaderFormat::Dxil if desc.bytes.is_empty() => {
                Err(Error::Shader("empty DXIL artifact".into()))
            }
            api::ShaderFormat::Dxil => Ok(Dx12Shader {
                bytes: desc.bytes.clone(),
                entries: desc.entries.clone(),
                source: None,
            }),
            api::ShaderFormat::Hlsl(_) if desc.bytes.is_empty() => {
                Err(Error::Shader("empty HLSL source".into()))
            }
            api::ShaderFormat::Hlsl(_) if !direct_hlsl_entries_supported(&desc) => {
                Err(Error::Unsupported("DX12 HLSL shader stage"))
            }
            api::ShaderFormat::Hlsl(model) if !(60..=69).contains(&model) => {
                Err(Error::Unsupported("DX12 HLSL shader model"))
            }
            api::ShaderFormat::Hlsl(_) => match std::str::from_utf8(&desc.bytes) {
                Ok(_) => Ok(Dx12Shader {
                    bytes: Arc::from([]),
                    entries: desc.entries.clone(),
                    source: Some(Dx12ShaderSource::Hlsl(desc.clone())),
                }),
                Err(_) => Err(Error::Shader("HLSL is not UTF-8".into())),
            },
            _ => Err(Error::Unsupported("DX12 direct shader format")),
        })
    }
    fn compile_naga<'a>(
        _: &'a Dx12Device,
        desc: &'a api::ShaderDesc,
        _: &'a api::ShaderTarget,
    ) -> Self::NagaFuture<'a> {
        let desc = desc.clone();
        pipeline::worker::submit(move || {
            shader_naga::validate_source(&desc).map(|()| Dx12Shader {
                bytes: Arc::from([]),
                entries: desc.entries.clone(),
                source: Some(Dx12ShaderSource::Wgsl(desc)),
            })
        })
    }
}

fn direct_hlsl_entries_supported(desc: &api::ShaderDesc) -> bool {
    !desc.entries.is_empty()
        && desc.entries.iter().all(|entry| {
            matches!(
                entry.stage,
                api::ShaderStage::Compute | api::ShaderStage::Vertex | api::ShaderStage::Fragment
            )
        })
}

fn shader_features_supported(
    features: &[String],
    ray_query_supported: bool,
    wave_ops_supported: bool,
) -> bool {
    features.iter().all(|feature| match feature.as_str() {
        api::SHADER_FEATURE_RAY_QUERY => ray_query_supported,
        "shaderClipDistance" => true,
        "subgroups" => wave_ops_supported,
        _ => false,
    })
}

fn max_committed_buffer_bytes(process_address_bits: u32) -> Result<u64> {
    // Committed resources use the per-process GPU VA range. Leave one native
    // 64 KiB placement unit unused so allocation-size rounding cannot cross
    // the queried limit; actual residency is still decided by the driver.
    let address_bytes = 1u64
        .checked_shl(process_address_bits)
        .ok_or_else(|| Error::Backend("invalid DX12 GPU virtual-address limit".into()))?;
    address_bytes
        .checked_sub(64 * 1024)
        .filter(|limit| *limit > 0)
        .ok_or_else(|| Error::Backend("DX12 GPU virtual-address range too small".into()))
}

#[cfg(test)]
#[path = "../tests/backend.rs"]
mod shader_feature_tests;

fn native_error(error: windows::core::Error) -> Error {
    Error::Backend(format!("Direct3D 12: {error}"))
}

impl fluxel_rhi_core::backend::ExecutionBackend for Dx12 {
    type Mode = fluxel_rhi_core::api::DeferredExecution;
}
