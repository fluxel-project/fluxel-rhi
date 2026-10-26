//! Metal instance and device ownership.
//!
//! This module owns adapter discovery and one native command queue per exposed lane.
//! Resource creation, command lowering, completion, and Surface lifetimes are
//! split into private modules so each native boundary owns its invariants.

use std::{
    future::{Ready, ready},
    sync::{Arc, Mutex},
};

use fluxel_rhi_core::api::{
    self, AdapterInfo, DeviceCapabilities, DeviceId, DeviceRequest, Error, InstanceId,
};
#[cfg(target_vendor = "apple")]
use fluxel_rhi_core::api::{AdapterId, LaneClass, LaneId, LaneInfo, WaitRoute};
use fluxel_rhi_core::backend::{Backend, InstanceBackend};
#[cfg(target_vendor = "apple")]
use objc2_metal::MTLDevice as _;

/// Static type for the native Apple Metal backend.
pub struct Metal;

/// Process-local Metal adapter snapshot.
#[derive(Clone)]
pub struct MetalInstance {
    _state: Arc<InstanceState>,
}

struct InstanceState {
    #[cfg(target_vendor = "apple")]
    devices: Vec<objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>>>,
}

/// Native Metal device and its direct command queue.
#[derive(Clone)]
pub struct MetalDevice {
    _state: Arc<DeviceState>,
}

mod argument;
mod binding_abi;
mod bindings;
mod cache;
#[cfg(target_vendor = "apple")]
mod capabilities;
mod cold;
mod cold_cpu;
mod compute;
mod copy;
mod encoder;
mod external;
mod interop;
mod mapping;
mod memory;
mod mesh;
mod multi_queue;
#[cfg(target_vendor = "apple")]
mod native_array_reflection;
mod pipeline;
mod query;
mod ray;
mod readback;
mod resource;
mod sampler;
mod shader;
mod shader_ray_query;
mod sparse;
mod submit;
mod surface;
mod texture;
#[cfg(target_vendor = "apple")]
mod upload;

pub(crate) use bindings::{MetalBindGroup, MetalBindGroupLayout, MetalPipelineLayout};
pub(crate) use encoder::MetalCommandEncoder;
pub use external::{MetalExternalHandle, MetalExternalSync};
pub(crate) use pipeline::{MetalComputePipeline, MetalRasterPipeline};
pub(crate) use resource::MetalBuffer;
pub(crate) use sampler::MetalSampler;
pub use surface::MetalSurface;
pub(crate) use texture::{MetalTexture, MetalTextureView};

struct DeviceState {
    #[cfg(target_vendor = "apple")]
    _device: objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>>,
    #[cfg(target_vendor = "apple")]
    _queue: objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandQueue>>,
    #[cfg(target_vendor = "apple")]
    compute_queue:
        objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandQueue>>,
    #[cfg(target_vendor = "apple")]
    ray_supported: bool,
    #[cfg(target_vendor = "apple")]
    queue_gate: std::sync::Mutex<()>,
    #[cfg(target_vendor = "apple")]
    lane_frontier: std::sync::Mutex<[Option<submit::Completion>; 2]>,
    #[cfg(target_vendor = "apple")]
    submission_tracker: std::sync::Mutex<submit::SubmissionTracker>,
    #[cfg(target_vendor = "apple")]
    indirect_dispatch_supported: bool,
    #[cfg(target_vendor = "apple")]
    indirect_count_supported: bool,
    #[cfg(target_vendor = "apple")]
    indirect_count_converter: Option<encoder::indirect_count::ConverterCache>,
    #[cfg(target_vendor = "apple")]
    visibility_merge_pipeline: query::VisibilityMergeCache,
    #[cfg(target_vendor = "apple")]
    upload_pool: upload::UploadPool,
}

#[cfg(target_vendor = "apple")]
impl Drop for DeviceState {
    fn drop(&mut self) {
        self.upload_pool.clear();
    }
}

#[derive(Clone)]
/// Compiled native Metal shader library.
pub struct MetalShader {
    #[cfg(target_vendor = "apple")]
    pub(crate) _library:
        Option<objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLLibrary>>>,
    #[cfg(target_vendor = "apple")]
    pub(crate) entries: Vec<api::ShaderEntryPoint>,
    #[cfg(target_vendor = "apple")]
    pub(crate) entry_names: Vec<(String, String)>,
    #[cfg(target_vendor = "apple")]
    pub(crate) wgsl_source: Option<Arc<[u8]>>,
}

/// A unique, generation-bound lease for an acquired `CAMetalDrawable`.
///
/// The lease stays private to the backend and transfers to native Present at
/// most once; the held drawable also keeps its texture alive during encoding.
pub struct MetalSurfaceImage {
    generation: u64,
    surface_generation: Arc<Mutex<u64>>,
    handed_off: bool,
    #[cfg(target_vendor = "apple")]
    drawable:
        objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_quartz_core::CAMetalDrawable>>,
    #[cfg(target_vendor = "apple")]
    texture: objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLTexture>>,
}

// SAFETY: the lease is exclusively borrowed at handoff and shares the origin
// Surface's generation lock, which serializes stale-lease validation with
// native reconfiguration.
unsafe impl fluxel_rhi_core::backend::NativeImageLease for MetalSurfaceImage {
    fn try_handoff(&mut self, generation: u64) -> api::Result<()> {
        let current = self
            .surface_generation
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if self.handed_off {
            return Err(Error::InvalidState);
        }
        // Reconfiguration may invalidate the lease after Core's preflight but
        // before this final, generation-locked native handoff.
        if self.generation != generation || *current != self.generation {
            return Err(Error::SurfaceOutdated);
        }
        self.handed_off = true;
        Ok(())
    }

    fn is_current(&self, generation: u64) -> bool {
        let current = self
            .surface_generation
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        lease_generation_is_current(self.generation, generation, *current, self.handed_off)
    }
}

fn lease_generation_is_current(
    image_generation: u64,
    core_generation: u64,
    current_generation: u64,
    handed_off: bool,
) -> bool {
    image_generation == core_generation && current_generation == image_generation && !handed_off
}

impl Backend for Metal {
    type NativeDevice = MetalDevice;
    type NativeBuffer = MetalBuffer;
    type NativeTexture = MetalTexture;
    type NativeTextureView = MetalTextureView;
    type NativeSampler = MetalSampler;
    type NativeBindGroupLayout = MetalBindGroupLayout;
    type NativeBindGroup = MetalBindGroup;
    type NativePipelineLayout = MetalPipelineLayout;
    type NativeShader = MetalShader;
    type NativeRasterPipeline = MetalRasterPipeline;
    type NativeComputePipeline = MetalComputePipeline;
    type NativeMeshPipeline = mesh::MetalMeshPipeline;
    type NativeRayPipeline = ();
    type NativePipelineCache = cache::MetalPipelineCache;
    type NativeAcceleration = ray::MetalAcceleration;
    type NativeSparseAllocation = sparse::MetalSparseAllocation;
    type NativeQuerySet = query::MetalQuerySet;
    type NativeEncoder = MetalCommandEncoder;
    type NativeCompletion = submit::Completion;
    type NativeSurface = MetalSurface;
    type NativeSurfaceImage = MetalSurfaceImage;
    type NativeUncertainty = ();
    type Compute = compute::Compute;
    type Mesh = mesh::MetalMesh;
    type Ray = ray::MetalRay;
    type Extras = encoder::indirect_count::RasterExtras;
}

impl InstanceBackend for Metal {
    type NativeInstance = MetalInstance;
    type OpenFuture<'a> = Ready<api::Result<(MetalDevice, DeviceCapabilities)>>;

    fn enumerate(instance: &MetalInstance, scope: InstanceId) -> api::Result<Vec<AdapterInfo>> {
        #[cfg(target_vendor = "apple")]
        {
            Ok(instance
                ._state
                .devices
                .iter()
                .enumerate()
                .map(|(ordinal, device)| AdapterInfo {
                    id: AdapterId {
                        instance: scope,
                        ordinal: ordinal as u64,
                    },
                    name: device.name().to_string(),
                })
                .collect())
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = (instance, scope);
            Ok(Vec::new())
        }
    }

    fn open<'a>(
        instance: &'a MetalInstance,
        id: DeviceId,
        request: &'a DeviceRequest,
    ) -> Self::OpenFuture<'a> {
        #[cfg(target_vendor = "apple")]
        {
            let result = (|| {
                let native = instance
                    ._state
                    .devices
                    .get(request.adapter.ordinal as usize)
                    .cloned()
                    .ok_or(Error::WrongDevice)?;
                if !request.required_features.is_empty() {
                    return Err(Error::Unsupported("requested Metal feature"));
                }
                let queue = native.newCommandQueue().ok_or(Error::OutOfMemory)?;
                let compute_queue = native.newCommandQueue().ok_or(Error::OutOfMemory)?;
                use objc2::runtime::NSObjectProtocol as _;
                let ray_supported = native.respondsToSelector(objc2::sel!(supportsRaytracing))
                    && native.supportsRaytracing();
                let multi_queue_supported = native.respondsToSelector(objc2::sel!(newSharedEvent))
                    && native.newSharedEvent().is_some();
                let indirect_dispatch_supported = compute::supports_indirect_dispatch(&native);
                let indirect_count_converter =
                    encoder::indirect_count::ConverterCache::new(&native)
                        .ok()
                        .filter(|converter| encoder::indirect_count::probe(&native, converter));
                let indirect_count_supported = indirect_count_converter.is_some();
                let device = MetalDevice {
                    _state: Arc::new(DeviceState {
                        _device: native,
                        _queue: queue,
                        compute_queue,
                        ray_supported,
                        queue_gate: std::sync::Mutex::new(()),
                        lane_frontier: std::sync::Mutex::new([None, None]),
                        submission_tracker: std::sync::Mutex::new(
                            submit::SubmissionTracker::default(),
                        ),
                        indirect_dispatch_supported,
                        indirect_count_supported,
                        indirect_count_converter,
                        visibility_merge_pipeline: Arc::new(Mutex::new(None)),
                        upload_pool: upload::UploadPool::default(),
                    }),
                };
                let caps = capabilities::describe(&device, multi_queue_supported);
                let graphics = LaneId {
                    device: id,
                    ordinal: 0,
                };
                let compute = LaneId {
                    device: id,
                    ordinal: 1,
                };
                let lanes: Vec<LaneInfo> = if multi_queue_supported {
                    vec![
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
                    ]
                } else {
                    vec![LaneInfo {
                        id: graphics,
                        class: LaneClass::Graphics,
                        routes: Vec::new(),
                    }]
                };
                let caps = DeviceCapabilities {
                    lanes: Arc::from(lanes),
                    ..caps
                };
                Ok((device, caps))
            })();
            ready(result)
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = (instance, id, request);
            ready(Err(Error::Unsupported("Metal requires an Apple target")))
        }
    }
}

impl MetalInstance {
    /// Discovers the Metal devices visible to this process.
    pub fn new() -> api::Result<Self> {
        #[cfg(target_vendor = "apple")]
        {
            // objc2-metal 0.3.2 wraps the platform availability boundary:
            // macOS/Catalyst enumerate; iOS/tvOS/visionOS call the default-
            // device API instead. This is not a direct unavailable iOS symbol.
            let devices = objc2_metal::MTLCopyAllDevices().to_vec();
            Ok(Self {
                _state: Arc::new(InstanceState { devices }),
            })
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            Ok(Self {
                _state: Arc::new(InstanceState {}),
            })
        }
    }

    /// Creates a typed Core instance with a fresh identity.
    pub fn instance(&self) -> api::Result<api::Instance<Metal>> {
        // SAFETY: the returned Core instance retains a clone of the native
        // instance state, and the allocated ID is unique for this process.
        Ok(unsafe { api::Instance::from_native(api::InstanceId::allocate()?, self.clone()) })
    }

    /// Opens the first Metal adapter exposed by this instance.
    pub async fn open_default(&self) -> api::Result<api::Device<Metal>> {
        let instance = self.instance()?;
        let adapter = instance
            .enumerate()?
            .into_iter()
            .next()
            .ok_or(api::Error::Unsupported("a Metal adapter"))?;
        instance
            .open(api::DeviceRequest {
                adapter: adapter.id,
                required_features: Vec::new(),
            })
            .await
    }
}

impl Metal {
    /// Creates a typed Surface for a host-owned AppKit `CAMetalLayer`.
    ///
    /// The host must keep the AppKit view alive until the Surface and all
    /// acquired drawable leases have been dropped.
    #[cfg(target_vendor = "apple")]
    pub fn surface_from_layer(
        device: &api::Device<Self>,
        layer: objc2::rc::Retained<objc2_quartz_core::CAMetalLayer>,
    ) -> api::Result<api::Surface<Self>> {
        // SAFETY: this is the owning backend; the layer association uses the
        // supplied device's native Metal device, and the Surface retains it.
        let native_device = unsafe { device.native_ref() };
        let native = MetalSurface::from_layer(native_device, layer)?;
        // SAFETY: `native` was created for `device`; the allocated surface ID
        // is unique and Surface owns the retained CAMetalLayer association.
        Ok(unsafe { api::Surface::from_native(device, api::SurfaceId::allocate()?, native) })
    }
}

#[cfg(test)]
#[path = "../tests/backend.rs"]
mod tests;

impl fluxel_rhi_core::backend::ExecutionBackend for Metal {
    type Mode = fluxel_rhi_core::api::DeferredExecution;
}
