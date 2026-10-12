//! Statically typed RHI contract component.
#![allow(missing_docs, dead_code, unused_variables, unused_imports)]
use super::*;
use std::{
    future::Future,
    marker::PhantomData,
    ops::Range,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};
pub type Result<T> = std::result::Result<T, Error>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    Unsupported(&'static str),
    WrongDevice,
    WrongLane,
    InvalidState,
    InvalidRange,
    DeviceLost,
    SurfaceLost,
    SurfaceOutdated,
    OutOfMemory,
    Shader(String),
    Backend(String),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeviceId(u64);
static NEXT_DEVICE: AtomicU64 = AtomicU64::new(1);
impl DeviceId {
    pub fn allocate() -> Result<Self> {
        NEXT_DEVICE
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .map(Self)
            .map_err(|_| Error::OutOfMemory)
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct InstanceId(u64);
static NEXT_INSTANCE: AtomicU64 = AtomicU64::new(1);
impl InstanceId {
    pub fn allocate() -> Result<Self> {
        NEXT_INSTANCE
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .map(Self)
            .map_err(|_| Error::OutOfMemory)
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AdapterId {
    pub instance: InstanceId,
    pub ordinal: u64,
}
pub struct AdapterInfo {
    pub id: AdapterId,
    pub name: String,
}
pub struct DeviceRequest {
    pub adapter: AdapterId,
    pub required_features: Vec<&'static str>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LaneId {
    pub device: DeviceId,
    pub ordinal: u32,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PlanId {
    pub device: DeviceId,
    pub ordinal: u64,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StepId {
    pub plan: PlanId,
    pub index: u32,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SubmissionIndex {
    pub device: DeviceId,
    pub value: u64,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LaneClass {
    Graphics,
    Compute,
    Transfer,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WaitRoute {
    SameQueue,
    NativeGpuWait,
    OrderedByNativeQueue,
    Unsupported,
}
#[derive(Clone)]
pub struct LaneInfo {
    pub id: LaneId,
    pub class: LaneClass,
    pub routes: Vec<(LaneId, WaitRoute)>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SurfaceId(u64);
static NEXT_SURFACE: AtomicU64 = AtomicU64::new(1);
impl SurfaceId {
    pub fn allocate() -> Result<Self> {
        NEXT_SURFACE
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .map(Self)
            .map_err(|_| Error::OutOfMemory)
    }
}

/// Every resource and hot-path encoder has a native type fixed by B. No Any,
/// downcast, backend enum, or mandatory trait-object dispatch is involved.
/// Associated types carry their own Send/Sync properties. A GL implementation
/// may put Rc state in NativeDevice; a modern implementation may use Arc.
pub trait Backend: Sized + 'static {
    type NativeDevice: Clone + 'static;
    type NativeBuffer: Clone + 'static;
    type NativeTexture: Clone + 'static;
    type NativeTextureView: Clone + 'static;
    type NativeSampler: Clone + 'static;
    type NativeBindGroupLayout: Clone + 'static;
    type NativeBindGroup: Clone + 'static;
    type NativePipelineLayout: Clone + 'static;
    type NativeShader: Clone + 'static;
    type NativeRasterPipeline: Clone + 'static;
    type NativeComputePipeline: Clone + 'static;
    type NativeMeshPipeline: Clone + 'static;
    type NativeRayPipeline: Clone + 'static;
    type NativePipelineCache: Clone + 'static;
    type NativeAcceleration: Clone + 'static;
    type NativeSparseAllocation: Clone + 'static;
    type NativeQuerySet: Clone + 'static;
    type NativeEncoder: 'static;
    type NativeCompletion: Clone + 'static;
    type NativeSurface: Clone + 'static;
    type NativeSurfaceImage: NativeImageLease + 'static;
    type NativeUncertainty: 'static;
    // Concrete static capability adapters. An absent feature uses Unavailable.
    // Capabilities still vary among devices of the same concrete Backend type.
    type Compute: ComputeOps<Self>;
    type Mesh: MeshOps<Self>;
    type Ray: RayOps<Self>;
    type Extras: RasterExtrasOps<Self>;
}

/// No backend implements missing optional domain methods merely to return errors.
/// Unavailable implements the tiny fallback once for every Backend type.
pub struct Unavailable;

/// Opening a device is a cold async operation. Runtime backend selection is
/// outside RHI Core; one concrete `Instance<B>` opens one concrete `Device<B>`.
pub trait InstanceBackend: Backend {
    type NativeInstance;
    type OpenFuture<'a>: Future<Output = Result<(Self::NativeDevice, DeviceCapabilities)>> + 'a
    where
        Self: 'a;
    fn enumerate(instance: &Self::NativeInstance, scope: InstanceId) -> Result<Vec<AdapterInfo>>;
    fn open<'a>(
        instance: &'a Self::NativeInstance,
        device: DeviceId,
        request: &'a DeviceRequest,
    ) -> Self::OpenFuture<'a>;
}
pub struct Instance<B: InstanceBackend> {
    id: InstanceId,
    native: B::NativeInstance,
}
impl<B: InstanceBackend> Instance<B> {
    /// # Safety
    /// The native instance must remain valid while opening its devices.
    pub unsafe fn from_native(id: InstanceId, native: B::NativeInstance) -> Self {
        Self { id, native }
    }
    pub fn enumerate(&self) -> Result<Vec<AdapterInfo>> {
        B::enumerate(&self.native, self.id)
    }
    pub async fn open(&self, request: DeviceRequest) -> Result<Device<B>> {
        if request.adapter.instance != self.id {
            return Err(Error::WrongDevice);
        }
        let id = DeviceId::allocate()?;
        let (native, caps) = B::open(&self.native, id, &request).await?;
        Ok(unsafe { Device::from_native(id, native, caps) })
    }
}

/// Native SurfaceImage is an exclusive RAII lease. Dropping an unhanded lease
/// abandons it once. Accepted or indeterminate native present handoff calls
/// try_handoff at the irrevocable native boundary; rejected work returns it live.
/// This is backend-native lease bookkeeping, not renderer frame management.
/// # Safety
/// Implementors must make generation validation, handoff, and abandonment
/// mutually exclusive with native reconfiguration of the origin surface.
pub unsafe trait NativeImageLease {
    fn try_handoff(&mut self, generation: u64) -> Result<()>;
    /// Checks the lease against its origin surface's current configuration.
    /// The lease keeps access to that native generation state; callers need
    /// not retain a mutable Surface merely to render or submit Present.
    fn is_current(&self, generation: u64) -> bool;
}

/// Marker for implementations that can actually cross threads. The bounds
/// are checked where a shared renderer is instantiated; Local has no marker.
pub trait SharedBackend: Backend + Send + Sync
where
    Self::NativeDevice: Send + Sync,
    Self::NativeBuffer: Send + Sync,
    Self::NativeTexture: Send + Sync,
    Self::NativeTextureView: Send + Sync,
    Self::NativeSampler: Send + Sync,
    Self::NativeBindGroupLayout: Send + Sync,
    Self::NativePipelineLayout: Send + Sync,
    Self::NativeBindGroup: Send + Sync,
    Self::NativeShader: Send + Sync,
    Self::NativeRasterPipeline: Send + Sync,
    Self::NativeComputePipeline: Send + Sync,
    Self::NativeMeshPipeline: Send + Sync,
    Self::NativeRayPipeline: Send + Sync,
    Self::NativePipelineCache: Send + Sync,
    Self::NativeAcceleration: Send + Sync,
    Self::NativeSparseAllocation: Send + Sync,
    Self::NativeQuerySet: Send + Sync,
    Self::NativeEncoder: Send,
    Self::NativeCompletion: Send + Sync,
    Self::NativeSurface: Send + Sync,
    Self::NativeSurfaceImage: Send,
    Self::NativeUncertainty: Send,
{
}

/// A multithreaded executor additionally places `Send` bounds on the cold
/// path futures it spawns. Local GL futures may remain `!Send`.

pub struct Device<B: Backend> {
    pub(crate) id: DeviceId,
    pub(crate) native: B::NativeDevice,
    pub(crate) caps: DeviceCapabilities,
}
impl<B: Backend> Clone for Device<B> {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            native: self.native.clone(),
            caps: self.caps.clone(),
        }
    }
}
impl<B: Backend> Device<B> {
    /// # Safety
    /// Native device, capabilities and ID must describe one real device.
    pub unsafe fn from_native(
        id: DeviceId,
        native: B::NativeDevice,
        caps: DeviceCapabilities,
    ) -> Self {
        assert!(
            !caps.compute || B::Compute::AVAILABLE,
            "compute capability without implementation"
        );
        assert!(
            !caps.mesh || B::Mesh::AVAILABLE,
            "mesh capability without implementation"
        );
        assert!(
            !caps.ray || B::Ray::AVAILABLE,
            "ray capability without implementation"
        );
        assert!(
            !(caps.indirect_draw
                || caps.multi_draw
                || caps.multi_draw_count
                || caps.push_constants)
                || B::Extras::AVAILABLE,
            "raster capability without implementation"
        );
        Self { id, native, caps }
    }
    pub fn id(&self) -> DeviceId {
        self.id
    }
    /// Backend bridge for cold native constructors such as window surfaces.
    /// # Safety
    /// Use only in the owning backend and preserve this Device's identity,
    /// thread, and lifetime rules. Native GPU state must stay under its SPI.
    #[doc(hidden)]
    pub unsafe fn native_ref(&self) -> &B::NativeDevice {
        &self.native
    }
    pub fn capabilities(&self) -> &DeviceCapabilities {
        &self.caps
    }
    pub fn lanes(&self) -> &[LaneInfo] {
        &self.caps.lanes
    }
    pub fn wait_route(&self, from: LaneId, to: LaneId) -> WaitRoute {
        if from.device != self.id || to.device != self.id {
            return WaitRoute::Unsupported;
        }
        self.caps
            .lanes
            .iter()
            .find(|lane| lane.id == from)
            .and_then(|lane| lane.routes.iter().find(|(target, _)| *target == to))
            .map_or(WaitRoute::Unsupported, |(_, route)| *route)
    }
}

/// Immutable facts negotiated at device open. Unsafe Device::from_native
/// requires these flags to match the implemented optional backend domains.
#[derive(Clone, Default)]
pub struct DeviceCapabilities {
    pub compute: bool,
    pub sparse: bool,
    pub mesh: bool,
    pub ray: bool,
    pub ray_query: bool,
    pub multi_queue: bool,
    pub indirect_draw: bool,
    pub multi_draw: bool,
    pub multi_draw_count: bool,
    pub push_constants: bool,
    pub storage_buffers: bool,
    pub storage_textures: bool,
    pub binding_arrays: bool,
    /// A binding array whose count is chosen at bind-group creation time.
    /// This is deliberately separate from fixed-size binding arrays: shader
    /// indexing examples require both the native descriptor range and its
    /// variable-count validation path.
    pub runtime_sized_binding_arrays: bool,
    /// Non-uniform indexing of sampled textures and storage buffers.
    /// Backends must not set this merely because they support binding arrays.
    pub nonuniform_sampled_texture_and_storage_buffer_indexing: bool,
    pub timestamp: bool,
    pub occlusion: bool,
    pub pipeline_statistics: bool,
    pub multiview: bool,
    pub placed_memory: bool,
    pub external_memory: bool,
    pub secondary_commands: bool,
    pub native_interop: bool,
    pub lanes: Arc<[LaneInfo]>,
    pub limits: DeviceLimits,
}

#[derive(Clone, Copy, Default)]
pub struct DeviceLimits {
    pub max_bind_groups: u32,
    pub max_bindings_per_group: u32,
    pub max_color_attachments: u32,
    pub max_texture_dimension_2d: u32,
    pub max_texture_array_layers: u32,
    pub max_buffer_bytes: u64,
    pub max_vertex_buffers: u32,
    pub max_vertex_attributes: u32,
    pub max_compute_workgroups: [u32; 3],
    pub max_compute_workgroup_size: [u32; 3],
    pub max_push_constant_bytes: u32,
    pub max_multiview_views: u32,
    pub uniform_offset_alignment: u64,
    pub storage_offset_alignment: u64,
}
