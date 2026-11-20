//! Backend identity, device ownership, capabilities, and common errors.
//!
//! `DeviceId` is the invariant joining every public resource and submission to
//! one backend-native device. Renderer policy, graph scheduling, and resource
//! retirement remain outside this layer.
#![allow(missing_docs, dead_code, unused_variables, unused_imports)]
use super::*;
use crate::backend::{Backend, InstanceBackend};
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

/// A portable failure that can be displayed or propagated as a standard error.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    Unsupported(&'static str),
    WrongDevice,
    WrongLane,
    InvalidState,
    InvalidRange,
    Timeout,
    DeviceLost,
    SurfaceLost,
    SurfaceOutdated,
    OutOfMemory,
    Shader(String),
    ShaderCompilation(ShaderCompilationReport),
    Backend(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unsupported(reason) => write!(f, "unsupported: {reason}"),
            Self::WrongDevice => f.write_str("resource belongs to a different device"),
            Self::WrongLane => f.write_str("work belongs to a different lane"),
            Self::InvalidState => f.write_str("invalid state"),
            Self::InvalidRange => f.write_str("invalid range"),
            Self::Timeout => f.write_str("completion wait timed out"),
            Self::DeviceLost => f.write_str("device lost"),
            Self::SurfaceLost => f.write_str("surface lost"),
            Self::SurfaceOutdated => f.write_str("surface outdated"),
            Self::OutOfMemory => f.write_str("out of memory"),
            Self::Shader(message) => write!(f, "shader error: {message}"),
            Self::ShaderCompilation(report) => write!(f, "shader error: {}", report.rendered),
            Self::Backend(message) => write!(f, "backend error: {message}"),
        }
    }
}

impl std::error::Error for Error {}

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

/// Pre-device evidence for a particular adapter and host target.
/// Creating and configuring the surface still qualifies the live native route.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SurfaceCompatibility {
    Compatible,
    Incompatible(&'static str),
    Unknown(SurfaceProbeUncertainty),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SurfaceProbeUncertainty {
    /// The adapter reports an attached output for the target's current monitor.
    OutputOnAdapter,
    /// The adapter does not report the target monitor among its outputs.
    OutputNotReportedOnAdapter,
    /// The current target or display topology could not be resolved.
    OutputUnavailable,
}

/// Selection hint used during adapter discovery. Software explicitly requests
/// a software adapter. The power hints do not authorize a software fallback.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AdapterPreference {
    #[default]
    Default,
    LowPower,
    HighPerformance,
    Software,
}

/// Diagnostic facts available without opening a device. Vendor and device
/// codes describe a model and must not be used as resource provenance.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdapterProperties {
    pub backend: &'static str,
    pub vendor_id: Option<u32>,
    pub device_id: Option<u32>,
    pub driver_version: Option<[u16; 4]>,
    pub is_software: Option<bool>,
    /// Local memory reported by the adapter, in bytes. This is not a live budget.
    pub dedicated_video_memory: Option<u64>,
    pub shared_system_memory: Option<u64>,
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
impl LaneInfo {
    /// Same-lane ordering is inherent to a lane; other routes must be
    /// explicitly advertised. This does not imply resource memory visibility.
    pub fn route_to(&self, target: LaneId) -> WaitRoute {
        if target == self.id {
            WaitRoute::SameQueue
        } else {
            self.routes
                .iter()
                .find(|(lane, _)| *lane == target)
                .map_or(WaitRoute::Unsupported, |(_, route)| *route)
        }
    }
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
    /// Selects an adapter in this instance's ID namespace.
    pub fn request_adapter(&self, preference: AdapterPreference) -> Result<AdapterInfo> {
        B::request_adapter(&self.native, self.id, preference)
    }
    pub fn adapter_properties(&self, adapter: AdapterId) -> Result<AdapterProperties> {
        if adapter.instance != self.id {
            return Err(Error::WrongDevice);
        }
        B::adapter_properties(&self.native, adapter)
    }
    pub async fn open(&self, request: DeviceRequest) -> Result<Device<B>> {
        self.open_with_capacities(request, &[]).await
    }
    /// Opens the requested adapter only if it satisfies the supplied minimum
    /// capacities. This leaves the actual device limits unchanged.
    pub async fn open_with_capacities(
        &self,
        request: DeviceRequest,
        minimum: &[DeviceCapacity],
    ) -> Result<Device<B>> {
        if request.adapter.instance != self.id {
            return Err(Error::WrongDevice);
        }
        let id = DeviceId::allocate()?;
        let (native, caps) = B::open_with_capacities(&self.native, id, &request, minimum).await?;
        // SAFETY: `open` returned the native device and capabilities for allocated `id`.
        let mut device = unsafe { Device::from_native(id, native, caps) };
        device.adapter = Some(request.adapter);
        Ok(device)
    }
}

impl<B: SurfaceProbeBackend> Instance<B> {
    /// Borrows the target only for this query. Unknown allows the caller to
    /// proceed with device opening and native surface creation.
    pub fn probe_surface(
        &self,
        adapter: AdapterId,
        target: B::Target<'_>,
    ) -> Result<SurfaceCompatibility> {
        if adapter.instance != self.id {
            return Err(Error::WrongDevice);
        }
        B::probe_surface(&self.native, adapter, target)
    }
}

/// A multithreaded executor additionally places `Send` bounds on the cold
/// path futures it spawns. Local GL futures may remain `!Send`.

pub struct Device<B: Backend> {
    pub(crate) id: DeviceId,
    adapter: Option<AdapterId>,
    pub(crate) native: B::NativeDevice,
    pub(crate) caps: DeviceCapabilities,
    pub(crate) stats: Arc<RhiStats>,
}
impl<B: Backend> Clone for Device<B> {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            adapter: self.adapter,
            native: self.native.clone(),
            caps: self.caps.clone(),
            stats: self.stats.clone(),
        }
    }
}
impl<B: Backend> Device<B> {
    /// # Safety
    /// Native device, capabilities and ID must describe one real device.
    pub unsafe fn from_native(
        id: DeviceId,
        native: B::NativeDevice,
        mut caps: DeviceCapabilities,
    ) -> Self {
        // The aggregate bit is a convenience summary. Keep it derived from
        // the independently supported recording scopes so inconsistent
        // backend initialization cannot hide an available scope or advertise
        // a marker route that has no lowering.
        caps.debug_markers = caps.debug_markers_encoder
            || caps.debug_markers_raster_pass
            || caps.debug_markers_compute_pass;
        // Capability bits are observable API facts. Checking their static
        // lowering domain here prevents a backend from publishing an optional
        // route which can only fail after renderer work has been recorded.
        assert!(
            !caps.compute || B::Compute::AVAILABLE,
            "compute capability without implementation"
        );
        assert!(
            !caps.memory_observation || B::MemoryObservation::AVAILABLE,
            "memory observation capability without implementation"
        );
        assert!(
            !caps.dynamic_depth_bias_clamp || caps.dynamic_depth_bias,
            "dynamic depth bias clamp capability without dynamic depth bias"
        );
        assert!(
            !caps.mesh || B::Mesh::AVAILABLE,
            "mesh capability without implementation"
        );
        assert!(
            !(caps.mesh_multi_indirect || caps.mesh_indirect_count) || caps.mesh,
            "mesh indirect capability without mesh support"
        );
        assert!(
            !caps.mesh_indirect_count
                || (caps.mesh_multi_indirect && caps.limits.max_mesh_indirect_count != 0),
            "mesh indirect-count capability without multi-indirect limit"
        );
        assert!(
            caps.mesh_multi_indirect || caps.limits.max_mesh_indirect_count == 0,
            "mesh indirect limit without multi-indirect capability"
        );
        assert!(
            !(caps.acceleration_structure || caps.ray || caps.ray_query) || B::Ray::AVAILABLE,
            "ray capability without implementation"
        );
        assert!(
            !caps.acceleration_structure_binding_arrays
                || (caps.acceleration_structure && caps.binding_arrays),
            "acceleration structure array capability without base array support"
        );
        assert!(
            !caps.runtime_sized_acceleration_structure_binding_arrays
                || (caps.acceleration_structure_binding_arrays
                    && caps.runtime_sized_binding_arrays),
            "runtime acceleration structure arrays without fixed/runtime array support"
        );
        assert!(
            !caps.partially_bound_acceleration_structure_binding_arrays
                || caps.acceleration_structure_binding_arrays,
            "partially-bound acceleration structure arrays without fixed array support"
        );
        assert!(
            !(caps.indirect_draw
                || caps.multi_draw
                || caps.multi_draw_count
                || caps.push_constants)
                || B::Extras::AVAILABLE,
            "raster capability without implementation"
        );
        assert!(
            caps.multi_draw_count_profile.is_none() || caps.multi_draw_count,
            "multi draw count profile without capability"
        );
        Self {
            id,
            adapter: None,
            native,
            caps,
            stats: Arc::new(RhiStats::new()),
        }
    }
    pub fn id(&self) -> DeviceId {
        self.id
    }
    /// The selected adapter when the device was opened through an Instance.
    pub fn adapter_id(&self) -> Option<AdapterId> {
        self.adapter
    }
    /// Returns this device's shared, cumulative RHI request counters.
    ///
    /// The returned object is available in all build profiles. Reading a
    /// snapshot performs relaxed atomic loads and does not query the GPU.
    pub fn stats(&self) -> &RhiStats {
        &self.stats
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
        // A lane ordinal has meaning only within its device. Rejecting foreign
        // IDs here keeps later submission validation from treating coincident
        // numeric ordinals as an interoperable synchronization route.
        if from.device != self.id || to.device != self.id {
            return WaitRoute::Unsupported;
        }
        self.caps
            .lanes
            .iter()
            .find(|lane| lane.id == from)
            .map_or(WaitRoute::Unsupported, |lane| lane.route_to(to))
    }
}

/// Scalar component type used by a native cooperative-matrix configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CooperativeMatrixComponentType {
    Float16,
    Float32,
    Float64,
    Sint8,
    Sint16,
    Sint32,
    Sint64,
    Uint8,
    Uint16,
    Uint32,
    Uint64,
    BFloat16,
}

/// Execution scope supported by a cooperative-matrix configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CooperativeMatrixScope {
    Subgroup,
    Workgroup,
    Device,
    QueueFamily,
}

/// One exact cooperative-matrix multiply-add configuration reported by a device.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CooperativeMatrixConfiguration {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub a_type: CooperativeMatrixComponentType,
    pub b_type: CooperativeMatrixComponentType,
    pub c_type: CooperativeMatrixComponentType,
    pub result_type: CooperativeMatrixComponentType,
    pub saturating_accumulation: bool,
    pub scope: CooperativeMatrixScope,
}
