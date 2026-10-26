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
    DeviceLost,
    SurfaceLost,
    SurfaceOutdated,
    OutOfMemory,
    Shader(String),
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
            Self::DeviceLost => f.write_str("device lost"),
            Self::SurfaceLost => f.write_str("surface lost"),
            Self::SurfaceOutdated => f.write_str("surface outdated"),
            Self::OutOfMemory => f.write_str("out of memory"),
            Self::Shader(message) => write!(f, "shader error: {message}"),
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
    pub async fn open(&self, request: DeviceRequest) -> Result<Device<B>> {
        if request.adapter.instance != self.id {
            return Err(Error::WrongDevice);
        }
        let id = DeviceId::allocate()?;
        let (native, caps) = B::open(&self.native, id, &request).await?;
        // SAFETY: `open` returned the native device and capabilities for allocated `id`.
        Ok(unsafe { Device::from_native(id, native, caps) })
    }
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
        // Capability bits are observable API facts. Checking their static
        // lowering domain here prevents a backend from publishing an optional
        // route which can only fail after renderer work has been recorded.
        assert!(
            !caps.compute || B::Compute::AVAILABLE,
            "compute capability without implementation"
        );
        assert!(
            !caps.mesh || B::Mesh::AVAILABLE,
            "mesh capability without implementation"
        );
        assert!(
            !(caps.acceleration_structure || caps.ray || caps.ray_query) || B::Ray::AVAILABLE,
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

/// Immutable facts negotiated at device open. Unsafe Device::from_native
/// requires these flags to match the implemented optional backend domains.
#[derive(Clone, Default)]
pub struct DeviceCapabilities {
    pub compute: bool,
    pub sparse: bool,
    pub mesh: bool,
    /// Acceleration structure sizing, creation, and build operations.
    pub acceleration_structure: bool,
    pub ray: bool,
    pub ray_query: bool,
    pub multi_queue: bool,
    pub indirect_draw: bool,
    /// Indirect records may contain a nonzero first-instance value, including
    /// its contribution to the shader instance index and instanced attributes.
    /// If false, callers must write zero to that argument field. Backends do
    /// not read GPU-produced records on the CPU to validate this precondition.
    pub indirect_first_instance: bool,
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
