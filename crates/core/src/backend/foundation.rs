//! Fundamental contracts implemented by RHI backend crates.
//!
//! These traits describe native object ownership and device opening. They are
//! intentionally separate from the renderer-facing values in `crate::api`.

use crate::api::{AdapterInfo, DeviceCapabilities, DeviceId, DeviceRequest, InstanceId, Result};
use crate::backend::{ComputeOps, MeshOps, RasterExtrasOps, RayOps};
use std::future::Future;

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
