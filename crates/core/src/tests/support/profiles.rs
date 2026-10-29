//! Shared and thread-local backend type profiles for Core contract tests.
use super::*;

pub(crate) struct NativeShared;
pub(crate) struct GlLocal;
impl Backend for NativeShared {
    type NativeDevice = Arc<()>;
    type NativeBuffer = Arc<()>;
    type NativeTexture = Arc<()>;
    type NativeTextureView = Arc<()>;
    type NativeVideoFrame = ();
    type NativeSampler = Arc<()>;
    type NativeBindGroupLayout = Arc<()>;
    type NativePipelineLayout = Arc<()>;
    type NativeBindGroup = Arc<()>;
    type NativeShader = Arc<()>;
    type NativeRasterPipeline = Arc<()>;
    type NativeComputePipeline = Arc<()>;
    type NativeMeshPipeline = Arc<()>;
    type NativeRayPipeline = Arc<()>;
    type NativePipelineCache = Arc<()>;
    type NativeAcceleration = Arc<()>;
    type NativeSparseAllocation = Arc<()>;
    type NativeQuerySet = Arc<()>;
    type NativeEncoder = ();
    type NativeCompletion = Arc<()>;
    type NativeSurface = Arc<()>;
    type NativeSurfaceImage = TestLease;
    type NativeUncertainty = ();
    type Compute = Unavailable;
    type Mesh = Unavailable;
    type Ray = Unavailable;
    type Extras = Unavailable;
    type MemoryObservation = Unavailable;
}
impl Backend for GlLocal {
    type NativeDevice = Rc<()>;
    type NativeBuffer = Rc<()>;
    type NativeTexture = Rc<()>;
    type NativeTextureView = Rc<()>;
    type NativeVideoFrame = ();
    type NativeSampler = Rc<()>;
    type NativeBindGroupLayout = Rc<()>;
    type NativePipelineLayout = Rc<()>;
    type NativeBindGroup = Rc<()>;
    type NativeShader = Rc<()>;
    type NativeRasterPipeline = Rc<()>;
    type NativeComputePipeline = Rc<()>;
    type NativeMeshPipeline = Rc<()>;
    type NativeRayPipeline = Rc<()>;
    type NativePipelineCache = Rc<()>;
    type NativeAcceleration = Rc<()>;
    type NativeSparseAllocation = Rc<()>;
    type NativeQuerySet = Rc<()>;
    type NativeEncoder = Rc<()>;
    type NativeCompletion = Rc<()>;
    type NativeSurface = Rc<()>;
    type NativeSurfaceImage = ();
    type NativeUncertainty = ();
    type Compute = Unavailable;
    type Mesh = Unavailable;
    type Ray = Unavailable;
    type Extras = Unavailable;
    type MemoryObservation = Unavailable;
}
