//! Backend implementation protocols for optional RHI operations.
//!
//! These traits are implemented by backend crates. Renderer-facing values and
//! facades remain in [`crate::api`].
use crate::{api::*, backend::*};
use std::future::Future;

/// Platform bridge chooses this concrete native handle type; RHI never stores
/// it in dyn Any or downcasts it on command paths.
pub trait ExternalBackend: CommandBackend {
    type NativeExternalHandle;
    type NativeExternalMemory: Clone;
    type NativeExternalTexture: Clone;
    type NativeExternalSync;
    /// Returns the imported buffer usable by ordinary command operations.
    fn external_buffer(memory: &Self::NativeExternalMemory) -> Result<Self::NativeBuffer>;
    /// Returns the imported texture usable by ordinary view and command operations.
    fn external_texture(texture: &Self::NativeExternalTexture) -> Result<Self::NativeTexture>;
    /// Imports a producer synchronization payload supplied by the platform bridge.
    ///
    /// # Safety
    /// The bridge must supply a valid native handle and obey its native signal
    /// and wait protocol, including the ownership rules for aliased payloads.
    unsafe fn import_external_sync(
        device: &Self::NativeDevice,
        sync: Self::NativeExternalHandle,
    ) -> Result<Self::NativeExternalSync>;
    /// Imports an externally owned buffer allocation.
    ///
    /// # Safety
    /// The bridge must supply a valid handle and exact allocation metadata,
    /// maintain native ownership and synchronization, and prevent concurrent
    /// host mappings through separately imported aliases.
    ///
    /// ```compile_fail
    /// use fluxel_rhi_core::{api::ExternalResourceDesc, backend::ExternalBackend};
    /// fn import_without_contract<B: ExternalBackend>(
    ///     device: &B::NativeDevice,
    ///     handle: B::NativeExternalHandle,
    ///     desc: &ExternalResourceDesc,
    /// ) {
    ///     let _ = B::import_memory(device, handle, desc);
    /// }
    /// ```
    unsafe fn import_memory(
        device: &Self::NativeDevice,
        handle: Self::NativeExternalHandle,
        desc: &ExternalResourceDesc,
    ) -> Result<Self::NativeExternalMemory>;
    /// Imports an externally owned texture.
    ///
    /// # Safety
    /// The bridge must supply a valid handle and exact texture metadata and
    /// coordinate native ownership transfers, layouts and synchronization.
    unsafe fn import_texture(
        device: &Self::NativeDevice,
        handle: Self::NativeExternalHandle,
        desc: &TextureDesc,
    ) -> Result<Self::NativeExternalTexture>;
    fn acquire_external(
        encoder: &mut Self::NativeEncoder,
        memory: &Self::NativeExternalMemory,
        sync: Self::NativeExternalSync,
    ) -> Result<()>;
    fn release_external(
        encoder: &mut Self::NativeEncoder,
        memory: &Self::NativeExternalMemory,
    ) -> Result<Self::NativeExternalSync>;
    fn acquire_external_texture(
        encoder: &mut Self::NativeEncoder,
        texture: &Self::NativeExternalTexture,
        sync: Self::NativeExternalSync,
        transition: &ExternalTextureTransition,
    ) -> Result<()>;
    fn release_external_texture(
        encoder: &mut Self::NativeEncoder,
        texture: &Self::NativeExternalTexture,
        transition: &ExternalTextureTransition,
    ) -> Result<Self::NativeExternalSync>;
}

pub trait NativeInteropBackend: CommandBackend {
    type NativeInteropGuard;
    fn acquire_native(
        device: &Self::NativeDevice,
        request: NativeInteropRequest,
    ) -> Result<Self::NativeInteropGuard>;
    /// Acquire the actual recording encoder. The caller of the public API
    /// keeps it exclusively borrowed for the lifetime of the returned guard.
    fn acquire_encoder(encoder: &mut Self::NativeEncoder) -> Result<Self::NativeInteropGuard>;
    /// # Safety
    /// The guard must remain alive and the backend must expose the returned
    /// pointer only under its native thread, lifetime, and synchronization rules.
    unsafe fn raw_pointer(guard: &Self::NativeInteropGuard) -> *mut std::ffi::c_void;
}

pub trait SecondaryBackend: RasterBackend + DeferredBackend {
    /// Dynamic depth bias is implemented for secondary raster command lists.
    const SECONDARY_DYNAMIC_DEPTH_BIAS_AVAILABLE: bool = false;
    type NativeSecondaryEncoder;
    type NativeSecondaryWork;
    fn begin_secondary(
        device: &Self::NativeDevice,
        desc: &SecondaryRasterDesc,
    ) -> Result<Self::NativeSecondaryEncoder>;
    fn secondary_set_pipeline(
        encoder: &mut Self::NativeSecondaryEncoder,
        pipeline: &Self::NativeRasterPipeline,
    ) -> Result<()>;
    /// Applies factors to a currently bound dynamic-depth-bias pipeline.
    fn secondary_depth_bias(
        _: &mut Self::NativeSecondaryEncoder,
        _: DynamicDepthBias,
    ) -> Result<()> {
        Err(Error::Unsupported("secondary dynamic depth bias"))
    }
    fn secondary_set_bind_group(
        encoder: &mut Self::NativeSecondaryEncoder,
        slot: u32,
        group: &Self::NativeBindGroup,
        offsets: &[u32],
    ) -> Result<()>;
    fn secondary_set_vertex_buffer(
        encoder: &mut Self::NativeSecondaryEncoder,
        slot: u32,
        buffer: &Self::NativeBuffer,
        offset: u64,
        size: u64,
    ) -> Result<()>;
    fn secondary_set_index_buffer(
        encoder: &mut Self::NativeSecondaryEncoder,
        buffer: &Self::NativeBuffer,
        offset: u64,
        size: u64,
        format: IndexFormat,
    ) -> Result<()>;
    fn secondary_viewport(
        encoder: &mut Self::NativeSecondaryEncoder,
        viewport: Viewport,
    ) -> Result<()>;
    fn secondary_scissor(encoder: &mut Self::NativeSecondaryEncoder, rect: Rect) -> Result<()>;
    fn secondary_set_push_constants(
        encoder: &mut Self::NativeSecondaryEncoder,
        stages: StageMask,
        offset: u32,
        bytes: &[u8],
    ) -> Result<()>;
    fn secondary_draw(
        encoder: &mut Self::NativeSecondaryEncoder,
        vertices: std::ops::Range<u32>,
        instances: std::ops::Range<u32>,
    ) -> Result<()>;
    fn secondary_draw_indexed(
        encoder: &mut Self::NativeSecondaryEncoder,
        indices: std::ops::Range<u32>,
        base_vertex: i32,
        instances: std::ops::Range<u32>,
    ) -> Result<()>;
    fn finish_secondary(encoder: Self::NativeSecondaryEncoder)
    -> Result<Self::NativeSecondaryWork>;
    fn execute_secondary(
        encoder: &mut Self::NativeEncoder,
        work: &Self::NativeSecondaryWork,
    ) -> Result<()>;
}

/// Per-backend fixed static domain implementation. Unavailable is one shared
/// zero-size implementation, not a fake resource or a dynamic vtable.
pub trait ComputeOps<B: Backend> {
    const AVAILABLE: bool;
    type PipelineFuture<'a>: Future<Output = Result<B::NativeComputePipeline>> + 'a
    where
        B: 'a;
    fn create_pipeline<'a>(
        device: &'a B::NativeDevice,
        desc: &'a ComputePipelineDesc<B>,
    ) -> Self::PipelineFuture<'a>;
    fn begin(encoder: &mut B::NativeEncoder) -> Result<()>;
    fn set_pipeline(
        encoder: &mut B::NativeEncoder,
        pipeline: &B::NativeComputePipeline,
    ) -> Result<()>;
    fn set_bind_group(
        encoder: &mut B::NativeEncoder,
        slot: u32,
        group: &B::NativeBindGroup,
        offsets: &[u32],
    ) -> Result<()>;
    fn dispatch(encoder: &mut B::NativeEncoder, groups: [u32; 3]) -> Result<()>;
    /// Dispatches group counts from a four-byte-aligned, 12-byte `[u32; 3]`
    /// record in a buffer with `BufferUsage::INDIRECT`.
    fn dispatch_indirect(
        encoder: &mut B::NativeEncoder,
        args: &B::NativeBuffer,
        offset: u64,
    ) -> Result<()>;
    fn set_push_constants(_: &mut B::NativeEncoder, _: u32, _: &[u8]) -> Result<()> {
        Err(Error::Unsupported("compute push constants"))
    }
    fn end(encoder: &mut B::NativeEncoder) -> Result<()>;
    fn abort(encoder: &mut B::NativeEncoder);
}

/// Explicit, cold-path memory observation for allocator and native reports.
pub trait MemoryObservationOps<B: Backend> {
    const AVAILABLE: bool;
    fn snapshot(device: &B::NativeDevice) -> Result<MemorySnapshot>;
}

impl<B: Backend> MemoryObservationOps<B> for Unavailable {
    const AVAILABLE: bool = false;
    fn snapshot(_: &B::NativeDevice) -> Result<MemorySnapshot> {
        Err(Error::Unsupported("memory observation"))
    }
}

pub trait MeshOps<B: Backend> {
    const AVAILABLE: bool;
    type PipelineFuture<'a>: Future<Output = Result<B::NativeMeshPipeline>> + 'a
    where
        B: 'a;
    fn create_pipeline<'a>(
        device: &'a B::NativeDevice,
        desc: &'a MeshPipelineDesc<B>,
    ) -> Self::PipelineFuture<'a>;
    fn bind_pipeline(
        encoder: &mut B::NativeEncoder,
        pipeline: &B::NativeMeshPipeline,
    ) -> Result<()>;
    fn dispatch(encoder: &mut B::NativeEncoder, groups: [u32; 3]) -> Result<()>;
    fn dispatch_indirect(
        encoder: &mut B::NativeEncoder,
        args: &B::NativeBuffer,
        offset: u64,
    ) -> Result<()>;
    /// Executes `count` tightly packed 12-byte `[u32; 3]` indirect records.
    fn dispatch_indirect_multi(
        encoder: &mut B::NativeEncoder,
        args: &B::NativeBuffer,
        offset: u64,
        count: u32,
    ) -> Result<()>;
    /// Executes at most `max_count` tightly packed 12-byte records, with the
    /// actual count read from a four-byte indirect count buffer.
    fn dispatch_indirect_count(
        encoder: &mut B::NativeEncoder,
        args: &B::NativeBuffer,
        offset: u64,
        count_buffer: &B::NativeBuffer,
        count_offset: u64,
        max_count: u32,
    ) -> Result<()>;
}

pub trait RayOps<B: Backend> {
    const AVAILABLE: bool;
    type PipelineFuture<'a>: Future<Output = Result<B::NativeRayPipeline>> + 'a
    where
        B: 'a;
    fn create_pipeline<'a>(
        device: &'a B::NativeDevice,
        desc: &'a RayPipelineDesc<B>,
    ) -> Self::PipelineFuture<'a>;
    fn set_pipeline(encoder: &mut B::NativeEncoder, pipeline: &B::NativeRayPipeline) -> Result<()>;
    fn set_bind_group(
        encoder: &mut B::NativeEncoder,
        slot: u32,
        group: &B::NativeBindGroup,
        offsets: &[u32],
    ) -> Result<()>;
    fn shader_table_properties(device: &B::NativeDevice) -> Result<ShaderTableProperties>;
    fn shader_group_identifier(
        device: &B::NativeDevice,
        pipeline: &B::NativeRayPipeline,
        group_index: u32,
    ) -> Result<Vec<u8>>;
    fn acceleration_sizes(
        device: &B::NativeDevice,
        desc: &AccelerationDesc<B>,
    ) -> Result<AccelerationSizes>;
    fn create_acceleration(
        device: &B::NativeDevice,
        desc: &AccelerationDesc<B>,
    ) -> Result<B::NativeAcceleration>;
    fn build_acceleration(
        encoder: &mut B::NativeEncoder,
        dst: &B::NativeAcceleration,
        scratch: &B::NativeBuffer,
        input: &AccelerationDesc<B>,
    ) -> Result<()>;
    fn trace(
        encoder: &mut B::NativeEncoder,
        pipeline: &B::NativeRayPipeline,
        shader_table: &ShaderTable<B>,
        dims: [u32; 3],
    ) -> Result<()>;
}

/// Explicitly optional modern raster variants. No checks or dynamic lookups per draw.
pub trait RasterExtrasOps<B: Backend> {
    const AVAILABLE: bool;
    fn draw_indirect(
        encoder: &mut B::NativeEncoder,
        buffer: &B::NativeBuffer,
        offset: u64,
        indexed: bool,
    ) -> Result<()>;
    fn multi_draw(
        encoder: &mut B::NativeEncoder,
        buffer: &B::NativeBuffer,
        offset: u64,
        count: u32,
        stride: u32,
        indexed: bool,
    ) -> Result<()>;
    fn multi_draw_count(
        encoder: &mut B::NativeEncoder,
        args: &B::NativeBuffer,
        args_offset: u64,
        count: &B::NativeBuffer,
        count_offset: u64,
        max: u32,
        stride: u32,
        indexed: bool,
    ) -> Result<()>;
    fn set_push_constants(
        encoder: &mut B::NativeEncoder,
        stages: StageMask,
        offset: u32,
        bytes: &[u8],
    ) -> Result<()>;
}

/// Sparse is plan-level work, not a command-buffer operation. A backend may
/// implement it only where its actual device has the capability.
pub trait SparseBackend: Backend {
    /// Creates a sparse texture with no initially mapped tiles.
    fn create_sparse_texture(
        _device: &Self::NativeDevice,
        _desc: &TextureDesc,
    ) -> Result<Self::NativeTexture> {
        Err(Error::Unsupported("sparse textures"))
    }
    /// Returns the native tile shape for a sparse texture.
    fn sparse_texture_tiling(
        _device: &Self::NativeDevice,
        _texture: &Self::NativeTexture,
    ) -> Result<SparseTextureTiling> {
        Err(Error::Unsupported("sparse texture tiling"))
    }
    fn create_sparse_allocation(
        device: &Self::NativeDevice,
        size: u64,
    ) -> Result<Self::NativeSparseAllocation>;
    /// Allocates backing memory compatible with a particular sparse texture.
    /// APIs whose memory type depends on the image requirements override this.
    fn create_sparse_allocation_for(
        device: &Self::NativeDevice,
        _texture: &Self::NativeTexture,
        size: u64,
    ) -> Result<Self::NativeSparseAllocation> {
        Self::create_sparse_allocation(device, size)
    }
}
pub trait ImmediateSparseBackend: SparseBackend + ImmediateBackend {
    fn sparse_immediate(
        device: &Self::NativeDevice,
        lane: LaneId,
        waits: &[StepWait],
        binds: &[SparseBind<Self>],
    ) -> ImmediateClose<Self>;
}
pub trait MultiQueueBackend: DeferredBackend {
    type NativeGpuSignal;
    /// # Safety
    /// Signals are installed before producer acceptance and bind one specific
    /// consumer execution. Returned metadata must describe that native signal
    /// truthfully; the backend must not substitute a signal from another run.
    unsafe fn submit_multi(
        device: &Self::NativeDevice,
        packet: MultiQueuePacket<Self>,
    ) -> MultiQueueSubmitOutcome<Self>;
}

/// Immediate-only SPI. Its command methods issue native calls synchronously.
/// No full CPU command list, replay stream, or owner-thread dispatcher exists.
/// # Safety
/// A failed or abandoned close must truthfully report whether work was issued.
/// Every emergency checkpoint must cover the complete reported prefix; if the
/// backend cannot prove coverage it must return Indeterminate.
pub unsafe trait ImmediateBackend: CommandBackend + CompletionBackend {
    /// Core calls this after validating that every wait names an earlier step
    /// of this plan and that every plan lane belongs to the device. Backends
    /// must still implement or reject cross-lane synchronization. A wait does
    /// not replace the resource visibility transitions supplied by the caller.
    /// Rejected means no native effect was issued. Any ambiguous native issue
    /// must return Indeterminate, never a plain error.
    fn install_waits(
        device: &Self::NativeDevice,
        lane: LaneId,
        waits: &[StepWait],
    ) -> WaitInstallOutcome<Self>;
    fn close_immediate(
        encoder: Self::NativeEncoder,
        reason: ImmediateCloseReason,
    ) -> ImmediateClose<Self>;
    /// Err is a clean native rejection with the still-live lease. Once native
    /// handoff is possible, return an issued/indeterminate close instead.
    fn present_immediate(
        device: &Self::NativeDevice,
        lane: LaneId,
        waits: &[StepWait],
        binding: PresentBinding<Self>,
    ) -> std::result::Result<ImmediateClose<Self>, (Error, PresentBinding<Self>)>;
    /// None creates no normal checkpoint. End covers every issued step, with
    /// multiple lane frontiers if needed. Steps covers the requested positions,
    /// possibly using later conservative proof. Failure must produce complete
    /// emergency coverage for the issued prefix or report Indeterminate.
    fn checkpoints(
        device: &Self::NativeDevice,
        plan: &SubmissionPlan<Self>,
        issued: &[StepId],
        request: &CompletionRequest,
    ) -> CheckpointOutcome<Self>;
    fn emergency_checkpoint(
        device: &Self::NativeDevice,
        plan: &SubmissionPlan<Self>,
        issued: &[StepId],
    ) -> CheckpointOutcome<Self>;
    /// An unclosed session is an executor error. Quarantine actual native GL
    /// issue effects, never silently assume completion or destroy resources.
    fn abandon_session(device: &Self::NativeDevice, plan: &SubmissionPlan<Self>, issued: &[StepId]);
}

/// Opts a backend into the common execution interface by selecting its native
/// execution model. Existing backend and encoder traits remain unchanged.
pub trait ExecutionBackend: CommandBackend + CompletionBackend {
    /// The backend's execution protocol.
    type Mode: ExecutionMode<Self>;
}

/// Borrowed facts from a model-specific result, without consuming recovery
/// payloads or interpreting an issue prefix as GPU completion.
pub trait ExecutionOutcome<B: Backend> {
    /// Portable execution observations.
    fn facts(&self) -> ExecutionFacts<'_, B>;
    /// The operation's failure, if one was reported.
    fn error(&self) -> Option<&Error>;
}

/// Static implementation of the common execution operations.
pub trait ExecutionMode<B: ExecutionBackend>: Sized {
    /// State owned by an open execution.
    type State;
    /// The original, model-specific owning result.
    type Outcome: ExecutionOutcome<B>;
    /// Starts execution under the caller's resource lifetime obligations.
    ///
    /// # Safety
    /// The plan must be realizable and all referenced resources retained until
    /// completion, including partial or uncertain native effects.
    unsafe fn begin(
        device: &Device<B>,
        plan: SubmissionPlan<B>,
        completion: CompletionRequest,
    ) -> Result<Self::State>;
    /// Encodes the next commands step. Immediate encoders call native APIs now.
    fn commands<F: FnOnce(&mut CommandEncoder<B>) -> Result<()>>(
        state: &mut Self::State,
        step: StepId,
        encode: F,
    ) -> Result<()>;
    /// Binds or performs the next presentation step.
    fn present(
        state: &mut Self::State,
        step: StepId,
        binding: PresentBinding<B>,
    ) -> std::result::Result<(), PresentError<B>>;
    /// Ends execution and returns all native execution facts and recovery state.
    ///
    /// # Safety
    /// The caller must satisfy the obligations of `begin` through completion.
    unsafe fn finish(state: Self::State) -> Self::Outcome;
}
