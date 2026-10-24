//! Deterministic Layer 1 recorder.  It models Fluxel-owned identities, never GL names.
//!
//! This file holds the recorder state, its shared validation helpers and the
//! `GlFamilyApi` lifetime domain.  Each remaining domain lives in its own
//! submodule so a change to one cannot silently invalidate another: objects
//! in `resource`, programs and their inputs in `program`, the pass in
//! `render_pass`, indirect command buffers in `indirect`, copies in
//! `transfer`, and observation in `lifecycle`.  The optional compute/storage
//! wrapper lives in `compute_storage`.
//!
//! Every domain here mirrors the executable providers rather than an
//! idealized contract, so a differential test can trust the trace: it checks
//! what a provider checks, at the point the provider checks it, and it never
//! rejects something a provider accepts.  Where a provider's check needs state
//! the recorder deliberately does not model -- the installed raster pipeline,
//! a VAO's index binding, GL's signed scalar widths -- the recorder checks the
//! strongest condition its own state supports and says so at the call site
//! rather than inventing a rule.

mod compute_storage;
mod indirect;
mod lifecycle;
mod program;
mod render_pass;
mod resource;
// The old cross-module test fixture suite depended on the retired WebGL2
// render-graph harness.  Keep this recorder itself as the reusable oracle;
// current backend conformance tests own fixtures beside the new lowering.
mod transfer;

use std::collections::{BTreeMap, BTreeSet};

use super::*;

/// Exact observable mock operations, deliberately expressed in domain vocabulary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MockCall {
    CreateBuffer(BufferId),
    DestroyBuffer(BufferId),
    CreateTexture(TextureId),
    DestroyTexture(TextureId),
    CreateRenderBuffer(RenderbufferId),
    DestroyRenderBuffer(RenderbufferId),
    CreateSampler(SamplerId),
    DestroySampler(SamplerId),
    CreateShader(ShaderId),
    DestroyShader(ShaderId),
    CreateProgram(ProgramId),
    DestroyProgram(ProgramId),
    CreateVertexArray(VertexArrayId),
    DestroyVertexArray(VertexArrayId),
    BindVertexArray(VertexArrayId),
    CreateFramebuffer(FramebufferId),
    DestroyFramebuffer(FramebufferId),
    BeginRenderPass(FramebufferId),
    EndRenderPass,
    SetRasterPipeline {
        program: ProgramId,
        vertex_array: VertexArrayId,
    },
    /// An immediate update to one VAO vertex-input slot.
    SetVertexBinding {
        vertex_array: VertexArrayId,
        binding: GlVertexBufferBinding,
    },
    /// An immediate update to the VAO element-buffer slot.
    SetIndexBinding {
        vertex_array: VertexArrayId,
        index: Option<GlIndexBinding>,
    },
    /// A dynamic viewport applied while a render pass is active.
    SetViewport(GlViewport),
    /// A dynamic scissor applied while a render pass is active.
    SetScissor(Option<GlScissorRect>),
    DrawRaster(GlDrawCommand),
    /// One draw carrying the optional base-vertex/base-instance offsets.
    DrawAdvancedRaster(GlAdvancedDrawCommand),
    /// One record read from a command buffer, drawn as one command.
    DrawIndirect(GlIndirectCommandRange),
    /// One work-group triple read from a command buffer.
    DispatchIndirect(GlDispatchIndirectCommand),
    /// A batch issued as one combined command, not as its single draws.
    ///
    /// The recorder only produces this variant on the route where a provider
    /// would issue a combined command.  Decomposing a batch is observable as
    /// the `DrawRaster` calls it really is, so the trace always says which
    /// submission shape happened instead of only which batch was requested.
    MultiDraw(GlMultiDraw),
    MultiDrawIndirect(GlIndirectCommandRange),
    MultiDrawIndirectCount {
        commands: GlIndirectCommandRange,
        count: GlIndirectCountRange,
    },
    CopyBuffer {
        source: BufferId,
        destination: BufferId,
        size: u64,
    },
    CopyTexture {
        source: TextureId,
        destination: TextureId,
    },
    UploadBuffer {
        buffer: BufferId,
        offset: u64,
        size: u64,
    },
    ReadBuffer {
        buffer: BufferId,
        offset: u64,
        size: u64,
    },
    UploadTexture(TextureId),
    ReadTexture(TextureId),
    ActiveTexture(u32),
    BindTexture {
        unit: u32,
        target: GlTextureTarget,
        texture: Option<TextureId>,
    },
    BindSampler {
        unit: u32,
        sampler: Option<SamplerId>,
    },
    BindUniformBuffer {
        index: u32,
        buffer: Option<BufferId>,
        offset: u64,
        size: u64,
    },
    BlitFramebuffer {
        source: FramebufferId,
        destination: FramebufferId,
    },
    CreateFence(GlFenceLease),
    DestroyFence(GlFenceLease),
    PollFence(GlFenceLease),
    WaitFence(GlFenceLease),
    Flush,
    AcquireSurface(GlSurfaceLease),
    ResizeSurface(GlSurfaceSize),
    SuspendSurface,
    ResumeSurface,
    PresentSurface(GlSurfaceLease),
    /// One frame published into the drawable from a texture.
    ///
    /// Recorded with both the lease and the source because the recorder has no
    /// drawable to blit into: what it can witness is that a caller paired this
    /// acquisition with this texture, which is the whole of what a publish
    /// claims before any driver is involved.
    PublishSurface {
        lease: GlSurfaceLease,
        source: TextureId,
    },
    CreateQuery(QueryId),
    DestroyQuery(QueryId),
    QueryResult(QueryId),
    BeginOcclusion(QueryId),
    EndOcclusion,
    BeginElapsed(QueryId),
    EndElapsed,
    QueryTimestamp(QueryId),
    SetComputeProgram(ProgramId),
    /// One selection of the driver's single current program.
    ///
    /// Recorded separately from the verb that asked for it because the two are
    /// not the same event: a pipeline install, a compute install, and the
    /// re-assertion a draw or dispatch performs when the slot was taken since
    /// all write this one piece of driver state.  A trace that named only the
    /// verbs could not show the re-assertion, which is the call a provider has
    /// to make for a dispatch to run the program the caller installed.
    SelectProgram(ProgramId),
    Dispatch(GlDispatchGroups),
    BindStorageBuffer {
        binding: u32,
        buffer: BufferId,
        offset: u64,
        size: u64,
    },
    BindStorageImage {
        binding: u32,
        texture: TextureId,
    },
    ContextLost,
    Error(GlError),
}

/// Common-profile mock. It intentionally does not implement compute or storage traits.
#[derive(Debug)]
pub struct MockGlFamilyApi {
    discovery: GlDiscoverySnapshot,
    stamp: ContextStamp,
    lifecycle: GlContextLifecycle,
    owner: OwnerThreadIdentity,
    next_slot: u32,
    buffers: BTreeMap<BufferId, GlBufferDesc>,
    textures: BTreeMap<TextureId, GlTextureDesc>,
    render_buffers: BTreeMap<RenderbufferId, GlRenderBufferDesc>,
    samplers: BTreeSet<SamplerId>,
    shaders: BTreeSet<ShaderId>,
    programs: BTreeSet<ProgramId>,
    vaos: BTreeSet<VertexArrayId>,
    framebuffers: BTreeMap<FramebufferId, GlFramebufferDescriptor>,
    queries: BTreeSet<QueryId>,
    fences: GlFenceLeaseBook,
    syncs: BTreeSet<SyncId>,
    surface: GlSurfaceLeaseBook,
    surface_size: GlSurfaceSize,
    surface_suspended: bool,
    pass_active: bool,
    /// The framebuffer the active pass renders into.
    ///
    /// Kept so an end can look it up, which is what both executable providers
    /// do after they have already consumed the pass: a mock that never checked
    /// would accept an end over a framebuffer the caller destroyed mid-pass.
    pass_framebuffer: Option<FramebufferId>,
    /// The compute program installed for dispatch work, if any.
    installed_compute_program: Option<ProgramId>,
    /// The program the modelled driver holds, or `None` when none is selected.
    ///
    /// GL has one current program and the recorder has to model that, not the
    /// intent of each verb: a pipeline install and a compute install write the
    /// same slot, and a link clears it because its reflection scope ends with no
    /// program selected.  A draw or dispatch that needs a program selects it
    /// again, which is why the trace can show a `SelectProgram` the caller never
    /// asked for -- that is the driver call the real provider makes.
    current_program: Option<ProgramId>,
    /// The raster program the active pass installed, if any.
    ///
    /// The recorder models the pipeline install as a program plus the vertex
    /// array it bound, and only the program half is needed to answer whether a
    /// draw has anything to draw with.  A pass end clears it, as the executable
    /// backends do.
    installed_raster_program: Option<ProgramId>,
    /// The vertex array the modelled driver holds, or `None` when it holds none.
    ///
    /// Separate from the program because the two go stale for different reasons:
    /// nothing but a compute install or a link touches the program slot, while
    /// the vertex-array slot is written by an input reconcile as well -- and an
    /// uncached reconcile destroys and replaces the array on *every* request.  So
    /// a recorder that kept the array the pipeline install named would refuse the
    /// array the driver is actually holding, which is precisely the defect the
    /// real provider had.  Both writers of the slot record it here: the pipeline
    /// install and `bind_vertex_array`.
    bound_vertex_array: Option<VertexArrayId>,
    /// The optional draw offsets this recorder treats as proved.
    ///
    /// A provider learns these from its own context queries, which the
    /// recorder has no equivalent of, so the test names them explicitly for the
    /// same reason fence completion and program reflection are injected.  The
    /// default proves nothing: an unconfigured recorder rejects a nonzero
    /// offset instead of accepting one no provider ever proved.
    advanced_raster: GlAdvancedRasterCapabilities,
    pixel_store: GlPixelStoreState,
    calls: Vec<MockCall>,
    next_error: Option<GlError>,
    /// Deterministic oracle answers for query observations.
    query_results: BTreeMap<QueryId, GlQueryResult>,
    /// Deterministic oracle answers for fence observations, keyed by the fence
    /// the observation names.  A fence absent from this map is reported
    /// `Pending`, never `Complete`: the recorder must not invent completion it
    /// was not told about, or every completion-safe release test would pass
    /// against a mock that never modelled the wait at all.
    fence_statuses: BTreeMap<SyncId, GlFenceStatus>,
    /// Deterministic reflection for the next `create_program` call.
    next_reflection: Option<GlProgramReflection>,
}

mod recorder;

impl GlFamilyApi for MockGlFamilyApi {
    fn profile(&self) -> GlFamilyProfile {
        self.discovery.context().profile()
    }
    fn context_stamp(&self) -> ContextStamp {
        self.stamp
    }
    fn lifecycle(&self) -> GlContextLifecycle {
        self.lifecycle
    }
    fn owner_thread(&self) -> OwnerThreadIdentity {
        self.owner
    }
    fn assert_owner_thread(&self, op: &'static str) -> Result<(), GlError> {
        self.owner(op)
    }
    fn discovery(&self) -> &GlDiscoverySnapshot {
        &self.discovery
    }
    fn context_lost(&mut self) -> Result<(), GlError> {
        self.ready("context-lost")?;
        self.lifecycle = GlContextLifecycle::Lost;
        self.reset_objects();
        self.calls.push(MockCall::ContextLost);
        Ok(())
    }
}

pub(crate) use compute_storage::MockComputeStorageApi;
