//! Native executable owner for every GL-family command domain.
//!
//! The Host owns the platform context; this type borrows its already-current
//! `glow` dispatch table and owns only Fluxel object tables, pass/raster
//! records, and the pixel-store snapshot. Raw GL names live inside these
//! records and never participate in identity comparisons.

use std::collections::BTreeMap;

use super::discovery::{NativeDiscoveryError, discover_current_glow};
use crate::backend::gl::api::GlFamilyApi as _;
use crate::backend::gl::api::{
    BufferId, ContextStamp, FramebufferId, GlBufferDesc, GlContextLifecycle, GlDiscoverySnapshot,
    GlError, GlFenceLeaseBook, GlIndexBinding, GlPixelStoreState, GlPrimitiveTopology,
    GlProgramDescriptor, GlRenderBufferDesc, GlSurfaceLeaseBook, GlSurfaceSize, GlTextureDesc,
    GlVertexLayout, OwnerThreadIdentity, ProgramId, QueryId, RenderbufferId, SamplerId, ShaderId,
    SyncId, TextureId, VertexArrayId,
};
use crate::backend::gl::state::{
    CanonicalBlockId, ContextState, ExecutionMode, GlObjectKey, RasterPipelineBlockInterner,
    RasterPipelinePacket, StateEvent,
};

#[cfg(any(feature = "native-gl-wgl", feature = "native-gles-egl"))]
pub(crate) struct NativeGlProvider {
    /// A provider owns its dispatch table.  The table contains loader-resolved
    /// entry points, not a platform context handle; the WGL/EGL owner still
    /// makes the actual context current before every call.  Owning this value
    /// avoids a self-reference between a thread-affine surface and its object
    /// tables when the pair is constructed on `NativeOwnerWorker`.
    pub(super) gl: glow::Context,
    pub(super) unmap_buffer: super::context::UnmapBufferFn,
    pub(super) discovery: GlDiscoverySnapshot,
    pub(super) lifecycle: GlContextLifecycle,
    pub(super) owner: OwnerThreadIdentity,
    pub(super) next_slot: u32,
    pub(super) buffers: BTreeMap<BufferId, (glow::NativeBuffer, GlBufferDesc)>,
    pub(super) textures: BTreeMap<TextureId, (glow::NativeTexture, GlTextureDesc)>,
    pub(super) renderbuffers:
        BTreeMap<RenderbufferId, (glow::NativeRenderbuffer, GlRenderBufferDesc)>,
    pub(super) samplers: BTreeMap<SamplerId, glow::NativeSampler>,
    pub(super) shaders: BTreeMap<ShaderId, glow::NativeShader>,
    pub(super) programs: BTreeMap<ProgramId, NativeProgram>,
    pub(super) vertex_arrays: BTreeMap<VertexArrayId, NativeVertexArray>,
    pub(super) framebuffers: BTreeMap<FramebufferId, NativeFramebuffer>,
    pub(super) queries: BTreeMap<QueryId, NativeQuery>,
    pub(super) syncs: BTreeMap<SyncId, glow::NativeFence>,
    pub(super) fences: GlFenceLeaseBook,
    pub(super) surface: GlSurfaceLeaseBook,
    pub(super) surface_suspended: bool,
    /// The Host-reported extent of the drawable, absent until reported.
    ///
    /// Not a default: this family has no core query for the default
    /// framebuffer's size, so an unreported extent is an unknown one, and the
    /// presentation domain answers it with suspension instead of a lease.
    pub(super) surface_extent: Option<GlSurfaceSize>,
    pub(super) pass: Option<ActivePass>,
    pub(super) raster: Option<ActiveRaster>,
    /// The program the driver is known to hold, or `None` when no program is
    /// selected.
    ///
    /// `None` is a fact and not a default: a freshly created context selects no
    /// program, and a link clears the selection when its reflection scope ends,
    /// so both are recorded here rather than assumed.  This is the one place the
    /// provider answers "which program is current", and
    /// [`Self::ensure_program`] is the only writer.
    pub(super) current_program: Option<ProgramId>,
    /// The vertex array the modelled driver holds, or `None` when it holds none.
    ///
    /// GL has one vertex-array binding slot, and the provider models the slot
    /// rather than the intent of each verb for the same reason
    /// [`Self::current_program`] exists: two verbs reach it -- a pipeline install
    /// and an input reconcile -- and the install's choice is the older of the
    /// two by the time a draw runs.  A raster draw resolves *this* record and not
    /// the array the pipeline install named, because the geometry domain may
    /// legitimately have replaced that array in between: under the uncached
    /// execution mode it replaces it on every request.  Reading the install-time
    /// identity instead made the draw validate a fact about the past and refuse
    /// the array the driver was actually holding, which is the defect that made
    /// the uncached differential unable to complete a single frame.
    ///
    /// Only the two verbs that bind a real array write it: the input domain's
    /// `bind_vertex_array` and [`Self::ensure_vertex_array`], which is where a
    /// pipeline install and both draw routes re-assert the binding.
    pub(super) bound_vertex_array: Option<VertexArrayId>,
    /// The query currently recording a measurement, if any.
    pub(super) active_query: Option<QueryId>,
    /// The compute program the caller selected for dispatch work, if any.
    ///
    /// This is the caller's choice, not a claim about the driver: which program
    /// the driver actually holds is [`Self::current_program`]'s to say, because a
    /// raster pipeline install writes that same slot.  A dispatch resolves this
    /// record and re-asserts it through [`Self::ensure_program`], which is why
    /// the two are separate fields rather than one.
    pub(super) active_compute_program: Option<ProgramId>,
    pub(super) pixel_store: GlPixelStoreState,
    /// The sole skip-decision authority for mutable GL state. The native
    /// fields above remain execution facts and must not make fixed-state
    /// elision decisions independently.
    pub(super) context_state: ContextState,
    /// Per-context canonicalization used only to form immutable state packets.
    pub(super) pipeline_blocks: RasterPipelineBlockInterner,
}

/// A linked raster or compute program record with its validated descriptor.
pub(super) struct NativeProgram {
    pub(super) generation: u32,
    pub(super) raw: glow::NativeProgram,
    pub(super) descriptor: GlProgramDescriptor,
    /// Link-time logical-to-native assignments.  Binding flushes must consult
    /// this instead of treating a RHI slot as a GL binding point.
    pub(super) reflection: crate::backend::gl::api::GlProgramReflection,
}

/// A created VAO with its structural layout and last recorded index binding.
pub(super) struct NativeVertexArray {
    pub(super) generation: u32,
    pub(super) raw: glow::NativeVertexArray,
    pub(super) layout: GlVertexLayout,
    pub(super) index: Option<GlIndexBinding>,
    pub(super) bindings: Vec<crate::backend::gl::api::GlVertexBufferBinding>,
}

/// A created framebuffer with its validated descriptor.
pub(super) struct NativeFramebuffer {
    pub(super) generation: u32,
    pub(super) raw: glow::NativeFramebuffer,
    pub(super) descriptor: crate::backend::gl::api::GlFramebufferDescriptor,
}

/// The target a query object last recorded a measurement for.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum QueryTarget {
    SamplesPassed,
    TimeElapsed,
    Timestamp,
}

pub(super) struct NativeQuery {
    pub(super) generation: u32,
    pub(super) raw: glow::NativeQuery,
    pub(super) target: Option<QueryTarget>,
}

/// Facts of the render pass currently recording on this context.
pub(super) struct ActivePass {
    pub(super) target: crate::backend::gl::api::GlRenderTarget,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) samples: u32,
    /// Per color attachment; `true` when `end_render_pass` must invalidate.
    pub(super) discard_color: Vec<bool>,
    /// `(depth, stencil)` discard decisions. The planes stay independent: a
    /// depth-stencil view may retain one plane while making the other undefined.
    pub(super) discard_depth_stencil: Option<(bool, bool)>,
}

/// The raster pipeline installed for the active pass.
pub(super) struct ActiveRaster {
    /// The program this pipeline installed.
    ///
    /// Recorded rather than assumed: a compute install or a link can leave
    /// another program current, so a draw has to be able to re-assert this one
    /// through [`NativeGlProvider::ensure_program`].
    pub(super) program: ProgramId,
    /// The topology this pipeline installed.
    ///
    /// The vertex array is deliberately *not* recorded here even though the
    /// install names one.  GL's vertex-array binding is a single slot that the
    /// input domain also writes, and the slot's owner is
    /// [`NativeGlProvider::bound_vertex_array`]; a copy kept per pipeline would
    /// go stale the moment the geometry domain reconciled inputs, which under the
    /// uncached execution mode is on every single request.
    pub(super) topology: GlPrimitiveTopology,
}

#[cfg(any(feature = "native-gl-wgl", feature = "native-gles-egl"))]
impl NativeGlProvider {
    /// # Safety
    ///
    /// Same as [`discover_current_glow`]: the caller keeps `gl` current and
    /// exclusively owned by this thread for the provider's entire lifetime.
    pub(crate) unsafe fn from_current(
        gl: glow::Context,
        stamp: ContextStamp,
        unmap_buffer: super::context::UnmapBufferFn,
    ) -> Result<Self, NativeDiscoveryError> {
        // SAFETY: forwarded from this constructor's current-context contract.
        let discovery = unsafe { discover_current_glow(&gl, stamp) }?;
        Ok(Self::assemble(gl, discovery, unmap_buffer))
    }

    /// Assembles the provider over an already-collected discovery snapshot.
    ///
    /// The caller must guarantee `gl` is the exact context that produced
    /// `discovery` and that it is current on the calling thread.
    ///
    /// # Safety
    ///
    /// Same as [`Self::from_current`].
    pub(crate) unsafe fn from_discovered(
        gl: glow::Context,
        discovery: GlDiscoverySnapshot,
        unmap_buffer: super::context::UnmapBufferFn,
    ) -> Self {
        Self::assemble(gl, discovery, unmap_buffer)
    }

    fn assemble(
        gl: glow::Context,
        discovery: GlDiscoverySnapshot,
        unmap_buffer: super::context::UnmapBufferFn,
    ) -> Self {
        Self {
            gl,
            unmap_buffer,
            discovery,
            lifecycle: GlContextLifecycle::Active,
            owner: OwnerThreadIdentity::current(),
            next_slot: 0,
            buffers: BTreeMap::new(),
            textures: BTreeMap::new(),
            renderbuffers: BTreeMap::new(),
            samplers: BTreeMap::new(),
            shaders: BTreeMap::new(),
            programs: BTreeMap::new(),
            vertex_arrays: BTreeMap::new(),
            framebuffers: BTreeMap::new(),
            queries: BTreeMap::new(),
            syncs: BTreeMap::new(),
            fences: GlFenceLeaseBook::default(),
            surface: GlSurfaceLeaseBook::new(),
            surface_suspended: false,
            // The Host has reported no extent yet, so the executor starts with
            // none: a provider that guessed a size here would hand out a lease
            // for a drawable it has never been told about.
            surface_extent: None,
            pass: None,
            raster: None,
            current_program: None,
            bound_vertex_array: None,
            active_query: None,
            active_compute_program: None,
            pixel_store: GlPixelStoreState::DEFAULT,
            context_state: ContextState::new(ExecutionMode::Optimized),
            pipeline_blocks: RasterPipelineBlockInterner::new(),
        }
    }

    /// Allocates one monotonically increasing slot.
    ///
    /// Slots are never reused within one context generation, so the `0`
    /// object generation stays sound; any future slot-reuse design must
    /// introduce per-slot generations first.
    pub(super) fn slot(&mut self, operation: &'static str) -> Result<u32, GlError> {
        let slot = self.next_slot;
        self.next_slot = self
            .next_slot
            .checked_add(1)
            .ok_or(GlError::OutOfMemory { operation })?;
        Ok(slot)
    }

    pub(super) fn validation(operation: &'static str, message: &'static str) -> GlError {
        GlError::Validation {
            operation,
            message: message.into(),
        }
    }

    pub(super) fn driver_error(&self, operation: &'static str) -> Result<(), GlError> {
        use glow::HasContext as _;
        // SAFETY: upheld by NativeGlProvider::from_current.
        let error = unsafe { self.gl.get_error() };
        (error == glow::NO_ERROR)
            .then_some(())
            .ok_or_else(|| GlError::Driver {
                operation,
                message: format!("GL error 0x{error:04x}"),
            })
    }

    /// Shared framebuffer-completeness observation for copy and pass work.
    pub(super) fn require_complete(&self, operation: &'static str) -> Result<(), GlError> {
        use glow::HasContext as _;
        // SAFETY: current-context contract.
        let status = unsafe { self.gl.check_framebuffer_status(glow::FRAMEBUFFER) };
        if status == glow::FRAMEBUFFER_COMPLETE {
            Ok(())
        } else {
            Err(GlError::IncompleteFramebuffer { operation, status })
        }
    }

    pub(super) fn buffer(
        &self,
        operation: &'static str,
        id: BufferId,
    ) -> Result<(glow::NativeBuffer, GlBufferDesc), GlError> {
        self.validate_object_context(operation, id.context)?;
        self.buffers
            .get(&id)
            .copied()
            .ok_or_else(|| Self::validation(operation, "buffer is not live"))
    }

    pub(super) fn texture(
        &self,
        operation: &'static str,
        id: TextureId,
    ) -> Result<(glow::NativeTexture, GlTextureDesc), GlError> {
        self.validate_object_context(operation, id.context)?;
        self.textures
            .get(&id)
            .copied()
            .ok_or_else(|| Self::validation(operation, "texture is not live"))
    }

    pub(super) fn renderbuffer(
        &self,
        operation: &'static str,
        id: RenderbufferId,
    ) -> Result<(glow::NativeRenderbuffer, GlRenderBufferDesc), GlError> {
        self.validate_object_context(operation, id.context)?;
        self.renderbuffers
            .get(&id)
            .copied()
            .ok_or_else(|| Self::validation(operation, "renderbuffer is not live"))
    }

    pub(super) fn sampler(
        &self,
        operation: &'static str,
        id: SamplerId,
    ) -> Result<glow::NativeSampler, GlError> {
        self.validate_object_context(operation, id.context)?;
        // Map keys are full identities, so a hit implies the same generation.
        self.samplers
            .get(&id)
            .copied()
            .ok_or_else(|| Self::validation(operation, "sampler is not live"))
    }

    pub(super) fn shader(
        &self,
        operation: &'static str,
        id: ShaderId,
    ) -> Result<glow::NativeShader, GlError> {
        self.validate_object_context(operation, id.context)?;
        self.shaders
            .get(&id)
            .copied()
            .ok_or_else(|| Self::validation(operation, "shader is not live"))
    }

    pub(super) fn program(
        &self,
        operation: &'static str,
        id: ProgramId,
    ) -> Result<&NativeProgram, GlError> {
        self.validate_object_context(operation, id.context)?;
        self.programs
            .get(&id)
            .filter(|entry| entry.generation == id.generation)
            .ok_or_else(|| Self::validation(operation, "program is not live"))
    }

    /// Makes `program` the driver's current program, if it is not already.
    ///
    /// GL has exactly one current program, and three verbs can change it: a
    /// link (which runs its reflection inside a bind scope and clears the
    /// selection when it ends), a raster pipeline install, and a compute program
    /// install.  [`Self::current_program`] records which one the driver holds,
    /// and this is the only function that writes both it and the driver, which is
    /// what makes "the driver holds this program" a fact rather than an
    /// assumption.
    ///
    /// It is called by every verb that *uses* a program rather than by every verb
    /// that selects one, because the two selections are not additive: a compute
    /// install between a raster install and a draw leaves the compute program
    /// current, and a raster install between a compute install and a dispatch
    /// leaves the raster one.  Re-asserting at the point of use is what makes each
    /// verb correct without either having to know that the other ran, and the
    /// comparison makes it free when nothing did.
    pub(super) fn ensure_program(
        &mut self,
        operation: &'static str,
        program: ProgramId,
    ) -> Result<(), GlError> {
        if self.current_program == Some(program) {
            return Ok(());
        }
        self.select_program(operation, program)
    }

    /// Selects a program after the state authority decided that the program
    /// leaf must be emitted. This deliberately makes no provider-local skip
    /// decision.
    pub(super) fn select_program(
        &mut self,
        operation: &'static str,
        program: ProgramId,
    ) -> Result<(), GlError> {
        use glow::HasContext as _;
        let raw = self.program(operation, program)?.raw;
        // SAFETY: current-context contract; the record was resolved live above,
        // so the name handed to GL belongs to this context.
        unsafe {
            self.gl.use_program(Some(raw));
        }
        match self.driver_error(operation) {
            Ok(()) => {
                self.current_program = Some(program);
                Ok(())
            }
            Err(error) => {
                // A failed useProgram may have reached the driver. Never let
                // the preceding native fact become a later skip.
                self.current_program = None;
                Err(error)
            }
        }
    }

    /// Builds the immutable state packet for one raster pipeline. Raster
    /// pipeline creation allocates a distinct linked program, so its complete
    /// context/slot/generation identity is collision-free here.
    pub(super) fn raster_state_packet(
        &mut self,
        pipeline: &crate::backend::gl::api::GlRasterPipeline,
    ) -> Result<RasterPipelinePacket, GlError> {
        let blocks = self
            .pipeline_blocks
            .intern(pipeline, "intern native raster pipeline")
            .map_err(|_| GlError::OutOfMemory {
                operation: "intern native raster pipeline",
            })?;
        Ok(RasterPipelinePacket {
            identity: GlObjectKey::new(
                pipeline.program.context,
                pipeline.program.slot,
                pipeline.program.generation,
            ),
            blocks,
        })
    }

    pub(super) const fn state_program_key(program: ProgramId) -> CanonicalBlockId {
        CanonicalBlockId::object(program)
    }

    pub(super) fn vertex_array(
        &self,
        operation: &'static str,
        id: VertexArrayId,
    ) -> Result<&NativeVertexArray, GlError> {
        self.validate_object_context(operation, id.context)?;
        self.vertex_arrays
            .get(&id)
            .filter(|entry| entry.generation == id.generation)
            .ok_or_else(|| Self::validation(operation, "vertex array is not live"))
    }

    /// Makes `array` the vertex array the driver holds, if it is not already.
    ///
    /// The mirror of [`Self::ensure_program`] for the other binding slot GL has,
    /// and it exists for the same reason: several verbs reach the vertex-array
    /// binding -- an input reconcile, a pipeline install, and the empty array a
    /// compute dispatch binds -- so "which array the driver holds" is a fact
    /// about the binding rather than about any one verb, and
    /// [`Self::bound_vertex_array`] is where it is recorded.  Every verb that
    /// *uses* the binding re-asserts it here rather than trusting whatever the
    /// last writer intended, and the comparison makes that free when nothing else
    /// ran in between.
    ///
    /// Returns the live record so the caller reads the index binding and layout
    /// from the array that is actually bound, which is the whole point: a raster
    /// install names the array it wants, but by draw time the geometry domain may
    /// legitimately have replaced it.
    pub(super) fn ensure_vertex_array(
        &mut self,
        operation: &'static str,
        array: VertexArrayId,
    ) -> Result<&NativeVertexArray, GlError> {
        use glow::HasContext as _;
        let raw = self.vertex_array(operation, array)?.raw;
        if self.bound_vertex_array != Some(array) {
            // SAFETY: current-context contract; the record was resolved live
            // above, so the name handed to GL belongs to this context.
            unsafe {
                self.gl.bind_vertex_array(Some(raw));
            }
            self.bound_vertex_array = Some(array);
        }
        self.vertex_array(operation, array)
    }

    pub(super) fn framebuffer(
        &self,
        operation: &'static str,
        id: FramebufferId,
    ) -> Result<&NativeFramebuffer, GlError> {
        self.validate_object_context(operation, id.context)?;
        self.framebuffers
            .get(&id)
            .filter(|entry| entry.generation == id.generation)
            .ok_or_else(|| Self::validation(operation, "framebuffer is not live"))
    }

    pub(super) fn query(
        &self,
        operation: &'static str,
        id: QueryId,
    ) -> Result<&NativeQuery, GlError> {
        self.validate_object_context(operation, id.context)?;
        self.queries
            .get(&id)
            .filter(|entry| entry.generation == id.generation)
            .ok_or_else(|| Self::validation(operation, "query is not live"))
    }

    /// Clears every executable table. Context loss makes every borrowed GL
    /// name invalid; only Fluxel-owned state is reset, and no driver call is
    /// made against a possibly-dead object.
    pub(super) fn reset_executable_state(&mut self) {
        self.buffers.clear();
        self.textures.clear();
        self.renderbuffers.clear();
        self.samplers.clear();
        self.shaders.clear();
        self.programs.clear();
        self.vertex_arrays.clear();
        self.framebuffers.clear();
        self.queries.clear();
        self.syncs.clear();
        self.fences.revoke_all();
        self.pass = None;
        self.raster = None;
        self.current_program = None;
        self.bound_vertex_array = None;
        self.active_query = None;
        self.active_compute_program = None;
        self.context_state.event(StateEvent::ContextLost);
        let _ = self.surface.invalidate_generation();
        // The recorded extent dies with the context generation it was observed
        // under: a replacement context may serve a differently sized drawable,
        // and a lease carrying the old extent would be a wrong size that nothing
        // reports. The Host re-reports the extent through `resize_surface`.
        self.surface_extent = None;
    }
}

#[cfg(any(feature = "native-gl-wgl", feature = "native-gles-egl"))]
impl crate::backend::gl::api::GlFamilyApi for NativeGlProvider {
    fn lifecycle(&self) -> GlContextLifecycle {
        self.lifecycle
    }
    fn owner_thread(&self) -> OwnerThreadIdentity {
        self.owner
    }
    fn assert_owner_thread(&self, operation: &'static str) -> Result<(), GlError> {
        let actual = OwnerThreadIdentity::current();
        (actual == self.owner)
            .then_some(())
            .ok_or(GlError::WrongThread {
                operation,
                expected: self.owner,
                actual,
            })
    }
    fn discovery(&self) -> &GlDiscoverySnapshot {
        &self.discovery
    }
    fn context_lost(&mut self) -> Result<(), GlError> {
        self.assert_ready("context-lost")?;
        self.lifecycle = GlContextLifecycle::Lost;
        self.reset_executable_state();
        Ok(())
    }
}
