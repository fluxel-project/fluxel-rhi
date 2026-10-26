//! The mutable driver-state authority for one real GL context.
//!
//! It remembers only successfully installed native state. It never owns GL
//! objects, records commands, schedules work, or infers resource hazards.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::backend::api::{
    BufferId, FramebufferId, ProgramId, QueryId, RenderbufferId, SamplerId, TextureId,
    VertexArrayId,
};

use super::{
    BindGroupKey, CanonicalBlockId, DriverKnowledge, ExecutionMode, GlObjectKey,
    RasterPipelineBlockInterner, RasterPipelineBlocks, RasterPipelinePacket, ResourceRef,
    StateDomain, StateEvent,
};

/// Pass-FBO state is not one opaque slot: read and draw targets as well as the
/// draw-buffer routing are independently mutable GL state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct PassPacket {
    pub(crate) draw_framebuffer: CanonicalBlockId,
    pub(crate) read_framebuffer: CanonicalBlockId,
    pub(crate) draw_buffers: CanonicalBlockId,
}
/// Pixel transfer has independent PACK/UNPACK settings and PBO bindings.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct PixelTransferPacket {
    pub(crate) pack: CanonicalBlockId,
    pub(crate) unpack: CanonicalBlockId,
    pub(crate) pack_buffer: CanonicalBlockId,
    pub(crate) unpack_buffer: CanonicalBlockId,
}

/// Differences the owner must lower. An exact pipeline hit has every flag
/// false. Geometry stays separate because a draw can replace the VAO after a
/// pipeline bind.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct RasterPipelineDiff {
    pub(crate) program: bool,
    pub(crate) raster: bool,
    pub(crate) depth_stencil: bool,
    pub(crate) blend: bool,
    pub(crate) multisample: bool,
}

/// Phase A must not associate two immutable packets with the same v13 object
/// identity. Rejecting that malformed private input keeps the O(1) fast path
/// sound instead of silently trusting a changed packet.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ContextStateError {
    UnknownPipeline,
    PipelineIdentityConflict,
}
impl RasterPipelineDiff {
    pub(crate) const fn is_empty(self) -> bool {
        !self.program && !self.raster && !self.depth_stencil && !self.blend && !self.multisample
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct InstalledRasterPipeline {
    identity: GlObjectKey,
    blocks: RasterPipelineBlocks,
}

/// A bound group is dirty until the owner has successfully flushed its private
/// packet. Resource uses deliberately do not appear here: visibility/barrier
/// tracking is not binding state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct BoundGroupPacket {
    pub(crate) group: BindGroupKey,
    pub(crate) index: u32,
    pub(crate) dynamic_offsets: Vec<u32>,
    /// Full program provenance, not a lossy `(slot, generation)` packing.
    /// Bind-group application must be revisited after a context replacement
    /// even where a provider happens to reuse both native name and slot.
    pub(crate) program_identity: CanonicalBlockId,
    pub(crate) dependencies: BTreeSet<ResourceRef>,
}

/// Immutable work to flush before a draw or dispatch. The owner calls
/// `acknowledge_bindings` only after all listed packets reached GL.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct BindingFlush {
    pub(crate) groups: Vec<BoundGroupPacket>,
}
impl BindingFlush {
    pub(crate) const fn is_empty(&self) -> bool {
        self.groups.is_empty()
    }
}

/// Private, structural keys for derived VAO/FBO cache entries. They are not
/// public GL objects and must be invalidated through their dependencies.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct DerivedCacheKey {
    pub(crate) kind: DerivedCacheKind,
    pub(crate) canonical: CanonicalBlockId,
}
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum DerivedCacheKind {
    VertexArray,
    DrawFramebuffer,
    ReadFramebuffer,
}

/// One real native/browser context's state authority. Submission, completion
/// and presentation remain separate execution concerns, but their context
/// owner calls `raw_access`/`event` whenever they touch mutable GL state.
pub(crate) struct ContextState {
    mode: ExecutionMode,
    pass: DriverKnowledge<PassPacket>,
    pipeline: DriverKnowledge<InstalledRasterPipeline>,
    current_program: DriverKnowledge<CanonicalBlockId>,
    geometry: DriverKnowledge<CanonicalBlockId>,
    compute: DriverKnowledge<CanonicalBlockId>,
    pixel_transfer: DriverKnowledge<PixelTransferPacket>,
    query: DriverKnowledge<CanonicalBlockId>,
    bindings: BTreeMap<u32, BoundGroupPacket>,
    dirty_bindings: BTreeSet<u32>,
    active_texture: DriverKnowledge<u32>,
    texture_slots: BTreeMap<u32, CanonicalBlockId>,
    // Samplers are independent texture-unit state in both desktop GL and
    // WebGL2.  They must not piggy-back on `texture_slots`: changing only a
    // compare/filter object is still a real `bind_sampler` mutation.
    sampler_slots: BTreeMap<u32, CanonicalBlockId>,
    uniform_slots: BTreeMap<u32, CanonicalBlockId>,
    storage_slots: BTreeMap<u32, CanonicalBlockId>,
    image_slots: BTreeMap<u32, CanonicalBlockId>,
    pipeline_packets: HashMap<GlObjectKey, RasterPipelineBlocks>,
    derived: HashMap<DerivedCacheKey, BTreeSet<ResourceRef>>,
}

impl ContextState {
    pub(crate) fn new(mode: ExecutionMode) -> Self {
        Self {
            mode,
            pass: DriverKnowledge::Unknown,
            pipeline: DriverKnowledge::Unknown,
            current_program: DriverKnowledge::Unknown,
            geometry: DriverKnowledge::Unknown,
            compute: DriverKnowledge::Unknown,
            pixel_transfer: DriverKnowledge::Unknown,
            query: DriverKnowledge::Unknown,
            bindings: BTreeMap::new(),
            dirty_bindings: BTreeSet::new(),
            active_texture: DriverKnowledge::Unknown,
            texture_slots: BTreeMap::new(),
            sampler_slots: BTreeMap::new(),
            uniform_slots: BTreeMap::new(),
            storage_slots: BTreeMap::new(),
            image_slots: BTreeMap::new(),
            pipeline_packets: HashMap::new(),
            derived: HashMap::new(),
        }
    }
    /// Pass begin carries load/clear/order semantics and is therefore never a
    /// cache hit. `pass_end`, draw, dispatch, query and barrier follow the same
    /// rule: this authority never elides execution commands.
    /// Returns `true` unconditionally so an owner cannot accidentally treat a
    /// structurally equal pass as a skipped GL command.
    pub(crate) const fn prepare_pass(&self, _: PassPacket) -> bool {
        true
    }
    pub(crate) fn commit_pass(&mut self, pass: PassPacket) {
        self.pass.set(pass);
    }
    pub(crate) fn pass_failed(&mut self) {
        self.pass.invalidate();
    }
    pub(crate) fn end_pass(&mut self) {
        self.pass.invalidate();
    }
    /// These are execution/visibility operations, never setters. Their `true`
    /// result is intentionally not cached and means the owner must emit GL.
    pub(crate) const fn draw(&self) -> bool {
        true
    }
    pub(crate) const fn dispatch(&self) -> bool {
        true
    }
    pub(crate) const fn clear(&self) -> bool {
        true
    }
    pub(crate) const fn query_command(&self) -> bool {
        true
    }
    pub(crate) const fn barrier(&self) -> bool {
        true
    }
    /// Registers a phase-A pipeline packet. Re-registration is legal only for
    /// exactly the same immutable packet, which makes identity hits sound.
    pub(crate) fn register_pipeline(
        &mut self,
        packet: RasterPipelinePacket,
    ) -> Result<(), ContextStateError> {
        match self.pipeline_packets.get(&packet.identity) {
            None => {
                self.pipeline_packets.insert(packet.identity, packet.blocks);
                Ok(())
            }
            Some(blocks) if *blocks == packet.blocks => Ok(()),
            Some(_) => Err(ContextStateError::PipelineIdentityConflict),
        }
    }
    pub(crate) fn prepare_pipeline(
        &self,
        wanted: RasterPipelinePacket,
    ) -> Result<RasterPipelineDiff, ContextStateError> {
        if self.pipeline_packets.get(&wanted.identity) != Some(&wanted.blocks) {
            return Err(if self.pipeline_packets.contains_key(&wanted.identity) {
                ContextStateError::PipelineIdentityConflict
            } else {
                ContextStateError::UnknownPipeline
            });
        }
        Ok(self.prepare_dynamic_pipeline(wanted))
    }

    /// The immediate backend includes dynamic viewport, stencil and blend
    /// values in its packet. Compare all canonical blocks instead of treating
    /// the program identity as an immutable pipeline identity.
    pub(crate) fn prepare_dynamic_pipeline(
        &self,
        wanted: RasterPipelinePacket,
    ) -> RasterPipelineDiff {
        if self.mode.may_skip()
            && self.pipeline.get().is_some_and(|have| {
                have.identity == wanted.identity && have.blocks == wanted.blocks
            })
            && self.current_program.agrees(&wanted.blocks.program)
        {
            return RasterPipelineDiff::default();
        }
        let prior = self.pipeline.get().copied();
        let diff = RasterPipelineDiff {
            // Compute and raster share GL_CURRENT_PROGRAM. A pipeline identity
            // miss caused solely by compute must still restore this leaf.
            program: !self.current_program.agrees(&wanted.blocks.program),
            raster: prior.is_none_or(|have| have.blocks.raster != wanted.blocks.raster),
            depth_stencil: prior
                .is_none_or(|have| have.blocks.depth_stencil != wanted.blocks.depth_stencil),
            blend: prior.is_none_or(|have| have.blocks.blend != wanted.blocks.blend),
            multisample: prior
                .is_none_or(|have| have.blocks.multisample != wanted.blocks.multisample),
        };
        diff
    }
    pub(crate) fn commit_pipeline(&mut self, wanted: RasterPipelinePacket) {
        self.current_program.set(wanted.blocks.program);
        self.pipeline.set(InstalledRasterPipeline {
            identity: wanted.identity,
            blocks: wanted.blocks,
        });
    }
    /// Must be called when lowering a reported leaf fails after a partial GL
    /// mutation. It never preserves a possibly stale exact-pipeline hit.
    pub(crate) fn pipeline_failed(&mut self) {
        self.pipeline.invalidate();
        self.current_program.invalidate();
    }
    pub(crate) fn prepare_geometry(&self, wanted: CanonicalBlockId) -> bool {
        if self.mode.may_skip() && self.geometry.agrees(&wanted) {
            false
        } else {
            true
        }
    }
    pub(crate) fn commit_geometry(&mut self, wanted: CanonicalBlockId) {
        self.geometry.set(wanted);
    }
    pub(crate) fn geometry_failed(&mut self) {
        self.geometry.invalidate();
    }
    pub(crate) fn prepare_compute_program(&self, wanted: CanonicalBlockId) -> bool {
        !(self.mode.may_skip() && self.current_program.agrees(&wanted))
    }
    pub(crate) fn commit_compute_program(&mut self, wanted: CanonicalBlockId) {
        self.current_program.set(wanted);
        self.compute.set(wanted);
    }
    /// A failed `useProgram` may have reached the driver. Clear both facts so
    /// neither a later dispatch nor a raster pipeline can skip a restore.
    pub(crate) fn compute_failed(&mut self) {
        self.current_program.invalidate();
        self.compute.invalidate();
    }
    /// Records an intentional `glUseProgram(0)` performed outside normal
    /// raster or compute installation.  The other state blocks remain valid,
    /// but neither pipeline may assume its program is still current.
    pub(crate) fn program_unbound(&mut self) {
        self.current_program.invalidate();
    }
    pub(crate) fn prepare_pixel_transfer(&self, wanted: PixelTransferPacket) -> bool {
        if self.mode.may_skip() && self.pixel_transfer.agrees(&wanted) {
            false
        } else {
            true
        }
    }
    pub(crate) fn commit_pixel_transfer(&mut self, wanted: PixelTransferPacket) {
        self.pixel_transfer.set(wanted);
    }
    /// Query begin/end are commands and are always emitted. This tracks only
    /// the active target for validation after a successful native begin/end.
    pub(crate) const fn prepare_query_command(&self, _: CanonicalBlockId) -> bool {
        true
    }
    pub(crate) fn commit_query_begin(&mut self, target: CanonicalBlockId) {
        self.query.set(target);
    }
    pub(crate) fn commit_query_end(&mut self) {
        self.query.invalidate();
    }
    pub(crate) fn query_failed(&mut self) {
        self.query.invalidate();
    }
    pub(crate) fn stage_bind_group(&mut self, packet: BoundGroupPacket) {
        let changed = self.bindings.get(&packet.index) != Some(&packet);
        if changed || !self.mode.may_skip() {
            self.dirty_bindings.insert(packet.index);
        }
        self.bindings.insert(packet.index, packet);
    }
    /// `active_texture` has exactly one authority. The owner prepares this
    /// before a texture slot bind and commits it only after `glActiveTexture`.
    pub(crate) fn prepare_texture_slot(
        &self,
        unit: u32,
        binding: CanonicalBlockId,
    ) -> (bool, bool) {
        (
            !(self.mode.may_skip() && self.active_texture.agrees(&unit)),
            !(self.mode.may_skip()
                && self
                    .texture_slots
                    .get(&unit)
                    .is_some_and(|known| *known == binding)),
        )
    }
    pub(crate) fn commit_texture_slot(&mut self, unit: u32, binding: CanonicalBlockId) {
        self.active_texture.set(unit);
        self.texture_slots.insert(unit, binding);
    }
    /// `glBindSampler` has no active-texture prerequisite, but is otherwise
    /// exactly the same kind of generation-safe unit cache as a texture bind.
    /// The caller commits only after the native/browser operation succeeds.
    pub(crate) fn prepare_sampler_slot(&self, unit: u32, binding: CanonicalBlockId) -> bool {
        !(self.mode.may_skip()
            && self
                .sampler_slots
                .get(&unit)
                .is_some_and(|known| *known == binding))
    }
    pub(crate) fn commit_sampler_slot(&mut self, unit: u32, binding: CanonicalBlockId) {
        self.sampler_slots.insert(unit, binding);
    }
    pub(crate) fn prepare_uniform_slot(&self, index: u32, binding: CanonicalBlockId) -> bool {
        !(self.mode.may_skip()
            && self
                .uniform_slots
                .get(&index)
                .is_some_and(|known| *known == binding))
    }
    pub(crate) fn commit_uniform_slot(&mut self, index: u32, binding: CanonicalBlockId) {
        self.uniform_slots.insert(index, binding);
    }
    pub(crate) fn prepare_storage_slot(&self, index: u32, binding: CanonicalBlockId) -> bool {
        !(self.mode.may_skip()
            && self
                .storage_slots
                .get(&index)
                .is_some_and(|known| *known == binding))
    }
    pub(crate) fn commit_storage_slot(&mut self, index: u32, binding: CanonicalBlockId) {
        self.storage_slots.insert(index, binding);
    }
    pub(crate) fn prepare_image_slot(&self, index: u32, binding: CanonicalBlockId) -> bool {
        !(self.mode.may_skip()
            && self
                .image_slots
                .get(&index)
                .is_some_and(|known| *known == binding))
    }
    pub(crate) fn commit_image_slot(&mut self, index: u32, binding: CanonicalBlockId) {
        self.image_slots.insert(index, binding);
    }
    pub(crate) fn binding_flush(&self) -> BindingFlush {
        BindingFlush {
            groups: self
                .dirty_bindings
                .iter()
                .filter_map(|index| self.bindings.get(index).cloned())
                .collect(),
        }
    }
    pub(crate) fn acknowledge_bindings(&mut self, flushed: &BindingFlush) {
        for packet in &flushed.groups {
            self.dirty_bindings.remove(&packet.index);
        }
    }
    /// Removes a logical bind-group from the applied-state mirror before its
    /// private packet is retired.  Keeping it would let a later draw treat a
    /// destroyed packet as already flushed.
    pub(crate) fn retire_bind_group(&mut self, group: BindGroupKey) {
        self.bindings.retain(|_, packet| packet.group != group);
        self.dirty_bindings
            .retain(|index| self.bindings.contains_key(index));
    }
    pub(crate) fn register_derived(
        &mut self,
        key: DerivedCacheKey,
        dependencies: BTreeSet<ResourceRef>,
    ) {
        self.derived.insert(key, dependencies);
    }
    pub(crate) fn event(&mut self, event: StateEvent) {
        match event {
            StateEvent::BufferRetired(id) => self.retire(ResourceRef::Buffer(id)),
            StateEvent::TextureRetired(id) => self.retire(ResourceRef::Texture(id)),
            StateEvent::RenderbufferRetired(id) => self.retire(ResourceRef::Renderbuffer(id)),
            StateEvent::FramebufferRetired(id) => self.retire(ResourceRef::Framebuffer(id)),
            StateEvent::QueryRetired(id) => self.retire(ResourceRef::Query(id)),
            StateEvent::SamplerRetired(id) => self.retire(ResourceRef::Sampler(id)),
            StateEvent::ProgramRetired(id) => self.retire(ResourceRef::Program(id)),
            StateEvent::VertexArrayRetired(id) => self.retire(ResourceRef::VertexArray(id)),
            StateEvent::DomainFailed(domain) => self.invalidate_domain(domain),
            StateEvent::ScopedRawAccess(access) => {
                for domain in access.domains().iter() {
                    self.invalidate_domain(domain);
                }
            }
            // Loss is terminal for this state mirror. A restored provider
            // constructs a new device and therefore a fresh `ContextState`.
            StateEvent::ContextLost => self.forget_context(),
        }
    }
    fn retire(&mut self, resource: ResourceRef) {
        self.derived
            .retain(|_, dependencies| !dependencies.contains(&resource));
        // A retired resource must not survive in a future flush packet.
        self.bindings
            .retain(|_, packet| !packet.dependencies.contains(&resource));
        self.dirty_bindings
            .retain(|index| self.bindings.contains_key(index));
        // Binding slots are state caches too.  Erase a matching object now;
        // although a complete identity would prevent a false hit after slot
        // reuse, eagerly forgetting it also prevents stale bookkeeping from
        // hiding a retirement while an application owns no replacement yet.
        self.texture_slots
            .retain(|_, value| !value.references(resource));
        self.sampler_slots
            .retain(|_, value| !value.references(resource));
        self.uniform_slots
            .retain(|_, value| !value.references(resource));
        self.storage_slots
            .retain(|_, value| !value.references(resource));
        self.image_slots
            .retain(|_, value| !value.references(resource));
        match resource {
            ResourceRef::Program(_) => {
                self.pipeline.invalidate();
                self.compute.invalidate();
                self.current_program.invalidate();
            }
            ResourceRef::VertexArray(_) | ResourceRef::Buffer(_) => self.geometry.invalidate(),
            ResourceRef::Texture(_)
            | ResourceRef::Renderbuffer(_)
            | ResourceRef::Framebuffer(_) => {
                self.pass.invalidate();
                self.pixel_transfer.invalidate();
            }
            ResourceRef::Query(_) => self.query.invalidate(),
            ResourceRef::Sampler(_) => self.pixel_transfer.invalidate(),
            _ => {}
        }
    }
    fn invalidate_domain(&mut self, domain: StateDomain) {
        match domain {
            StateDomain::PassFramebuffer => self.pass.invalidate(),
            StateDomain::RasterPipeline => {
                self.pipeline.invalidate();
                self.current_program.invalidate();
            }
            StateDomain::Geometry => self.geometry.invalidate(),
            StateDomain::Bindings => {
                self.dirty_bindings.extend(self.bindings.keys().copied());
                self.active_texture.invalidate();
                self.texture_slots.clear();
                self.sampler_slots.clear();
                self.uniform_slots.clear();
                self.storage_slots.clear();
                self.image_slots.clear();
            }
            StateDomain::Compute => {
                self.compute.invalidate();
                self.current_program.invalidate();
            }
            StateDomain::PixelTransferCopy => self.pixel_transfer.invalidate(),
            StateDomain::Query => self.query.invalidate(),
            StateDomain::DerivedCaches => self.derived.clear(),
        }
    }
    fn forget_context(&mut self) {
        self.pass.invalidate();
        self.pipeline.invalidate();
        self.current_program.invalidate();
        self.geometry.invalidate();
        self.compute.invalidate();
        self.pixel_transfer.invalidate();
        self.query.invalidate();
        self.dirty_bindings.extend(self.bindings.keys().copied());
        self.derived.clear();
        self.pipeline_packets.clear();
        self.active_texture.invalidate();
        self.texture_slots.clear();
        self.sampler_slots.clear();
        self.uniform_slots.clear();
        self.storage_slots.clear();
        self.image_slots.clear();
    }
}

impl CanonicalBlockId {
    fn references(self, resource: ResourceRef) -> bool {
        let (context, slot, generation) = match self {
            Self::Object {
                context,
                slot,
                generation,
            }
            | Self::UniformRange {
                context,
                slot,
                generation,
                ..
            } => (context, slot, generation),
            Self::Allocated(_) => return false,
        };
        // The cache is intentionally type-erased.  A same-number identity of
        // another GL object kind may cause an extra bind, never an unsafe skip.
        // Context/slot/generation equality is nevertheless mandatory.
        (match resource {
            ResourceRef::Buffer(id) => (id.context, id.slot, id.generation),
            ResourceRef::Texture(id) => (id.context, id.slot, id.generation),
            ResourceRef::Renderbuffer(id) => (id.context, id.slot, id.generation),
            ResourceRef::Sampler(id) => (id.context, id.slot, id.generation),
            ResourceRef::Shader(id) => (id.context, id.slot, id.generation),
            ResourceRef::Program(id) => (id.context, id.slot, id.generation),
            ResourceRef::VertexArray(id) => (id.context, id.slot, id.generation),
            ResourceRef::Framebuffer(id) => (id.context, id.slot, id.generation),
            ResourceRef::Query(id) => (id.context, id.slot, id.generation),
            ResourceRef::Sync(id) => (id.context, id.slot, id.generation),
        }) == (context, slot, generation)
    }
}

#[cfg(test)]
#[path = "../../tests/backend/state/context_tests.rs"]
mod tests;
