//! Compiled submission plans, presentation leases, and completion evidence.
//!
//! RenderGraph supplies an ordered realizable plan and lifetime decisions.
//! Backends execute it as submitted and report facts; this module builds no
//! DAG, hazard tracker, scheduler, or CPU command replay layer.
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
#[path = "submission_plan.rs"]
mod plan;
pub use plan::{IssuedSteps, StepDesc, StepKind, StepWait, SubmissionPlan};

/// Host observation only. It cannot be used as a GPU queue wait.
pub struct CompletionPoint<B: Backend> {
    pub(crate) device: DeviceId,
    pub(crate) native: B::NativeCompletion,
}
impl<B: Backend> Clone for CompletionPoint<B> {
    fn clone(&self) -> Self {
        Self {
            device: self.device,
            native: self.native.clone(),
        }
    }
}
impl<B: Backend> CompletionPoint<B> {
    /// # Safety
    /// The native point must be valid completion evidence on this device.
    pub unsafe fn from_native(device: DeviceId, native: B::NativeCompletion) -> Self {
        Self { device, native }
    }
    pub fn device(&self) -> DeviceId {
        self.device
    }
}
pub enum CompletionState {
    Pending,
    Complete,
    Failed(Error),
}
impl<B: CompletionBackend> Device<B> {
    pub fn completion_status(&self, point: &CompletionPoint<B>) -> Result<CompletionState> {
        if point.device != self.id {
            return Err(Error::WrongDevice);
        }
        Ok(B::completion_status(&self.native, &point.native))
    }
    pub async fn wait_completion(&self, point: &CompletionPoint<B>) -> Result<()> {
        if point.device != self.id {
            return Err(Error::WrongDevice);
        }
        B::wait_completion(&self.native, &point.native).await
    }
    pub async fn wait_idle(&self) -> Result<()> {
        B::wait_idle(&self.native).await
    }
}

/// The request is per execution, never part of the reusable plan topology.
/// None yields no normal-path host evidence. End may yield multiple lanes.
pub enum CompletionRequest {
    None,
    End,
    Steps(Arc<[StepId]>),
}
impl Default for CompletionRequest {
    fn default() -> Self {
        Self::End
    }
}
impl CompletionRequest {
    pub(crate) fn validate<B: Backend>(&self, plan: &SubmissionPlan<B>) -> Result<()> {
        if let Self::Steps(steps) = self {
            if steps
                .iter()
                .any(|s| s.plan != plan.id || s.index as usize >= plan.steps.len())
            {
                return Err(Error::InvalidState);
            }
        }
        Ok(())
    }
}
pub struct CompletionCoverage {
    pub lane_frontiers: Vec<(LaneId, StepId)>,
}
pub struct Checkpoint<B: Backend> {
    pub point: CompletionPoint<B>,
    pub covers: CompletionCoverage,
}
pub struct PresentReceipt {
    pub step: Option<StepId>,
    pub accepted: bool,
}
pub struct SubmissionReceipt<B: Backend> {
    pub device: DeviceId,
    pub index: Option<SubmissionIndex>,
    pub accepted: Vec<StepId>,
    pub checkpoints: Vec<Checkpoint<B>>,
    pub presents: Vec<PresentReceipt>,
}
/// A true issue prefix with no mandatory per-step fence.
pub struct IssuedPrefix<B: Backend> {
    pub steps: IssuedSteps,
    pub checkpoints: Vec<Checkpoint<B>>,
}
pub struct NativeUncertainty<B: Backend> {
    pub device: DeviceId,
    pub native: B::NativeUncertainty,
}
impl<B: Backend> NativeUncertainty<B> {
    /// Carries backend recovery state after an irrevocable native issue whose
    /// acceptance cannot be determined. Never use it for a clean rejection.
    /// # Safety
    /// The native state must belong to this Device and identify the uncertain
    /// issue boundary truthfully.
    #[doc(hidden)]
    pub unsafe fn from_native(device: DeviceId, native: B::NativeUncertainty) -> Self {
        Self { device, native }
    }
}

pub struct RecordedWork<B: DeferredBackend> {
    pub(crate) device: DeviceId,
    pub(crate) lane: LaneId,
    pub(crate) native: B::NativeRecordedWork,
}
impl<B: DeferredBackend> RecordedWork<B> {
    /// Borrows the backend's one-shot recording for submission validation.
    ///
    /// # Safety
    /// Only the owning backend may call this while validating a packet for
    /// its device and lane. The reference must not be used to submit, mutate,
    /// retain beyond this borrow, or replay the recording.
    #[doc(hidden)]
    pub unsafe fn native_ref(&self) -> &B::NativeRecordedWork {
        &self.native
    }
    pub fn device(&self) -> DeviceId {
        self.device
    }
    pub fn lane(&self) -> LaneId {
        self.lane
    }
    /// Transfers this single-use native recording to the owning backend's
    /// submission path. The caller retains every referenced GPU resource until
    /// actual completion, including on partial or indeterminate acceptance.
    /// # Safety
    /// Call only while lowering a validated packet for this work's Device and
    /// Lane; never replay or submit the returned native work twice.
    #[doc(hidden)]
    pub unsafe fn into_native(self) -> B::NativeRecordedWork {
        self.native
    }
}
/// A standard mip tile rectangle, or the entire opaque packed mip tail.
/// For a packed tail use its `first_mip`, layer zero, origin `[0, 0, 0]`,
/// and count `[tile_count, 1, 1]`. Partial packed-tail mapping is invalid.
pub struct SparseTile {
    /// Exact native aspect group, or None for the tiling's primary group.
    pub aspect: Option<TextureAspects>,
    pub mip: u32,
    pub layer: u32,
    pub origin: [u32; 3],
    pub count: [u32; 3],
}
impl SparseTile {
    /// Selects all tiles in the packed mip tail of layer zero. When the tiling
    /// reports a shared tail this maps the tail for every array layer.
    pub fn packed_tail(tail: SparsePackedMipTail) -> Self {
        Self {
            aspect: None,
            mip: tail.first_mip,
            layer: 0,
            origin: [0, 0, 0],
            count: [tail.tile_count, 1, 1],
        }
    }
    /// Selects the packed tail of an explicitly reported native aspect group.
    pub fn packed_tail_for(aspect: TextureAspects, tail: SparsePackedMipTail) -> Self {
        Self {
            aspect: Some(aspect),
            ..Self::packed_tail(tail)
        }
    }
}
/// Mapping changes are ordered with commands on their submission lane. The caller
/// must ensure mapped residency, resource states, and aliasing hazards for every
/// GPU access; mapping does not initialize tile contents.
/// Mapping alone does not initialize texture contents or native attachment
/// metadata. Before using newly mapped render-target or depth/stencil memory,
/// the caller must initialize affected subresources using a complete copy or
/// clear, or a supported transition from `ImageLayout::Undefined` on their
/// first use. Preserve existing contents when remapping by choosing an
/// initialization operation that is valid for the intended data; backends do
/// not infer a discard from a mapping or from `ImageLayout::Common`.
pub struct SparseBind<B: Backend> {
    pub resource: Texture<B>,
    /// Regions are traversed in order; standard rectangles are x-major. A
    /// packed tail contributes its reported number of consecutive tiles.
    pub tiles: Vec<SparseTile>,
    /// `None` unmaps the regions. New mappings do not initialize their contents.
    pub allocation: Option<SparseAllocation<B>>,
    /// Byte offset in the allocation for the first mapped tile. Must be a
    /// multiple of the texture's reported tile size; zero for an unmap.
    pub allocation_offset: u64,
}
#[must_use = "present or explicitly abandon this exclusive surface lease"]
pub struct SurfaceImage<B: Backend> {
    pub(crate) device: DeviceId,
    pub(crate) surface: SurfaceId,
    pub(crate) generation: u64,
    pub(crate) native: B::NativeSurfaceImage,
    pub(crate) stats: Arc<RhiStats>,
}
impl<B: Backend> SurfaceImage<B> {
    pub fn device(&self) -> DeviceId {
        self.device
    }
    pub fn surface(&self) -> SurfaceId {
        self.surface
    }
    pub fn generation(&self) -> u64 {
        self.generation
    }
    pub fn is_current(&self) -> bool {
        self.native.is_current(self.generation)
    }
    /// # Safety: native RHI backend only; never persist a reference beyond
    /// the source surface lease or mutate its state outside native protocol.
    #[doc(hidden)]
    pub unsafe fn native_ref(&self) -> &B::NativeSurfaceImage {
        &self.native
    }
    /// # Safety
    /// Native submission/presentation accepts this single-use lease. Native
    /// owner invokes try_handoff() exactly when acceptance becomes irrevocable;
    /// dropping an unhanded native lease invokes abandon exactly once.
    pub unsafe fn into_native(self) -> B::NativeSurfaceImage {
        self.native
    }
}
/// One-shot plan payload. The native lease carries private access to its
/// origin surface's current generation; validation does not require a second
/// mutable Surface borrow during deferred submission.
pub struct PresentBinding<B: Backend> {
    pub(crate) image: SurfaceImage<B>,
    pub(crate) handed_off: bool,
}
impl<B: Backend> PresentBinding<B> {
    pub fn new(image: SurfaceImage<B>) -> Self {
        Self {
            image,
            handed_off: false,
        }
    }
    pub fn image(&self) -> &SurfaceImage<B> {
        &self.image
    }
    pub fn into_image(self) -> Result<SurfaceImage<B>> {
        if self.handed_off {
            return Err(Error::InvalidState);
        }
        Ok(self.image)
    }
    /// # Safety
    /// Backend-only irrevocable native present boundary. After success the
    /// binding cannot be returned as a clean rejection.
    pub unsafe fn try_handoff(&mut self) -> Result<()> {
        if self.handed_off {
            return Err(Error::InvalidState);
        }
        self.image.native.try_handoff(self.image.generation)?;
        self.image.stats.record_present_accepted();
        self.handed_off = true;
        Ok(())
    }
}

pub enum StepPayload<B: DeferredBackend> {
    Commands(Vec<RecordedWork<B>>),
    SparseBind(Vec<SparseBind<B>>),
    Present(PresentBinding<B>),
}
pub struct BoundStep<B: DeferredBackend> {
    pub step: StepId,
    pub payload: StepPayload<B>,
}
impl<B: DeferredBackend> BoundStep<B> {
    fn validate(&self, plan: &SubmissionPlan<B>, index: usize, device: DeviceId) -> Result<()> {
        let desc = &plan.steps[index];
        if self.step != plan.step(index) {
            return Err(Error::InvalidState);
        }
        match (&self.payload, desc.kind) {
            (StepPayload::Commands(work), StepKind::Commands) => {
                if work.iter().any(|w| w.device != device) {
                    return Err(Error::WrongDevice);
                }
                if work.iter().any(|w| w.lane != desc.lane) {
                    return Err(Error::WrongLane);
                }
            }
            (StepPayload::SparseBind(binds), StepKind::SparseBind) => {
                if binds.iter().any(|b| {
                    b.resource.device != device
                        || b.allocation.as_ref().is_some_and(|a| a.device != device)
                }) {
                    return Err(Error::WrongDevice);
                }
            }
            (StepPayload::Present(binding), StepKind::Present) => {
                if binding.image.device != device {
                    return Err(Error::WrongDevice);
                }
                if binding.handed_off {
                    return Err(Error::InvalidState);
                }
                if !binding.image.is_current() {
                    return Err(Error::SurfaceOutdated);
                }
            }
            _ => return Err(Error::InvalidState),
        }
        Ok(())
    }
}
pub struct SubmitPacket<B: DeferredBackend> {
    pub plan: SubmissionPlan<B>,
    pub steps: Vec<BoundStep<B>>,
    pub completion: CompletionRequest,
}
impl<B: DeferredBackend> SubmitPacket<B> {
    /// Checks only binding integrity. The compiled graph and synchronization
    /// choices remain the RenderGraph's responsibility.
    pub(super) fn validate(&self, device: DeviceId) -> Result<()> {
        // Do this before native submission so a clean rejection can return the
        // unchanged owning packet. Once native submission begins, only the
        // backend can state which prefix has become externally observable.
        if self.plan.id.device != device {
            return Err(Error::WrongDevice);
        }
        if self.steps.len() != self.plan.steps.len() {
            return Err(Error::InvalidState);
        }
        for (index, bound) in self.steps.iter().enumerate() {
            bound.validate(&self.plan, index, device)?;
        }
        self.completion.validate(&self.plan)
    }
}
pub struct DeferredTail<B: DeferredBackend> {
    pub plan: SubmissionPlan<B>,
    pub next_step: u32,
    pub pending: Vec<BoundStep<B>>,
    /// Backend-provided emergency evidence covering the accepted frontier.
    pub accepted_frontier: Vec<Checkpoint<B>>,
    pub completion: CompletionRequest,
}
impl<B: DeferredBackend> DeferredTail<B> {
    fn validate(&self, device: DeviceId) -> Result<()> {
        // A tail is not fresh work: it carries an accepted frontier from an
        // earlier call. Preserve that provenance so callers cannot resume it
        // on a lookalike queue or a rebuilt plan.
        if self.plan.id.device != device {
            return Err(Error::WrongDevice);
        }
        let first = self.next_step as usize;
        if first > self.plan.steps.len() || self.pending.len() != self.plan.steps.len() - first {
            return Err(Error::InvalidState);
        }
        for (offset, bound) in self.pending.iter().enumerate() {
            let index = first + offset;
            bound.validate(&self.plan, index, device)?;
        }
        for checkpoint in &self.accepted_frontier {
            if checkpoint.point.device != device {
                return Err(Error::WrongDevice);
            }
            for &(lane, step) in &checkpoint.covers.lane_frontiers {
                if step.plan != self.plan.id || step.index as usize >= first {
                    return Err(Error::InvalidState);
                }
                if self.plan.steps[step.index as usize].lane != lane {
                    return Err(Error::WrongLane);
                }
            }
        }
        self.completion.validate(&self.plan)
    }
}
pub enum RetryableSubmit<B: DeferredBackend> {
    Fresh(SubmitPacket<B>),
    Tail(DeferredTail<B>),
}
pub enum SubmitOutcome<B: DeferredBackend> {
    Accepted(SubmissionReceipt<B>),
    Rejected {
        error: Error,
        work: RetryableSubmit<B>,
    },
    PartiallyAccepted {
        error: Error,
        accepted: SubmissionReceipt<B>,
        tail: DeferredTail<B>,
    },
    Indeterminate {
        error: Error,
        uncertain: NativeUncertainty<B>,
    },
}

impl<B: DeferredBackend> Device<B> {
    /// Native command recording; Immediate backends can only obtain an
    /// encoder from an already opened ImmediateSession plan slot.
    pub fn encoder(&self, lane: LaneId) -> Result<CommandEncoder<B>> {
        CommandEncoder::new(self, lane)
    }
}
impl<B: DeferredBackend> CommandEncoder<B> {
    pub fn finish(mut self) -> Result<RecordedWork<B>> {
        // Finishing an incomplete pass or a poisoned encoder would turn a
        // failed recording path into apparently reusable native work.
        if self.poisoned || self.in_pass || self.debug_group_depth != 0 {
            return Err(Error::InvalidState);
        }
        let native = B::finish_encoder(self.native.take().ok_or(Error::InvalidState)?, self.lane)?;
        Ok(RecordedWork {
            device: self.device,
            lane: self.lane,
            native,
        })
    }
}
pub struct Queue<B: DeferredBackend> {
    device: Device<B>,
}
impl<B: DeferredBackend> Device<B> {
    pub fn queue(&self) -> Queue<B> {
        Queue {
            device: self.clone(),
        }
    }
}
impl<B: DeferredBackend> Queue<B> {
    /// # Safety
    /// Renderer proves resource lifetimes and all compiled synchronization.
    pub unsafe fn submit(&self, packet: SubmitPacket<B>) -> SubmitOutcome<B> {
        if let Err(error) = packet.validate(self.device.id) {
            return SubmitOutcome::Rejected {
                error,
                work: RetryableSubmit::Fresh(packet),
            };
        }
        // SAFETY: packet validation above established this queue's device ownership.
        let outcome = unsafe { B::submit(&self.device.native, packet) };
        if matches!(
            &outcome,
            SubmitOutcome::Accepted(_) | SubmitOutcome::PartiallyAccepted { .. }
        ) {
            self.device.stats.record_submit();
        }
        outcome
    }
    /// # Safety
    /// Caller has observed every checkpoint in the accepted frontier.
    pub unsafe fn resume(&self, tail: DeferredTail<B>) -> SubmitOutcome<B> {
        // Revalidation guards Core-owned structural facts; the caller's unsafe
        // obligation remains proving completion of the accepted frontier.
        if let Err(error) = tail.validate(self.device.id) {
            return SubmitOutcome::Rejected {
                error,
                work: RetryableSubmit::Tail(tail),
            };
        }
        // SAFETY: tail validation above established the same device and accepted frontier.
        let outcome = unsafe { B::resume(&self.device.native, tail) };
        if matches!(
            &outcome,
            SubmitOutcome::Accepted(_) | SubmitOutcome::PartiallyAccepted { .. }
        ) {
            self.device.stats.record_submit();
        }
        outcome
    }
}
