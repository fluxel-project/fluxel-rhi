//! Optional sparse-resource mapping and submission contract.
//!
//! The renderer supplies mappings and synchronization explicitly. Backends
//! reject this API when their native sparse capability is unavailable.
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

#[cfg(test)]
#[path = "../tests/api/sparse.rs"]
mod tests;

/// Tile dimensions, texture grid, and allocation alignment for a sparse texture.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SparseTextureTiling {
    /// Native binding groups. The remaining fields mirror the first group;
    /// select a group explicitly with `SparseTile::aspect` when needed.
    pub aspects: Vec<SparseAspectTiling>,
    pub tile_extent: [u32; 3],
    /// Standard mip zero grid, or `[0, 0, 0]` when mip zero is packed.
    pub tile_grid: [u32; 3],
    pub tile_bytes: u64,
    /// Grid for each standard mip, ordered from mip zero. Empty when all
    /// mips belong to the opaque packed tail.
    pub standard_mip_grids: Vec<[u32; 3]>,
    /// Packed mips share an opaque allocation and must be mapped or unmapped
    /// together, using the packed-tail SparseTile representation. See
    /// `packed_tail_shared_by_layers` for its array-layer scope.
    pub packed_tail: Option<SparsePackedMipTail>,
    /// A single opaque tail covers all array layers. Bind it once with layer zero.
    pub packed_tail_shared_by_layers: bool,
}
/// Tiling for one native binding group; a combined mask means the aspects
/// share one indivisible mapping rather than independently bindable planes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SparseAspectTiling {
    pub aspect: TextureAspects,
    pub tile_extent: [u32; 3],
    pub tile_grid: [u32; 3],
    pub tile_bytes: u64,
    pub standard_mip_grids: Vec<[u32; 3]>,
    pub packed_tail: Option<SparsePackedMipTail>,
    pub packed_tail_shared_by_layers: bool,
}
impl SparseTextureTiling {
    /// Construct the compatibility fields from the primary native group.
    pub fn from_aspects(aspects: Vec<SparseAspectTiling>) -> Option<Self> {
        let first = aspects.first()?;
        Some(Self {
            tile_extent: first.tile_extent,
            tile_grid: first.tile_grid,
            tile_bytes: first.tile_bytes,
            standard_mip_grids: first.standard_mip_grids.clone(),
            packed_tail: first.packed_tail,
            packed_tail_shared_by_layers: first.packed_tail_shared_by_layers,
            aspects,
        })
    }
}
/// Opaque packed mip tail. Its array-layer scope is reported by the tiling.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SparsePackedMipTail {
    pub first_mip: u32,
    pub mip_count: u32,
    pub tile_count: u32,
}
pub struct SparseDomain<'a, B: SparseBackend> {
    device: &'a Device<B>,
}
impl<B: SparseBackend> Device<B> {
    pub fn sparse(&self) -> Option<SparseDomain<'_, B>> {
        self.caps.sparse.then_some(SparseDomain { device: self })
    }
}
impl<B: SparseBackend> SparseDomain<'_, B> {
    /// Creates a sparse texture; newly created tiles are unmapped.
    pub fn texture(&self, desc: &TextureDesc) -> Result<Texture<B>> {
        validate_transient_texture_desc(desc)?;
        if desc.usage.contains(TextureUsage::TRANSIENT_ATTACHMENT) {
            return Err(Error::InvalidState);
        }
        // SAFETY: the backend creates this native object for this device.
        Ok(unsafe {
            Texture::from_native(
                self.device.id,
                B::create_sparse_texture(&self.device.native, desc)?,
            )
        })
    }
    /// Reports the physical tile shape for this particular sparse texture.
    pub fn tiling(&self, texture: &Texture<B>) -> Result<SparseTextureTiling> {
        if texture.device != self.device.id {
            return Err(Error::WrongDevice);
        }
        B::sparse_texture_tiling(&self.device.native, &texture.native)
    }
    pub fn allocation(&self, size: u64) -> Result<SparseAllocation<B>> {
        if size == 0 {
            return Err(Error::InvalidRange);
        }
        // SAFETY: SPI allocation and the checked backing allocation share this device.
        Ok(unsafe {
            SparseAllocation::from_native(
                self.device.id,
                B::create_sparse_allocation(&self.device.native, size)?,
            )
        })
    }
    /// Allocates memory using the texture's native memory requirements.
    /// The resulting allocation can only be bound to textures compatible with
    /// its native memory type and alignment; the backend checks each bind.
    pub fn allocation_for(&self, texture: &Texture<B>, size: u64) -> Result<SparseAllocation<B>> {
        if texture.device != self.device.id {
            return Err(Error::WrongDevice);
        }
        if size == 0 {
            return Err(Error::InvalidRange);
        }
        // SAFETY: the backend creates the allocation on the checked device.
        Ok(unsafe {
            SparseAllocation::from_native(
                self.device.id,
                B::create_sparse_allocation_for(&self.device.native, &texture.native, size)?,
            )
        })
    }
}
impl<B: ImmediateSparseBackend> SparseDomain<'_, B> {
    pub fn bind_immediate(
        &self,
        session: &mut ImmediateSession<B>,
        binds: &[SparseBind<B>],
    ) -> ImmediateRunOutcome<B> {
        if session.device.id != self.device.id {
            return session.fail(Error::WrongDevice);
        }
        session.sparse(binds)
    }
}

pub struct GpuSignalRequest {
    pub producer: StepId,
    pub consumer: StepId,
    pub consumer_lane: LaneId,
}
pub struct GpuWait<B: MultiQueueBackend> {
    pub signal: GpuSignalPoint<B>,
    pub destination: AccessScope,
}
/// A single-use signal for an explicitly selected consumer execution.
///
/// Step IDs identify positions in reusable plans, not individual executions.
/// The renderer must associate this token with the correct producer and
/// consumer executions, including when plan executions overlap. Core checks
/// the static identities but does not keep an execution registry.
pub struct GpuSignalPoint<B: MultiQueueBackend> {
    device: DeviceId,
    producer: StepId,
    producer_lane: LaneId,
    consumer: StepId,
    consumer_lane: LaneId,
    native: B::NativeGpuSignal,
}
impl<B: MultiQueueBackend> GpuSignalPoint<B> {
    /// # Safety
    /// The accepted producer installed a real GPU-waitable signal for exactly
    /// this particular consumer execution before returning this point. The
    /// metadata must describe the native signal and its actual producer and
    /// consumer executions, not merely matching reusable plan positions.
    /// Native signal storage stays valid until consumed or safely abandoned.
    pub unsafe fn from_native(
        device: DeviceId,
        producer: StepId,
        producer_lane: LaneId,
        consumer: StepId,
        consumer_lane: LaneId,
        native: B::NativeGpuSignal,
    ) -> Self {
        Self {
            device,
            producer,
            producer_lane,
            consumer,
            consumer_lane,
            native,
        }
    }
    /// Producer position in the reusable plan, not an execution identifier.
    pub fn producer(&self) -> StepId {
        self.producer
    }
    /// Native producer lane.
    pub fn producer_lane(&self) -> LaneId {
        self.producer_lane
    }
    /// Consumer position in the reusable plan, not an execution identifier.
    pub fn consumer(&self) -> StepId {
        self.consumer
    }
    /// Native consumer lane.
    pub fn consumer_lane(&self) -> LaneId {
        self.consumer_lane
    }
    /// Borrows the native signal for side-effect-free submission validation.
    ///
    /// # Safety
    /// Only the owning backend may inspect this value while lowering a
    /// validated packet. Inspection must not consume the one-shot wait, signal
    /// a fence, or treat a reusable plan position as an execution identity.
    #[doc(hidden)]
    pub unsafe fn native_ref(&self) -> &B::NativeGpuSignal {
        &self.native
    }
    /// Transfers the one-shot native signal to the backend wait installation.
    ///
    /// # Safety
    /// Only the owning backend may consume this signal while lowering a
    /// validated packet. Install it for its declared consumer execution at most
    /// once and keep the native signal alive until the native wait completes or
    /// the backend safely abandons it.
    #[doc(hidden)]
    pub unsafe fn into_native(self) -> B::NativeGpuSignal {
        self.native
    }
}
pub struct MultiQueuePacket<B: MultiQueueBackend> {
    pub work: SubmitPacket<B>,
    pub signals: Vec<GpuSignalRequest>,
    pub waits: Vec<GpuWait<B>>,
}
pub struct MultiQueueReceipt<B: MultiQueueBackend> {
    pub work: SubmissionReceipt<B>,
    pub signals: Vec<GpuSignalPoint<B>>,
}
/// Diagnostic unaccepted work after partial acceptance. There is deliberately
/// no resume method: the renderer must rebuild a valid signal path or await
/// the accepted prefix itself before submitting dependent work again.
pub struct MultiQueueTail<B: MultiQueueBackend> {
    pub plan: SubmissionPlan<B>,
    pub pending: Vec<BoundStep<B>>,
    pub remaining_signals: Vec<GpuSignalRequest>,
    pub remaining_waits: Vec<GpuWait<B>>,
    pub completion: CompletionRequest,
}
pub enum MultiQueueSubmitOutcome<B: MultiQueueBackend> {
    Accepted(MultiQueueReceipt<B>),
    Rejected {
        error: Error,
        packet: MultiQueuePacket<B>,
    },
    PartiallyAccepted {
        error: Error,
        accepted: MultiQueueReceipt<B>,
        unaccepted: MultiQueueTail<B>,
    },
    Indeterminate {
        error: Error,
        uncertain: NativeUncertainty<B>,
    },
}
pub struct MultiQueueDomain<'a, B: MultiQueueBackend> {
    device: &'a Device<B>,
}
impl<B: MultiQueueBackend> Device<B> {
    pub fn multi_queue(&self) -> Option<MultiQueueDomain<'_, B>> {
        self.caps
            .multi_queue
            .then_some(MultiQueueDomain { device: self })
    }
}
impl<B: MultiQueueBackend> MultiQueueDomain<'_, B> {
    /// # Safety
    /// Same resource lifetime and scheduling obligations as Queue::submit. The
    /// caller must route each consumed signal to its intended consumer execution
    /// at most once, retaining the true producer/consumer relationship even
    /// when the same plan is submitted more than once. StepId equality alone
    /// cannot prove that relationship; Core does not track execution instances.
    pub unsafe fn submit(&self, packet: MultiQueuePacket<B>) -> MultiQueueSubmitOutcome<B> {
        if let Err(error) = packet.work.validate(self.device.id) {
            return MultiQueueSubmitOutcome::Rejected { error, packet };
        }
        let plan = &packet.work.plan;
        let validate = || -> Result<()> {
            for signal in &packet.signals {
                if signal.producer.plan.device != self.device.id
                    || signal.consumer.plan.device != self.device.id
                    || signal.consumer_lane.device != self.device.id
                {
                    return Err(Error::WrongDevice);
                }
                if signal.producer.plan != plan.id {
                    return Err(Error::InvalidState);
                }
                let producer = plan
                    .steps
                    .get(signal.producer.index as usize)
                    .ok_or(Error::InvalidState)?;
                if producer.lane.device != self.device.id {
                    return Err(Error::WrongDevice);
                }
                if self.device.wait_route(producer.lane, signal.consumer_lane)
                    == WaitRoute::Unsupported
                {
                    return Err(Error::WrongLane);
                }
                // A signal may target a later submission with a different
                // plan, whose slots are unavailable here. If it names this
                // packet's plan, however, validate the concrete consumer
                // slot and its declared lane before the backend sees it.
                if signal.consumer.plan == plan.id {
                    let consumer = plan
                        .steps
                        .get(signal.consumer.index as usize)
                        .ok_or(Error::InvalidState)?;
                    if consumer.lane != signal.consumer_lane {
                        return Err(Error::WrongLane);
                    }
                }
            }
            for wait in &packet.waits {
                let signal = &wait.signal;
                if signal.device != self.device.id
                    || signal.producer.plan.device != self.device.id
                    || signal.consumer.plan.device != self.device.id
                    || signal.producer_lane.device != self.device.id
                    || signal.consumer_lane.device != self.device.id
                {
                    return Err(Error::WrongDevice);
                }
                if signal.consumer.plan != plan.id {
                    return Err(Error::InvalidState);
                }
                let consumer = plan
                    .steps
                    .get(signal.consumer.index as usize)
                    .ok_or(Error::InvalidState)?;
                if consumer.lane != signal.consumer_lane
                    || self
                        .device
                        .wait_route(signal.producer_lane, signal.consumer_lane)
                        == WaitRoute::Unsupported
                {
                    return Err(Error::WrongLane);
                }
            }
            Ok(())
        };
        if let Err(error) = validate() {
            return MultiQueueSubmitOutcome::Rejected { error, packet };
        }
        // SAFETY: packet validation above established this queue's device ownership.
        unsafe { B::submit_multi(&self.device.native, packet) }
    }
}
