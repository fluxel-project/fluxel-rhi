//! Statically typed RHI contract component.
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

/// Sparse is plan-level work, not a command-buffer operation. A backend may
/// implement it only where its actual device has the capability.
pub trait SparseBackend: Backend {
    fn create_sparse_allocation(
        device: &Self::NativeDevice,
        size: u64,
    ) -> Result<Self::NativeSparseAllocation>;
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
    pub fn allocation(&self, size: u64) -> Result<SparseAllocation<B>> {
        Ok(unsafe {
            SparseAllocation::from_native(
                self.device.id,
                B::create_sparse_allocation(&self.device.native, size)?,
            )
        })
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
/// No Clone: one producer signal is declared for one exact consumer.
pub struct GpuSignalPoint<B: MultiQueueBackend> {
    device: DeviceId,
    pub producer: StepId,
    pub producer_lane: LaneId,
    pub consumer: StepId,
    pub consumer_lane: LaneId,
    native: B::NativeGpuSignal,
}
impl<B: MultiQueueBackend> GpuSignalPoint<B> {
    /// # Safety
    /// The accepted producer installed a real GPU-waitable signal for exactly
    /// this consumer before returning this point. Native signal storage stays
    /// valid until consumed or safely abandoned.
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
/// Cross-submission GPU synchronization is optional; ordinary Queue::submit
/// has no signal/wait fields or hidden CPU wait.
pub trait MultiQueueBackend: DeferredBackend {
    type NativeGpuSignal;
    /// # Safety
    /// Signals are installed before producer acceptance and bind one consumer.
    unsafe fn submit_multi(
        device: &Self::NativeDevice,
        packet: MultiQueuePacket<Self>,
    ) -> MultiQueueSubmitOutcome<Self>;
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
    /// Same resource lifetime and scheduling obligations as Queue::submit.
    pub unsafe fn submit(&self, packet: MultiQueuePacket<B>) -> MultiQueueSubmitOutcome<B> {
        let plan = &packet.work.plan;
        let valid = plan.id.device == self.device.id
            && packet.signals.iter().all(|signal| {
                signal.producer.plan == plan.id
                    && signal.consumer.plan.device == self.device.id
                    && signal.consumer_lane.device == self.device.id
                    && plan
                        .steps
                        .get(signal.producer.index as usize)
                        .is_some_and(|producer| {
                            self.device.wait_route(producer.lane, signal.consumer_lane)
                                != WaitRoute::Unsupported
                        })
            })
            && packet.waits.iter().all(|wait| {
                let signal = &wait.signal;
                signal.device == self.device.id
                    && signal.producer.plan.device == self.device.id
                    && signal.producer_lane.device == self.device.id
                    && signal.consumer.plan == plan.id
                    && plan
                        .steps
                        .get(signal.consumer.index as usize)
                        .is_some_and(|step| {
                            step.lane == signal.consumer_lane
                                && self
                                    .device
                                    .wait_route(signal.producer_lane, signal.consumer_lane)
                                    != WaitRoute::Unsupported
                        })
            });
        if !valid {
            return MultiQueueSubmitOutcome::Rejected {
                error: Error::WrongDevice,
                packet,
            };
        }
        unsafe { B::submit_multi(&self.device.native, packet) }
    }
}
