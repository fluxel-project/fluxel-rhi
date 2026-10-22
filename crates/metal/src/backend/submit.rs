//! One-shot Metal queue submission and host completion evidence.
//!
//! Core supplies the complete order and owns graph policy. This module accepts
//! only the implemented single graphics lane, Commands, and one optional
//! Present step, committing each native command buffer in Core's order. It does
//! not build a CPU command list or schedule/reorder GPU work. Completion
//! handlers observe Metal's terminal command-buffer result; cancellation only
//! drops the observer.

use super::{Metal, MetalDevice};
use fluxel_rhi_core::api::{
    self, CompletionBackend, CompletionState, DeferredBackend, Error, LaneId, Result, StepKind,
    SubmissionPlan, SubmitOutcome,
};
#[cfg(target_vendor = "apple")]
use fluxel_rhi_core::api::{CompletionCoverage, CompletionPoint, StepId, StepPayload};
use std::{
    future::Future,
    pin::Pin,
    sync::{Arc, Mutex},
    task::{Context, Poll, Waker},
};

#[cfg(target_vendor = "apple")]
type NativeCommandBuffer =
    objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandBuffer>>;

#[cfg(target_vendor = "apple")]
#[derive(Default)]
pub(super) struct SubmissionTracker {
    next: u64,
    pending: std::collections::HashMap<u64, Completion>,
    failure: Option<Error>,
}

/// Register before commit while holding queue_gate. The tracker retains the
/// buffer until its completion callback has recorded its terminal result.
#[cfg(target_vendor = "apple")]
pub(super) fn track_command(
    device: &MetalDevice,
    command: &NativeCommandBuffer,
    query: Option<(Arc<super::query::QueryCommit>, u32)>,
) -> Completion {
    use objc2_metal::{MTLCommandBuffer as _, MTLCommandBufferStatus};
    let mut tracker = device
        ._state
        .submission_tracker
        .lock()
        .unwrap_or_else(|p| p.into_inner());
    let id = tracker.next;
    tracker.next = tracker
        .next
        .checked_add(1)
        .expect("Metal submission tracker exhausted");
    let completion = Completion::new();
    tracker.pending.insert(id, completion.clone());
    let state = Arc::downgrade(&device._state);
    let tracked = completion.clone();
    let callback = block2::RcBlock::new(
        move |pointer: std::ptr::NonNull<
            objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        >| {
            let Some(state) = state.upgrade() else { return };
            // SAFETY: Metal supplies a live command buffer to its terminal callback.
            let command = unsafe { pointer.as_ref() };
            if command.status() != MTLCommandBufferStatus::Completed {
                if let Some((commit, lane)) = &query {
                    let _gate = state.queue_gate.lock().unwrap_or_else(|p| p.into_inner());
                    super::query::invalidate_ops(commit, *lane);
                }
            }
            let mut tracker = state
                .submission_tracker
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            let outcome = if command.status() != MTLCommandBufferStatus::Completed {
                let error = if command.error().is_some_and(|e| {
                    e.code() == objc2_metal::MTLCommandBufferError::DeviceRemoved.0 as isize
                }) {
                    Error::DeviceLost
                } else {
                    Error::Backend("Metal command buffer execution failed".into())
                };
                if tracker.failure.is_none() || matches!(error, Error::DeviceLost) {
                    tracker.failure = Some(error.clone());
                }
                Err(error)
            } else {
                Ok(())
            };
            tracker.pending.remove(&id);
            drop(tracker);
            completion.publish(outcome);
        },
    );
    // SAFETY: Metal copies the block; the weak device breaks retention cycles.
    unsafe { command.addCompletedHandler(block2::RcBlock::as_ptr(&callback).cast()) };
    tracked
}
#[cfg(target_vendor = "apple")]
type CheckpointMarker = (
    StepId,
    NativeCommandBuffer,
    Completion,
    Vec<(LaneId, StepId)>,
);

#[derive(Clone)]
pub struct Completion(Arc<CompletionInner>);

struct CompletionInner {
    result: Mutex<Option<Result<()>>>,
    wakers: Mutex<Vec<Waker>>,
}

impl Completion {
    pub(super) fn new() -> Self {
        Self(Arc::new(CompletionInner {
            result: Mutex::new(None),
            wakers: Mutex::new(Vec::new()),
        }))
    }

    pub(super) fn result(&self) -> Option<Result<()>> {
        self.0
            .result
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    pub(super) fn watch(&self, waker: &Waker) -> Option<Result<()>> {
        let mut wakers = self.0.wakers.lock().unwrap_or_else(|p| p.into_inner());
        let result = self.result();
        if result.is_none() && !wakers.iter().any(|w| w.will_wake(waker)) {
            wakers.push(waker.clone());
        }
        result
    }

    pub(super) fn publish(&self, result: Result<()>) {
        let mut slot = self
            .0
            .result
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if slot.is_some() {
            return;
        }
        *slot = Some(result);
        drop(slot);

        let wakers = std::mem::take(
            &mut *self
                .0
                .wakers
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()),
        );
        for waker in wakers {
            // A user-provided waker must not unwind through Metal's Objective-C
            // callback boundary.
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| waker.wake()));
        }
    }

    #[cfg(target_vendor = "apple")]
    pub(super) fn install_group(
        commands: &[NativeCommandBuffer],
        initial_failure: Option<Error>,
    ) -> Self {
        use objc2_metal::{MTLCommandBuffer as _, MTLCommandBufferStatus};
        let result = Self::new();
        let remaining = Arc::new(std::sync::atomic::AtomicUsize::new(commands.len()));
        let failure = Arc::new(Mutex::new(initial_failure));
        for command in commands {
            let result = result.clone();
            let remaining = remaining.clone();
            let failure = failure.clone();
            let callback = block2::RcBlock::new(
                move |pointer: std::ptr::NonNull<
                    objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
                >| {
                    // SAFETY: Metal invokes the callback while its command buffer is live.
                    let native = unsafe { pointer.as_ref() };
                    if native.status() != MTLCommandBufferStatus::Completed {
                        let error = if native.error().is_some_and(|e| {
                            e.code() == objc2_metal::MTLCommandBufferError::DeviceRemoved.0 as isize
                        }) {
                            Error::DeviceLost
                        } else {
                            Error::Backend("Metal command buffer execution failed".into())
                        };
                        let mut recorded = failure.lock().unwrap_or_else(|p| p.into_inner());
                        if recorded.is_none() || matches!(error, Error::DeviceLost) {
                            *recorded = Some(error);
                        }
                    }
                    if remaining.fetch_sub(1, std::sync::atomic::Ordering::AcqRel) == 1 {
                        result.publish(
                            failure
                                .lock()
                                .unwrap_or_else(|p| p.into_inner())
                                .clone()
                                .map_or(Ok(()), Err),
                        );
                    }
                },
            );
            // SAFETY: Metal copies this owning block before returning.
            unsafe { command.addCompletedHandler(block2::RcBlock::as_ptr(&callback).cast()) };
        }
        result
    }
}

struct CompletionJoin {
    children: Vec<Completion>,
    output: Completion,
    #[cfg(any(target_vendor = "apple", test))]
    ray_builds: Vec<super::ray::BuildCompletion>,
}
impl CompletionJoin {
    fn update(self: &Arc<Self>) {
        let waker = Waker::from(self.clone());
        let mut failure = None;
        let mut pending = false;
        for child in &self.children {
            let mut wakers = child.0.wakers.lock().unwrap_or_else(|p| p.into_inner());
            match child.result() {
                Some(Err(error)) => {
                    if failure.is_none() || matches!(error, Error::DeviceLost) {
                        failure = Some(error);
                    }
                }
                Some(Ok(())) => {}
                None => {
                    pending = true;
                    if !wakers.iter().any(|w| w.will_wake(&waker)) {
                        wakers.push(waker.clone());
                    }
                }
            }
        }
        if !pending {
            #[cfg(any(target_vendor = "apple", test))]
            for build in &self.ray_builds {
                build.complete(failure.is_none());
            }
            self.output.publish(failure.map_or(Ok(()), Err));
        }
    }
}
impl std::task::Wake for CompletionJoin {
    fn wake(self: Arc<Self>) {
        self.update();
    }
    fn wake_by_ref(self: &Arc<Self>) {
        self.update();
    }
}
pub(super) fn join_into(output: &Completion, children: Vec<Completion>) {
    Arc::new(CompletionJoin {
        children,
        output: output.clone(),
        #[cfg(any(target_vendor = "apple", test))]
        ray_builds: Vec::new(),
    })
    .update();
}

pub struct CompletionFuture(Vec<Completion>);

pub struct MetalRecordedWork {
    #[cfg(target_vendor = "apple")]
    pub(super) wait_slots:
        Vec<objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLSharedEvent>>>,
    #[cfg(target_vendor = "apple")]
    pub(super) command:
        objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandBuffer>>,
    #[cfg(target_vendor = "apple")]
    pub(super) prior_commands: Vec<NativeCommandBuffer>,
    #[cfg(target_vendor = "apple")]
    prepared_draws: Vec<super::encoder::indirect_count::PreparedDraws>,
    #[cfg(target_vendor = "apple")]
    pub(super) lane: u32,
    #[cfg(target_vendor = "apple")]
    pub(super) query_ops: Vec<super::query::QueryOp>,
    #[cfg(target_vendor = "apple")]
    accelerations: Vec<super::ray::MetalAcceleration>,
    #[cfg(target_vendor = "apple")]
    pub(super) ray_builds: Vec<Arc<super::ray::BuildReservation>>,
    #[cfg(target_vendor = "apple")]
    indirect_arguments:
        Vec<objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>>,
}

impl MetalRecordedWork {
    #[cfg(target_vendor = "apple")]
    pub(super) fn submit_ray_builds(&self) {
        if self.ray_builds.is_empty() {
            return;
        }
        for build in &self.ray_builds {
            build.submit();
        }
    }

    #[cfg(target_vendor = "apple")]
    pub(super) fn guard_ray_completion(&self, covered: &mut Vec<Completion>) {
        if self.ray_builds.is_empty() {
            return;
        }
        let output = Completion::new();
        Arc::new(CompletionJoin {
            children: std::mem::take(covered),
            output: output.clone(),
            ray_builds: self
                .ray_builds
                .iter()
                .map(|build| build.completion())
                .collect(),
        })
        .update();
        covered.push(output);
    }
    #[cfg(target_vendor = "apple")]
    pub(super) fn retain_indirect_arguments_until_completed(&self) {
        use objc2_metal::MTLCommandBuffer as _;
        if self.indirect_arguments.is_empty()
            && self.accelerations.is_empty()
            && self.prepared_draws.is_empty()
        {
            return;
        }
        let retained = self.indirect_arguments.clone();
        let accelerations = self.accelerations.clone();
        let prepared = self.prepared_draws.clone();
        let callback = block2::RcBlock::new(
            move |_: std::ptr::NonNull<
                objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
            >| {
                let _ = retained.len();
                let _ = accelerations.len();
                let _ = prepared.len();
            },
        );
        // SAFETY: the owning block captures strong references to every indirect
        // argument buffer and is copied by Metal. It is installed before commit
        // and released only after the command buffer reaches terminal status.
        unsafe {
            self.command
                .addCompletedHandler(block2::RcBlock::as_ptr(&callback).cast())
        };
    }
}

impl Future for CompletionFuture {
    type Output = Result<()>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut pending = false;
        let mut failure = None;
        for completion in &self.get_mut().0 {
            let mut wakers = completion
                .0
                .wakers
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if let Some(result) = completion.result() {
                if let Err(error) = result {
                    if failure.is_none() || matches!(error, Error::DeviceLost) {
                        failure = Some(error);
                    }
                }
            } else {
                pending = true;
                if !wakers.iter().any(|waker| waker.will_wake(cx.waker())) {
                    wakers.push(cx.waker().clone());
                }
            }
        }
        if pending {
            Poll::Pending
        } else {
            Poll::Ready(failure.map_or(Ok(()), Err))
        }
    }
}

impl CompletionBackend for Metal {
    type WaitFuture<'a> = CompletionFuture;

    fn completion_status(_: &MetalDevice, point: &Completion) -> CompletionState {
        match point.result() {
            Some(Ok(())) => CompletionState::Complete,
            Some(Err(error)) => CompletionState::Failed(error),
            None => CompletionState::Pending,
        }
    }

    fn wait_completion<'a>(_: &'a MetalDevice, point: &'a Completion) -> Self::WaitFuture<'a> {
        CompletionFuture(vec![point.clone()])
    }

    fn wait_idle(device: &MetalDevice) -> Self::WaitFuture<'_> {
        #[cfg(target_vendor = "apple")]
        {
            use objc2_metal::MTLCommandQueue as _;
            let completion = (|| {
                let command = device
                    ._state
                    ._queue
                    .commandBuffer()
                    .ok_or(Error::OutOfMemory)?;
                let compute = device
                    ._state
                    .compute_queue
                    .commandBuffer()
                    .ok_or(Error::OutOfMemory)?;
                let markers = [command, compute];
                let completion = {
                    let _gate = device
                        ._state
                        .queue_gate
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    // Snapshot pending command buffers and sticky failures under
                    // the issue gate. Their callbacks may run in any order.
                    let tracker = device
                        ._state
                        .submission_tracker
                        .lock()
                        .unwrap_or_else(|p| p.into_inner());
                    let mut pending: Vec<_> = tracker.pending.values().cloned().collect();
                    let failure = tracker.failure.clone();
                    drop(tracker);
                    if let Some(sparse) = device
                        ._state
                        .sparse_pending
                        .lock()
                        .unwrap_or_else(|p| p.into_inner())
                        .as_ref()
                    {
                        pending.push(sparse.clone());
                    }
                    let completion = Completion::install_group(&markers, failure);
                    pending.push(completion);
                    use objc2_metal::MTLCommandBuffer as _;
                    for command in &markers {
                        command.commit();
                    }
                    pending
                };
                Ok(completion)
            })();
            CompletionFuture(completion.unwrap_or_else(|error| {
                let failed = Completion::new();
                failed.publish(Err(error));
                vec![failed]
            }))
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = device;
            let failed = Completion::new();
            failed.publish(Err(Error::Unsupported("Metal requires an Apple target")));
            CompletionFuture(vec![failed])
        }
    }
}

impl DeferredBackend for Metal {
    type NativeRecordedWork = MetalRecordedWork;

    fn finish_encoder(
        encoder: super::MetalCommandEncoder,
        lane: LaneId,
    ) -> Result<Self::NativeRecordedWork> {
        if lane.ordinal > 1 {
            return Err(Error::InvalidState);
        }
        #[cfg(target_vendor = "apple")]
        {
            if encoder.render.is_some() || encoder.compute.is_some() {
                return Err(Error::InvalidState);
            }
            if encoder.lane != lane.ordinal {
                return Err(Error::WrongLane);
            }
            // The queue's commandBuffer() factory retains native objects used
            // in encoded commands. The backend does not duplicate every bind
            // in a Rust keep-alive vector; Core still owns public resource
            // lifetime decisions through actual GPU completion.
            Ok(MetalRecordedWork {
                command: encoder.command,
                prior_commands: encoder.prior_commands,
                prepared_draws: encoder.prepared_draws,
                wait_slots: encoder.wait_slots,
                lane: encoder.lane,
                query_ops: encoder.query_ops,
                accelerations: encoder.accelerations,
                ray_builds: encoder.ray_builds,
                indirect_arguments: encoder.indirect_arguments,
            })
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = encoder;
            Err(Error::Unsupported("Metal requires an Apple target"))
        }
    }

    // SAFETY: Core's submit contract requires the caller to keep every resource
    // and synchronization object reachable from `packet` alive until the
    // returned completion frontier is truly complete. This implementation
    // preserves the packet's Step order and reports only command buffers that
    // were committed to Metal; partial acceptance carries a completion marker.
    unsafe fn submit(device: &MetalDevice, packet: api::SubmitPacket<Self>) -> SubmitOutcome<Self> {
        if packet
            .plan
            .steps()
            .iter()
            .any(|step| step.kind == StepKind::SparseBind)
        {
            // SAFETY: the caller's packet and lifetime obligations are unchanged;
            // the multi-queue lowering preflights every mapping before handoff.
            let outcome = unsafe {
                <Metal as api::MultiQueueBackend>::submit_multi(
                    device,
                    api::MultiQueuePacket {
                        work: packet,
                        signals: Vec::new(),
                        waits: Vec::new(),
                    },
                )
            };
            return match outcome {
                api::MultiQueueSubmitOutcome::Accepted(receipt) => {
                    SubmitOutcome::Accepted(receipt.work)
                }
                api::MultiQueueSubmitOutcome::Rejected { error, packet } => {
                    SubmitOutcome::Rejected {
                        error,
                        work: api::RetryableSubmit::Fresh(packet.work),
                    }
                }
                api::MultiQueueSubmitOutcome::Indeterminate { error, uncertain } => {
                    SubmitOutcome::Indeterminate { error, uncertain }
                }
                // This backend's multi-queue path preallocates before handoff and
                // has no fallible native issue after accepting its first step.
                api::MultiQueueSubmitOutcome::PartiallyAccepted { .. } => {
                    unreachable!("Metal multi-queue never returns a partial prefix")
                }
            };
        }
        // Validate every unsupported shape before consuming RecordedWork. This
        // preserves Core's clean-rejection ownership contract.
        if !supports_command_plan(&packet.plan) {
            return SubmitOutcome::Rejected {
                error: Error::Unsupported("Metal supports Commands on one graphics lane"),
                work: api::RetryableSubmit::Fresh(packet),
            };
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = device;
            SubmitOutcome::Rejected {
                error: Error::Unsupported("Metal requires an Apple target"),
                work: api::RetryableSubmit::Fresh(packet),
            }
        }
        #[cfg(target_vendor = "apple")]
        {
            use objc2_metal::{MTLCommandBuffer as _, MTLCommandQueue as _};
            let _gate = device
                ._state
                .queue_gate
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            if device
                ._state
                .sparse_pending
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .as_ref()
                .is_some_and(|pending| pending.result().is_none())
            {
                return SubmitOutcome::Rejected {
                    error: Error::InvalidState,
                    work: api::RetryableSubmit::Fresh(packet),
                };
            }
            let ops = packet
                .steps
                .iter()
                .flat_map(|step| match &step.payload {
                    StepPayload::Commands(records) => records.as_slice(),
                    _ => &[],
                })
                .flat_map(|record| {
                    // SAFETY: Core validates the backend provenance of each recording.
                    let work = unsafe { record.native_ref() };
                    work.query_ops.iter().map(move |op| (work.lane, op))
                });
            if let Err(error) = super::query::validate_ops(ops) {
                return SubmitOutcome::Rejected {
                    error,
                    work: api::RetryableSubmit::Fresh(packet),
                };
            }
            let present_steps = packet
                .steps
                .iter()
                .filter_map(|step| {
                    matches!(step.payload, StepPayload::Present(_)).then_some(step.step)
                })
                .collect::<Vec<_>>();
            let mut present_commands = Vec::with_capacity(present_steps.len());
            for bound in &packet.steps {
                if let StepPayload::Present(binding) = &bound.payload {
                    let Some(command) = device._state._queue.commandBuffer() else {
                        return SubmitOutcome::Rejected {
                            error: Error::OutOfMemory,
                            work: api::RetryableSubmit::Fresh(packet),
                        };
                    };
                    // SAFETY: Core validated this current exclusive lease for
                    // the plan's device; the drawable stays owned by binding.
                    let image = unsafe { binding.image().native_ref() };
                    let drawable =
                        objc2::runtime::ProtocolObject::<dyn objc2_metal::MTLDrawable>::from_ref(
                            &*image.drawable,
                        );
                    command.presentDrawable(drawable);
                    present_commands.push((bound.step, command));
                }
            }
            let api::SubmitPacket {
                plan,
                mut steps,
                completion: request,
            } = packet;
            let requested_steps = match &request {
                api::CompletionRequest::Steps(steps) => steps.to_vec(),
                _ => Vec::new(),
            };

            // Allocate every checkpoint marker and install every callback
            // before consuming the packet or committing native work. A setup
            // failure therefore remains a clean rejection with the fresh plan.
            let mut markers: Vec<CheckpointMarker> = Vec::new();
            let marker_targets = match &request {
                api::CompletionRequest::Steps(targets) => {
                    let mut unique = Vec::new();
                    for target in targets.iter().copied() {
                        if !unique.contains(&target) {
                            unique.push(target);
                        }
                    }
                    unique
                }
                api::CompletionRequest::End if steps.is_empty() => {
                    vec![StepId {
                        plan: plan.id(),
                        index: 0,
                    }]
                }
                api::CompletionRequest::End => vec![plan.step(steps.len() - 1)],
                api::CompletionRequest::None => Vec::new(),
            };
            for target in marker_targets {
                let Some(command) = device._state._queue.commandBuffer() else {
                    return SubmitOutcome::Rejected {
                        error: Error::OutOfMemory,
                        work: api::RetryableSubmit::Fresh(api::SubmitPacket {
                            plan,
                            steps,
                            completion: request,
                        }),
                    };
                };
                let completion = Completion::new();
                markers.push((target, command, completion, Vec::new()));
            }

            // All setup that can return an error is complete. Limit this path
            // to one Present step so lease handoff cannot partially succeed.
            for bound in &mut steps {
                if let StepPayload::Present(binding) = &mut bound.payload {
                    // SAFETY: this is the single native present handoff after
                    // command buffers and completion handlers are prepared.
                    if let Err(error) = unsafe { binding.try_handoff() } {
                        return SubmitOutcome::Rejected {
                            error,
                            work: api::RetryableSubmit::Fresh(api::SubmitPacket {
                                plan,
                                steps,
                                completion: request,
                            }),
                        };
                    }
                }
            }

            let mut native_steps = Vec::with_capacity(steps.len());
            for bound in steps {
                let work = match bound.payload {
                    StepPayload::Commands(work) => work,
                    StepPayload::Present(_) => Vec::new(),
                    StepPayload::SparseBind(_) => unreachable!("prevalidated Metal plan"),
                };
                let native = work
                    .into_iter()
                    .map(|record| {
                        // SAFETY: Core validated device/lane provenance and this
                        // one-shot packet consumes each native recording once.
                        unsafe { record.into_native() }
                    })
                    .collect::<Vec<_>>();
                native_steps.push((bound.step, native));
            }

            let mut accepted = Vec::with_capacity(native_steps.len());
            let mut frontiers: Vec<(LaneId, StepId)> = Vec::new();
            let mut prefix = Vec::new();
            if let Some(previous) = device
                ._state
                .lane_frontier
                .lock()
                .unwrap_or_else(|p| p.into_inner())[0]
                .clone()
            {
                prefix.push(previous);
            }
            for (step, commands) in &native_steps {
                let lane = plan.steps()[step.index as usize].lane;
                for work in commands {
                    use objc2_metal::MTLSharedEvent as _;
                    for slot in &work.wait_slots {
                        slot.setSignaledValue(1);
                    }
                    let query_commit =
                        Arc::new(super::query::commit_ops(&work.query_ops, work.lane));
                    work.retain_indirect_arguments_until_completed();
                    work.submit_ray_builds();
                    for command in &work.prior_commands {
                        prefix.push(track_command(
                            device,
                            command,
                            Some((query_commit.clone(), work.lane)),
                        ));
                        command.commit();
                    }
                    prefix.push(track_command(
                        device,
                        &work.command,
                        Some((query_commit, work.lane)),
                    ));
                    work.command.commit();
                    work.guard_ray_completion(&mut prefix);
                }
                for (present_step, command) in &present_commands {
                    if present_step == step {
                        prefix.push(track_command(device, command, None));
                        command.commit();
                    }
                }
                accepted.push(*step);
                record_frontier(&mut frontiers, lane, *step);
                let lane_completion = Completion::new();
                join_into(&lane_completion, prefix.clone());
                device
                    ._state
                    .lane_frontier
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())[0] = Some(lane_completion);

                for (target, marker, completion, coverage) in markers
                    .iter_mut()
                    .filter(|(target, _, _, _)| target == step)
                {
                    *coverage = frontiers.clone();
                    let marker_completion = track_command(device, marker, None);
                    let mut covered = prefix.clone();
                    covered.push(marker_completion);
                    join_into(completion, covered);
                    marker.commit();
                    let _ = target;
                }
            }
            if native_steps.is_empty() {
                for (_, marker, completion, _) in &markers {
                    let tracked = track_command(device, marker, None);
                    join_into(completion, vec![tracked]);
                    marker.commit();
                }
            }

            // A marker after the final plan step is naturally placed after all
            // command buffers. For requested checkpoints each marker was
            // committed immediately after its matching Commands step.
            let mut checkpoints = Vec::new();
            for requested in requested_steps {
                if let Some((_, _, completion, coverage)) = markers
                    .iter()
                    .find(|(target, _, _, _)| *target == requested)
                {
                    checkpoints.push(api::Checkpoint {
                        // SAFETY: this callback belongs to the queue marker
                        // issued at the requested plan boundary.
                        point: unsafe {
                            CompletionPoint::from_native(plan.id().device, completion.clone())
                        },
                        covers: CompletionCoverage {
                            lane_frontiers: coverage.clone(),
                        },
                    });
                }
            }
            if matches!(request, api::CompletionRequest::End) {
                if let Some((_, _, completion, coverage)) = markers.last() {
                    checkpoints.push(api::Checkpoint {
                        // SAFETY: End marker follows the full accepted queue prefix.
                        point: unsafe {
                            CompletionPoint::from_native(plan.id().device, completion.clone())
                        },
                        covers: CompletionCoverage {
                            lane_frontiers: coverage.clone(),
                        },
                    });
                }
            }
            SubmitOutcome::Accepted(api::SubmissionReceipt {
                device: plan.id().device,
                index: None,
                accepted,
                checkpoints,
                presents: present_steps
                    .into_iter()
                    .map(|step| api::PresentReceipt {
                        step: Some(step),
                        accepted: true,
                    })
                    .collect(),
            })
        }
    }

    // SAFETY: resuming requires proof that the accepted prefix completed and
    // that referenced resources remain alive. Metal does not implement tails,
    // so it rejects the tail unchanged and performs no native operation.
    unsafe fn resume(_: &MetalDevice, tail: api::DeferredTail<Self>) -> SubmitOutcome<Self> {
        SubmitOutcome::Rejected {
            error: Error::Unsupported("Metal submission tails are not implemented"),
            work: api::RetryableSubmit::Tail(tail),
        }
    }
}

fn supports_command_plan(plan: &SubmissionPlan<Metal>) -> bool {
    let mut present_count = 0;
    plan.steps().iter().enumerate().all(|(index, desc)| {
        if desc.kind == StepKind::Present {
            present_count += 1;
        }
        matches!(desc.kind, StepKind::Commands | StepKind::Present)
            && present_count <= 1
            && desc.lane.device == plan.id().device
            && desc.lane.ordinal == 0
            && desc.waits.iter().all(|wait| {
                wait.previous.plan == plan.id()
                    && (wait.previous.index as usize) < index
                    && plan.steps()[wait.previous.index as usize].lane == desc.lane
            })
    })
}

/// Keep the latest accepted step for each lane. A completion marker snapshots
/// this list at its issue boundary, so later steps cannot widen earlier proof.
fn record_frontier(frontiers: &mut Vec<(LaneId, api::StepId)>, lane: LaneId, step: api::StepId) {
    frontiers.retain(|(existing, _)| *existing != lane);
    frontiers.push((lane, step));
}

#[cfg(test)]
mod tests;
