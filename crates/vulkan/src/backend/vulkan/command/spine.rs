//! Transactional Vulkan submission spine.
//!
//! A Fluxel submission has a stronger contract than Vulkan's incremental queue
//! API: returning `Err` proves that no work from the plan was accepted.  Vulkan
//! native encoders finish command buffers before submission, so all fallible
//! lowering has already completed before the first `vkQueueSubmit`.
//!
//! Once that first submit is made, failure is execution-domain state, never the
//! return value of `submit`: a Vulkan implementation may have accepted part of
//! the call before reporting an error, and reporting `Err` would lie to the
//! portable caller.  The spine poisons the accepted serial range and reports
//! `Failed` (or `DeviceLost`) from completion instead.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::task::Waker;

use ash::vk;

use crate::api::command::ResourceUse;
use crate::api::platform::DeviceLossInfo;
use crate::api::presentation::{FrameAttachment, PresentState};
use crate::api::resource::transfer::ReadbackStatus;
use crate::api::submission::backend::{SubmissionOutcome, SubmissionRequest};
use crate::api::submission::plan::PlanBatch;
use crate::api::submission::{CompletionFailure, CompletionState};
use crate::backend::vulkan::failure::VulkanFailure;
use crate::backend::vulkan::ffi;
use crate::backend::vulkan::platform::device::VulkanShared;
use crate::backend::vulkan::resource::{VulkanBuffer, VulkanQuerySet};

use super::transfer::{self, TransferRetention};

#[cfg(any(windows, target_os = "android"))]
use crate::backend::vulkan::presentation::{VulkanFrameAttachment, VulkanPresentSync};
#[cfg(not(any(windows, target_os = "android")))]
#[derive(Clone, Copy)]
struct VulkanPresentSync {
    acquire_wait: vk::Semaphore,
    render_finished: vk::Semaphore,
}

/// One single-queue Vulkan command domain.
pub(in crate::backend::vulkan) struct VulkanCommandSpine {
    inner: Arc<SpineInner>,
    waiter: Option<std::thread::JoinHandle<()>>,
}

/// Queue-local native objects retained by completion waiter threads.
struct SpineInner {
    shared: Arc<VulkanShared>,
    state: Mutex<SpineState>,
    pending_changed: Condvar,
    shutdown: AtomicBool,
}

/// Mutable submission state guarded as one transaction.
struct SpineState {
    /// Last serial that has entered Phase B.  Zero is deliberately never issued.
    issued: u64,
    /// Last serial known complete by an explicit fence query.
    completed: u64,
    /// Fence waiters are independent host threads and may report a later
    /// single-queue fence before an earlier waiter gets scheduled. Keep those
    /// observations here; the public completion frontier advances only across
    /// a contiguous prefix, after each batch's readbacks have been published.
    finished: BTreeSet<u64>,
    /// First serial whose queue outcome cannot safely be observed.  All later
    /// serials are behind it on the same queue and are terminal for the same
    /// reason.
    poison: Option<(u64, CompletionFailure)>,
    /// One fence per accepted batch.  Per-batch fences preserve v13's finer
    /// completion without exposing a Vulkan fence as a public token.
    pending: BTreeMap<u64, PendingBatch>,
}

struct PendingBatch {
    fence: vk::Fence,
    /// A directly encoded recorder owns its own pool. Keeping these finished
    /// buffers here keeps both that pool and every native list alive through the
    /// fence, without making the submission spine responsible for them.
    _native_buffers: Vec<super::native::FinishedBuffer>,
    /// Portable buffer handles and backend staging allocations referenced by
    /// the accepted command buffer.  Native Vulkan handles alone do not extend
    /// Fluxel resource lifetime, so releasing this only after the fence is
    /// terminal is part of the v13 ownership contract.
    retention: TransferRetention,
}

#[derive(Default)]
struct BatchPresentation {
    waits: Vec<vk::Semaphore>,
    signals: Vec<vk::Semaphore>,
    presents: Vec<usize>,
}

/// A fence wait is deliberately bounded even though the normal completion path
/// has no deadline.  `VulkanCommandSpine::drop` must be able to join its sole
/// waiter before it destroys pending native objects, and Vulkan has no operation that
/// cancels an in-progress `vkWaitForFences`.  A finite wait is therefore the
/// shutdown observation point; it is *not* a completion timeout.
const FENCE_WAITER_POLL_NS: u64 = 50_000_000;

impl VulkanCommandSpine {
    pub(in crate::backend::vulkan) fn new(
        shared: Arc<VulkanShared>,
    ) -> Result<Self, VulkanFailure> {
        let inner = Arc::new(SpineInner {
            shared: Arc::clone(&shared),
            pending_changed: Condvar::new(),
            shutdown: AtomicBool::new(false),
            state: Mutex::new(SpineState {
                issued: 0,
                completed: 0,
                finished: BTreeSet::new(),
                poison: None,
                pending: BTreeMap::new(),
            }),
        });
        let worker = Arc::clone(&inner);
        let waiter = std::thread::Builder::new()
            .name("fluxel-vulkan-fence".into())
            .spawn(move || fence_wait_loop(worker))
            .map_err(|_| {
                VulkanFailure::Native(ffi::NativeError::new(
                    vk::Result::ERROR_OUT_OF_HOST_MEMORY,
                    "VulkanCommandSpine::spawn fence waiter",
                ))
            })?;
        Ok(Self {
            inner,
            waiter: Some(waiter),
        })
    }

    fn lock(&self) -> MutexGuard<'_, SpineState> {
        self.inner
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    pub(in crate::backend::vulkan) fn poll(&self) {
        // Phase B installs a blocking waiter for every fence. Avoid querying a
        // fence here: host access to a fence is externally synchronized, and a
        // poll racing that waiter would violate Vulkan's synchronization rule.
    }

    /// The only blocking operation on this spine. Holding the queue-domain lock
    /// satisfies Vulkan's external synchronization rule for queue idle against
    /// concurrent submission on the same logical device.
    pub(in crate::backend::vulkan) fn wait_idle(&self) -> Result<(), VulkanFailure> {
        let issued = self.lock().issued;
        if issued == 0 {
            return Ok(());
        }
        {
            let _queue = self.inner.shared.queue_guard();
            unsafe { self.inner.shared.device.device_wait_idle() }.map_err(|result| {
                VulkanFailure::Native(ffi::NativeError::new(
                    result,
                    "VulkanCommandSpine::wait_idle",
                ))
            })?;
        }

        // `vkDeviceWaitIdle` proves the queue is idle, but it does not transfer
        // ownership of the worker's host fence wait or of its `PendingBatch`.
        // Let that one authority observe the signalled fences, publish readbacks,
        // wake futures, and free the native objects. Returning before this loop
        // would make `wait_idle().await` falsely imply a still-Pending completion
        // or readback. The bounded wait also observes a loss discovered by another
        // native entry point even if it did not notify this spine's condition
        // variable.
        let mut state = self.lock();
        while state.completed < issued {
            if self.inner.shared.loss_info().is_some() {
                return Err(VulkanFailure::Native(ffi::NativeError::new(
                    vk::Result::ERROR_DEVICE_LOST,
                    "VulkanCommandSpine::wait_idle after device loss",
                )));
            }
            let (next, _) = self
                .inner
                .pending_changed
                .wait_timeout(state, std::time::Duration::from_nanos(FENCE_WAITER_POLL_NS))
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            state = next;
        }
        Ok(())
    }

    /// Lowers one complete plan under the v13 two-phase acceptance rule.
    ///
    /// Buffer copy/upload/readback lowerings are completed entirely in Phase A.
    /// Any unsupported payload, allocation failure, map failure, or command
    /// recording failure therefore returns before the first `vkQueueSubmit`.
    pub(in crate::backend::vulkan) fn submit(
        &self,
        request: &SubmissionRequest<'_>,
    ) -> Result<SubmissionOutcome, VulkanFailure> {
        let state = self.lock();
        if self.inner.shared.loss_info().is_some() {
            return Err(VulkanFailure::Native(ffi::NativeError::new(
                vk::Result::ERROR_DEVICE_LOST,
                "VulkanCommandSpine::submit after device loss",
            )));
        }

        // An empty plan is a legal portable no-op with its own plan identity.
        // Keep native command-buffer allocation and queue submission entirely
        // out of this path; serial zero is the completed identity frontier of a
        // fresh device, and a later empty plan observes the current frontier.
        if request.batches.is_empty() {
            return Ok(SubmissionOutcome {
                completion: state.issued,
                points: Vec::new(),
            });
        }

        self.submit_native_locked(request, state)
    }

    fn submit_native_locked(
        &self,
        request: &SubmissionRequest<'_>,
        mut state: MutexGuard<'_, SpineState>,
    ) -> Result<SubmissionOutcome, VulkanFailure> {
        let presentation = prepare_presentations(request)?;
        let first_serial = state
            .issued
            .checked_add(1)
            .ok_or(VulkanFailure::Unsupported {
                what: "another Vulkan submission",
                why: "the backend completion serial space is exhausted",
            })?;
        let last_serial = first_serial
            .checked_add(request.batches.len() as u64)
            .and_then(|value| value.checked_sub(1))
            .ok_or(VulkanFailure::Unsupported {
                what: "a submission with too many batches",
                why: "the backend completion serial space is exhausted",
            })?;
        // Refuse non-Vulkan, already-consumed, and duplicate buffers before
        // allocating fences or handing any work to the queue.
        let mut seen = HashSet::new();
        for batch in request.batches {
            for work in &batch.work {
                let native = work
                    .native()
                    .as_any()
                    .downcast_ref::<super::native::VulkanNativeCommandBuffer>()
                    .ok_or(VulkanFailure::Unsupported {
                        what: "a command buffer from another backend",
                        why: "a Vulkan queue can execute only Vulkan command buffers",
                    })?;
                if !native.is_available() {
                    return Err(VulkanFailure::Unsupported {
                        what: "a command buffer submitted more than once",
                        why: "a Vulkan command buffer may be submitted only once by this RHI backend",
                    });
                }
                if !seen.insert(native as *const super::native::VulkanNativeCommandBuffer) {
                    return Err(VulkanFailure::Unsupported {
                        what: "the same Vulkan command buffer more than once in one submission",
                        why: "a command buffer has one ownership transfer to one queue submission",
                    });
                }
            }
        }
        let mut fences = Vec::with_capacity(request.batches.len());
        for _ in request.batches {
            let info = vk::FenceCreateInfo::default();
            match unsafe { self.inner.shared.device.create_fence(&info, None) } {
                Ok(fence) => fences.push(fence),
                Err(result) => {
                    for fence in fences {
                        unsafe { self.inner.shared.device.destroy_fence(fence, None) };
                    }
                    return Err(VulkanFailure::Native(ffi::NativeError::new(
                        result,
                        "VulkanCommandSpine::create_fence for native command buffers",
                    )));
                }
            }
        }
        let mut recorded = Vec::with_capacity(request.batches.len());
        for batch in request.batches {
            let mut buffers = Vec::with_capacity(batch.work.len());
            let mut retention = TransferRetention::default();
            for work in &batch.work {
                let native = work
                    .native()
                    .as_any()
                    .downcast_ref::<super::native::VulkanNativeCommandBuffer>()
                    .expect("native buffers were preflighted");
                let mut finished = native
                    .take()
                    .expect("preflighted native buffer was consumed once");
                retention.append(std::mem::take(&mut finished.retention));
                buffers.push(finished);
            }
            recorded.push((buffers, retention));
        }
        state.issued = last_serial;
        let mut poisoned = None;
        let mut presented = vec![false; request.presents.len()];
        let mut recorded = recorded.into_iter();
        let mut fences = fences.into_iter();
        for index in 0..request.batches.len() {
            let (buffers, retention) = recorded
                .next()
                .expect("native command buffers were preflighted");
            let fence = fences
                .next()
                .expect("native completion fences were allocated");
            let serial = first_serial + index as u64;
            let command_buffers = buffers
                .iter()
                .map(|buffer| buffer.command_buffer)
                .collect::<Vec<_>>();
            let presentation = &presentation[index];
            let wait_stages = vec![vk::PipelineStageFlags::ALL_COMMANDS; presentation.waits.len()];
            let submit = vk::SubmitInfo::default()
                .command_buffers(&command_buffers)
                .wait_semaphores(&presentation.waits)
                .wait_dst_stage_mask(&wait_stages)
                .signal_semaphores(&presentation.signals);
            let result = unsafe {
                let _queue = self.inner.shared.queue_guard();
                self.inner.shared.device.queue_submit(
                    self.inner.shared.graphics_queue,
                    &[submit],
                    fence,
                )
            };
            if let Err(result) = result {
                let failure = VulkanFailure::Native(ffi::NativeError::new(
                    result,
                    "VulkanCommandSpine::queue_submit native command buffers",
                ));
                state.poison = Some((serial, CompletionFailure::new(failure.message())));
                poisoned = Some(DeviceLossInfo::new(format!(
                    "Vulkan queue submission failed after work may have been accepted: {}",
                    failure.message()
                )));
                // A failing queue submit can still have accepted this batch.
                // Retain its native objects through execution-domain teardown;
                // never feed later batches to a queue whose state is unknown.
                let readbacks = retention.readback_tickets();
                state.pending.insert(
                    serial,
                    PendingBatch {
                        fence,
                        _native_buffers: buffers,
                        retention,
                    },
                );
                self.inner.shared.register_readbacks(&readbacks);
                self.inner.pending_changed.notify_one();
                break;
            }
            mark_batch_buffer_uses(&request.batches[index], serial);
            let readbacks = retention.readback_tickets();
            state.pending.insert(
                serial,
                PendingBatch {
                    fence,
                    _native_buffers: buffers,
                    retention,
                },
            );
            self.inner.shared.register_readbacks(&readbacks);
            self.inner.pending_changed.notify_one();
            for &present_index in &presentation.presents {
                let present = &request.presents[present_index];
                present.attachment.present(present.receipt);
                presented[present_index] = true;
            }
            // Presentation can be the first native operation that discovers
            // loss. Later command buffers were never submitted and their
            // frame receipts are terminated below.
            if self.inner.shared.loss_info().is_some() {
                break;
            }
        }
        let points = request
            .batches
            .iter()
            .enumerate()
            .map(|(index, batch)| (batch.point, first_serial + index as u64))
            .collect();
        drop(state);
        if let Some(info) = poisoned {
            self.inner.shared.mark_lost(info);
            self.inner.pending_changed.notify_all();
        }
        if let Some(info) = self.inner.shared.loss_info() {
            for (index, present) in request.presents.iter().enumerate() {
                if !presented[index] {
                    present
                        .attachment
                        .terminate_present(present.receipt, PresentState::DeviceLost(info.clone()));
                }
            }
        }
        Ok(SubmissionOutcome {
            completion: last_serial,
            points,
        })
    }

    /// Non-blocking completion query.  This path never waits for the GPU.
    pub(in crate::backend::vulkan) fn completion(&self, serial: u64) -> CompletionState {
        let state = self.lock();
        let answer = completion_from_state(&state, serial);
        let known_complete = serial != 0 && serial <= state.completed;
        drop(state);
        if let Some(info) = self.inner.shared.loss_info() {
            return if known_complete {
                CompletionState::Complete
            } else {
                CompletionState::DeviceLost(info)
            };
        }
        answer
    }

    /// Samples and registers a future waker under the same mutex used by
    /// completion advancement, preventing a completed fence from being missed
    /// between the sample and registration.
    pub(in crate::backend::vulkan) fn completion_or_register_waker(
        &self,
        serial: u64,
        waker: &Waker,
    ) -> CompletionState {
        let state = self.lock();
        let answer = completion_from_state(&state, serial);
        let known_complete = serial != 0 && serial <= state.completed;
        drop(state);
        if let Some(info) = self.inner.shared.loss_info() {
            return if known_complete {
                CompletionState::Complete
            } else {
                CompletionState::DeviceLost(info)
            };
        }
        if matches!(answer, CompletionState::Pending) {
            if let Err(info) = self.inner.shared.register_completion_waker(serial, waker) {
                return CompletionState::DeviceLost(info);
            }
            // Completion may have raced registration. Re-sample so a fence that
            // became signalled just before registration cannot strand a future.
            let state = self.lock();
            let answer = completion_from_state(&state, serial);
            drop(state);
            if !matches!(answer, CompletionState::Pending) {
                self.inner.shared.wake_completion(serial);
            }
            return answer;
        }
        answer
    }
}

pub(super) fn query_result_stride(set: &crate::api::query::QuerySet) -> Result<u64, VulkanFailure> {
    use crate::api::query::{PipelineStatistics, QueryType};
    let values = match set.descriptor().ty {
        QueryType::Occlusion | QueryType::Timestamp => 1,
        QueryType::PipelineStatistics(selection) => [
            PipelineStatistics::VERTEX_SHADER_INVOCATIONS,
            PipelineStatistics::CLIPPER_INVOCATIONS,
            PipelineStatistics::CLIPPER_PRIMITIVES_OUT,
            PipelineStatistics::FRAGMENT_SHADER_INVOCATIONS,
            PipelineStatistics::COMPUTE_SHADER_INVOCATIONS,
        ]
        .into_iter()
        .filter(|counter| selection.contains(*counter))
        .count(),
        _ => crate::unknown_portable_variant(),
    };
    u64::try_from(values)
        .ok()
        .and_then(|values| values.checked_mul(8))
        .ok_or(VulkanFailure::Unsupported {
            what: "a Vulkan query-result stride",
            why: "the selected portable result layout overflowed Vulkan's stride type",
        })
}

impl Drop for VulkanCommandSpine {
    fn drop(&mut self) {
        self.inner.shutdown.store(true, Ordering::Release);
        self.inner.pending_changed.notify_all();
        if let Some(waiter) = self.waiter.take() {
            let _ = waiter.join();
        }
    }
}

impl Drop for SpineInner {
    fn drop(&mut self) {
        // Unlike COM-backed APIs, Vulkan command buffers and their pools may not
        // be destroyed while submitted work still uses them. The last public
        // Device owner can disappear without the caller awaiting its receipt,
        // so destruction itself must close that native lifetime. A lost device
        // may reject the wait; destruction is still the only remaining cleanup
        // path in that terminal domain.
        let _queue = self.shared.queue_guard();
        let _ = unsafe { self.shared.device.device_wait_idle() };
        let state = self
            .state
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // Device shutdown drops the spine before VulkanShared.  The device may
        // already be lost; Vulkan destruction is still the required ownership
        // cleanup and does not turn a destructor into a recovery operation.
        for (_, pending) in std::mem::take(&mut state.pending) {
            unsafe { self.shared.device.destroy_fence(pending.fence, None) };
        }
    }
}

/// One device-owned progress worker replaces one OS thread per batch. The
/// single queue issues serials in order, so waiting for the earliest pending
/// fence advances the exact portable frontier without sacrificing per-batch
/// completion points.
fn fence_wait_loop(inner: Arc<SpineInner>) {
    loop {
        let (serial, fence) = {
            let mut state = inner
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            loop {
                if inner.shutdown.load(Ordering::Acquire) || inner.shared.loss_info().is_some() {
                    return;
                }
                if let Some((&serial, pending)) = state.pending.first_key_value() {
                    break (serial, pending.fence);
                }
                state = inner
                    .pending_changed
                    .wait(state)
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
            }
        };
        // SAFETY: the pending map retains this fence until this sole worker
        // publishes its terminal result. Shutdown joins this worker before the
        // command pool or fences are destroyed.
        let result = unsafe {
            inner
                .shared
                .device
                .wait_for_fences(&[fence], true, FENCE_WAITER_POLL_NS)
        };
        // A finite host wait only means this worker should resample shutdown,
        // loss, and the same earliest fence. It proves neither completion nor
        // failure, so in particular it must not remove `pending` or release the
        // command buffer/staging retained by that accepted batch.
        if fence_wait_timed_out(&result) {
            continue;
        }
        finish_waited_batch(&inner, serial, result);
        // The synchronous `wait_idle` path waits for the worker, rather than
        // touching a fence concurrently with it. Notify after every terminal
        // observation, including loss, so it need not wait for its bounded
        // resample interval in the usual case.
        inner.pending_changed.notify_all();
    }
}

/// Classifies the one non-terminal result of the bounded worker wait.
///
/// Keeping this separate makes it hard for a future refactor to accidentally
/// hand `TIMEOUT` to `finish_waited_batch`, where any error is deliberately a
/// terminal execution-domain failure.
fn fence_wait_timed_out(result: &Result<(), vk::Result>) -> bool {
    matches!(result, Err(vk::Result::TIMEOUT))
}

fn finish_waited_batch(inner: &SpineInner, serial: u64, result: Result<(), vk::Result>) {
    let mut state = inner
        .state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(pending) = state.pending.remove(&serial) else {
        return;
    };
    match result {
        Ok(()) => {
            let tickets = pending.retention.readback_tickets();
            let committed = inner.shared.commit_completion(&tickets, || {
                let mut terminal_loss = None;
                for readback in &pending.retention.readbacks {
                    if let Err(result) = transfer::publish_readback(readback) {
                        if result == vk::Result::ERROR_DEVICE_LOST {
                            terminal_loss = Some(
                                "Vulkan reported VK_ERROR_DEVICE_LOST while publishing completed readback data",
                            );
                            break;
                        }
                    }
                }
                terminal_loss
            });
            let terminal_loss = match committed {
                Ok(loss) => loss,
                Err(_) => {
                    // Device loss linearized first. The successful native wait
                    // does not get to change the terminal answer observers have
                    // already received for this still-pending logical point.
                    for readback in &pending.retention.readbacks {
                        readback.ticket.set_status(ReadbackStatus::DeviceLost);
                    }
                    unsafe { inner.shared.device.destroy_fence(pending.fence, None) };
                    drop(state);
                    return;
                }
            };
            unsafe { inner.shared.device.destroy_fence(pending.fence, None) };
            if let Some(message) = terminal_loss {
                for other in state.pending.values() {
                    for readback in &other.retention.readbacks {
                        readback.ticket.set_status(ReadbackStatus::DeviceLost);
                    }
                }
                drop(state);
                inner
                    .shared
                    .mark_lost(DeviceLossInfo::new(message.to_owned()));
                return;
            }
            state.finished.insert(serial);
            let previous_frontier = state.completed;
            loop {
                let Some(next) = state.completed.checked_add(1) else {
                    break;
                };
                if !state.finished.remove(&next) {
                    break;
                }
                state.completed += 1;
            }
            let completed_frontier = state.completed;
            inner.shared.advance_completed_serial(completed_frontier);
            drop(state);
            for completed in previous_frontier + 1..=completed_frontier {
                inner.shared.wake_completion(completed);
            }
        }
        Err(result) if result == vk::Result::ERROR_DEVICE_LOST => {
            for readback in &pending.retention.readbacks {
                readback.ticket.set_status(ReadbackStatus::DeviceLost);
            }
            for other in state.pending.values() {
                for readback in &other.retention.readbacks {
                    readback.ticket.set_status(ReadbackStatus::DeviceLost);
                }
            }
            // Loss is not proof that DMA has stopped. Keep the native fence,
            // command buffer, staging, and portable resource owners retained
            // through execution-domain teardown rather than freeing memory a
            // removed device could conceivably still touch.
            state.pending.insert(serial, pending);
            drop(state);
            inner.shared.mark_lost(DeviceLossInfo::new(
                "Vulkan reported VK_ERROR_DEVICE_LOST while waiting for a completion fence"
                    .to_owned(),
            ));
        }
        Err(result) => {
            for readback in &pending.retention.readbacks {
                readback.ticket.set_status(ReadbackStatus::DeviceLost);
            }
            state.poison = Some((
                serial,
                CompletionFailure::new(format!(
                    "Vulkan fence wait became unobservable after submission: {result:?}"
                )),
            ));
            // An unobservable fence is likewise not permission to retire the
            // work's native ownership. It is released only during teardown.
            state.pending.insert(serial, pending);
            drop(state);
            inner.shared.mark_lost(DeviceLossInfo::new(format!(
                "Vulkan completion fence failed after submission and the execution domain can no longer prove progress: {result:?}"
            )));
        }
    }
}

fn completion_from_state(state: &SpineState, serial: u64) -> CompletionState {
    if serial == 0 {
        CompletionState::Complete
    } else if serial <= state.completed {
        CompletionState::Complete
    } else if let Some((first, failure)) = &state.poison {
        if serial >= *first {
            CompletionState::Failed(failure.clone())
        } else {
            CompletionState::Pending
        }
    } else if serial <= state.issued {
        CompletionState::Pending
    } else {
        CompletionState::Failed(CompletionFailure::new(
            "Vulkan completion was queried for a serial this device never issued",
        ))
    }
}

/// Records the completion serial only after `vkQueueSubmit` accepted this
/// batch. Phase-A failures never reach here, preserving the map future's
/// guarantee that it waits for actual accepted GPU use rather than recorded
/// intent. Duplicate uses are harmless because the native buffer atomically
/// keeps the greatest serial.
fn mark_batch_buffer_uses(batch: &PlanBatch, serial: u64) {
    for work in &batch.work {
        for use_ in work.resource_uses() {
            if let ResourceUse::Buffer(buffer) = use_ {
                if let Some(native) = buffer
                    .buffer
                    .native()
                    .as_any()
                    .downcast_ref::<VulkanBuffer>()
                {
                    native.mark_accepted(serial);
                }
            }
        }
    }
}

fn prepare_presentations(
    request: &SubmissionRequest<'_>,
) -> Result<Vec<BatchPresentation>, VulkanFailure> {
    let mut batches = (0..request.batches.len())
        .map(|_| BatchPresentation::default())
        .collect::<Vec<_>>();
    for (present_index, present) in request.presents.iter().enumerate() {
        let sync = frame_present_sync(&present.attachment)?;
        let frame = present.attachment.frame_id();
        let first_use = request
            .batches
            .iter()
            .position(|batch| {
                batch.work.iter().any(|work| {
                    work.resource_uses()
                        .iter()
                        .any(|use_| matches!(use_, ResourceUse::Frame(use_) if use_.frame == frame))
                })
            })
            .ok_or(VulkanFailure::Unsupported {
                what: "a Vulkan presentation relation without frame work",
                why: "portable plan validation should require the presented frame to be used",
            })?;
        let present_batch = request
            .batches
            .iter()
            .position(|batch| batch.point == present.after)
            .ok_or(VulkanFailure::Unsupported {
                what: "a Vulkan presentation point outside the submitted plan",
                why: "portable plan validation should resolve every present-after point",
            })?;
        if first_use > present_batch {
            return Err(VulkanFailure::Unsupported {
                what: "a Vulkan presentation ordered before its first frame use",
                why: "portable plan validation should order every frame use before presentation",
            });
        }
        batches[first_use].waits.push(sync.acquire_wait);
        batches[present_batch].signals.push(sync.render_finished);
        batches[present_batch].presents.push(present_index);
    }
    Ok(batches)
}

#[cfg(any(windows, target_os = "android"))]
fn frame_present_sync(attachment: &FrameAttachment) -> Result<VulkanPresentSync, VulkanFailure> {
    attachment
        .native()
        .as_any()
        .downcast_ref::<VulkanFrameAttachment>()
        .map(VulkanFrameAttachment::sync)
        .ok_or(VulkanFailure::Unsupported {
            what: "a Vulkan presentation attachment",
            why: "its native drawable belongs to another backend",
        })
}

#[cfg(not(any(windows, target_os = "android")))]
fn frame_present_sync(_: &FrameAttachment) -> Result<VulkanPresentSync, VulkanFailure> {
    Err(VulkanFailure::Unsupported {
        what: "a Vulkan presentation attachment",
        why: "this platform has no Vulkan presentation lowering",
    })
}

pub(super) fn native_query_pool(
    set: &crate::api::query::QuerySet,
) -> Result<vk::QueryPool, VulkanFailure> {
    set.native()
        .as_any()
        .downcast_ref::<VulkanQuerySet>()
        .map(VulkanQuerySet::pool)
        .ok_or(VulkanFailure::Unsupported {
            what: "a query set this Vulkan device did not create",
            why: "its VkQueryPool belongs to another backend",
        })
}

pub(super) fn native_buffer(
    buffer: &crate::api::resource::Buffer,
) -> Result<vk::Buffer, VulkanFailure> {
    buffer
        .native()
        .as_any()
        .downcast_ref::<VulkanBuffer>()
        .map(VulkanBuffer::buffer)
        .ok_or(VulkanFailure::Unsupported {
            what: "a query resolve buffer this Vulkan device did not create",
            why: "its VkBuffer belongs to another backend",
        })
}

#[cfg(test)]
mod tests {
    use ash::vk;

    use super::fence_wait_timed_out;

    #[test]
    fn bounded_fence_wait_timeout_is_not_an_execution_failure() {
        assert!(fence_wait_timed_out(&Err(vk::Result::TIMEOUT)));
        assert!(!fence_wait_timed_out(&Ok(())));
        assert!(!fence_wait_timed_out(&Err(vk::Result::ERROR_DEVICE_LOST)));
    }
}
