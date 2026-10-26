//! Completion tracking for native Metal command buffers and queue frontiers.

#[cfg(target_vendor = "apple")]
use super::NativeCommandBuffer;
use super::{Metal, MetalDevice};
use fluxel_rhi_core::api::{CompletionState, Error, Result};
use fluxel_rhi_core::backend::CompletionBackend;
use std::{
    future::Future,
    pin::Pin,
    sync::{Arc, Mutex},
    task::{Context, Poll, Waker},
};

#[cfg(target_vendor = "apple")]
#[derive(Default)]
pub(crate) struct SubmissionTracker {
    next: u64,
    pending: std::collections::HashMap<u64, Completion>,
    failure: Option<Error>,
}

/// Register before commit while holding queue_gate. The tracker retains the
/// buffer until its completion callback has recorded its terminal result.
#[cfg(target_vendor = "apple")]
pub(crate) fn track_command(
    device: &MetalDevice,
    command: &NativeCommandBuffer,
    query: Option<(Arc<super::super::query::QueryCommit>, u32)>,
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
            // SAFETY: Metal supplies a live command buffer to its terminal callback.
            let command = unsafe { pointer.as_ref() };
            let state = state.upgrade();
            let outcome = if command.status() != MTLCommandBufferStatus::Completed {
                let error = if command.error().is_some_and(|e| {
                    e.code() == objc2_metal::MTLCommandBufferError::DeviceRemoved.0 as isize
                }) {
                    Error::DeviceLost
                } else {
                    Error::Backend("Metal command buffer execution failed".into())
                };
                if let Some((commit, lane)) = &query {
                    if let Some(state) = &state {
                        let _gate = state.queue_gate.lock().unwrap_or_else(|p| p.into_inner());
                        super::super::query::invalidate_ops(commit, *lane);
                    } else {
                        super::super::query::invalidate_ops(commit, *lane);
                    }
                }
                Err(error)
            } else {
                Ok(())
            };
            if let Some(state) = state {
                let mut tracker = state
                    .submission_tracker
                    .lock()
                    .unwrap_or_else(|p| p.into_inner());
                if let Err(error) = &outcome {
                    if tracker.failure.is_none() || matches!(error, Error::DeviceLost) {
                        tracker.failure = Some(error.clone());
                    }
                }
                tracker.pending.remove(&id);
            }
            completion.publish(outcome);
        },
    );
    // SAFETY: Metal copies the block; the weak device breaks retention cycles.
    unsafe { command.addCompletedHandler(block2::RcBlock::as_ptr(&callback).cast()) };
    tracked
}
#[derive(Clone)]
pub struct Completion(Arc<CompletionInner>);

struct CompletionInner {
    result: Mutex<Option<Result<()>>>,
    wakers: Mutex<Vec<Waker>>,
}

impl Completion {
    pub(crate) fn new() -> Self {
        Self(Arc::new(CompletionInner {
            result: Mutex::new(None),
            wakers: Mutex::new(Vec::new()),
        }))
    }

    pub(crate) fn result(&self) -> Option<Result<()>> {
        self.0
            .result
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    pub(crate) fn watch(&self, waker: &Waker) -> Option<Result<()>> {
        let mut wakers = self.0.wakers.lock().unwrap_or_else(|p| p.into_inner());
        let result = self.result();
        if result.is_none() && !wakers.iter().any(|w| w.will_wake(waker)) {
            wakers.push(waker.clone());
        }
        result
    }

    pub(crate) fn publish(&self, result: Result<()>) {
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
    pub(crate) fn install_group(
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

pub(super) struct CompletionJoin {
    pub(super) children: Vec<Completion>,
    pub(super) output: Completion,
    #[cfg(any(target_vendor = "apple", test))]
    pub(super) ray_builds: Vec<super::super::ray::BuildCompletion>,
}
impl CompletionJoin {
    pub(super) fn update(self: &Arc<Self>) {
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
pub(crate) fn join_into(output: &Completion, children: Vec<Completion>) {
    Arc::new(CompletionJoin {
        children,
        output: output.clone(),
        #[cfg(any(target_vendor = "apple", test))]
        ray_builds: Vec::new(),
    })
    .update();
}

pub struct CompletionFuture(pub(super) Vec<Completion>);

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
