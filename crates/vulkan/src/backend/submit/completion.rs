//! Vulkan fence completion observation.
//!
//! The waiter owns completion observation and post-submit retention only. It
//! never records or schedules GPU work; submission remains in the parent.

use super::{DeviceState, RecordedWork, Vulkan, VulkanDevice, vk_error};
use ash::vk;
use fluxel_rhi_core::api::{self, CompletionState, Error, Result};
use std::{
    future::Future,
    pin::Pin,
    sync::{
        Arc, Mutex, OnceLock,
        mpsc::{self, Receiver, RecvTimeoutError, Sender, SyncSender},
    },
    task::{Context, Poll, Waker},
    time::Duration,
};

struct WaitState {
    result: Option<Result<()>>,
    wakers: Vec<Waker>,
}

#[derive(Clone)]
pub struct Completion(Arc<FenceState>);
struct FenceState {
    device: Arc<DeviceState>,
    fence: vk::Fence,
    wait: Mutex<WaitState>,
}
impl Drop for FenceState {
    fn drop(&mut self) {
        // SAFETY: FenceState exclusively owns this fence and retains its device
        // until all Completion clones, including the observer's, have dropped.
        unsafe { self.device.device.destroy_fence(self.fence, None) };
    }
}
impl Completion {
    pub(super) fn new(device: Arc<DeviceState>, fence: vk::Fence) -> Self {
        Self(Arc::new(FenceState {
            device,
            fence,
            wait: Mutex::new(WaitState {
                result: None,
                wakers: Vec::new(),
            }),
        }))
    }

    pub(super) fn fence(&self) -> vk::Fence {
        self.0.fence
    }

    fn publish(&self, result: Result<()>) {
        let wakers = {
            let mut wait = self.0.wait.lock().unwrap_or_else(|p| p.into_inner());
            wait.result = Some(result);
            std::mem::take(&mut wait.wakers)
        };
        for waker in wakers {
            waker.wake();
        }
    }

    fn result(&self) -> Option<Result<()>> {
        self.0
            .wait
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .result
            .clone()
    }
}
pub struct FenceFuture {
    completion: Option<Completion>,
    immediate: Option<Result<()>>,
}
impl FenceFuture {
    fn pending(completion: Completion) -> Self {
        Self {
            completion: Some(completion),
            immediate: None,
        }
    }
    fn immediate(result: Result<()>) -> Self {
        Self {
            completion: None,
            immediate: Some(result),
        }
    }
}
impl Future for FenceFuture {
    type Output = Result<()>;
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        if let Some(result) = &this.immediate {
            return Poll::Ready(result.clone());
        }
        let completion = this
            .completion
            .as_ref()
            .expect("completion future has a fence");
        let mut wait = completion.0.wait.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(result) = &wait.result {
            Poll::Ready(result.clone())
        } else {
            if !wait.wakers.iter().any(|waker| waker.will_wake(cx.waker())) {
                wait.wakers.push(cx.waker().clone());
            }
            Poll::Pending
        }
    }
}

/// One bounded observer for submitted Vulkan fences.
///
/// It owns only fences and already-recorded native work.  In particular it
/// never chooses queue order, submits commands, or interprets recorded work.
/// A dropped `FenceFuture` leaves its `Completion` in this queue, so command
/// pools and upload allocations still remain alive until the GPU is finished.
pub(super) struct FenceWaiter {
    sender: Sender<FenceJob>,
}

struct FenceJob {
    completion: Completion,
    work: Vec<RecordedWork>,
    sync: Vec<Arc<super::multi_queue::Timeline>>,
    sparse: Vec<super::super::sparse::PreparedSparseBind>,
}
static FENCE_WAITER: OnceLock<std::result::Result<FenceWaiter, String>> = OnceLock::new();

// This bounds notification latency while keeping the observer asleep whenever
// no producer sends a job.  It is deliberately a timed receive, rather than a
// Future polling loop: `FenceFuture` is only woken by `Completion::publish`.
const OBSERVATION_INTERVAL: Duration = Duration::from_millis(2);

/// Completes one job after its native fence has supplied a definitive result.
///
/// Publication precedes optional capture and retirement.  A waiter therefore
/// observes GPU completion independently of slow filesystem I/O, while the
/// capture worker retains its command resources until it has read them.
fn finish_job(job: FenceJob, result: Result<()>, captures: &SyncSender<Vec<RecordedWork>>) {
    let FenceJob {
        completion,
        work,
        sync,
        sparse,
    } = job;
    let completed = result.is_ok();
    completion.publish(result);
    if completed {
        // Capture is optional instrumentation.  The dedicated bounded worker
        // must not make a completed fence delay another fence's notification.
        // If that worker is unavailable, dropping is safe because completion
        // was already established; capture failure never changes RHI result.
        if work.iter().any(|item| item.encoder.capture.is_some()) {
            // Never let diagnostic filesystem throughput head-of-line-block
            // native completion observation. There is normally only one
            // process-wide capture, but a full bounded queue may drop it.
            let _ = captures.try_send(work);
        }
        drop(sync);
        drop(sparse);
    } else {
        // A failed status query is not retirement evidence. Keep every native
        // owner (including the fence/device) quarantined exactly as before.
        std::mem::forget(work);
        std::mem::forget(sync);
        std::mem::forget(sparse);
        std::mem::forget(completion);
    }
}

fn capture_worker(receiver: Receiver<Vec<RecordedWork>>) {
    for work in receiver {
        // Capture is diagnostic-only; errors must not retroactively fail a
        // completion that has already been published to the caller.
        let _ = super::super::capture::write_completed(&work);
    }
}

fn observe_fences(receiver: Receiver<FenceJob>, captures: SyncSender<Vec<RecordedWork>>) {
    let mut active = Vec::new();
    loop {
        // A finite channel wait lets newly submitted jobs join the active set
        // promptly, even while an earlier fence is intentionally unsignaled.
        match receiver.recv_timeout(OBSERVATION_INTERVAL) {
            Ok(job) => active.push(job),
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) if active.is_empty() => return,
            // The process-global sender normally prevents this case. If it is
            // ever dropped while jobs remain, retain the finite observation
            // cadence rather than spin on an immediately-disconnected channel.
            Err(RecvTimeoutError::Disconnected) => std::thread::park_timeout(OBSERVATION_INTERVAL),
        }
        while let Ok(job) = receiver.try_recv() {
            active.push(job);
        }

        let mut index = 0;
        while index < active.len() {
            let completion = &active[index].completion;
            // SAFETY: every active job owns its fence and retains the logical
            // device. A zero-time status query neither accesses a queue nor
            // changes fence state, so all independent devices can be scanned.
            let result = unsafe {
                completion
                    .0
                    .device
                    .device
                    .wait_for_fences(&[completion.0.fence], true, 0)
            };
            match result {
                Ok(()) => {
                    let job = active.swap_remove(index);
                    finish_job(job, Ok(()), &captures);
                }
                Err(vk::Result::TIMEOUT) => index += 1,
                Err(error) => {
                    let job = active.swap_remove(index);
                    finish_job(job, Err(vk_error(error)), &captures);
                }
            }
        }
    }
}

pub(super) fn fence_waiter() -> Result<&'static FenceWaiter> {
    match FENCE_WAITER.get_or_init(|| {
        let (sender, receiver) = mpsc::channel::<FenceJob>();
        // Capture is optional test instrumentation. A capacity of one bounds
        // retained capture work without making the completion observer wait
        // behind filesystem I/O.
        let (capture_sender, capture_receiver) = mpsc::sync_channel::<Vec<RecordedWork>>(1);
        std::thread::Builder::new()
            .name("fluxel-vulkan-capture-writer".into())
            .spawn(move || capture_worker(capture_receiver))
            .map_err(|error| format!("Vulkan capture worker: {error}"))?;
        std::thread::Builder::new()
            .name("fluxel-vulkan-fence-waiter".into())
            .spawn(move || observe_fences(receiver, capture_sender))
            .map(|_| FenceWaiter { sender })
            .map_err(|error| format!("Vulkan fence waiter: {error}"))
    }) {
        Ok(waiter) => Ok(waiter),
        Err(message) => Err(Error::Backend(message.clone())),
    }
}

// The waiter is only a native completion observer and owner of already issued
// command pools/uploads. It never schedules GPU commands or replays CPU work.
pub(super) fn retain_until_fence(completion: Completion, work: Vec<RecordedWork>) {
    retain_with_sync(completion, work, Vec::new(), Vec::new());
}

pub(super) fn retain_with_sync(
    completion: Completion,
    work: Vec<RecordedWork>,
    sync: Vec<Arc<super::multi_queue::Timeline>>,
    sparse: Vec<super::super::sparse::PreparedSparseBind>,
) {
    let job = FenceJob {
        completion: completion.clone(),
        work,
        sync,
        sparse,
    };
    match fence_waiter() {
        Ok(waiter) => match waiter.sender.send(job) {
            Ok(()) => {}
            Err(error) => {
                // A native submission may already reference this work. Never
                // drop it on an observer failure; leaking this exceptional
                // ownership is safer than a synchronous host wait or an early
                // Vulkan command-pool destruction.
                std::mem::forget(error.0);
                completion.publish(Err(Error::Backend(
                    "Vulkan fence waiter unavailable".into(),
                )));
            }
        },
        Err(error) => {
            std::mem::forget(job);
            completion.publish(Err(error));
        }
    }
}

impl api::CompletionBackend for Vulkan {
    type WaitFuture<'a> = FenceFuture;

    fn completion_status(_: &VulkanDevice, point: &Completion) -> CompletionState {
        match point.result() {
            Some(Ok(())) => CompletionState::Complete,
            Some(Err(error)) => CompletionState::Failed(error),
            // SAFETY: Completion retains its device and fence; status polling
            // does not require queue ownership or mutate either object.
            None => match unsafe { point.0.device.device.get_fence_status(point.0.fence) } {
                Ok(true) => CompletionState::Complete,
                Ok(false) => CompletionState::Pending,
                Err(error) => CompletionState::Failed(vk_error(error)),
            },
        }
    }

    fn wait_completion<'a>(_: &'a VulkanDevice, point: &'a Completion) -> Self::WaitFuture<'a> {
        FenceFuture::pending(point.clone())
    }

    fn wait_idle(device: &VulkanDevice) -> Self::WaitFuture<'_> {
        // SAFETY: DeviceState owns the logical device and Completion immediately
        // becomes the sole owner of the newly created fence on success.
        let fence = match unsafe {
            device
                .0
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)
        } {
            Ok(fence) => fence,
            Err(error) => return FenceFuture::immediate(Err(vk_error(error))),
        };
        let completion = Completion::new(device.0.clone(), fence);
        let result = {
            let _submit = device
                .0
                .submit_gate
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            let _queue = device
                .0
                .queue_gate
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            // SAFETY: submit_gate and queue_gate serialize this empty marker
            // submission; the new fence belongs to this logical device.
            unsafe {
                device
                    .0
                    .device
                    .queue_submit(device.0.graphics_queue, &[], fence)
            }
        };
        if let Err(error) = result {
            // Queue submission failure is not retirement evidence for this fence.
            std::mem::forget(completion);
            return FenceFuture::immediate(Err(vk_error(error)));
        }
        retain_until_fence(completion.clone(), Vec::new());
        FenceFuture::pending(completion)
    }
}

#[cfg(all(test, feature = "hardware-tests"))]
mod hardware_tests {
    //! Real-driver regression for independent-device fence observation.
    //!
    //! The first queue waits on an unsignaled timeline value, so its marker
    //! fence cannot retire. The second logical device has an independent
    //! queue and marker fence; it must still publish before the timeline is
    //! released. This exercises the actual global waiter rather than a mock
    //! readiness source.

    use super::{Completion, FenceFuture, Vulkan, retain_until_fence};
    use crate::backend::VulkanInstance;
    use ash::vk;
    use fluxel_rhi_core::api::Device;
    use std::{
        future::Future,
        sync::{Arc, mpsc::sync_channel},
        task::{Context, Poll, Wake, Waker},
        time::Duration,
    };

    struct ThreadWake(std::thread::Thread);
    impl Wake for ThreadWake {
        fn wake(self: Arc<Self>) {
            self.0.unpark();
        }
        fn wake_by_ref(self: &Arc<Self>) {
            self.0.unpark();
        }
    }

    fn block_on<F: Future>(future: F) -> F::Output {
        let waker = Waker::from(Arc::new(ThreadWake(std::thread::current())));
        let mut context = Context::from_waker(&waker);
        let mut future = std::pin::pin!(future);
        loop {
            match future.as_mut().poll(&mut context) {
                Poll::Ready(value) => return value,
                Poll::Pending => std::thread::park(),
            }
        }
    }

    fn device() -> Device<Vulkan> {
        block_on(VulkanInstance::new().unwrap().open_default()).unwrap()
    }

    #[test]
    fn held_timeline_fence_does_not_head_of_line_block_another_device() {
        let held_device = device();
        let independent_device = device();
        // SAFETY: these native handles are only used below while their typed
        // Core devices remain alive; this test does not expose them elsewhere.
        let held_native = unsafe { held_device.native_ref() }.clone();
        let independent_native = unsafe { independent_device.native_ref() }.clone();
        let held_state = held_native.0.clone();
        let independent_state = independent_native.0.clone();
        if !held_state.timeline_semaphore {
            // The regression requires a native timeline wait to hold a fence
            // without adding arbitrary GPU work. Older real devices simply do
            // not expose that Vulkan mechanism.
            return;
        }

        let mut timeline_type = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        let timeline_info = vk::SemaphoreCreateInfo::default().push_next(&mut timeline_type);
        // SAFETY: held_state owns the logical device and this test owns the
        // resulting semaphore until both submissions have completed.
        let timeline = unsafe { held_state.device.create_semaphore(&timeline_info, None) }.unwrap();
        let held = Completion::new(
            held_state.clone(),
            // SAFETY: the new fence belongs exclusively to this Completion.
            unsafe {
                held_state
                    .device
                    .create_fence(&vk::FenceCreateInfo::default(), None)
            }
            .unwrap(),
        );
        let independent = Completion::new(
            independent_state.clone(),
            // SAFETY: the new fence belongs exclusively to this Completion.
            unsafe {
                independent_state
                    .device
                    .create_fence(&vk::FenceCreateInfo::default(), None)
            }
            .unwrap(),
        );

        let wait_values = [1];
        let waits = [timeline];
        let stages = [vk::PipelineStageFlags::ALL_COMMANDS];
        let mut wait_timeline =
            vk::TimelineSemaphoreSubmitInfo::default().wait_semaphore_values(&wait_values);
        let held_submit = vk::SubmitInfo::default()
            .wait_semaphores(&waits)
            .wait_dst_stage_mask(&stages)
            .push_next(&mut wait_timeline);
        {
            let _queue = held_state
                .queue_gate
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            // SAFETY: the queue gate serializes this test submission; the
            // timeline wait prevents `held` from signalling until below.
            unsafe {
                held_state.device.queue_submit(
                    held_state.graphics_queue,
                    &[held_submit],
                    held.fence(),
                )
            }
            .unwrap();
        }
        {
            let _queue = independent_state
                .queue_gate
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            // SAFETY: independent_state owns its queue and fence; this empty
            // marker has no dependency on the held timeline submission.
            unsafe {
                independent_state.device.queue_submit(
                    independent_state.graphics_queue,
                    &[],
                    independent.fence(),
                )
            }
            .unwrap();
        }

        retain_until_fence(held.clone(), Vec::new());
        retain_until_fence(independent.clone(), Vec::new());
        let independent_published = independent.clone();
        let (done, notified) = sync_channel(1);
        std::thread::spawn(move || {
            done.send(block_on(FenceFuture::pending(independent)))
                .unwrap();
        });
        // A one-worker infinite wait would time out here behind `held`.
        notified
            .recv_timeout(Duration::from_secs(2))
            .expect("independent marker must publish while the first fence is held")
            .unwrap();
        assert!(matches!(independent_published.result(), Some(Ok(()))));
        assert!(
            held.result().is_none(),
            "held marker retired before its timeline signal"
        );

        let signal = vk::SemaphoreSignalInfo::default()
            .semaphore(timeline)
            .value(1);
        // SAFETY: this host signal supplies exactly the value awaited by the
        // earlier queue submission; unlike a later same-queue submission it
        // cannot be ordered behind that wait.
        unsafe { held_state.device.signal_semaphore(&signal) }.unwrap();
        block_on(FenceFuture::pending(held.clone())).unwrap();
        assert!(matches!(held.result(), Some(Ok(()))));
        // SAFETY: the held marker proves both submissions consumed `timeline`.
        unsafe { held_state.device.destroy_semaphore(timeline, None) };
    }
}
