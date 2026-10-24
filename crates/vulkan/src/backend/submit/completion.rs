//! Vulkan fence completion observation.
//!
//! The waiter owns completion observation and post-submit retention only. It
//! never records or schedules GPU work; submission remains in the parent.

use super::{DeviceState, RecordedWork, Vulkan, VulkanDevice, vk_error};
use ash::vk;
use fluxel_rhi_core::api::{self, CompletionState, Error, Result};
use std::{
    collections::BTreeMap,
    future::Future,
    pin::Pin,
    sync::{Arc, Mutex},
    task::{Context, Poll, Waker},
};

struct WaitState {
    result: Option<Result<()>>,
    wakers: BTreeMap<u64, Waker>,
    next_registration: u64,
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
                wakers: BTreeMap::new(),
                next_registration: 0,
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
        for waker in wakers.into_values() {
            // A caller-provided waker must not kill the retirement worker.
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| waker.wake()));
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
/// Independent native fence observations; polling never calls into the driver.
pub struct FenceFuture {
    registrations: Vec<(Completion, Option<u64>)>,
    immediate: Option<Result<()>>,
}
impl FenceFuture {
    fn pending(completion: Completion) -> Self {
        Self::pending_many(vec![completion])
    }
    fn pending_many(completions: Vec<Completion>) -> Self {
        Self {
            registrations: completions.into_iter().map(|point| (point, None)).collect(),
            immediate: None,
        }
    }
    fn immediate(result: Result<()>) -> Self {
        Self {
            registrations: Vec::new(),
            immediate: Some(result),
        }
    }
}
impl Drop for FenceFuture {
    fn drop(&mut self) {
        for (point, id) in &mut self.registrations {
            if let Some(id) = id.take() {
                point
                    .0
                    .wait
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .wakers
                    .remove(&id);
            }
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
        let mut pending = false;
        let mut failure = None;
        for (point, registration) in &mut this.registrations {
            let mut wait = point.0.wait.lock().unwrap_or_else(|p| p.into_inner());
            if let Some(result) = &wait.result {
                if let Err(error) = result {
                    failure.get_or_insert_with(|| error.clone());
                }
                if let Some(id) = registration.take() {
                    wait.wakers.remove(&id);
                }
                continue;
            }
            pending = true;
            let id = match *registration {
                Some(id) => id,
                None => {
                    let Some(next) = wait.next_registration.checked_add(1) else {
                        return Poll::Ready(Err(Error::InvalidState));
                    };
                    let id = wait.next_registration;
                    wait.next_registration = next;
                    *registration = Some(id);
                    id
                }
            };
            match wait.wakers.get_mut(&id) {
                Some(waker) if waker.will_wake(cx.waker()) => {}
                Some(waker) => *waker = cx.waker().clone(),
                None => {
                    wait.wakers.insert(id, cx.waker().clone());
                }
            }
        }
        if let Some(error) = failure {
            Poll::Ready(Err(error))
        } else if pending {
            Poll::Pending
        } else {
            Poll::Ready(Ok(()))
        }
    }
}

mod observer;
pub(super) use observer::{fence_waiter, retain_until_fence, retain_with_sync};

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
        // Install independent markers under the submission gate. Waiting on both
        // is a CPU completion aggregate, never an implicit GPU queue join.
        if let Err(error) = fence_waiter() {
            return FenceFuture::immediate(Err(error));
        }
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
        let mut queues = vec![device.0.graphics_queue];
        if let Some(compute) = device.0.compute_queue {
            if compute != device.0.graphics_queue {
                queues.push(compute);
            }
        }
        let mut points = Vec::with_capacity(queues.len());
        for queue in queues {
            // SAFETY: the logical device is retained and owns the new fence.
            let fence = match unsafe {
                device
                    .0
                    .device
                    .create_fence(&vk::FenceCreateInfo::default(), None)
            } {
                Ok(fence) => fence,
                Err(error) => return FenceFuture::immediate(Err(vk_error(error))),
            };
            let point = Completion::new(device.0.clone(), fence);
            // SAFETY: both native queue external-synchronization gates are held.
            if let Err(error) = unsafe { device.0.device.queue_submit(queue, &[], fence) } {
                // Failed submit does not establish whether the fence was used.
                std::mem::forget(point);
                return FenceFuture::immediate(Err(vk_error(error)));
            }
            retain_until_fence(point.clone(), Vec::new());
            points.push(point);
        }
        FenceFuture::pending_many(points)
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
