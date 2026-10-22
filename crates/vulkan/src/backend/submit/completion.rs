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
        mpsc::{self, Sender},
    },
    task::{Context, Poll, Waker},
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

pub(super) fn fence_waiter() -> Result<&'static FenceWaiter> {
    match FENCE_WAITER.get_or_init(|| {
        let (sender, receiver) = mpsc::channel::<FenceJob>();
        std::thread::Builder::new()
            .name("fluxel-vulkan-fence-waiter".into())
            .spawn(move || {
                for job in receiver {
                    // SAFETY: this observer owns an accepted fence, retains its
                    // device, and only waits; it does not access the queue.
                    let result = unsafe {
                        job.completion.0.device.device.wait_for_fences(
                            &[job.completion.0.fence],
                            true,
                            u64::MAX,
                        )
                    }
                    .map_err(vk_error);
                    // Completion represents GPU evidence only.  Optional
                    // visual capture is test instrumentation and must never
                    // turn completed GPU work into a failed RHI completion.
                    let completed = result.is_ok();
                    job.completion.publish(result);
                    if completed {
                        let _ = super::super::capture::write_completed(&job.work);
                    }
                    if completed {
                        drop(job.work);
                        drop(job.sync);
                        drop(job.sparse);
                    } else {
                        // A failed wait does not prove the queue has retired native objects.
                        std::mem::forget(job.work);
                        std::mem::forget(job.sync);
                        std::mem::forget(job.sparse);
                        std::mem::forget(job.completion);
                    }
                }
            })
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
