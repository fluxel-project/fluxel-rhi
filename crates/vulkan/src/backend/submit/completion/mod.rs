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

impl fluxel_rhi_core::backend::CompletionBackend for Vulkan {
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
#[path = "../../../tests/backend/submit/completion/hardware_tests.rs"]
mod hardware_tests;
