//! Unit tests for the corresponding implementation module.

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
