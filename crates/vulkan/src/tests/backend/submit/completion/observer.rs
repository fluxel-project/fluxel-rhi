//! Tests for the fence-observer worker and its Future registration hygiene.

#[cfg(feature = "hardware-tests")]
use super::{
    super::{FenceFuture, Vulkan},
    Completion, FenceJob,
};
use super::{ObserverProbe, PROBE, observe_fences};
use std::{
    sync::{Arc, atomic::Ordering, mpsc},
    thread,
    time::{Duration, Instant},
};

#[cfg(feature = "hardware-tests")]
use crate::backend::VulkanInstance;
#[cfg(feature = "hardware-tests")]
use ash::vk;
#[cfg(feature = "hardware-tests")]
use fluxel_rhi_core::api::Device;
#[cfg(feature = "hardware-tests")]
use std::{
    future::Future,
    pin::Pin,
    sync::atomic::AtomicUsize,
    task::{Context, Poll, Wake, Waker},
};

#[cfg(feature = "hardware-tests")]
struct ThreadWake(thread::Thread);
#[cfg(feature = "hardware-tests")]
impl Wake for ThreadWake {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }
    fn wake_by_ref(self: &Arc<Self>) {
        self.0.unpark();
    }
}

#[cfg(feature = "hardware-tests")]
fn block_on<F: Future>(future: F) -> F::Output {
    let waker = Waker::from(Arc::new(ThreadWake(thread::current())));
    let mut context = Context::from_waker(&waker);
    let mut future = std::pin::pin!(future);
    loop {
        match future.as_mut().poll(&mut context) {
            Poll::Ready(value) => return value,
            Poll::Pending => thread::park(),
        }
    }
}

fn wait_until(label: &str, predicate: impl Fn() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(2);
    while !predicate() {
        assert!(Instant::now() < deadline, "timed out waiting for {label}");
        thread::sleep(Duration::from_millis(1));
    }
}

#[cfg(feature = "hardware-tests")]
fn device() -> Device<Vulkan> {
    block_on(VulkanInstance::new().unwrap().open_default()).unwrap()
}

/// Owns the test-only native queue hold. Drop signals the awaited value so a
/// test panic cannot strand later users of the same queue behind it.
#[cfg(feature = "hardware-tests")]
struct TimelineGate {
    state: Arc<super::super::DeviceState>,
    semaphore: vk::Semaphore,
    released: bool,
}

#[cfg(feature = "hardware-tests")]
impl TimelineGate {
    fn release(&mut self) {
        if !std::mem::replace(&mut self.released, true) {
            let info = vk::SemaphoreSignalInfo::default()
                .semaphore(self.semaphore)
                .value(1);
            // SAFETY: this test owns the only host signal for the value held
            // by its private queue submission.
            unsafe { self.state.device.signal_semaphore(&info) }
                .expect("release Vulkan observer test gate");
        }
    }
}

#[cfg(feature = "hardware-tests")]
impl Drop for TimelineGate {
    fn drop(&mut self) {
        if !std::mem::replace(&mut self.released, true) {
            let info = vk::SemaphoreSignalInfo::default()
                .semaphore(self.semaphore)
                .value(1);
            // SAFETY: best-effort panic cleanup releases the queue dependency;
            // the semaphore is destroyed only after normal fence retirement.
            let _ = unsafe { self.state.device.signal_semaphore(&info) };
        }
    }
}

#[cfg(feature = "hardware-tests")]
fn hold_graphics_queue(device: &Device<Vulkan>) -> Option<TimelineGate> {
    // SAFETY: the typed device remains live through this private test queue
    // submission and all following fence observations.
    let state = unsafe { device.native_ref() }.0.clone();
    if !state.timeline_semaphore {
        return None;
    }
    let mut kind = vk::SemaphoreTypeCreateInfo::default()
        .semaphore_type(vk::SemaphoreType::TIMELINE)
        .initial_value(0);
    let create = vk::SemaphoreCreateInfo::default().push_next(&mut kind);
    // SAFETY: timeline semaphore support was enabled for this logical device.
    let semaphore = unsafe { state.device.create_semaphore(&create, None) }.ok()?;
    let waits = [semaphore];
    let stages = [vk::PipelineStageFlags::ALL_COMMANDS];
    let values = [1];
    let mut timeline = vk::TimelineSemaphoreSubmitInfo::default().wait_semaphore_values(&values);
    let submit = vk::SubmitInfo::default()
        .wait_semaphores(&waits)
        .wait_dst_stage_mask(&stages)
        .push_next(&mut timeline);
    {
        let _queue = state.queue_gate.lock().unwrap_or_else(|p| p.into_inner());
        // SAFETY: queue_gate provides the required host synchronization and
        // the unsignaled timeline value holds subsequent queue markers.
        unsafe {
            state
                .device
                .queue_submit(state.graphics_queue, &[submit], vk::Fence::null())
        }
        .expect("enqueue Vulkan observer test hold");
    }
    Some(TimelineGate {
        state,
        semaphore,
        released: false,
    })
}

#[test]
fn idle_observer_blocks_on_recv_without_periodic_wakeups() {
    let (sender, receiver) = mpsc::channel();
    let (capture_sender, _capture_receiver) = mpsc::sync_channel(1);
    let probe = Arc::new(ObserverProbe::default());
    let thread_probe = probe.clone();
    let worker = thread::spawn(move || {
        PROBE.with(|slot| *slot.borrow_mut() = Some(thread_probe));
        observe_fences(receiver, capture_sender);
        PROBE.with(|slot| *slot.borrow_mut() = None);
    });
    wait_until("observer to enter idle receive", || {
        probe.idle_receives.load(Ordering::Relaxed) == 1
    });
    thread::sleep(Duration::from_millis(3));
    assert_eq!(
        probe.idle_receives.load(Ordering::Relaxed),
        1,
        "idle observer must remain in blocking recv"
    );
    drop(sender);
    worker.join().unwrap();
}

#[test]
#[cfg(feature = "hardware-tests")]
fn same_device_held_fences_are_batched_without_status_queries() {
    const FENCES: usize = 64;
    let device = device();
    let Some(mut gate) = hold_graphics_queue(&device) else {
        eprintln!("Vulkan timeline semaphore unavailable; skipping observer batch test");
        return;
    };
    let state = gate.state.clone();
    let (sender, receiver) = mpsc::channel();
    let (capture_sender, _capture_receiver) = mpsc::sync_channel(1);
    let probe = Arc::new(ObserverProbe::default());
    let thread_probe = probe.clone();
    let mut points = Vec::with_capacity(FENCES);
    {
        let _queue = state.queue_gate.lock().unwrap_or_else(|p| p.into_inner());
        for _ in 0..FENCES {
            // SAFETY: each fence is freshly allocated for one Completion and
            // all empty submissions follow the held queue dependency.
            let fence = unsafe {
                state
                    .device
                    .create_fence(&vk::FenceCreateInfo::default(), None)
            }
            .unwrap();
            let point = Completion::new(state.clone(), fence);
            // SAFETY: the gate serializes all raw queue submissions in this
            // fixture, and the marker fence belongs to this logical device.
            unsafe {
                state
                    .device
                    .queue_submit(state.graphics_queue, &[], point.fence())
            }
            .unwrap();
            sender
                .send(FenceJob {
                    completion: point.clone(),
                    work: Vec::new(),
                    sync: Vec::new(),
                    sparse: Vec::new(),
                })
                .unwrap();
            points.push(point);
        }
    }
    drop(sender);
    let worker = thread::spawn(move || {
        PROBE.with(|slot| *slot.borrow_mut() = Some(thread_probe));
        observe_fences(receiver, capture_sender);
        PROBE.with(|slot| *slot.borrow_mut() = None);
    });
    wait_until("first native batch query", || {
        probe.batch_queries.load(Ordering::Relaxed) != 0
    });
    assert!(probe.max_batch.load(Ordering::Relaxed) >= FENCES);
    assert_eq!(
        probe.status_queries.load(Ordering::Relaxed),
        0,
        "timeout batch must not issue per-fence status queries"
    );
    gate.release();
    block_on(FenceFuture::pending_many(points)).unwrap();
    worker.join().unwrap();
    // SAFETY: every held marker completed before this private timeline object
    // is destroyed, so no queue operation can retain its handle.
    unsafe { state.device.destroy_semaphore(gate.semaphore, None) };
    gate.released = true;
}

#[cfg(feature = "hardware-tests")]
struct CountWake(AtomicUsize);
#[cfg(feature = "hardware-tests")]
impl Wake for CountWake {
    fn wake(self: Arc<Self>) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }
    fn wake_by_ref(self: &Arc<Self>) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }
}

#[test]
#[cfg(feature = "hardware-tests")]
fn fence_future_replaces_and_removes_its_waker_registration() {
    let device = device();
    // SAFETY: the Completion is the only owner of this unsubmitted fence and
    // drops it before the typed device can be destroyed.
    let state = unsafe { device.native_ref() }.0.clone();
    let fence = unsafe {
        state
            .device
            .create_fence(&vk::FenceCreateInfo::default(), None)
    }
    .unwrap();
    let completion = Completion::new(state, fence);
    let old = Arc::new(CountWake(AtomicUsize::new(0)));
    let replacement = Arc::new(CountWake(AtomicUsize::new(0)));
    let old_waker = Waker::from(old.clone());
    let replacement_waker = Waker::from(replacement.clone());
    let mut future = FenceFuture::pending(completion.clone());
    assert!(matches!(
        Pin::new(&mut future).poll(&mut Context::from_waker(&old_waker)),
        Poll::Pending
    ));
    assert_eq!(completion.0.wait.lock().unwrap().wakers.len(), 1);
    assert!(matches!(
        Pin::new(&mut future).poll(&mut Context::from_waker(&replacement_waker)),
        Poll::Pending
    ));
    assert_eq!(completion.0.wait.lock().unwrap().wakers.len(), 1);
    drop(old_waker);
    assert_eq!(Arc::strong_count(&old), 1, "old waker must be released");
    drop(future);
    assert!(completion.0.wait.lock().unwrap().wakers.is_empty());
}
