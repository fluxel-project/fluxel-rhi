//! Owner-thread completion timing and spurious-poll evidence on native WGL.
use super::*;
use crate::backend::api::GlWaitBound;
use crate::{Gl, GlInstance};
use fluxel_host::{Window, WindowConfig};
use fluxel_rhi_core::api::{AdapterId, Device, DeviceRequest, Instance, InstanceId};
use std::{
    sync::Arc,
    task::{Wake, Waker},
    time::Instant,
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
fn block<F: Future>(future: F) -> F::Output {
    let waker = Waker::from(Arc::new(ThreadWake(std::thread::current())));
    let mut context = Context::from_waker(&waker);
    let mut future = std::pin::pin!(future);
    loop {
        match future.as_mut().poll(&mut context) {
            Poll::Ready(result) => return result,
            Poll::Pending => std::thread::park(),
        }
    }
}
fn open(window: &Window) -> Device<Gl> {
    // SAFETY: fixture retains this HWND and uses the resulting device on its owner thread.
    let native = unsafe { GlInstance::from_wgl_window(window, [32, 32]) }.unwrap();
    let id = InstanceId::allocate().unwrap();
    // SAFETY: the fresh identity belongs exclusively to this adopted native context.
    let instance = unsafe { Instance::<Gl>::from_native(id, native) };
    block(instance.open(DeviceRequest {
        adapter: AdapterId {
            instance: id,
            ordinal: 0,
        },
        required_features: vec![],
    }))
    .unwrap()
}
#[test]
fn native_retry_backoff_has_no_zero_or_unbounded_interval() {
    let mut delay = Duration::from_millis(1);
    let mut values = Vec::new();
    for _ in 0..8 {
        values.push(delay.as_millis());
        delay = next_retry_interval(delay);
    }
    assert_eq!(values, [1, 2, 4, 8, 8, 8, 8, 8]);
}
#[link(name = "kernel32")]
unsafe extern "system" {
    fn GetCurrentThread() -> *mut std::ffi::c_void;
    fn QueryThreadCycleTime(thread: *mut std::ffi::c_void, cycles: *mut u64) -> i32;
}
fn owner_cycles() -> u64 {
    let mut value = 0;
    // SAFETY: the pseudo-handle denotes this live test thread; output is writable.
    assert_ne!(
        unsafe { QueryThreadCycleTime(GetCurrentThread(), &mut value) },
        0
    );
    value
}
#[test]
#[ignore = "requires an explicit WGL hardware run"]
fn wgl_pending_retry_latency_and_spurious_polls() {
    let window = Window::new(WindowConfig::new("GL completion timing", 32, 32).unwrap()).unwrap();
    let device = open(&window);
    // SAFETY: immutable native borrowing is confined to this device's test owner thread.
    let native = unsafe { device.native_ref() };
    let point = GlCompletion::after_issued(native.0.clone()).unwrap();
    let mut spurious = NativeWaitFuture::new(point);
    spurious.retry = Some(Delay::new(Duration::from_secs(1)));
    for _ in 0..20 {
        assert!(
            Pin::new(&mut spurious)
                .poll(&mut Context::from_waker(Waker::noop()))
                .is_pending()
        );
    }
    assert_eq!(
        spurious.observations, 0,
        "a sleeping timer must not query GL on spurious polls"
    );
    drop(spurious);

    let mut wall: [Vec<u128>; 2] = Default::default();
    let mut cycles: [Vec<u64>; 2] = Default::default();
    for round in 0..16 {
        for policy in [round % 2, 1 - round % 2] {
            let point = GlCompletion::after_issued(native.0.clone()).unwrap();
            let GlCompletionEvidence::Native(fence) = &point.evidence else {
                unreachable!()
            };
            let lease = fence.borrow().unwrap();
            // A test-only explicit bounded wait ensures the fence is already
            // signaled. We measure observation *after a pending retry timer*,
            // not GPU duration or a renderer frame-rate improvement.
            assert_eq!(
                native
                    .0
                    .with_native("settle timing fixture", |provider| provider.wait_fence(
                        lease,
                        GlWaitBound {
                            nanoseconds: 1_000_000_000
                        }
                    ))
                    .unwrap(),
                GlFenceStatus::Complete
            );
            let mut wait = NativeWaitFuture::new(point);
            wait.retry = Some(Delay::new(Duration::from_millis(if policy == 0 {
                16
            } else {
                1
            })));
            let tick = Instant::now();
            let start = owner_cycles();
            block(&mut wait).unwrap();
            cycles[policy].push(owner_cycles() - start);
            wall[policy].push(tick.elapsed().as_micros());
            assert_eq!(wait.observations, 1);
        }
    }
    for policy in 0..2 {
        wall[policy].sort_unstable();
        cycles[policy].sort_unstable();
        println!(
            "completion-retry policy={} samples=16 median_us={} owner_thread_median_cycles={} native_observations=1 timer_thread_cpu_not_measured=true",
            if policy == 0 {
                "fixed16ms"
            } else {
                "adaptive-first1ms"
            },
            wall[policy][8],
            cycles[policy][8]
        );
    }
    assert!(
        wall[1][8] < wall[0][8],
        "short retry must improve observation latency on this runner"
    );
}
