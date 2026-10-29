//! Fence completion, wakeup, cancellation, and device-loss regressions.

use super::{Dx12FenceFuture, fence_progress};
use fluxel_rhi_core::api::Error;

#[test]
fn device_removed_is_never_completion() {
    assert_eq!(fence_progress(u64::MAX, 1), Err(Error::DeviceLost));
    assert_eq!(fence_progress(4, 5), Ok(false));
    assert_eq!(fence_progress(5, 5), Ok(true));
}

#[test]
fn fence_future_is_send() {
    fn assert_send<T: Send>() {}
    assert_send::<Dx12FenceFuture>();
}

#[cfg(feature = "hardware-tests")]
#[test]
fn native_wait_yields_wakes_and_can_be_cancelled() {
    use crate::backend::{Dx12, Dx12Instance};
    use fluxel_rhi_core::api::{AdapterId, DeviceId, DeviceRequest, InstanceId};
    use fluxel_rhi_core::backend::{CompletionBackend, InstanceBackend};
    use std::{
        future::Future,
        pin::Pin,
        sync::{Arc, Mutex, atomic::Ordering},
        task::{Context, Poll, Wake, Waker},
        thread,
        time::{Duration, Instant},
    };

    struct Notify(thread::Thread);
    impl Wake for Notify {
        fn wake(self: Arc<Self>) {
            self.0.unpark();
        }
        fn wake_by_ref(self: &Arc<Self>) {
            self.0.unpark();
        }
    }

    let instance = Dx12Instance::new().unwrap();
    let request = DeviceRequest {
        adapter: AdapterId {
            instance: InstanceId::allocate().unwrap(),
            ordinal: 0,
        },
        required_features: Vec::new(),
    };
    let mut open =
        <Dx12 as InstanceBackend>::open(&instance, DeviceId::allocate().unwrap(), &request);
    let waker = Waker::from(Arc::new(Notify(thread::current())));
    let mut context = Context::from_waker(&waker);
    let (device, _) = match Pin::new(&mut open).poll(&mut context) {
        Poll::Ready(Ok(value)) => value,
        _ => panic!("DX12 device creation did not complete"),
    };
    let mut future = Dx12FenceFuture::new(device.clone(), device.0.fence.clone(), 1);
    let start = Instant::now();
    assert!(matches!(
        Pin::new(&mut future).poll(&mut context),
        Poll::Pending
    ));
    assert!(
        start.elapsed() < Duration::from_secs(1),
        "poll blocked the caller"
    );

    {
        let _queue = device.0.queue_lock.lock().unwrap();
        let value = device.0.next_fence.fetch_add(1, Ordering::Relaxed);
        assert_eq!(value, 1);
        // SAFETY: test holds the same queue lock used in production, and
        // the test device owns both queue and fence interfaces.
        unsafe { device.0.queue.Signal(&device.0.fence, value) }.unwrap();
    }
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        match Pin::new(&mut future).poll(&mut context) {
            Poll::Ready(Ok(())) => break,
            Poll::Ready(Err(error)) => panic!("fence wait failed: {error:?}"),
            Poll::Pending => {
                assert!(Instant::now() < deadline, "fence waker did not run");
                thread::park_timeout(Duration::from_millis(100));
            }
        }
    }
    drop(future);

    let mut idle = <Dx12 as CompletionBackend>::wait_idle(&device);
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        match Pin::new(&mut idle).poll(&mut context) {
            Poll::Ready(Ok(())) => break,
            Poll::Ready(Err(error)) => panic!("idle wait failed: {error:?}"),
            Poll::Pending => {
                assert!(Instant::now() < deadline, "idle waker did not run");
                thread::park_timeout(Duration::from_millis(100));
            }
        }
    }
    drop(idle);

    let mut cancelled = Dx12FenceFuture::new(device.clone(), device.0.fence.clone(), 3);
    assert!(matches!(
        Pin::new(&mut cancelled).poll(&mut context),
        Poll::Pending
    ));
    let start = Instant::now();
    drop(cancelled);
    assert!(
        start.elapsed() < Duration::from_secs(1),
        "cancellation blocked"
    );
    assert_eq!(device.0.abandoned_fence_events.lock().unwrap().len(), 1);

    let values = Mutex::new(Vec::new());
    thread::scope(|scope| {
        let workers: Vec<_> = (0..16)
            .map(|_| {
                scope.spawn(|| {
                    let mut future = <Dx12 as CompletionBackend>::wait_idle(&device);
                    values.lock().unwrap().push(future.value);
                    let waker = Waker::from(Arc::new(Notify(thread::current())));
                    let mut context = Context::from_waker(&waker);
                    let deadline = Instant::now() + Duration::from_secs(5);
                    loop {
                        match Pin::new(&mut future).poll(&mut context) {
                            Poll::Ready(Ok(())) => break,
                            Poll::Ready(Err(error)) => {
                                panic!("concurrent idle failed: {error:?}")
                            }
                            Poll::Pending => {
                                assert!(Instant::now() < deadline, "concurrent wait timed out");
                                thread::park_timeout(Duration::from_millis(100));
                            }
                        }
                    }
                })
            })
            .collect();
        for worker in workers {
            worker.join().unwrap();
        }
    });
    let mut issued = values.into_inner().unwrap();
    issued.sort_unstable();
    assert_eq!(issued, (3..19).collect::<Vec<_>>());
    // SAFETY: the test device retains the fence; this final query is read-only.
    assert_eq!(unsafe { device.0.fence.GetCompletedValue() }, 18);
    assert!(device.0.abandoned_fence_events.lock().unwrap().is_empty());
}
