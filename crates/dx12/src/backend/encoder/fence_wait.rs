//! A native fence event wakes the executor without blocking its polling thread.

use std::{
    ffi::c_void,
    future::Future,
    pin::Pin,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    task::{Context, Poll, Waker},
};

use fluxel_rhi_core::api::Error;
use windows::Win32::{
    Foundation::{CloseHandle, HANDLE},
    Graphics::Direct3D12::ID3D12Fence,
    System::Threading::{
        CloseThreadpoolWait, CreateEventW, CreateThreadpoolWait,
        DisassociateCurrentThreadFromCallback, PTP_CALLBACK_INSTANCE, PTP_WAIT, SetThreadpoolWait,
        WaitForThreadpoolWaitCallbacks,
    },
};

use super::{Dx12Device, native_error, submit::retire};

pub(super) fn fence_progress(completed: u64, target: u64) -> Result<bool, Error> {
    if completed == u64::MAX {
        Err(Error::DeviceLost)
    } else {
        Ok(completed >= target)
    }
}

struct WakeState {
    notified: AtomicBool,
    waker: Mutex<Option<Waker>>,
}

unsafe extern "system" fn wake_on_fence(
    callback: PTP_CALLBACK_INSTANCE,
    context: *mut c_void,
    _: PTP_WAIT,
    _: u32,
) {
    // The registration owns this Arc reference until callbacks have drained.
    let waker = {
        let state = unsafe { &*(context.cast::<WakeState>()) };
        state.notified.store(true, Ordering::Release);
        state.waker.lock().unwrap_or_else(|p| p.into_inner()).take()
    };
    // Waker::wake may synchronously finish and drop this future. Release the
    // waiter's cleanup gate before invoking user executor code; no context is
    // accessed after this point.
    unsafe { DisassociateCurrentThreadFromCallback(callback) };
    if let Some(waker) = waker {
        waker.wake();
    }
}

struct Registration {
    wait: PTP_WAIT,
    event: Option<HANDLE>,
    context: *const WakeState,
    state: Arc<WakeState>,
}

impl Drop for Registration {
    fn drop(&mut self) {
        unsafe {
            SetThreadpoolWait(self.wait, None, None);
            WaitForThreadpoolWaitCallbacks(self.wait, true);
            CloseThreadpoolWait(self.wait);
            drop(Arc::from_raw(self.context));
            if let Some(event) = self.event.take() {
                let _ = CloseHandle(event);
            }
        }
    }
}

// D3D12 has no API to cancel a SetEventOnCompletion registration. A dropped
// future transfers its event to the device until the fence reaches the target.
pub(crate) struct AbandonedFenceEvent {
    pub(super) value: u64,
    event: HANDLE,
}

// A HANDLE is an OS value; this type only transfers and closes it. It never
// dereferences the handle from another thread.
unsafe impl Send for AbandonedFenceEvent {}

impl Drop for AbandonedFenceEvent {
    fn drop(&mut self) {
        unsafe {
            let _ = CloseHandle(self.event);
        }
    }
}

pub struct Dx12FenceFuture {
    device: Dx12Device,
    fence: ID3D12Fence,
    value: u64,
    registration: Option<Registration>,
    immediate: Option<Result<(), Error>>,
}

impl Dx12FenceFuture {
    pub(super) fn immediate(
        device: Dx12Device,
        fence: ID3D12Fence,
        value: u64,
        result: Result<(), Error>,
    ) -> Self {
        Self {
            device,
            fence,
            value,
            registration: None,
            immediate: Some(result),
        }
    }

    pub(super) fn new(device: Dx12Device, fence: ID3D12Fence, value: u64) -> Self {
        let completed = unsafe { fence.GetCompletedValue() };
        match fence_progress(completed, value) {
            Err(error) => return Self::immediate(device, fence, value, Err(error)),
            Ok(true) => return Self::immediate(device, fence, value, Ok(())),
            Ok(false) => {}
        }
        let event = match unsafe { CreateEventW(None, false, false, None) } {
            Ok(event) => event,
            Err(error) => {
                return Self::immediate(device, fence, value, Err(native_error(error)));
            }
        };
        let state = Arc::new(WakeState {
            notified: AtomicBool::new(false),
            waker: Mutex::new(None),
        });
        let context = Arc::into_raw(state.clone());
        let wait = match unsafe {
            CreateThreadpoolWait(Some(wake_on_fence), Some(context.cast_mut().cast()), None)
        } {
            Ok(wait) => wait,
            Err(error) => {
                unsafe {
                    drop(Arc::from_raw(context));
                    let _ = CloseHandle(event);
                }
                return Self::immediate(device, fence, value, Err(native_error(error)));
            }
        };
        let registration = Registration {
            wait,
            event: Some(event),
            context,
            state,
        };
        unsafe { SetThreadpoolWait(wait, Some(event), None) };
        match unsafe { fence.SetEventOnCompletion(value, event) } {
            Ok(()) => Self {
                device,
                fence,
                value,
                registration: Some(registration),
                immediate: None,
            },
            Err(error) => Self::immediate(device, fence, value, Err(native_error(error))),
        }
    }
}

impl Drop for Dx12FenceFuture {
    fn drop(&mut self) {
        let mut registration = self.registration.take();
        let event = registration.as_mut().and_then(|entry| entry.event.take());
        // Close the threadpool wait before any path is allowed to close the
        // event, including a concurrent retire pass.
        drop(registration);
        if let Some(event) = event {
            let completed = unsafe { self.fence.GetCompletedValue() };
            if matches!(fence_progress(completed, self.value), Ok(false)) {
                self.device
                    .0
                    .abandoned_fence_events
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .push(AbandonedFenceEvent {
                        value: self.value,
                        event,
                    });
            } else {
                unsafe {
                    let _ = CloseHandle(event);
                }
            }
        }
        retire(&self.device);
    }
}

impl Future for Dx12FenceFuture {
    type Output = Result<(), Error>;

    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        if let Some(result) = &this.immediate {
            if result.is_ok() {
                retire(&this.device);
            }
            return Poll::Ready(result.clone());
        }
        let completed = unsafe { this.fence.GetCompletedValue() };
        match fence_progress(completed, this.value) {
            Err(error) => return Poll::Ready(Err(error)),
            Ok(true) => {
                retire(&this.device);
                return Poll::Ready(Ok(()));
            }
            Ok(false) => {}
        }
        let registration = this
            .registration
            .as_ref()
            .expect("pending fence has a wait");
        *registration
            .state
            .waker
            .lock()
            .unwrap_or_else(|p| p.into_inner()) = Some(context.waker().clone());
        if registration.state.notified.load(Ordering::Acquire) {
            let completed = unsafe { this.fence.GetCompletedValue() };
            if !fence_progress(completed, this.value)? {
                return Poll::Ready(Err(Error::Backend(
                    "DX12 fence event signaled before the target value".into(),
                )));
            }
            retire(&this.device);
            return Poll::Ready(Ok(()));
        }
        Poll::Pending
    }
}

#[cfg(test)]
mod tests {
    use super::fence_progress;
    use fluxel_rhi_core::api::Error;

    #[test]
    fn device_removed_is_never_completion() {
        assert_eq!(fence_progress(u64::MAX, 1), Err(Error::DeviceLost));
        assert_eq!(fence_progress(4, 5), Ok(false));
        assert_eq!(fence_progress(5, 5), Ok(true));
    }

    #[cfg(feature = "hardware-tests")]
    #[test]
    fn native_wait_yields_wakes_and_can_be_cancelled() {
        use super::Dx12FenceFuture;
        use crate::backend::{Dx12, Dx12Instance};
        use fluxel_rhi_core::api::{
            AdapterId, CompletionBackend, DeviceId, DeviceRequest, InstanceBackend, InstanceId,
        };
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
        assert_eq!(unsafe { device.0.fence.GetCompletedValue() }, 18);
        assert!(device.0.abandoned_fence_events.lock().unwrap().is_empty());
    }
}
