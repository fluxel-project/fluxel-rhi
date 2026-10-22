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
        // SAFETY: `Registration` converts its owning Arc into this exact raw
        // pointer and drains every thread-pool callback before recovering it.
        // Therefore this callback observes a live WakeState for its duration.
        let state = unsafe { &*(context.cast::<WakeState>()) };
        state.notified.store(true, Ordering::Release);
        state.waker.lock().unwrap_or_else(|p| p.into_inner()).take()
    };
    // Waker::wake may synchronously finish and drop this future. Release the
    // waiter's cleanup gate before invoking user executor code; no context is
    // accessed after this point.
    // SAFETY: this is the thread-pool callback instance supplied by Windows;
    // disassociating it before wake prevents Drop from deadlocking on itself.
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
        // SAFETY: this Registration exclusively owns `wait`, its event, and
        // the Arc raw pointer. Cancelling and draining callbacks first proves
        // no callback can read `context` before it is converted back to Arc.
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
    pub(super) fence: ID3D12Fence,
    event: HANDLE,
}

// A HANDLE is an OS value; this type only transfers and closes it. It never
// dereferences the handle from another thread.
// SAFETY: the event HANDLE is transferred as an owned OS value. No thread
// dereferences it, and retirement closes it only after fence observation.
unsafe impl Send for AbandonedFenceEvent {}

impl Drop for AbandonedFenceEvent {
    fn drop(&mut self) {
        // SAFETY: this type owns the event HANDLE after the future transfers
        // it; retirement only drops it once its fence can no longer signal it.
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

// D3D12 fence/device interfaces are free-threaded. Registration owns the
// thread-pool wait and drains its callback before either native handle or the
// callback context can be released, so moving a pending future only changes
// which executor thread polls or drops it.
// SAFETY: D3D12 fence/device interfaces are free-threaded; registration Drop
// drains the callback before its context or HANDLE can be released.
unsafe impl Send for Dx12FenceFuture {}

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
        // SAFETY: this future owns a cloned fence COM interface; querying it
        // is side-effect free and `fence_progress` handles device removal.
        let completed = unsafe { fence.GetCompletedValue() };
        match fence_progress(completed, value) {
            Err(error) => return Self::immediate(device, fence, value, Err(error)),
            Ok(true) => return Self::immediate(device, fence, value, Ok(())),
            Ok(false) => {}
        }
        // SAFETY: no borrowed pointers are supplied; the returned event HANDLE
        // is owned by this future/registration on every success path.
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
        // SAFETY: `context` was produced by Arc::into_raw and remains owned by
        // Registration until callback drain; callback ABI matches wake_on_fence.
        let wait = match unsafe {
            CreateThreadpoolWait(Some(wake_on_fence), Some(context.cast_mut().cast()), None)
        } {
            Ok(wait) => wait,
            Err(error) => {
                // SAFETY: threadpool wait was not created, so this path solely
                // recovers the raw Arc once and closes the still-owned event.
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
        // SAFETY: Registration owns wait/event and callback context; Windows
        // borrows the event until registration Drop cancels and drains the wait.
        unsafe { SetThreadpoolWait(wait, Some(event), None) };
        // SAFETY: fence is retained by this future and event lifetime is
        // extended by either the registration or abandoned-event device store.
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
            // SAFETY: the future owns this cloned fence; query is read-only.
            let completed = unsafe { self.fence.GetCompletedValue() };
            if matches!(fence_progress(completed, self.value), Ok(false)) {
                self.device
                    .0
                    .abandoned_fence_events
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .push(AbandonedFenceEvent {
                        value: self.value,
                        fence: self.fence.clone(),
                        event,
                    });
            } else {
                // SAFETY: completion already proves Windows will not signal
                // this event again; this future still owns the HANDLE exactly once.
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
        // SAFETY: `this` owns a cloned fence COM interface and this read has no
        // queue-side effect.
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
            // SAFETY: as above, the future retains the fence for this poll.
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
}
