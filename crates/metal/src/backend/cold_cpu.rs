//! Bounded execution of CPU-only Metal shader translation.
//!
//! Jobs own source and numeric binding descriptions, never native encoders or
//! Objective-C descriptors. Idle workers block on channel receive; full
//! admission yields a cancellation-safe future rather than blocking a caller.
//! Native library and pipeline compilation uses Metal completion handlers.

use std::{
    collections::BTreeMap,
    future::Future,
    panic::{AssertUnwindSafe, catch_unwind},
    pin::Pin,
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicBool, Ordering},
        mpsc::{Receiver, SyncSender, TrySendError, sync_channel},
    },
    task::{Context, Poll, Waker},
    thread,
};

use fluxel_rhi_core::api::{Error, Result};

/// Bounds native compiler memory and driver work already admitted to the
/// worker. A full queue is normal backpressure: the Future retains its job
/// and remains pending, rather than rejecting a valid caller burst.
const QUEUE_LIMIT: usize = 16;

struct Job {
    run: Box<dyn FnOnce() + Send + 'static>,
    fail: Box<dyn FnOnce() + Send + 'static>,
}

#[derive(Clone)]
struct Worker {
    sender: SyncSender<Job>,
    /// Serializes observing a full channel with registration for its next
    /// capacity wake, preventing a lost wake between those two steps.
    admission: Arc<Mutex<Admission>>,
}

struct Admission {
    next_registration: u64,
    /// One replaceable current waker per Future, not an append-only list.
    waiters: BTreeMap<u64, Waker>,
}

struct Registration {
    id: u64,
    admission: Arc<Mutex<Admission>>,
}

impl Worker {
    fn new(limit: usize) -> std::result::Result<Self, String> {
        let (sender, receiver) = sync_channel(limit);
        let admission = Arc::new(Mutex::new(Admission {
            next_registration: 0,
            waiters: BTreeMap::new(),
        }));
        let wake_admission = admission.clone();
        thread::Builder::new()
            .name("fluxel-metal-shader-cpu".into())
            .spawn(move || run(receiver, wake_admission))
            .map_err(|error| error.to_string())?;
        Ok(Self { sender, admission })
    }

    fn try_submit(
        &self,
        job: Job,
        waker: &Waker,
        registration: &mut Option<Registration>,
    ) -> std::result::Result<Option<Job>, Error> {
        let mut admission = self.admission.lock().unwrap_or_else(|e| e.into_inner());
        match self.sender.try_send(job) {
            Ok(()) => {
                remove_registration(&mut admission, registration);
                Ok(None)
            }
            Err(TrySendError::Full(job)) => {
                let registration = registration.get_or_insert_with(|| {
                    let id = admission.next_registration;
                    admission.next_registration = admission.next_registration.wrapping_add(1);
                    Registration {
                        id,
                        admission: self.admission.clone(),
                    }
                });
                admission.waiters.insert(registration.id, waker.clone());
                Ok(Some(job))
            }
            Err(TrySendError::Disconnected(_)) => {
                remove_registration(&mut admission, registration);
                Err(Error::Backend("Metal CPU pipeline worker stopped".into()))
            }
        }
    }

    #[cfg(test)]
    fn waiter_count(&self) -> usize {
        self.admission
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .waiters
            .len()
    }
}

fn remove_registration(admission: &mut Admission, registration: &mut Option<Registration>) {
    if let Some(registration) = registration.take() {
        admission.waiters.remove(&registration.id);
    }
}

fn cancel_registration(registration: &mut Option<Registration>) {
    if let Some(registration) = registration.take() {
        registration
            .admission
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .waiters
            .remove(&registration.id);
    }
}

fn run(receiver: Receiver<Job>, admission: Arc<Mutex<Admission>>) {
    // `recv` sleeps while idle. There is no polling loop or GPU wait here.
    while let Ok(Job {
        run: operation,
        fail,
    }) = receiver.recv()
    {
        // Coordinate after every dequeue. A full observer either registers
        // before this point or sees the freed slot on its next poll.
        let waiters = {
            let mut admission = admission.lock().unwrap_or_else(|e| e.into_inner());
            std::mem::take(&mut admission.waiters)
        };
        for (_, waker) in waiters {
            wake_caller(waker);
        }
        if catch_unwind(AssertUnwindSafe(operation)).is_err() {
            // A panic must resolve the current future; otherwise it could
            // strand callers while this worker continues processing jobs.
            fail();
        }
    }
}

struct Inner<T> {
    result: Option<Result<T>>,
    waker: Option<Waker>,
}

struct Shared<T> {
    cancelled: AtomicBool,
    inner: Mutex<Inner<T>>,
}

impl<T> Shared<T> {
    fn pending() -> Self {
        Self {
            cancelled: AtomicBool::new(false),
            inner: Mutex::new(Inner {
                result: None,
                waker: None,
            }),
        }
    }
}

fn finish<T>(state: &Shared<T>, result: Result<T>) {
    if state.cancelled.load(Ordering::Acquire) {
        return;
    }
    let waker = {
        let mut inner = state.inner.lock().unwrap_or_else(|e| e.into_inner());
        if state.cancelled.load(Ordering::Acquire) {
            return;
        }
        inner.result = Some(result);
        inner.waker.take()
    };
    if let Some(waker) = waker {
        wake_caller(waker);
    }
}

/// Caller-supplied executor code is outside this worker's trust boundary. A
/// panicking waker cannot be allowed to terminate the sole cold worker.
fn wake_caller(waker: Waker) {
    let _ = catch_unwind(AssertUnwindSafe(|| waker.wake()));
}

/// Concrete future exposed only through Core's associated backend traits.
pub struct CpuFuture<T> {
    state: Arc<Shared<T>>,
    worker: Option<Worker>,
    pending_job: Option<Job>,
    registration: Option<Registration>,
}

impl<T> CpuFuture<T> {
    fn ready(result: Result<T>) -> Self {
        Self {
            state: Arc::new(Shared {
                cancelled: AtomicBool::new(false),
                inner: Mutex::new(Inner {
                    result: Some(result),
                    waker: None,
                }),
            }),
            worker: None,
            pending_job: None,
            registration: None,
        }
    }
}

impl<T> Future for CpuFuture<T> {
    type Output = Result<T>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        {
            let mut inner = self.state.inner.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(result) = inner.result.take() {
                return Poll::Ready(result);
            }
            // Sharing this lock with `finish` closes result-before-waker races.
            inner.waker = Some(cx.waker().clone());
        }
        if let Some(job) = self.pending_job.take() {
            let Some(worker) = self.worker.as_ref().cloned() else {
                return Poll::Ready(Err(Error::Backend(
                    "Metal CPU pipeline worker state lost".into(),
                )));
            };
            match worker.try_submit(job, cx.waker(), &mut self.registration) {
                Ok(None) => {}
                Ok(Some(job)) => self.pending_job = Some(job),
                Err(error) => return Poll::Ready(Err(error)),
            }
        }
        Poll::Pending
    }
}

impl<T> Drop for CpuFuture<T> {
    fn drop(&mut self) {
        // A native create already executing cannot be interrupted. Cancellation
        // suppresses queued execution/result retention and frees its waker slot.
        self.state.cancelled.store(true, Ordering::Release);
        cancel_registration(&mut self.registration);
    }
}

fn worker() -> std::result::Result<Worker, String> {
    static WORKER: OnceLock<std::result::Result<Worker, String>> = OnceLock::new();
    WORKER
        .get_or_init(|| Worker::new(QUEUE_LIMIT))
        .as_ref()
        .map(Clone::clone)
        .map_err(Clone::clone)
}

/// Snapshots one fully-owned cold operation. Construction never starts native
/// work; only polling attempts nonblocking worker admission.
pub(crate) fn spawn_cpu<T: Send + 'static>(
    operation: impl FnOnce() -> Result<T> + Send + 'static,
) -> CpuFuture<T> {
    let worker = match worker() {
        Ok(worker) => worker,
        Err(error) => {
            return CpuFuture::ready(Err(Error::Backend(format!(
                "Metal CPU pipeline worker could not start: {error}"
            ))));
        }
    };
    let state = Arc::new(Shared::pending());
    let run_state = state.clone();
    let fail_state = state.clone();
    let job = Job {
        run: Box::new(move || {
            if !run_state.cancelled.load(Ordering::Acquire) {
                finish(&run_state, operation());
            }
        }),
        fail: Box::new(move || {
            finish(
                &fail_state,
                Err(Error::Backend(
                    "Metal CPU pipeline worker job panicked".into(),
                )),
            )
        }),
    };
    CpuFuture {
        state,
        worker: Some(worker),
        pending_job: Some(job),
        registration: None,
    }
}
