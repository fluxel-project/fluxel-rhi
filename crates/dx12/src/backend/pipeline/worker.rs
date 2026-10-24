//! Bounded blocking worker for cold DX12 compilation and PSO creation.
//!
//! D3D12 accepts device and pipeline creation from worker threads. Callers
//! snapshot API descriptors before enqueueing, so no job borrows Core-owned
//! descriptor memory. This is only a cold-object factory: it neither records
//! commands nor schedules GPU work.

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

/// A single native compiler lane avoids unbounded simultaneous DXC/PSO memory
/// spikes. The bounded native queue limits work that has actually entered the
/// worker. Overflow remains a normal, nonblocking pending Future so valid
/// bursts do not acquire a new failure mode.
const QUEUE_LIMIT: usize = 16;

struct Job {
    run: Box<dyn FnOnce() + Send + 'static>,
    fail: Box<dyn FnOnce() + Send + 'static>,
}

#[derive(Clone)]
struct Worker {
    sender: SyncSender<Job>,
    /// Serializes `try_send(Full)` with capacity notification so a Future
    /// cannot miss the only wake after it observed a full queue.
    admission: Arc<Mutex<Admission>>,
}

struct Admission {
    next_registration: u64,
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
            .name("fluxel-dx12-pipeline".into())
            .spawn(move || run(receiver, wake_admission))
            .map_err(|error| error.to_string())?;
        Ok(Self { sender, admission })
    }

    /// Tries admission without waiting for worker capacity. The returned job
    /// remains owned by the Future when the queue is full.
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
                // Replace, rather than append, when an executor changes a
                // Future's waker. A dropped Future removes this exact slot.
                admission.waiters.insert(registration.id, waker.clone());
                Ok(Some(job))
            }
            Err(TrySendError::Disconnected(_)) => {
                remove_registration(&mut admission, registration);
                Err(Error::Backend("DX12 pipeline worker stopped".into()))
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
    // `recv` sleeps while idle. There is no timer, spin loop, or GPU wait in
    // this worker: jobs perform only the synchronous native cold path.
    while let Ok(job) = receiver.recv() {
        let Job { run, fail } = job;
        // Coordinate after each dequeue so every Future that observed Full
        // either registers before this point or sees the newly free slot.
        let waiters = {
            let mut admission = admission.lock().unwrap_or_else(|e| e.into_inner());
            std::mem::take(&mut admission.waiters)
        };
        for (_, waker) in waiters {
            waker.wake();
        }
        if catch_unwind(AssertUnwindSafe(run)).is_err() {
            // A panicking native/compiler call must resolve its own Future;
            // otherwise a worker panic would strand every queued caller.
            fail();
        }
    }
}

struct Inner<T> {
    result: Option<Result<T>>,
    waker: Option<Waker>,
}

impl<T> Default for Inner<T> {
    fn default() -> Self {
        Self {
            result: None,
            waker: None,
        }
    }
}

struct Shared<T> {
    cancelled: AtomicBool,
    inner: Mutex<Inner<T>>,
}

impl<T> Default for Shared<T> {
    fn default() -> Self {
        Self {
            cancelled: AtomicBool::new(false),
            inner: Mutex::new(Inner::default()),
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
        waker.wake();
    }
}

/// A nonblocking future for one owned cold-path result.
///
/// Public only because Core's public backend traits expose their concrete
/// associated future type; construction stays private to this backend.
pub struct ColdFuture<T> {
    state: Arc<Shared<T>>,
    worker: Option<Worker>,
    /// Kept here until a poll can admit it to the bounded worker queue.
    pending_job: Option<Job>,
    registration: Option<Registration>,
}

impl<T> ColdFuture<T> {
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

impl<T> Future for ColdFuture<T> {
    type Output = Result<T>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        {
            let mut inner = self.state.inner.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(result) = inner.result.take() {
                return Poll::Ready(result);
            }
            // Sharing this lock with `finish` closes the result-before-waker
            // race: a completed job either supplied the result above or sees
            // this waker and wakes the executor.
            inner.waker = Some(cx.waker().clone());
        }
        if let Some(job) = self.pending_job.take() {
            let worker = self
                .worker
                .as_ref()
                .expect("pending job has worker")
                .clone();
            match worker.try_submit(job, cx.waker(), &mut self.registration) {
                Ok(None) => {}
                Ok(Some(job)) => self.pending_job = Some(job),
                Err(error) => return Poll::Ready(Err(error)),
            }
        }
        Poll::Pending
    }
}

impl<T> Drop for ColdFuture<T> {
    fn drop(&mut self) {
        // Native compilation cannot be interrupted once it entered D3D12/DXC,
        // but queued work is skipped and a finished result is discarded. The
        // captured device/descriptors then release on the worker thread.
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

/// Creates a Future for one fully-owned DX12 cold-path operation. Its first
/// poll attempts queue admission; construction itself never starts compiling.
/// `T` and the closure are `Send` so borrowed Core descriptors and
/// thread-affine native state cannot cross this boundary.
pub(crate) fn submit<T: Send + 'static>(
    operation: impl FnOnce() -> Result<T> + Send + 'static,
) -> ColdFuture<T> {
    submit_on(worker(), operation)
}

fn submit_on<T: Send + 'static>(
    worker: std::result::Result<Worker, String>,
    operation: impl FnOnce() -> Result<T> + Send + 'static,
) -> ColdFuture<T> {
    let worker = match worker {
        Ok(worker) => worker,
        Err(error) => {
            return ColdFuture::ready(Err(Error::Backend(format!(
                "DX12 pipeline worker could not start: {error}"
            ))));
        }
    };
    let state = Arc::new(Shared::default());
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
                Err(Error::Backend("DX12 pipeline worker job panicked".into())),
            );
        }),
    };
    ColdFuture {
        state,
        worker: Some(worker),
        pending_job: Some(job),
        registration: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{Dx12Device, Dx12Shader};
    use std::{
        sync::{Arc, Barrier, atomic::AtomicUsize},
        task::{Context, Wake},
    };

    fn poll_once<T>(future: &mut ColdFuture<T>) -> Poll<Result<T>> {
        let waker = Waker::noop();
        let mut context = Context::from_waker(waker);
        Pin::new(future).poll(&mut context)
    }

    fn poll_with<T>(future: &mut ColdFuture<T>, waker: &Waker) -> Poll<Result<T>> {
        let mut context = Context::from_waker(waker);
        Pin::new(future).poll(&mut context)
    }

    struct CountWake(AtomicUsize);
    impl Wake for CountWake {
        fn wake(self: Arc<Self>) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
        fn wake_by_ref(self: &Arc<Self>) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[test]
    fn poll_is_nonblocking_until_worker_finishes() {
        let gate = Arc::new(Barrier::new(2));
        let wait = gate.clone();
        let worker = Worker::new(1).unwrap();
        let mut future = submit_on(Ok(worker), move || {
            wait.wait();
            Ok(7usize)
        });
        assert!(matches!(poll_once(&mut future), Poll::Pending));
        gate.wait();
        loop {
            if let Poll::Ready(result) = poll_once(&mut future) {
                assert_eq!(result.unwrap(), 7);
                break;
            }
            thread::yield_now();
        }
    }

    #[test]
    fn full_queue_uses_pending_backpressure_not_an_error() {
        let gate = Arc::new(Barrier::new(2));
        let wait = gate.clone();
        let worker = Worker::new(1).unwrap();
        let mut first = submit_on(Ok(worker.clone()), move || {
            wait.wait();
            Ok(1usize)
        });
        assert!(matches!(poll_once(&mut first), Poll::Pending));
        let mut second = submit_on(Ok(worker.clone()), || Ok(2usize));
        assert!(matches!(poll_once(&mut second), Poll::Pending));
        // The first job is running and the second occupies the one-slot queue,
        // so this third Future must retain its job locally under backpressure.
        let mut third = submit_on(Ok(worker), || Ok(3usize));
        assert!(matches!(poll_once(&mut third), Poll::Pending));
        gate.wait();
        loop {
            match poll_once(&mut third) {
                Poll::Ready(result) => {
                    assert_eq!(result.unwrap(), 3);
                    break;
                }
                Poll::Pending => thread::yield_now(),
            }
        }
    }

    #[test]
    fn full_queue_replaces_and_removes_each_future_registration() {
        let started = Arc::new(Barrier::new(2));
        let release = Arc::new(Barrier::new(2));
        let started_job = started.clone();
        let release_job = release.clone();
        let worker = Worker::new(1).unwrap();
        let mut running = submit_on(Ok(worker.clone()), move || {
            started_job.wait();
            release_job.wait();
            Ok(1usize)
        });
        assert!(matches!(poll_once(&mut running), Poll::Pending));
        started.wait();
        let mut queued = submit_on(Ok(worker.clone()), || Ok(2usize));
        assert!(matches!(poll_once(&mut queued), Poll::Pending));

        let old = Arc::new(CountWake(AtomicUsize::new(0)));
        let replacement = Arc::new(CountWake(AtomicUsize::new(0)));
        let old_waker = Waker::from(old.clone());
        let replacement_waker = Waker::from(replacement.clone());
        let mut cancelled = submit_on(Ok(worker.clone()), || Ok(3usize));
        assert!(matches!(
            poll_with(&mut cancelled, &old_waker),
            Poll::Pending
        ));
        assert_eq!(worker.waiter_count(), 1);
        assert!(matches!(
            poll_with(&mut cancelled, &replacement_waker),
            Poll::Pending
        ));
        assert_eq!(worker.waiter_count(), 1);
        drop(old_waker);
        assert_eq!(Arc::strong_count(&old), 1, "old waker must be released");
        drop(cancelled);
        assert_eq!(worker.waiter_count(), 0, "drop removes its slot");

        let live_count = Arc::new(CountWake(AtomicUsize::new(0)));
        let live_waker = Waker::from(live_count.clone());
        let mut live = submit_on(Ok(worker), || Ok(4usize));
        assert!(matches!(poll_with(&mut live, &live_waker), Poll::Pending));
        assert_eq!(live.worker.as_ref().unwrap().waiter_count(), 1);
        release.wait();
        for _ in 0..10_000 {
            if live_count.0.load(Ordering::Relaxed) != 0 {
                break;
            }
            thread::yield_now();
        }
        assert!(live_count.0.load(Ordering::Relaxed) != 0, "capacity wake");
        loop {
            match poll_with(&mut live, &live_waker) {
                Poll::Ready(result) => {
                    assert_eq!(result.unwrap(), 4);
                    break;
                }
                Poll::Pending => thread::yield_now(),
            }
        }
    }

    #[test]
    fn dropped_queued_job_is_not_run() {
        let started = Arc::new(Barrier::new(2));
        let release = Arc::new(Barrier::new(2));
        let started_job = started.clone();
        let release_job = release.clone();
        let worker = Worker::new(2).unwrap();
        let mut first = submit_on(Ok(worker.clone()), move || {
            started_job.wait();
            release_job.wait();
            Ok(())
        });
        assert!(matches!(poll_once(&mut first), Poll::Pending));
        started.wait();
        let ran = Arc::new(AtomicUsize::new(0));
        let observed = ran.clone();
        let mut cancelled = submit_on(Ok(worker.clone()), move || {
            observed.fetch_add(1, Ordering::Relaxed);
            Ok(())
        });
        assert!(matches!(poll_once(&mut cancelled), Poll::Pending));
        drop(cancelled);
        // FIFO completion of this sentinel proves the worker already took and
        // skipped the preceding cancelled job, rather than merely not having
        // been scheduled before a bounded-yield assertion.
        let mut sentinel = submit_on(Ok(worker), || Ok(()));
        assert!(matches!(poll_once(&mut sentinel), Poll::Pending));
        release.wait();
        drop(first);
        loop {
            match poll_once(&mut sentinel) {
                Poll::Ready(result) => {
                    result.unwrap();
                    break;
                }
                Poll::Pending => thread::yield_now(),
            }
        }
        assert_eq!(ran.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn worker_panic_resolves_its_future_as_an_error() {
        let worker = Worker::new(1).unwrap();
        let mut future = submit_on(Ok(worker), || -> Result<()> { panic!("fixture") });
        assert!(matches!(poll_once(&mut future), Poll::Pending));
        loop {
            match poll_once(&mut future) {
                Poll::Ready(result) => {
                    assert!(result.is_err());
                    break;
                }
                Poll::Pending => thread::yield_now(),
            }
        }
    }

    #[test]
    fn worker_boundary_accepts_only_thread_safe_native_ownership() {
        fn assert_send_sync<T: Send + Sync>() {}
        fn assert_send<T: Send>() {}

        assert_send_sync::<Dx12Device>();
        assert_send::<Dx12Shader>();
        assert_send::<ColdFuture<Dx12Shader>>();
    }
}
