//! Bounded worker for Vulkan shader translation and pipeline creation.
//!
//! Vulkan permits these host operations from a worker thread. Jobs own cloned
//! API descriptions and a `VulkanDevice`, so no Core borrow or encoder state
//! crosses the boundary. This is deliberately a cold-object factory, not a
//! GPU scheduler: it never records commands, submits work, or waits on GPU
//! completion.

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
            .name("fluxel-vulkan-pipeline".into())
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
                Err(Error::Backend("Vulkan pipeline worker stopped".into()))
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
pub struct ColdFuture<T> {
    state: Arc<Shared<T>>,
    worker: Option<Worker>,
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
            // Sharing this lock with `finish` closes result-before-waker races.
            inner.waker = Some(cx.waker().clone());
        }
        if let Some(job) = self.pending_job.take() {
            let Some(worker) = self.worker.as_ref().cloned() else {
                return Poll::Ready(Err(Error::Backend(
                    "Vulkan pipeline worker state lost".into(),
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

impl<T> Drop for ColdFuture<T> {
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
pub(crate) fn submit<T: Send + 'static>(
    operation: impl FnOnce() -> Result<T> + Send + 'static,
) -> ColdFuture<T> {
    let worker = match worker() {
        Ok(worker) => worker,
        Err(error) => {
            return ColdFuture::ready(Err(Error::Backend(format!(
                "Vulkan pipeline worker could not start: {error}"
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
                Err(Error::Backend("Vulkan pipeline worker job panicked".into())),
            )
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
    use crate::backend::{VulkanDevice, VulkanShader};
    use std::{
        sync::{Barrier, atomic::AtomicUsize},
        task::Wake,
    };

    fn poll_once<T>(future: &mut ColdFuture<T>) -> Poll<Result<T>> {
        let waker = Waker::noop();
        Pin::new(future).poll(&mut Context::from_waker(waker))
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

    struct ParkWake(thread::Thread);
    impl Wake for ParkWake {
        fn wake(self: Arc<Self>) {
            self.0.unpark();
        }
        fn wake_by_ref(self: &Arc<Self>) {
            self.0.unpark();
        }
    }

    struct PanicWake;
    impl Wake for PanicWake {
        fn wake(self: Arc<Self>) {
            panic!("fixture completion waker");
        }
    }

    #[test]
    fn full_queue_backpressures_and_dropped_waiter_is_removed() {
        let started = Arc::new(Barrier::new(2));
        let release = Arc::new(Barrier::new(2));
        let worker = Worker::new(1).unwrap();
        let s = started.clone();
        let r = release.clone();
        let mut running = submit_on(worker.clone(), move || {
            s.wait();
            r.wait();
            Ok(1usize)
        });
        assert!(matches!(poll_once(&mut running), Poll::Pending));
        started.wait();
        let mut queued = submit_on(worker.clone(), || Ok(2usize));
        assert!(matches!(poll_once(&mut queued), Poll::Pending));
        let old = Arc::new(CountWake(AtomicUsize::new(0)));
        let replacement = Arc::new(CountWake(AtomicUsize::new(0)));
        let old_waker = Waker::from(old.clone());
        let replacement_waker = Waker::from(replacement.clone());
        let mut blocked = submit_on(worker.clone(), || Ok(3usize));
        assert!(matches!(
            Pin::new(&mut blocked).poll(&mut Context::from_waker(&old_waker)),
            Poll::Pending
        ));
        assert_eq!(worker.waiter_count(), 1);
        assert!(matches!(
            Pin::new(&mut blocked).poll(&mut Context::from_waker(&replacement_waker)),
            Poll::Pending
        ));
        assert_eq!(worker.waiter_count(), 1, "one slot per Future");
        drop(old_waker);
        assert_eq!(
            Arc::strong_count(&old),
            1,
            "replaced waker must be released"
        );
        drop(blocked);
        assert_eq!(worker.waiter_count(), 0);
        release.wait();
    }

    #[test]
    fn worker_panic_resolves_result() {
        let worker = Worker::new(1).unwrap();
        let mut future = submit_on(worker, || -> Result<()> { panic!("fixture") });
        let waker = Waker::from(Arc::new(ParkWake(thread::current())));
        let mut context = Context::from_waker(&waker);
        assert!(matches!(
            Pin::new(&mut future).poll(&mut context),
            Poll::Pending
        ));
        for _ in 0..10 {
            thread::park_timeout(std::time::Duration::from_secs(1));
            if let Poll::Ready(result) = Pin::new(&mut future).poll(&mut context) {
                assert!(result.is_err());
                return;
            }
        }
        panic!("worker did not resolve panic");
    }

    #[test]
    fn panicking_caller_waker_does_not_kill_later_jobs() {
        let worker = Worker::new(2).unwrap();
        let (first_done, observed) = std::sync::mpsc::sync_channel(1);
        let mut first = submit_on(worker.clone(), move || {
            first_done.send(()).unwrap();
            Ok(1usize)
        });
        let panic_waker = Waker::from(Arc::new(PanicWake));
        assert!(matches!(
            Pin::new(&mut first).poll(&mut Context::from_waker(&panic_waker)),
            Poll::Pending
        ));
        observed
            .recv_timeout(std::time::Duration::from_secs(1))
            .unwrap();
        // The notification panic is isolated after `finish` stored the real
        // result, so it must not replace successful completion with an error.
        let mut first_completed = false;
        for _ in 0..10 {
            if let Poll::Ready(result) =
                Pin::new(&mut first).poll(&mut Context::from_waker(Waker::noop()))
            {
                assert_eq!(result.unwrap(), 1);
                first_completed = true;
                break;
            }
            thread::park_timeout(std::time::Duration::from_millis(1));
        }
        assert!(
            first_completed,
            "panicking wake prevented successful result"
        );

        let mut second = submit_on(worker, || Ok(2usize));
        let waker = Waker::from(Arc::new(ParkWake(thread::current())));
        let mut context = Context::from_waker(&waker);
        assert!(matches!(
            Pin::new(&mut second).poll(&mut context),
            Poll::Pending
        ));
        for _ in 0..10 {
            thread::park_timeout(std::time::Duration::from_secs(1));
            if let Poll::Ready(result) = Pin::new(&mut second).poll(&mut context) {
                assert_eq!(result.unwrap(), 2);
                return;
            }
        }
        panic!("later cold job was stranded by a panicking caller waker");
    }

    #[test]
    fn panicking_admission_waker_does_not_kill_later_jobs() {
        let started = Arc::new(Barrier::new(2));
        let release = Arc::new(Barrier::new(2));
        let worker = Worker::new(1).unwrap();
        let entered = started.clone();
        let unblock = release.clone();
        let mut running = submit_on(worker.clone(), move || {
            entered.wait();
            unblock.wait();
            Ok(1usize)
        });
        assert!(matches!(poll_once(&mut running), Poll::Pending));
        started.wait();
        let mut queued = submit_on(worker.clone(), || Ok(2usize));
        assert!(matches!(poll_once(&mut queued), Poll::Pending));
        let mut blocked = submit_on(worker, || Ok(3usize));
        let panic_waker = Waker::from(Arc::new(PanicWake));
        assert!(matches!(
            Pin::new(&mut blocked).poll(&mut Context::from_waker(&panic_waker)),
            Poll::Pending
        ));
        release.wait();

        let waker = Waker::from(Arc::new(ParkWake(thread::current())));
        let mut context = Context::from_waker(&waker);
        for _ in 0..10 {
            match Pin::new(&mut blocked).poll(&mut context) {
                Poll::Ready(result) => {
                    assert_eq!(result.unwrap(), 3);
                    return;
                }
                Poll::Pending => thread::park_timeout(std::time::Duration::from_secs(1)),
            }
        }
        panic!("admission wake panic stranded a later cold job");
    }

    #[test]
    fn dropped_queued_job_is_skipped_before_a_sentinel() {
        let started = Arc::new(Barrier::new(2));
        let release = Arc::new(Barrier::new(2));
        let worker = Worker::new(2).unwrap();
        let started_job = started.clone();
        let release_job = release.clone();
        let mut first = submit_on(worker.clone(), move || {
            started_job.wait();
            release_job.wait();
            Ok(())
        });
        assert!(matches!(poll_once(&mut first), Poll::Pending));
        started.wait();
        let ran = Arc::new(AtomicUsize::new(0));
        let observed = ran.clone();
        let mut cancelled = submit_on(worker.clone(), move || {
            observed.fetch_add(1, Ordering::Relaxed);
            Ok(())
        });
        assert!(matches!(poll_once(&mut cancelled), Poll::Pending));
        drop(cancelled);
        // FIFO completion proves the worker has passed the cancelled job;
        // merely yielding could pass before it was ever dequeued.
        let mut sentinel = submit_on(worker, || Ok(()));
        let waker = Waker::from(Arc::new(ParkWake(thread::current())));
        let mut context = Context::from_waker(&waker);
        assert!(matches!(
            Pin::new(&mut sentinel).poll(&mut context),
            Poll::Pending
        ));
        release.wait();
        for _ in 0..10 {
            thread::park_timeout(std::time::Duration::from_secs(1));
            if let Poll::Ready(result) = Pin::new(&mut sentinel).poll(&mut context) {
                result.unwrap();
                assert_eq!(ran.load(Ordering::Relaxed), 0);
                return;
            }
        }
        panic!("sentinel was not completed");
    }

    #[test]
    fn worker_boundary_requires_thread_safe_native_ownership() {
        fn assert_send_sync<T: Send + Sync>() {}
        fn assert_send<T: Send>() {}
        assert_send_sync::<VulkanDevice>();
        assert_send::<VulkanShader>();
        assert_send::<ColdFuture<VulkanShader>>();
    }

    fn submit_on<T: Send + 'static>(
        worker: Worker,
        operation: impl FnOnce() -> Result<T> + Send + 'static,
    ) -> ColdFuture<T> {
        let state = Arc::new(Shared::pending());
        let run_state = state.clone();
        let fail_state = state.clone();
        let job = Job {
            run: Box::new(move || {
                if !run_state.cancelled.load(Ordering::Acquire) {
                    finish(&run_state, operation())
                }
            }),
            fail: Box::new(move || finish(&fail_state, Err(Error::Backend("panic".into())))),
        };
        ColdFuture {
            state,
            worker: Some(worker),
            pending_job: Some(job),
            registration: None,
        }
    }
}
