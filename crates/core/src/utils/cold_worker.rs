//! Bounded worker for CPU-only cold object creation shared by native backends.
//!
//! The worker owns only `Send + 'static` CPU jobs. Backend modules retain
//! responsibility for choosing which native operations are safe to enqueue;
//! GPU command recording, submission, completion, and platform callbacks stay
//! outside this crate.

use std::{
    collections::BTreeMap,
    future::Future,
    panic::{AssertUnwindSafe, catch_unwind},
    pin::Pin,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc::{Receiver, SyncSender, TrySendError, sync_channel},
    },
    task::{Context, Poll, Waker},
    thread,
};

use crate::api::{Error, Result};

struct Job {
    run: Box<dyn FnOnce() + Send + 'static>,
    fail: Box<dyn FnOnce() + Send + 'static>,
}

/// One bounded worker lane. Backend modules normally keep one in a `OnceLock`.
#[derive(Clone)]
pub struct Worker {
    sender: SyncSender<Job>,
    admission: Arc<Mutex<Admission>>,
    stopped_error: &'static str,
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
    /// Starts a single sleeping worker with bounded queue admission.
    pub fn new(
        limit: usize,
        thread_name: &'static str,
        stopped_error: &'static str,
    ) -> std::io::Result<Self> {
        assert!(limit > 0, "cold worker queue must be non-empty");
        let (sender, receiver) = sync_channel(limit);
        let admission = Arc::new(Mutex::new(Admission {
            next_registration: 0,
            waiters: BTreeMap::new(),
        }));
        let wake_admission = admission.clone();
        thread::Builder::new()
            .name(thread_name.into())
            .spawn(move || run(receiver, wake_admission))?;
        Ok(Self {
            sender,
            admission,
            stopped_error,
        })
    }

    fn try_submit(
        &self,
        job: Job,
        waker: &Waker,
        registration: &mut Option<Registration>,
    ) -> std::result::Result<Option<Job>, Error> {
        let mut admission = self
            .admission
            .lock()
            .unwrap_or_else(|error| error.into_inner());
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
                Err(Error::Backend(self.stopped_error.into()))
            }
        }
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
            .unwrap_or_else(|error| error.into_inner())
            .waiters
            .remove(&registration.id);
    }
}

fn run(receiver: Receiver<Job>, admission: Arc<Mutex<Admission>>) {
    while let Ok(Job {
        run: operation,
        fail,
    }) = receiver.recv()
    {
        // A full observer either registered before this point or will observe
        // the newly available capacity on its next poll.
        let waiters = {
            let mut admission = admission.lock().unwrap_or_else(|error| error.into_inner());
            std::mem::take(&mut admission.waiters)
        };
        for (_, waker) in waiters {
            wake_caller(waker);
        }
        if catch_unwind(AssertUnwindSafe(operation)).is_err() {
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
        let mut inner = state
            .inner
            .lock()
            .unwrap_or_else(|error| error.into_inner());
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

/// Notifies an executor without letting a caller panic kill the worker lane.
fn wake_caller(waker: Waker) {
    let _ = catch_unwind(AssertUnwindSafe(|| waker.wake()));
}

/// Nonblocking future for one fully-owned CPU cold-path operation.
pub struct ColdFuture<T> {
    state: Arc<Shared<T>>,
    worker: Option<Worker>,
    pending_job: Option<Job>,
    registration: Option<Registration>,
    state_lost_error: String,
}

impl<T> ColdFuture<T> {
    fn ready(result: Result<T>, state_lost_error: String) -> Self {
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
            state_lost_error,
        }
    }
}

impl<T> Future for ColdFuture<T> {
    type Output = Result<T>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        {
            let mut inner = self
                .state
                .inner
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            if let Some(result) = inner.result.take() {
                return Poll::Ready(result);
            }
            // Shared with `finish` to close the result-before-waker race.
            inner.waker = Some(cx.waker().clone());
        }
        if let Some(job) = self.pending_job.take() {
            let Some(worker) = self.worker.as_ref().cloned() else {
                return Poll::Ready(Err(Error::Backend(self.state_lost_error.clone())));
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
        self.state.cancelled.store(true, Ordering::Release);
        cancel_registration(&mut self.registration);
    }
}

/// Creates a future whose first poll attempts nonblocking queue admission.
/// Worker startup is kept in backend-owned `OnceLock`s so each backend retains
/// its own thread name and error messages.
pub fn submit<T: Send + 'static>(
    worker: std::result::Result<Worker, String>,
    worker_label: &'static str,
    operation: impl FnOnce() -> Result<T> + Send + 'static,
) -> ColdFuture<T> {
    let worker = match worker {
        Ok(worker) => worker,
        Err(error) => {
            return ColdFuture::ready(
                Err(Error::Backend(format!(
                    "{worker_label} could not start: {error}"
                ))),
                format!("{worker_label} state lost"),
            );
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
                Err(Error::Backend(format!("{worker_label} job panicked"))),
            );
        }),
    };
    ColdFuture {
        state,
        worker: Some(worker),
        pending_job: Some(job),
        registration: None,
        state_lost_error: format!("{worker_label} state lost"),
    }
}

#[cfg(test)]
#[path = "../tests/utils/cold_worker.rs"]
mod tests;
