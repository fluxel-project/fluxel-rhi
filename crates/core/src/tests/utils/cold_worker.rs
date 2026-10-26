//! Unit tests for the parent module.

use super::*;
use std::{
    sync::{Arc, Barrier, atomic::AtomicUsize},
    task::Wake,
};

fn worker(limit: usize) -> Worker {
    Worker::new(limit, "fluxel-test-cold-worker", "test worker stopped").unwrap()
}

fn poll_with<T>(future: &mut ColdFuture<T>, waker: &Waker) -> Poll<Result<T>> {
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

struct PanicWake;
impl Wake for PanicWake {
    fn wake(self: Arc<Self>) {
        panic!("fixture caller waker");
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

fn submit_on<T: Send + 'static>(
    worker: Worker,
    operation: impl FnOnce() -> Result<T> + Send + 'static,
) -> ColdFuture<T> {
    submit(Ok(worker), "test worker", operation)
}

#[test]
fn bounded_queue_replaces_and_cancels_waker_registrations() {
    let started = Arc::new(Barrier::new(2));
    let release = Arc::new(Barrier::new(2));
    let started_job = started.clone();
    let release_job = release.clone();
    let worker = worker(1);
    let mut running = submit_on(worker.clone(), move || {
        started_job.wait();
        release_job.wait();
        Ok(1usize)
    });
    assert!(matches!(
        poll_with(&mut running, Waker::noop()),
        Poll::Pending
    ));
    started.wait();
    let mut queued = submit_on(worker.clone(), || Ok(2usize));
    assert!(matches!(
        poll_with(&mut queued, Waker::noop()),
        Poll::Pending
    ));
    let old = Arc::new(CountWake(AtomicUsize::new(0)));
    let replacement = Arc::new(CountWake(AtomicUsize::new(0)));
    let old_waker = Waker::from(old.clone());
    let new_waker = Waker::from(replacement);
    let mut blocked = submit_on(worker.clone(), || Ok(3usize));
    assert!(matches!(poll_with(&mut blocked, &old_waker), Poll::Pending));
    assert!(matches!(poll_with(&mut blocked, &new_waker), Poll::Pending));
    assert_eq!(worker.admission.lock().unwrap().waiters.len(), 1);
    drop(old_waker);
    assert_eq!(Arc::strong_count(&old), 1);
    drop(blocked);
    assert!(worker.admission.lock().unwrap().waiters.is_empty());
    release.wait();
}

#[test]
fn dropped_queued_job_is_skipped_and_panic_does_not_stop_worker() {
    fn assert_send<T: Send>() {}
    assert_send::<ColdFuture<usize>>();

    let worker = worker(2);
    let started = Arc::new(Barrier::new(2));
    let release = Arc::new(Barrier::new(2));
    let started_job = started.clone();
    let release_job = release.clone();
    let mut first = submit_on(worker.clone(), move || {
        started_job.wait();
        release_job.wait();
        Ok(())
    });
    assert!(matches!(
        poll_with(&mut first, Waker::noop()),
        Poll::Pending
    ));
    started.wait();
    let ran = Arc::new(AtomicUsize::new(0));
    let observed = ran.clone();
    let mut cancelled = submit_on(worker.clone(), move || {
        observed.fetch_add(1, Ordering::Relaxed);
        Ok(())
    });
    assert!(matches!(
        poll_with(&mut cancelled, Waker::noop()),
        Poll::Pending
    ));
    drop(cancelled);
    let mut panic_job = submit_on(worker.clone(), || -> Result<()> {
        panic!("fixture");
    });
    let wake = Waker::from(Arc::new(ParkWake(thread::current())));
    assert!(matches!(poll_with(&mut panic_job, &wake), Poll::Pending));
    release.wait();
    drop(first);
    let mut sentinel = submit_on(worker.clone(), || Ok(()));
    assert!(matches!(poll_with(&mut sentinel, &wake), Poll::Pending));
    for _ in 0..10 {
        thread::park_timeout(std::time::Duration::from_millis(100));
        if let Poll::Ready(result) = poll_with(&mut sentinel, &wake) {
            result.unwrap();
            assert_eq!(ran.load(Ordering::Relaxed), 0);
            assert!(matches!(
                poll_with(&mut panic_job, Waker::noop()),
                Poll::Ready(Err(_))
            ));
            return;
        }
    }
    panic!("worker did not complete sentinel after panic");
}

#[test]
fn panicking_completion_and_admission_wakers_are_isolated() {
    let worker = worker(2);
    let mut first = submit_on(worker.clone(), || Ok(7usize));
    assert!(matches!(
        poll_with(&mut first, &Waker::from(Arc::new(PanicWake))),
        Poll::Pending
    ));
    let ready_wake = Waker::from(Arc::new(ParkWake(thread::current())));
    let mut ready = false;
    for _ in 0..10 {
        thread::park_timeout(std::time::Duration::from_millis(100));
        if let Poll::Ready(result) = poll_with(&mut first, &ready_wake) {
            assert_eq!(result.unwrap(), 7);
            ready = true;
            break;
        }
    }
    assert!(ready);

    let gate = Arc::new(Barrier::new(2));
    let release = Arc::new(Barrier::new(2));
    let gate_job = gate.clone();
    let release_job = release.clone();
    let mut running = submit_on(worker.clone(), move || {
        gate_job.wait();
        release_job.wait();
        Ok(())
    });
    assert!(matches!(
        poll_with(&mut running, Waker::noop()),
        Poll::Pending
    ));
    gate.wait();
    let mut queued = submit_on(worker.clone(), || Ok(()));
    assert!(matches!(
        poll_with(&mut queued, Waker::noop()),
        Poll::Pending
    ));
    let mut blocked = submit_on(worker, || Ok(9usize));
    assert!(matches!(
        poll_with(&mut blocked, &Waker::from(Arc::new(PanicWake))),
        Poll::Pending
    ));
    release.wait();
    for _ in 0..10 {
        thread::park_timeout(std::time::Duration::from_millis(100));
        if let Poll::Ready(result) = poll_with(&mut blocked, &ready_wake) {
            assert_eq!(result.unwrap(), 9);
            return;
        }
    }
    panic!("admission waker panic stranded a queued job");
}
