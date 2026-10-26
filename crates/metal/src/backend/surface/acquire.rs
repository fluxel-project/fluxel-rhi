//! Asynchronous, cancellable native `CAMetalLayer` drawable acquisition.
//!
//! This module owns the only blocking call in Metal Surface acquisition. The
//! parent surface module owns configuration and epochs; requests carry a
//! snapshot of that epoch and are rejected when it no longer matches.

use super::{MetalSurface, MetalSurfaceImage};
use fluxel_rhi_core::api;
use std::{
    future::Future,
    marker::PhantomData,
    pin::Pin,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc::{Receiver, SyncSender, TrySendError, sync_channel},
    },
    task::{Context, Poll, Waker},
};

/// One retained Core Animation layer moved to the acquisition worker.
///
/// `CAMetalLayer` is a Core Animation layer and can be used off its creator
/// thread. `raw-window-metal` makes the same narrowly-scoped assertion for
/// this exact type, citing Apple's `CATransaction::lock` documentation. The
/// worker never touches the host view: it only invokes `nextDrawable` on this
/// retained layer. Layer configuration remains on the caller thread.
struct WorkerLayer(objc2::rc::Retained<objc2_quartz_core::CAMetalLayer>);

unsafe impl Send for WorkerLayer {}

/// A drawable and texture transferred once from worker to Future executor.
///
/// Metal drawables are designed for command encoding and presentation from
/// arbitrary Metal submission threads. This bridge is `Send`, never `Sync`,
/// and is private to the acquisition worker.
struct WorkerDrawable {
    drawable:
        objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_quartz_core::CAMetalDrawable>>,
    texture: objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLTexture>>,
}

unsafe impl Send for WorkerDrawable {}

struct AcquireTask {
    generation: u64,
    surface_generation: Arc<Mutex<u64>>,
    cancelled: AtomicBool,
    result: Mutex<Option<api::Result<WorkerDrawable>>>,
    waker: Mutex<Option<Waker>>,
}

impl AcquireTask {
    fn new(generation: u64, surface_generation: Arc<Mutex<u64>>) -> Self {
        Self {
            generation,
            surface_generation,
            cancelled: AtomicBool::new(false),
            result: Mutex::new(None),
            waker: Mutex::new(None),
        }
    }

    fn cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }

    fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
        self.waker
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take();
    }

    fn complete(&self, result: api::Result<WorkerDrawable>) {
        if self.cancelled() {
            return;
        }
        let mut output = self
            .result
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if self.cancelled() {
            return;
        }
        *output = Some(result);
        drop(output);
        let waker = self
            .waker
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take();
        if let Some(waker) = waker {
            // A custom executor waker is user-provided code. A panic must not
            // terminate the sole native-acquire worker and strand later jobs.
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| waker.wake()));
        }
    }
}

struct AcquireJob {
    layer: WorkerLayer,
    task: Arc<AcquireTask>,
}

/// A single blocking `nextDrawable` lane plus one queued request.
///
/// The queue is deliberately bounded. A cancelled active call cannot be
/// interrupted by Core Animation, but it cannot retain a lease; cancellation
/// of a queued call is observed before `nextDrawable` is invoked.
pub(super) struct AcquireWorker {
    sender: SyncSender<AcquireJob>,
    admission_wakers: Arc<Mutex<Vec<(Arc<AcquireTask>, Waker)>>>,
}

enum EnqueueError {
    Full(AcquireJob),
    Disconnected,
}

impl AcquireWorker {
    pub(super) fn new() -> api::Result<Arc<Self>> {
        let (sender, receiver) = sync_channel(1);
        let admission_wakers = Arc::new(Mutex::new(Vec::new()));
        let worker = Arc::new(Self {
            sender,
            admission_wakers: admission_wakers.clone(),
        });
        std::thread::Builder::new()
            .name("fluxel-metal-drawable".into())
            .spawn(move || Self::run(receiver, admission_wakers))
            .map_err(|error| {
                api::Error::Backend(format!("failed to start Metal drawable worker: {error}"))
            })?;
        Ok(worker)
    }

    fn run(
        receiver: Receiver<AcquireJob>,
        admission_wakers: Arc<Mutex<Vec<(Arc<AcquireTask>, Waker)>>>,
    ) {
        while let Ok(job) = receiver.recv() {
            let wakers = std::mem::take(
                &mut *admission_wakers
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner()),
            );
            for (_, waker) in wakers {
                let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| waker.wake()));
            }

            // This thread is created by Rust, so it has no ambient Foundation
            // autorelease pool. Keep every job (including a cancelled one) in
            // a short-lived pool; only explicitly retained native handles can
            // leave it through `WorkerDrawable`.
            objc2::rc::autoreleasepool(|_| {
                if job.task.cancelled() {
                    return;
                }
                use objc2_quartz_core::CAMetalDrawable as _;
                let result = match job.layer.0.nextDrawable() {
                    Some(drawable) => {
                        let current = *job
                            .task
                            .surface_generation
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner());
                        if current != job.task.generation {
                            Err(api::Error::SurfaceOutdated)
                        } else {
                            Ok(WorkerDrawable {
                                texture: drawable.texture(),
                                drawable,
                            })
                        }
                    }
                    None => Err(api::Error::SurfaceOutdated),
                };
                job.task.complete(result);
            });
        }
    }

    fn enqueue(&self, job: AcquireJob, waker: &Waker) -> Result<(), EnqueueError> {
        // Keep this lock across `try_send`: a receiver that frees the bounded
        // slot either observes this waker or leaves enough room for this send.
        // That closes the otherwise lost wake between `Full` and registration.
        let mut admission_wakers = self
            .admission_wakers
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let task = job.task.clone();
        if let Some((_, registered)) = admission_wakers
            .iter_mut()
            .find(|(registered, _)| Arc::ptr_eq(registered, &task))
        {
            *registered = waker.clone();
        } else {
            admission_wakers.push((task.clone(), waker.clone()));
        }
        match self.sender.try_send(job) {
            Ok(()) => {
                admission_wakers.retain(|(registered, _)| !Arc::ptr_eq(registered, &task));
                Ok(())
            }
            Err(TrySendError::Full(job)) => Err(EnqueueError::Full(job)),
            Err(TrySendError::Disconnected(_)) => {
                admission_wakers.retain(|(registered, _)| !Arc::ptr_eq(registered, &task));
                Err(EnqueueError::Disconnected)
            }
        }
    }

    fn cancel_admission(&self, task: &Arc<AcquireTask>) {
        self.admission_wakers
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .retain(|(registered, _)| !Arc::ptr_eq(registered, task));
    }
}

/// The Metal drawable-acquisition operation exposed through the public
/// `SurfaceBackend` associated type.
pub struct MetalAcquireFuture<'a> {
    worker: Arc<AcquireWorker>,
    task: Arc<AcquireTask>,
    pending_job: Option<AcquireJob>,
    // The native Surface remains exclusively borrowed until this future is
    // ready or cancelled, preventing a configure call from racing a delivered
    // lease. The worker itself owns all native values it needs.
    _surface: PhantomData<&'a mut MetalSurface>,
}

/// Creates an acquire Future without touching `nextDrawable`.
pub(super) fn begin<'a>(
    worker: Arc<AcquireWorker>,
    layer: objc2::rc::Retained<objc2_quartz_core::CAMetalLayer>,
    generation: u64,
    surface_generation: Arc<Mutex<u64>>,
) -> MetalAcquireFuture<'a> {
    let task = Arc::new(AcquireTask::new(generation, surface_generation));
    let pending_job = Some(AcquireJob {
        layer: WorkerLayer(layer),
        task: task.clone(),
    });
    MetalAcquireFuture {
        worker,
        task,
        pending_job,
        _surface: PhantomData,
    }
}

/// Creates an already-failed acquire Future without scheduling native work.
pub(super) fn failed<'a>(
    worker: Arc<AcquireWorker>,
    surface_generation: Arc<Mutex<u64>>,
    error: api::Error,
) -> MetalAcquireFuture<'a> {
    let task = Arc::new(AcquireTask::new(0, surface_generation));
    task.complete(Err(error));
    MetalAcquireFuture {
        worker,
        task,
        pending_job: None,
        _surface: PhantomData,
    }
}

impl Future for MetalAcquireFuture<'_> {
    type Output = api::Result<MetalSurfaceImage>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if let Some(job) = self.pending_job.take() {
            match self.worker.enqueue(job, cx.waker()) {
                Ok(()) => {}
                Err(EnqueueError::Full(job)) => {
                    self.pending_job = Some(job);
                    return Poll::Pending;
                }
                Err(EnqueueError::Disconnected) => {
                    return Poll::Ready(Err(api::Error::Backend(
                        "Metal drawable worker terminated".into(),
                    )));
                }
            }
        }

        let mut result = self
            .task
            .result
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(result) = result.take() {
            // A backend Surface clone may have reconfigured after the worker
            // published its result. Recheck immediately before minting a lease.
            if *self
                .task
                .surface_generation
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                != self.task.generation
            {
                return Poll::Ready(Err(api::Error::SurfaceOutdated));
            }
            return Poll::Ready(result.map(|drawable| MetalSurfaceImage {
                generation: self.task.generation,
                surface_generation: self.task.surface_generation.clone(),
                handed_off: false,
                drawable: drawable.drawable,
                texture: drawable.texture,
            }));
        }
        *self
            .task
            .waker
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(cx.waker().clone());
        drop(result);
        Poll::Pending
    }
}

impl Drop for MetalAcquireFuture<'_> {
    fn drop(&mut self) {
        // A request still outside the bounded channel owns no native work;
        // remove its admission waker immediately rather than retaining a
        // cancelled Future until another request is dequeued.
        self.worker.cancel_admission(&self.task);
        self.task.cancel();
    }
}
