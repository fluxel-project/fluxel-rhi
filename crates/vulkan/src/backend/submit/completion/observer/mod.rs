//! Batched native-fence observation and post-submit owner retirement.
//!
//! No GPU work is submitted here. Idle observation blocks on job arrival;
//! active observation uses bounded, zero-time native fence checks by device.

use super::{Completion, RecordedWork, vk_error};
use ash::vk;
use fluxel_rhi_core::api::{Error, Result};
use std::{
    collections::HashMap,
    sync::{
        Arc, OnceLock,
        mpsc::{self, Receiver, RecvTimeoutError, Sender, SyncSender},
    },
    time::Duration,
};

/// One shared observer for submitted Vulkan fences.
///
/// It owns only fences and already-recorded native work.  In particular it
/// never chooses queue order, submits commands, or interprets recorded work.
/// A dropped `FenceFuture` leaves its `Completion` in this queue, so command
/// pools and upload allocations still remain alive until the GPU is finished.
pub(in crate::backend::submit) struct FenceWaiter {
    sender: Sender<FenceJob>,
}

struct FenceJob {
    completion: Completion,
    work: Vec<RecordedWork>,
    sync: Vec<Arc<super::super::multi_queue::Timeline>>,
    sparse: Vec<super::super::super::sparse::PreparedSparseBind>,
}
static FENCE_WAITER: OnceLock<std::result::Result<FenceWaiter, String>> = OnceLock::new();

// Active observations back off within a bounded notification interval; idle
// observation blocks until job arrival. Futures only register completion wakes.
const FIRST_OBSERVATION: Duration = Duration::from_micros(250);
const MAX_OBSERVATION: Duration = Duration::from_millis(2);
const INTAKE_LIMIT: usize = 256;

/// Completes one job after its native fence has supplied a definitive result.
///
/// Publication precedes optional capture and retirement.  A waiter therefore
/// observes GPU completion independently of slow filesystem I/O, while the
/// capture worker retains its command resources until it has read them.
fn finish_job(job: FenceJob, result: Result<()>, captures: &SyncSender<Vec<RecordedWork>>) {
    let FenceJob {
        completion,
        work,
        sync,
        sparse,
    } = job;
    let completed = result.is_ok();
    completion.publish(result);
    if completed {
        // Capture is optional instrumentation.  The dedicated bounded worker
        // must not make a completed fence delay another fence's notification.
        // If that worker is unavailable, dropping is safe because completion
        // was already established; capture failure never changes RHI result.
        if work.iter().any(|item| item.encoder.capture.is_some()) {
            // Never let diagnostic filesystem throughput head-of-line-block
            // native completion observation. There is normally only one
            // process-wide capture, but a full bounded queue may drop it.
            let _ = captures.try_send(work);
        }
        drop(sync);
        drop(sparse);
    } else {
        // A failed status query is not retirement evidence. Keep every native
        // owner (including the fence/device) quarantined exactly as before.
        std::mem::forget(work);
        std::mem::forget(sync);
        std::mem::forget(sparse);
        std::mem::forget(completion);
    }
}

fn capture_worker(receiver: Receiver<Vec<RecordedWork>>) {
    for work in receiver {
        // Capture is diagnostic-only; errors must not retroactively fail a
        // completion that has already been published to the caller.
        let _ = super::super::super::capture::write_completed(&work);
    }
}

#[derive(Default)]
struct DeviceBatch {
    // Index into `active`, not an Arc: idle scratch must not retain a device.
    first: usize,
    positions: Vec<usize>,
    fences: Vec<vk::Fence>,
}

fn observe_fences(receiver: Receiver<FenceJob>, captures: SyncSender<Vec<RecordedWork>>) {
    let mut active: Vec<FenceJob> = Vec::new();
    let mut batches: Vec<DeviceBatch> = Vec::new();
    let mut devices = HashMap::<usize, usize>::new();
    let mut ready = Vec::<(usize, Result<()>)>::new();
    let mut interval = FIRST_OBSERVATION;
    let mut disconnected = false;
    loop {
        let mut received = false;
        if active.is_empty() {
            if disconnected {
                return;
            }
            // No timer and no driver calls when there is nothing to observe.
            #[cfg(test)]
            diagnostic(|probe| {
                probe
                    .idle_receives
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            });
            match receiver.recv() {
                Ok(job) => {
                    active.push(job);
                    received = true;
                }
                Err(_) => return,
            }
        } else if disconnected {
            std::thread::park_timeout(interval);
        } else {
            match receiver.recv_timeout(interval) {
                Ok(job) => {
                    active.push(job);
                    received = true;
                }
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => disconnected = true,
            }
        }
        // An unbounded producer burst must not starve actual completion work.
        for _ in 1..INTAKE_LIMIT {
            match receiver.try_recv() {
                Ok(job) => {
                    active.push(job);
                    received = true;
                }
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    disconnected = true;
                    break;
                }
            }
        }
        devices.clear();
        for batch in &mut batches {
            batch.positions.clear();
            batch.fences.clear();
        }
        for (index, job) in active.iter().enumerate() {
            let identity = Arc::as_ptr(&job.completion.0.device) as usize;
            let next = devices.len();
            let slot = *devices.entry(identity).or_insert(next);
            if slot == batches.len() {
                batches.push(DeviceBatch::default());
            }
            let batch = &mut batches[slot];
            if batch.positions.is_empty() {
                batch.first = index;
            }
            batch.positions.push(index);
            batch.fences.push(job.completion.0.fence);
        }
        for batch in batches.iter().take(devices.len()) {
            let device = &active[batch.first].completion.0.device.device;
            #[cfg(test)]
            diagnostic(|probe| {
                use std::sync::atomic::Ordering::Relaxed;
                probe.batch_queries.fetch_add(1, Relaxed);
                probe.max_batch.fetch_max(batch.fences.len(), Relaxed);
            });
            // SAFETY: every fence belongs to this exact logical device and is
            // held by an active Completion. Zero timeout never waits on GPU.
            match unsafe { device.wait_for_fences(&batch.fences, false, 0) } {
                Err(vk::Result::TIMEOUT) => {}
                Err(error) => {
                    for &index in &batch.positions {
                        ready.push((index, Err(vk_error(error))));
                    }
                }
                Ok(()) => {
                    // Only a batch with at least one signaled fence needs
                    // per-fence status queries to identify ready owners.
                    for &index in &batch.positions {
                        #[cfg(test)]
                        diagnostic(|probe| {
                            probe
                                .status_queries
                                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        });
                        // SAFETY: the active job pins this native fence.
                        match unsafe { device.get_fence_status(active[index].completion.0.fence) } {
                            Ok(true) => ready.push((index, Ok(()))),
                            Ok(false) => {}
                            Err(error) => ready.push((index, Err(vk_error(error)))),
                        }
                    }
                }
            }
        }
        let progressed = !ready.is_empty();
        // Descending removals preserve every remaining original job index.
        ready.sort_unstable_by_key(|(index, _)| std::cmp::Reverse(*index));
        for (index, result) in ready.drain(..) {
            finish_job(active.swap_remove(index), result, &captures);
        }
        interval = if received || progressed {
            FIRST_OBSERVATION
        } else {
            interval.saturating_mul(2).min(MAX_OBSERVATION)
        };
    }
}

#[cfg(test)]
#[derive(Default)]
struct ObserverProbe {
    idle_receives: std::sync::atomic::AtomicUsize,
    batch_queries: std::sync::atomic::AtomicUsize,
    status_queries: std::sync::atomic::AtomicUsize,
    max_batch: std::sync::atomic::AtomicUsize,
}
#[cfg(test)]
thread_local! {
    static PROBE: std::cell::RefCell<Option<Arc<ObserverProbe>>> = const { std::cell::RefCell::new(None) };
}
#[cfg(test)]
fn diagnostic(f: impl FnOnce(&ObserverProbe)) {
    PROBE.with(|probe| {
        if let Some(probe) = probe.borrow().as_ref() {
            f(probe);
        }
    });
}

pub(in crate::backend::submit) fn fence_waiter() -> Result<&'static FenceWaiter> {
    match FENCE_WAITER.get_or_init(|| {
        let (sender, receiver) = mpsc::channel::<FenceJob>();
        // Capture is optional test instrumentation. A capacity of one bounds
        // retained capture work without making the completion observer wait
        // behind filesystem I/O.
        let (capture_sender, capture_receiver) = mpsc::sync_channel::<Vec<RecordedWork>>(1);
        std::thread::Builder::new()
            .name("fluxel-vulkan-capture-writer".into())
            .spawn(move || capture_worker(capture_receiver))
            .map_err(|error| format!("Vulkan capture worker: {error}"))?;
        std::thread::Builder::new()
            .name("fluxel-vulkan-fence-waiter".into())
            .spawn(move || observe_fences(receiver, capture_sender))
            .map(|_| FenceWaiter { sender })
            .map_err(|error| format!("Vulkan fence waiter: {error}"))
    }) {
        Ok(waiter) => Ok(waiter),
        Err(message) => Err(Error::Backend(message.clone())),
    }
}

// The waiter is only a native completion observer and owner of already issued
// command pools/uploads. It never schedules GPU commands or replays CPU work.
pub(in crate::backend::submit) fn retain_until_fence(
    completion: Completion,
    work: Vec<RecordedWork>,
) {
    retain_with_sync(completion, work, Vec::new(), Vec::new());
}

pub(in crate::backend::submit) fn retain_with_sync(
    completion: Completion,
    work: Vec<RecordedWork>,
    sync: Vec<Arc<super::super::multi_queue::Timeline>>,
    sparse: Vec<super::super::super::sparse::PreparedSparseBind>,
) {
    let job = FenceJob {
        completion: completion.clone(),
        work,
        sync,
        sparse,
    };
    match fence_waiter() {
        Ok(waiter) => match waiter.sender.send(job) {
            Ok(()) => {}
            Err(error) => {
                // A native submission may already reference this work. Never
                // drop it on an observer failure; leaking this exceptional
                // ownership is safer than a synchronous host wait or an early
                // Vulkan command-pool destruction.
                std::mem::forget(error.0);
                completion.publish(Err(Error::Backend(
                    "Vulkan fence waiter unavailable".into(),
                )));
            }
        },
        Err(error) => {
            std::mem::forget(job);
            completion.publish(Err(error));
        }
    }
}

#[cfg(test)]
#[path = "../../../../tests/backend/submit/completion/observer.rs"]
mod tests;
