//! Process-wide native-call counters for isolated GPU cost measurements.
//!
//! Counters exist only with `hardware-tests`. Normal builds compile `record`
//! to an empty inline function. Measurements use snapshot deltas in a single
//! test process; these counts do not describe concurrent unrelated devices.

#[derive(Clone, Copy)]
pub(crate) enum Operation {
    QueueSignal,
    ExecuteCalls,
    SubmittedLists,
    UploadResources,
    UploadMaps,
    DescriptorHeapSets,
    UploadBytes,
    UploadCapacityBytes,
}

#[cfg(feature = "hardware-tests")]
static COUNTERS: [std::sync::atomic::AtomicU64; 8] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 8];

#[cfg(feature = "hardware-tests")]
static SUBMISSION_GATE_WAIT_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
#[cfg(feature = "hardware-tests")]
static SUBMISSION_GATE_HOLD_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Starts the held-time interval after a submission acquired its query gate.
///
/// This type is only compiled for the opt-in hardware-test build, so release
/// submissions neither read a clock nor carry a timing guard.
#[cfg(feature = "hardware-tests")]
pub(crate) struct SubmissionGateTiming {
    acquired: std::time::Instant,
}

#[cfg(feature = "hardware-tests")]
pub(crate) fn submission_gate_acquired(wait_started: std::time::Instant) -> SubmissionGateTiming {
    use std::sync::atomic::Ordering::Relaxed;
    let acquired = std::time::Instant::now();
    SUBMISSION_GATE_WAIT_NS.fetch_add(
        acquired.duration_since(wait_started).as_nanos() as u64,
        Relaxed,
    );
    SubmissionGateTiming { acquired }
}

#[cfg(feature = "hardware-tests")]
impl Drop for SubmissionGateTiming {
    fn drop(&mut self) {
        use std::sync::atomic::Ordering::Relaxed;
        SUBMISSION_GATE_HOLD_NS.fetch_add(self.acquired.elapsed().as_nanos() as u64, Relaxed);
    }
}

/// The query-state gate plus opt-in timing ownership for one submission.
/// Field order drops the timer before the guard, so held time includes the
/// whole critical section while the mutex remains held.
#[cfg(feature = "hardware-tests")]
pub(crate) struct SubmissionGateGuard<'a, T> {
    _timing: SubmissionGateTiming,
    _guard: std::sync::MutexGuard<'a, T>,
}

/// Production carries exactly the pre-existing mutex guard and no clock state.
#[cfg(not(feature = "hardware-tests"))]
pub(crate) struct SubmissionGateGuard<'a, T> {
    _guard: std::sync::MutexGuard<'a, T>,
}

#[inline]
pub(crate) fn lock_submission_gate<T>(gate: &std::sync::Mutex<T>) -> SubmissionGateGuard<'_, T> {
    #[cfg(feature = "hardware-tests")]
    let wait_started = std::time::Instant::now();
    let guard = gate.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    #[cfg(feature = "hardware-tests")]
    {
        return SubmissionGateGuard {
            _timing: submission_gate_acquired(wait_started),
            _guard: guard,
        };
    }
    #[cfg(not(feature = "hardware-tests"))]
    SubmissionGateGuard { _guard: guard }
}

#[inline]
pub(crate) fn record(operation: Operation, count: u64) {
    #[cfg(feature = "hardware-tests")]
    COUNTERS[operation as usize].fetch_add(count, std::sync::atomic::Ordering::Relaxed);
    #[cfg(not(feature = "hardware-tests"))]
    let _ = (operation, count);
}

/// Attempted native operations since process start. Use deltas without reset.
#[cfg(feature = "hardware-tests")]
#[derive(Clone, Copy, Debug, Default)]
pub struct Snapshot {
    pub queue_signals: u64,
    pub execute_calls: u64,
    pub submitted_lists: u64,
    pub upload_resources: u64,
    pub upload_maps: u64,
    pub descriptor_heap_sets: u64,
    pub upload_bytes: u64,
    /// Sum of resource widths successfully allocated by the buffer upload arena.
    pub upload_capacity_bytes: u64,
    /// Cumulative time submitting threads spent waiting for the query gate.
    pub submission_gate_wait_ns: u64,
    /// Cumulative time holding the query gate during submission.
    pub submission_gate_hold_ns: u64,
}

#[cfg(feature = "hardware-tests")]
pub fn snapshot() -> Snapshot {
    use std::sync::atomic::Ordering::Relaxed;
    let load = |operation: Operation| COUNTERS[operation as usize].load(Relaxed);
    Snapshot {
        queue_signals: load(Operation::QueueSignal),
        execute_calls: load(Operation::ExecuteCalls),
        submitted_lists: load(Operation::SubmittedLists),
        upload_resources: load(Operation::UploadResources),
        upload_maps: load(Operation::UploadMaps),
        descriptor_heap_sets: load(Operation::DescriptorHeapSets),
        upload_bytes: load(Operation::UploadBytes),
        upload_capacity_bytes: load(Operation::UploadCapacityBytes),
        submission_gate_wait_ns: SUBMISSION_GATE_WAIT_NS.load(Relaxed),
        submission_gate_hold_ns: SUBMISSION_GATE_HOLD_NS.load(Relaxed),
    }
}
