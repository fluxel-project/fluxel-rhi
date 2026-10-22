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
    }
}
