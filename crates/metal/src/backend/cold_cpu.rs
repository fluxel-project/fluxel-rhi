//! Metal configuration for CPU-only cold shader translation.
//!
//! Objective-C library/pipeline creation and completion handlers use the
//! separate native asynchronous path; only owned Naga translation jobs enter
//! this shared CPU worker.

use std::sync::OnceLock;

use fluxel_rhi_core::api::Result;
use fluxel_rhi_core::utils::cold_worker::Worker;

pub type CpuFuture<T> = fluxel_rhi_core::utils::cold_worker::ColdFuture<T>;

const QUEUE_LIMIT: usize = 16;

fn worker() -> std::result::Result<Worker, String> {
    static WORKER: OnceLock<std::result::Result<Worker, String>> = OnceLock::new();
    WORKER
        .get_or_init(|| {
            Worker::new(
                QUEUE_LIMIT,
                "fluxel-metal-shader-cpu",
                "Metal CPU pipeline worker stopped",
            )
            .map_err(|error| error.to_string())
        })
        .clone()
}

/// Enqueues CPU-only Naga translation without transferring native Metal state.
pub(crate) fn spawn_cpu<T: Send + 'static>(
    operation: impl FnOnce() -> Result<T> + Send + 'static,
) -> CpuFuture<T> {
    fluxel_rhi_core::utils::cold_worker::submit(worker(), "Metal CPU pipeline worker", operation)
}
