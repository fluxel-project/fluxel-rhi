//! Vulkan configuration for the shared CPU cold-object worker.
//!
//! Jobs contain owned shader and pipeline descriptions. They create cold
//! native objects only; queue submission and GPU completion stay elsewhere.

use std::sync::OnceLock;

use fluxel_rhi_core::api::Result;
use fluxel_rhi_core::utils::cold_worker::Worker;

pub use fluxel_rhi_core::utils::cold_worker::ColdFuture;

const QUEUE_LIMIT: usize = 16;

fn worker() -> std::result::Result<Worker, String> {
    static WORKER: OnceLock<std::result::Result<Worker, String>> = OnceLock::new();
    WORKER
        .get_or_init(|| {
            Worker::new(
                QUEUE_LIMIT,
                "fluxel-vulkan-pipeline",
                "Vulkan pipeline worker stopped",
            )
            .map_err(|error| error.to_string())
        })
        .clone()
}

/// Creates a future for one fully-owned Vulkan cold-path operation.
pub(crate) fn submit<T: Send + 'static>(
    operation: impl FnOnce() -> Result<T> + Send + 'static,
) -> ColdFuture<T> {
    fluxel_rhi_core::utils::cold_worker::submit(worker(), "Vulkan pipeline worker", operation)
}
