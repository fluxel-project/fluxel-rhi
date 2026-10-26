//! DX12 configuration for the shared CPU cold-object worker.
//!
//! Only fully-owned shader/pipeline construction is admitted here. Native
//! command recording and GPU retirement remain owned by their backend paths.

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
                "fluxel-dx12-pipeline",
                "DX12 pipeline worker stopped",
            )
            .map_err(|error| error.to_string())
        })
        .clone()
}

/// Creates a future for one fully-owned DX12 cold-path operation.
pub(crate) fn submit<T: Send + 'static>(
    operation: impl FnOnce() -> Result<T> + Send + 'static,
) -> ColdFuture<T> {
    fluxel_rhi_core::utils::cold_worker::submit(worker(), "DX12 pipeline worker", operation)
}
