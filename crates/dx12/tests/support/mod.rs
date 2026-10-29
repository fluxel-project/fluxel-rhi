//! Test-only synchronous polling for DX12 integration fixtures.
//!
//! The GPU fixtures intentionally do not introduce an async runtime: each one
//! waits for explicit RHI completion, and a no-op waker plus a short sleep is
//! sufficient for those futures on the test thread.

use std::future::Future;

pub fn block<F: Future>(future: F) -> F::Output {
    let mut context = std::task::Context::from_waker(std::task::Waker::noop());
    let mut future = std::pin::pin!(future);
    loop {
        match future.as_mut().poll(&mut context) {
            std::task::Poll::Ready(value) => return value,
            std::task::Poll::Pending => std::thread::sleep(std::time::Duration::from_millis(1)),
        }
    }
}
