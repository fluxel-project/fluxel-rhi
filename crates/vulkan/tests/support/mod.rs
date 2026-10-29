//! Shared polling helpers for Vulkan integration-test targets.

use std::future::Future;

/// Polls a test future with a no-op waker and a short retry delay.
pub fn block_on<F: Future>(future: F) -> F::Output {
    let mut context = std::task::Context::from_waker(std::task::Waker::noop());
    let mut future = std::pin::pin!(future);
    loop {
        match future.as_mut().poll(&mut context) {
            std::task::Poll::Ready(value) => return value,
            std::task::Poll::Pending => std::thread::sleep(std::time::Duration::from_millis(1)),
        }
    }
}

/// Alias for targets that historically named the shared polling helper `block`.
#[allow(unused_imports)]
pub use block_on as block;
