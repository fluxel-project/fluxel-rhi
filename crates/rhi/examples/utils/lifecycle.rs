//! Example-owned lifetime boundary for immediate native/browser execution.
//!
//! A failed idle wait is not completion evidence. Standalone examples stop
//! execution without unwinding rather than release an unproven in-flight scene.
use fluxel_rhi::{api::Device, backend::CompletionBackend};

pub(super) fn unsettled(reason: &str) -> ! {
    eprintln!("RHI example cannot safely release in-flight resources: {reason}");
    // Native abort terminates the process; WASM abort traps without running
    // Rust destructors. The browser must discard/reload that failed instance.
    // Neither route invents completion or tries to continue with unknown state.
    std::process::abort()
}

/// Declare after the scene so this guard runs before its resources are dropped.
/// Browser callers must await `finish`; cancellation cannot asynchronously drain
/// in Drop, so it traps and retains the failed WASM instance's heap instead.
pub struct GpuScope<B: CompletionBackend> {
    device: Device<B>,
    armed: bool,
}
impl<B: CompletionBackend> GpuScope<B> {
    pub fn new(device: &Device<B>) -> Self {
        Self {
            device: device.clone(),
            armed: true,
        }
    }
    pub async fn finish(mut self) {
        if let Err(error) = self.device.wait_idle().await {
            unsettled(&format!("scene drain failed: {error}"));
        }
        self.armed = false;
    }
}
impl<B: CompletionBackend> Drop for GpuScope<B> {
    fn drop(&mut self) {
        if self.armed {
            #[cfg(not(target_arch = "wasm32"))]
            {
                if let Err(error) = super::block_on(self.device.wait_idle()) {
                    unsettled(&format!("scene teardown drain failed: {error}"));
                }
            }
            #[cfg(target_arch = "wasm32")]
            unsettled("immediate scene exited or was cancelled before GPU drain");
        }
    }
}
