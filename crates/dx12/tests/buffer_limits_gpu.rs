//! The device-reported GPU virtual-address limit governs buffer support.
#![cfg(all(windows, feature = "hardware-tests"))]

use fluxel_rhi_core::api::{BufferDesc, BufferUsage, MemoryClass, Support};
use fluxel_rhi_dx12::Dx12Instance;
use std::future::Future;

fn block<F: Future>(future: F) -> F::Output {
    let mut context = std::task::Context::from_waker(std::task::Waker::noop());
    let mut future = std::pin::pin!(future);
    loop {
        if let std::task::Poll::Ready(value) = future.as_mut().poll(&mut context) {
            return value;
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
}

#[test]
fn device_reports_native_buffer_limit_above_old_one_gib_cap() {
    let device = block(Dx12Instance::new().unwrap().open_default()).unwrap();
    let size = (1u64 << 30) + 65536;
    assert!(device.capabilities().limits.max_buffer_bytes >= size);
    assert!(matches!(
        device.buffer_support(&BufferDesc {
            label: None,
            size,
            usage: BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        }),
        Support::Supported(_)
    ));
}
