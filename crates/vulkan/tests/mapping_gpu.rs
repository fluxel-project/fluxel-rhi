//! Real-device checks for mapping lifetime, ranges, and cache operations.
#![cfg(all(windows, feature = "hardware-tests"))]
use fluxel_rhi_core::api::*;
use fluxel_rhi_vulkan::VulkanInstance;
use std::{
    future::Future,
    sync::Arc,
    task::{Context, Poll, Wake, Waker},
};
struct WakeThread(std::thread::Thread);
impl Wake for WakeThread {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }
}
fn block_on<F: Future>(f: F) -> F::Output {
    let waker = Waker::from(Arc::new(WakeThread(std::thread::current())));
    let mut cx = Context::from_waker(&waker);
    let mut f = std::pin::pin!(f);
    loop {
        match f.as_mut().poll(&mut cx) {
            Poll::Ready(v) => return v,
            Poll::Pending => std::thread::park(),
        }
    }
}
#[test]
fn mapping_lease_drop_and_cancel() {
    let device = block_on(VulkanInstance::new().unwrap().open_default()).unwrap();
    let buffer = device
        .create_buffer(&BufferDesc {
            label: None,
            size: 1024,
            usage: BufferUsage::COPY_SRC | BufferUsage::MAP_WRITE,
            memory: MemoryClass::Upload,
        })
        .unwrap();
    // SAFETY: this test never submits GPU work using the buffer.
    let mut lease =
        block_on(unsafe { device.map_buffer(&buffer, 13..25, MapMode::Write) }).unwrap();
    lease.bytes_mut().unwrap().copy_from_slice(&[7; 12]);
    lease.flush(1..11).unwrap();
    assert!(matches!(lease.flush(0..13), Err(Error::InvalidRange)));
    // SAFETY: no GPU use exists; the live first lease must reject a second map.
    assert!(matches!(
        block_on(unsafe { device.map_buffer(&buffer, 0..1, MapMode::Write) }),
        Err(Error::InvalidState)
    ));
    drop(lease);
    // SAFETY: cancellation before polling cannot leave a native mapping behind.
    drop(unsafe { device.map_buffer(&buffer, 0..1, MapMode::Write) });
    // SAFETY: all previous leases have ended and there is no GPU work.
    let lease = block_on(unsafe { device.map_buffer(&buffer, 13..25, MapMode::Write) }).unwrap();
    assert_eq!(lease.bytes(), &[7; 12]);
}
#[test]
fn placed_buffers_share_mapping_exclusion_and_respect_offsets() {
    let device = block_on(VulkanInstance::new().unwrap().open_default()).unwrap();
    let memory = device.memory().expect("Vulkan placed memory");
    let desc = BufferDesc {
        label: None,
        size: 1024,
        usage: BufferUsage::COPY_SRC | BufferUsage::MAP_WRITE,
        memory: MemoryClass::Upload,
    };
    let req = memory.buffer_requirements(&desc).unwrap();
    assert!(!req.dedicated_only);
    let index = req.native_type_bits.trailing_zeros();
    assert!(index < 64);
    let heap = memory
        .heap(&HeapDesc {
            size: req.size * 2,
            native_memory_class: index,
        })
        .unwrap();
    assert!(matches!(
        memory.place_buffer(&heap, 1, &desc),
        Err(Error::InvalidRange)
    ));
    let invalid_type = (0..64).find(|index| req.native_type_bits & (1u64 << index) == 0);
    if let Some(index) = invalid_type {
        if let Ok(wrong_heap) = memory.heap(&HeapDesc {
            size: req.size * 2,
            native_memory_class: index,
        }) {
            assert!(matches!(
                memory.place_buffer(&wrong_heap, 0, &desc),
                Err(Error::Unsupported(_))
            ));
        }
    }
    let a = memory.place_buffer(&heap, 0, &desc).unwrap();
    let b = memory.place_buffer(&heap, req.size, &desc).unwrap();
    drop(heap);
    // SAFETY: these resources have no GPU use in this test.
    let mut first = block_on(unsafe { device.map_buffer(&a, 13..21, MapMode::Write) }).unwrap();
    first.bytes_mut().unwrap().copy_from_slice(&[1; 8]);
    first.flush(0..8).unwrap();
    // SAFETY: same heap must reject simultaneous maps even for distinct buffers.
    assert!(matches!(
        block_on(unsafe { device.map_buffer(&b, 13..21, MapMode::Write) }),
        Err(Error::InvalidState)
    ));
    drop(first);
    // SAFETY: previous map ended and no GPU work has been submitted.
    let mut second = block_on(unsafe { device.map_buffer(&b, 13..21, MapMode::Write) }).unwrap();
    second.bytes_mut().unwrap().copy_from_slice(&[2; 8]);
    second.flush(0..8).unwrap();
    drop(second);
    // SAFETY: previous map ended and no GPU work has been submitted.
    let first = block_on(unsafe { device.map_buffer(&a, 13..21, MapMode::Write) }).unwrap();
    assert_eq!(first.bytes(), &[1; 8]);
    drop(first);
    // SAFETY: previous map ended and no GPU work has been submitted.
    let second = block_on(unsafe { device.map_buffer(&b, 13..21, MapMode::Write) }).unwrap();
    assert_eq!(second.bytes(), &[2; 8]);
}
