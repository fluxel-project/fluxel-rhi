//! Default resource pool free-range, budget, and lease-lifetime invariants.

use super::{free_range, take_range};

#[test]
fn aligned_ranges_split_coalesce_and_reuse() {
    let mut free = vec![(0, 1024)];
    assert_eq!(take_range(&mut free, 100, 256), Some(0));
    assert_eq!(take_range(&mut free, 100, 256), Some(256));
    assert_eq!(take_range(&mut free, 100, 256), Some(512));
    assert_eq!(take_range(&mut free, 100, 256), Some(768));
    assert_eq!(take_range(&mut free, 200, 1), None);

    let mut free = Vec::new();
    free_range(&mut free, 256, 100);
    free_range(&mut free, 0, 256);
    free_range(&mut free, 356, 156);
    assert_eq!(free, vec![(0, 512)]);
    assert_eq!(take_range(&mut free, 512, 256), Some(0));
}
#[test]
fn alignment_overflow_fails_without_mutating_free_list() {
    let mut free = vec![(u64::MAX - 3, 3)];
    assert_eq!(take_range(&mut free, 8, 8), None);
    assert_eq!(free, vec![(u64::MAX - 3, 3)]);
}
#[test]
fn budget_gate_accepts_exact_fit_and_rejects_overflow_or_excess() {
    assert!(super::within_budget(100, 200, 100));
    assert!(!super::within_budget(101, 200, 100));
    assert!(!super::within_budget(u64::MAX - 3, u64::MAX, 8));
    assert!(!super::within_budget(0, 0, 1));
}

#[cfg(all(windows, feature = "hardware-tests"))]
#[test]
fn direct_rhi_memory_snapshot_tracks_pool_ranges_and_dxgi_scopes() {
    use crate::backend::Dx12Instance;
    use fluxel_rhi_core::api::{
        AllocatorMemorySource, BufferDesc, BufferUsage, MemoryClass, MemoryReportScope,
        MemoryReportSource,
    };
    use std::future::Future;

    fn block<F: Future>(future: F) -> F::Output {
        let mut context = std::task::Context::from_waker(std::task::Waker::noop());
        let mut future = std::pin::pin!(future);
        loop {
            match future.as_mut().poll(&mut context) {
                std::task::Poll::Ready(value) => return value,
                std::task::Poll::Pending => std::thread::sleep(std::time::Duration::from_millis(1)),
            }
        }
    }

    let device = block(Dx12Instance::new().unwrap().open_default()).unwrap();
    let observation = device
        .memory_observation()
        .expect("DX12 observation capability");
    let initial = observation.snapshot().unwrap();
    let initial_pool = initial.allocator.expect("DX12 default pool summary");
    assert_eq!(
        initial_pool.source,
        AllocatorMemorySource::Dx12DefaultResourcePool
    );
    assert_eq!(
        initial_pool.reserved_bytes,
        initial_pool.live_range_bytes + initial_pool.free_bytes
    );
    assert_eq!(initial.native_reports.len(), 2);
    assert!(initial.native_reports.iter().all(|report| {
        report.source == MemoryReportSource::DxgiProcessSegment
            && matches!(
                report.scope,
                MemoryReportScope::DxgiLocalSegment | MemoryReportScope::DxgiNonLocalSegment
            )
            && match (report.usage_bytes, report.budget_bytes) {
                (Some(_), Some(budget)) => budget > 0,
                (None, None) => true,
                _ => false,
            }
    }));

    let desc = BufferDesc {
        label: None,
        size: 8 * 1024 * 1024,
        usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
        memory: MemoryClass::Device,
    };
    let buffers = (0..3)
        .map(|_| device.create_buffer(&desc).unwrap())
        .collect::<Vec<_>>();
    let live = observation.snapshot().unwrap().allocator.unwrap();
    assert!(live.live_allocation_count >= initial_pool.live_allocation_count + 3);
    assert!(live.live_range_bytes > initial_pool.live_range_bytes);
    assert!(live.page_count >= 1);
    assert_eq!(live.reserved_bytes, live.live_range_bytes + live.free_bytes);
    assert!(live.peak_reserved_bytes >= live.reserved_bytes);
    assert!(live.peak_live_range_bytes >= live.live_range_bytes);

    drop(buffers);
    let released = observation.snapshot().unwrap().allocator.unwrap();
    assert_eq!(
        released.live_allocation_count,
        initial_pool.live_allocation_count
    );
    assert_eq!(released.live_range_bytes, initial_pool.live_range_bytes);
    assert_eq!(
        released.reserved_bytes,
        released.live_range_bytes + released.free_bytes
    );
    assert!(released.peak_live_range_bytes >= live.live_range_bytes);
}

#[cfg(all(windows, feature = "hardware-tests"))]
#[test]
fn native_resource_clone_keeps_lease_until_last_drop_then_reuses_range() {
    use crate::backend::{Dx12, Dx12Instance};
    use fluxel_rhi_core::api::{self};
    use fluxel_rhi_core::backend::ResourceBackend;
    use std::future::Future;

    fn block<F: Future>(future: F) -> F::Output {
        let mut context = std::task::Context::from_waker(std::task::Waker::noop());
        let mut future = std::pin::pin!(future);
        loop {
            match future.as_mut().poll(&mut context) {
                std::task::Poll::Ready(value) => return value,
                std::task::Poll::Pending => std::thread::sleep(std::time::Duration::from_millis(1)),
            }
        }
    }

    let public_device = block(Dx12Instance::new().unwrap().open_default()).unwrap();
    // SAFETY: the public device owns this native device for the test scope.
    let device = unsafe { public_device.native_ref() };
    let desc = api::BufferDesc {
        label: None,
        size: 4096,
        usage: api::BufferUsage::COPY_SRC | api::BufferUsage::COPY_DST,
        memory: api::MemoryClass::Device,
    };

    let first = <Dx12 as ResourceBackend>::create_buffer(device, &desc).unwrap();
    let lease = first.default_allocation.as_ref().expect("pooled buffer");
    let first_page = lease.0.page_id;
    let first_offset = lease.0.offset;
    let clones: Vec<_> = (0..16).map(|_| first.clone()).collect();
    drop(first);
    {
        let pool = device.0.default_resources.inner.lock().unwrap();
        let page = pool
            .pages
            .iter()
            .find(|page| page.id == first_page)
            .unwrap();
        assert_eq!(page.live, 1, "clone keeps its range leased");
    }
    let drops: Vec<_> = clones
        .into_iter()
        .map(|buffer| std::thread::spawn(move || drop(buffer)))
        .collect();
    for drop in drops {
        drop.join().unwrap();
    }
    {
        let pool = device.0.default_resources.inner.lock().unwrap();
        let page = pool
            .pages
            .iter()
            .find(|page| page.id == first_page)
            .unwrap();
        assert_eq!(page.live, 0, "final owner returns its range");
    }

    let reused = <Dx12 as ResourceBackend>::create_buffer(device, &desc).unwrap();
    let lease = reused.default_allocation.as_ref().expect("pooled buffer");
    assert_eq!(
        (lease.0.page_id, lease.0.offset),
        (first_page, first_offset)
    );
}
