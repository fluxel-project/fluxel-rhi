//! Free-range coalescing diagnostics for descriptor-table churn.

use super::release_range;
use std::ops::Range;

/// The pre-change release algorithm, retained only as a test oracle and
/// diagnostic baseline. It drains every range into a newly allocated vector.
fn release_range_baseline(free: &mut Vec<Range<u32>>, returned: Range<u32>) -> usize {
    let index = free.partition_point(|range| range.start < returned.start);
    free.insert(index, returned);
    let mut merged: Vec<Range<u32>> = Vec::with_capacity(free.len());
    let mut visits = 0;
    for range in free.drain(..) {
        visits += 1;
        if let Some(previous) = merged.last_mut() {
            if range.start <= previous.end {
                previous.end = previous.end.max(range.end);
                continue;
            }
        }
        merged.push(range);
    }
    *free = merged;
    visits
}

fn assert_free_list(free: &[Range<u32>]) {
    for range in free {
        assert!(range.start < range.end, "empty free range: {range:?}");
    }
    for pair in free.windows(2) {
        assert!(pair[0].end < pair[1].start, "unmerged ranges: {pair:?}");
    }
}

fn shuffled_slots(count: u32) -> Vec<u32> {
    let mut slots: Vec<_> = (0..count).collect();
    let mut state = 0x4d59_5df4_u64;
    for index in (1..slots.len()).rev() {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let swap = (state as usize) % (index + 1);
        slots.swap(index, swap);
    }
    slots
}

#[test]
fn adjacent_returns_coalesce_without_replacing_the_free_list() {
    let mut free = vec![4..8, 12..16];
    let storage = free.as_ptr();
    release_range(&mut free, 8..12);
    assert_eq!(free, vec![4..16]);
    assert_eq!(free.as_ptr(), storage, "adjacent merge should not allocate");
    assert_free_list(&free);
}

#[test]
fn randomized_table_churn_matches_drain_baseline_and_recovers_capacity() {
    // This models descriptor-table drops after a churny bind-group workload:
    // allocations are single slots, while retirements arrive in arbitrary
    // completion order. It does not claim GPU timing; it isolates free-list
    // traversal and allocation work.
    const TABLES: u32 = 1024;
    let mut baseline = Vec::new();
    let mut candidate = Vec::new();
    let mut baseline_visits = 0;
    let mut candidate_neighbor_checks = 0;
    for slot in shuffled_slots(TABLES) {
        baseline_visits += release_range_baseline(&mut baseline, slot..slot + 1);
        release_range(&mut candidate, slot..slot + 1);
        candidate_neighbor_checks += 2;
        assert_eq!(candidate, baseline, "release slot {slot}");
        assert_free_list(&candidate);
    }
    assert_eq!(candidate, vec![0..TABLES]);
    println!(
        "descriptor-free-churn diagnostic: tables={TABLES} baseline_drain_visits={baseline_visits} baseline_vec_allocations={TABLES} candidate_neighbor_checks={candidate_neighbor_checks} candidate_vec_allocations=0"
    );
    assert!(baseline_visits > candidate_neighbor_checks);
}

#[cfg(feature = "hardware-tests")]
#[test]
fn native_descriptor_heap_churn_reclaims_real_descriptor_slots() -> fluxel_rhi_core::api::Result<()>
{
    use crate::backend::{Dx12, Dx12Instance};
    use fluxel_rhi_core::api::{DeviceId, DeviceRequest, Error, InstanceBackend};
    use std::{
        future::Future,
        task::{Context, Poll, Waker},
    };

    use super::{VIEW_CAPACITY, create_heaps};

    // This is intentionally a release-bound diagnostic, not an FPS benchmark:
    // it uses a real shader-visible heap and exercises Heap::allocate/release
    // in a bind-group-sized single-descriptor churn pattern.
    let native_instance = Dx12Instance::new()?;
    let instance = native_instance.instance()?;
    let Some(adapter) = instance.enumerate()?.into_iter().next() else {
        eprintln!("SKIP descriptor heap churn: DXGI reported no adapters");
        return Ok(());
    };
    let request = DeviceRequest {
        adapter: adapter.id,
        required_features: Vec::new(),
    };
    let mut open = std::pin::pin!(<Dx12 as InstanceBackend>::open(
        &native_instance,
        DeviceId::allocate()?,
        &request,
    ));
    let mut context = Context::from_waker(Waker::noop());
    let (device, _) = match open.as_mut().poll(&mut context) {
        Poll::Ready(Ok(device)) => device,
        Poll::Ready(Err(Error::Unsupported(reason))) => {
            eprintln!("SKIP descriptor heap churn: {reason}");
            return Ok(());
        }
        Poll::Ready(Err(error)) => return Err(error),
        Poll::Pending => panic!("DX12 open must be immediately ready"),
    };
    let heaps = create_heaps(&device.0.native)?;
    const TABLES: u32 = 1024;
    let mut allocated = Vec::with_capacity(TABLES as usize);
    for _ in 0..TABLES {
        allocated.push(heaps.views.allocate(1)?);
    }
    for slot in shuffled_slots(TABLES) {
        heaps.views.release(allocated[slot as usize], 1);
    }
    let free = heaps.views.free.lock().map_err(|_| Error::DeviceLost)?;
    assert_eq!(*free, vec![0..VIEW_CAPACITY]);
    eprintln!(
        "descriptor-native-heap-churn diagnostic: tables={TABLES} allocations={TABLES} releases={TABLES} final_ranges={} release_bound_only=true",
        free.len()
    );
    Ok(())
}
