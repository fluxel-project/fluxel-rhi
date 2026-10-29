//! Range accounting, budget gates, and allocation reuse invariants.

use super::{aggregate_totals, apply_peaks, coalesce, take_range};

#[test]
fn snapshot_totals_cover_multiple_pages_idle_cache_and_peaks() {
    // The first page holds two padded live ranges. The second remains as the
    // pool's one retained idle cache page for its key.
    let pages = [(16 * 1024, 4 * 1024, 2usize), (16 * 1024, 16 * 1024, 0)];
    assert_eq!(
        aggregate_totals(&pages),
        (32 * 1024, 12 * 1024, 20 * 1024, 1, 2)
    );

    let mut peak_reserved = 0;
    let mut peak_live = 0;
    apply_peaks(&mut peak_reserved, &mut peak_live, 32 * 1024, 12 * 1024);
    // Releasing the two ranges lowers the current total but never rewrites
    // the device-lifetime high-water facts.
    apply_peaks(&mut peak_reserved, &mut peak_live, 32 * 1024, 0);
    assert_eq!((peak_reserved, peak_live), (32 * 1024, 12 * 1024));
}

#[test]
fn aligned_ranges_split_coalesce_and_reuse() {
    let mut split = vec![(0, 1024)];
    assert_eq!(take_range(&mut split, 128, 256), Some(0));
    assert_eq!(take_range(&mut split, 128, 256), Some(256));
    assert_eq!(take_range(&mut split, 128, 256), Some(512));
    assert_eq!(take_range(&mut split, 128, 256), Some(768));
    assert!(take_range(&mut split, 129, 1).is_none());

    let mut free = vec![(0, 256), (256, 256), (512, 256)];
    coalesce(&mut free);
    assert!(free.contains(&(0, 768)));
    assert_eq!(take_range(&mut free, 768, 256), Some(0));
}

#[test]
fn overflow_does_not_consume_a_range() {
    let mut free = vec![(u64::MAX - 7, 7)];
    assert_eq!(take_range(&mut free, 16, 16), None);
    assert_eq!(free, vec![(u64::MAX - 7, 7)]);
}

#[test]
fn budget_gate_accepts_exact_fit_and_rejects_overflow_or_excess() {
    assert!(super::within_budget(100, 200, 100));
    assert!(!super::within_budget(101, 200, 100));
    assert!(!super::within_budget(u64::MAX - 3, u64::MAX, 8));
    assert!(super::within_budget(0, 0, 16));
}
