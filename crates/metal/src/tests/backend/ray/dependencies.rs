//! Unit tests for the corresponding implementation module.

use super::FrozenDependencies;
use std::sync::{Arc, Barrier};

#[test]
fn replacement_publishes_only_after_completion() {
    let dependencies = FrozenDependencies::default();
    assert!(dependencies.freeze(1, vec![1, 2]).is_ok());
    dependencies.submit(1);
    dependencies.complete(vec![1, 2], true);
    assert!(dependencies.freeze(2, vec![3]).is_ok());
    assert_eq!(
        dependencies
            .with_snapshot(2, |values| values.to_vec())
            .unwrap(),
        [3]
    );
}

#[test]
fn abandoned_recording_releases_its_snapshot() {
    let dependencies = FrozenDependencies::default();
    let value = Arc::new(());
    let weak = Arc::downgrade(&value);
    dependencies.freeze(7, vec![value]).unwrap();
    dependencies.abandon(7);
    assert!(weak.upgrade().is_none());
}

#[test]
fn only_one_recording_owner_wins() {
    let dependencies = Arc::new(FrozenDependencies::default());
    let barrier = Arc::new(Barrier::new(3));
    let workers: Vec<_> = (0..2)
        .map(|id| {
            let (dependencies, barrier) = (dependencies.clone(), barrier.clone());
            std::thread::spawn(move || {
                barrier.wait();
                dependencies.freeze(id + 1, vec![id])
            })
        })
        .collect();
    barrier.wait();
    let successes = workers
        .into_iter()
        .map(|worker| worker.join().unwrap())
        .filter(|result| result.is_ok())
        .count();
    assert_eq!(successes, 1);
}
