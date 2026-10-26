//! Unit tests for the corresponding implementation module.

use super::*;
use ash::vk::Handle;

#[test]
fn cross_queue_metadata_wait_coalesces_timeline_values() {
    let a = vk::Queue::from_raw(1);
    let b = vk::Queue::from_raw(2);
    let semaphore = vk::Semaphore::from_raw(3);
    let mut waits = Vec::new();
    metadata_wait(a, a, semaphore, 5, &mut waits);
    assert!(waits.is_empty());
    metadata_wait(a, b, semaphore, 5, &mut waits);
    metadata_wait(a, b, semaphore, 3, &mut waits);
    metadata_wait(a, b, semaphore, 7, &mut waits);
    assert_eq!(waits, vec![(semaphore, 7)]);
}
