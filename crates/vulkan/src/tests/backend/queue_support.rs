//! Unit tests for the corresponding implementation module.

use super::*;

#[test]
fn selected_compute_lane_can_supply_sparse_binding() {
    let support = sparse_queues(
        true,
        vk::QueueFlags::GRAPHICS,
        Some(vk::QueueFlags::COMPUTE | vk::QueueFlags::SPARSE_BINDING),
    );
    assert!(support.images_available);
    assert!(!support.graphics_binding);
    assert!(support.compute_binding);
}

#[test]
fn an_uncreated_compute_family_cannot_supply_sparse_binding() {
    assert!(!sparse_queues(true, vk::QueueFlags::GRAPHICS, None).images_available);
    assert!(
        sparse_queues(
            true,
            vk::QueueFlags::GRAPHICS | vk::QueueFlags::SPARSE_BINDING,
            None,
        )
        .images_available
    );
    assert!(
        !sparse_queues(
            false,
            vk::QueueFlags::GRAPHICS | vk::QueueFlags::SPARSE_BINDING,
            Some(vk::QueueFlags::COMPUTE | vk::QueueFlags::SPARSE_BINDING),
        )
        .images_available
    );
}
