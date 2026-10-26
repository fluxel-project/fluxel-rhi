//! Native queue-family facts for the lanes actually created by this device.
//!
//! Resource capability admission may use either exposed lane. Submission still
//! checks the selected lane independently; no queue or work is selected here.

use ash::vk;

#[derive(Debug, PartialEq, Eq)]
pub(super) struct SparseQueues {
    pub(super) images_available: bool,
    pub(super) graphics_binding: bool,
    pub(super) compute_binding: bool,
}

pub(super) fn sparse_queues(
    image_features: bool,
    graphics: vk::QueueFlags,
    compute: Option<vk::QueueFlags>,
) -> SparseQueues {
    let graphics_binding = graphics.contains(vk::QueueFlags::SPARSE_BINDING);
    let compute_binding =
        compute.is_some_and(|flags| flags.contains(vk::QueueFlags::SPARSE_BINDING));
    SparseQueues {
        images_available: image_features && (graphics_binding || compute_binding),
        graphics_binding,
        compute_binding,
    }
}

#[cfg(test)]
#[path = "../tests/backend/queue_support.rs"]
mod tests;
