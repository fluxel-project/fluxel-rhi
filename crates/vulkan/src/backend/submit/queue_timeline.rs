//! Reusable per-queue Vulkan timeline semaphores.
//!
//! DeviceState owns raw semaphores and their monotonic counters. Submitted
//! jobs and signal tokens retain lightweight wrappers with an `Arc<DeviceState>`
//! but do not own or destroy the raw semaphore, avoiding a retention cycle.

use super::*;
use std::sync::atomic::{AtomicU64, Ordering};

pub(in crate::backend) struct RawQueueTimeline {
    pub(in crate::backend) semaphore: vk::Semaphore,
    next_value: AtomicU64,
}

impl RawQueueTimeline {
    pub(in crate::backend) fn new_pair(device: &ash::Device) -> Result<[Self; 2]> {
        let make = || {
            let mut kind = vk::SemaphoreTypeCreateInfo::default()
                .semaphore_type(vk::SemaphoreType::TIMELINE)
                .initial_value(0);
            let info = vk::SemaphoreCreateInfo::default().push_next(&mut kind);
            // SAFETY: timeline semaphore support was enabled for this device.
            unsafe { device.create_semaphore(&info, None) }.map_err(vk_error)
        };
        let first = make()?;
        match make() {
            Ok(second) => Ok([
                Self {
                    semaphore: first,
                    next_value: AtomicU64::new(0),
                },
                Self {
                    semaphore: second,
                    next_value: AtomicU64::new(0),
                },
            ]),
            Err(error) => {
                // SAFETY: this function still exclusively owns `first`.
                unsafe { device.destroy_semaphore(first, None) };
                Err(error)
            }
        }
    }
}

impl Timeline {
    pub(in crate::backend) fn acquire(device: &VulkanDevice) -> Result<[Arc<Self>; 2]> {
        if device.0.queue_timelines.is_none() {
            return Err(Error::Unsupported("Vulkan timeline semaphore"));
        }
        Ok(std::array::from_fn(|lane| {
            Arc::new(Self {
                device: device.0.clone(),
                lane,
                semaphore: device.0.queue_timelines.as_ref().expect("timeline enabled")[lane]
                    .semaphore,
            })
        }))
    }

    fn raw(&self) -> &RawQueueTimeline {
        &self
            .device
            .queue_timelines
            .as_ref()
            .expect("timeline enabled")[self.lane]
    }

    pub(in crate::backend) fn next_value(&self) -> Result<u64> {
        self.raw()
            .next_value
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                value.checked_add(1)
            })
            .map(|value| value + 1)
            .map_err(|_| Error::InvalidRange)
    }
}

pub(in crate::backend) struct Timeline {
    device: Arc<DeviceState>,
    lane: usize,
    pub(in crate::backend) semaphore: vk::Semaphore,
}
