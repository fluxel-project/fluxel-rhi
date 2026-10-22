//! Internal residency for implementation-defined sparse image metadata.
use super::*;

pub(super) struct Metadata {
    device: Arc<DeviceState>,
    memory: vk::DeviceMemory,
    binds: Vec<vk::SparseMemoryBind>,
    frontier: Mutex<
        Option<(
            vk::Queue,
            Arc<super::super::submit::multi_queue::Timeline>,
            u64,
        )>,
    >,
}
impl Metadata {
    pub(super) fn new(
        device: &VulkanDevice,
        req: vk::MemoryRequirements,
        sparse: &[vk::SparseImageMemoryRequirements],
        layers: u32,
    ) -> Result<Option<Self>> {
        if req.alignment == 0 {
            return Err(Error::InvalidState);
        }
        let mut regions = Vec::new();
        let mut total = 0u64;
        for item in sparse
            .iter()
            .filter(|s| s.format_properties.aspect_mask == vk::ImageAspectFlags::METADATA)
        {
            let count = if item
                .format_properties
                .flags
                .contains(vk::SparseImageFormatFlags::SINGLE_MIPTAIL)
            {
                1
            } else {
                layers
            };
            if item.image_mip_tail_size == 0 || item.image_mip_tail_size % req.alignment != 0 {
                return Err(Error::Unsupported("Vulkan sparse metadata alignment"));
            }
            for layer in 0..count {
                let offset = item
                    .image_mip_tail_offset
                    .checked_add(
                        u64::from(layer)
                            .checked_mul(item.image_mip_tail_stride)
                            .ok_or(Error::InvalidRange)?,
                    )
                    .ok_or(Error::InvalidRange)?;
                regions.push((offset, item.image_mip_tail_size, total));
                total = total
                    .checked_add(item.image_mip_tail_size)
                    .ok_or(Error::InvalidRange)?;
            }
        }
        if regions.is_empty() {
            return Ok(None);
        }
        let mut metadata_req = req;
        metadata_req.size = total;
        let memory = resource::allocate_with_address(
            device,
            metadata_req,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            false,
        )?;
        let binds = regions
            .into_iter()
            .map(|(offset, size, memory_offset)| vk::SparseMemoryBind {
                resource_offset: offset,
                size,
                memory,
                memory_offset,
                flags: vk::SparseMemoryBindFlags::METADATA,
            })
            .collect();
        Ok(Some(Self {
            device: device.0.clone(),
            memory,
            binds,
            frontier: Mutex::new(None),
        }))
    }
    pub(super) fn prepare(
        &self,
        queue: vk::Queue,
        binds: &mut Vec<vk::SparseMemoryBind>,
        waits: &mut Vec<(vk::Semaphore, u64)>,
    ) -> bool {
        let frontier = self.frontier.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((producer, timeline, value)) = &*frontier {
            metadata_wait(*producer, queue, timeline.semaphore, *value, waits);
            false
        } else {
            binds.extend_from_slice(&self.binds);
            true
        }
    }
    pub(super) fn mark_bound(
        &self,
        queue: vk::Queue,
        timeline: Arc<super::super::submit::multi_queue::Timeline>,
        value: u64,
    ) {
        *self.frontier.lock().unwrap_or_else(|e| e.into_inner()) = Some((queue, timeline, value));
    }
}

// Caller holds the texture mapping lock across prepare, native submission and
// mark_bound, so concurrent binds cannot both install metadata.
fn metadata_wait(
    producer: vk::Queue,
    queue: vk::Queue,
    semaphore: vk::Semaphore,
    value: u64,
    waits: &mut Vec<(vk::Semaphore, u64)>,
) {
    if producer != queue {
        if let Some(wait) = waits.iter_mut().find(|wait| wait.0 == semaphore) {
            wait.1 = wait.1.max(value);
        } else {
            waits.push((semaphore, value));
        }
    }
}

#[cfg(test)]
mod tests {
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
}
impl Drop for Metadata {
    fn drop(&mut self) {
        // SAFETY: TextureState destroys the image before dropping metadata;
        // submitted work retains that state through completion.
        unsafe { self.device.device.free_memory(self.memory, None) }
    }
}
