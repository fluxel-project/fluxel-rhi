//! Native Vulkan occlusion-query ownership and direct command lowering.
//!
//! A query set owns one `VkQueryPool`. Queries are begun and ended directly on
//! the active command buffer. Result copying and the next reset are encoded
//! after raster rendering, while reset is encoded explicitly before the raster
//! pass, so no CPU-side draw list or query scheduler exists.

use std::{ops::Range, sync::Arc};

use ash::vk;
use fluxel_rhi_core::api::{self, Error, Result};

use super::{DeviceState, VulkanBuffer, VulkanDevice, VulkanEncoder, vk_error};

#[derive(Clone)]
pub struct VulkanQuerySet(Arc<QuerySetState>);

impl VulkanQuerySet {
    pub(super) fn pool(&self) -> vk::QueryPool {
        self.0.pool
    }
}

struct QuerySetState {
    device: Arc<DeviceState>,
    pool: vk::QueryPool,
    count: u32,
}

impl Drop for QuerySetState {
    fn drop(&mut self) {
        unsafe { self.device.device.destroy_query_pool(self.pool, None) };
    }
}

pub(super) fn create(device: &VulkanDevice, desc: &api::QueryDesc) -> Result<VulkanQuerySet> {
    if !matches!(desc.kind, api::QueryKind::Occlusion) || desc.count == 0 {
        return Err(Error::Unsupported("Vulkan query type/count"));
    }
    let info = vk::QueryPoolCreateInfo::default()
        .query_type(vk::QueryType::OCCLUSION)
        .query_count(desc.count);
    let pool = unsafe { device.0.device.create_query_pool(&info, None) }.map_err(vk_error)?;
    Ok(VulkanQuerySet(Arc::new(QuerySetState {
        device: device.0.clone(),
        pool,
        count: desc.count,
    })))
}

pub(super) fn begin_occlusion(
    encoder: &mut VulkanEncoder,
    set: &VulkanQuerySet,
    slot: u32,
) -> Result<()> {
    if !encoder.raster
        || encoder.secondary_signature.is_some()
        || slot >= set.0.count
        || encoder.active_occlusion.is_some()
    {
        return Err(Error::InvalidState);
    }
    unsafe {
        encoder.device.device.cmd_begin_query(
            encoder.command,
            set.0.pool,
            slot,
            vk::QueryControlFlags::empty(),
        );
    }
    encoder.active_occlusion = Some((set.clone(), slot));
    encoder.queries.push(set.clone());
    Ok(())
}

pub(super) fn reset(
    encoder: &mut VulkanEncoder,
    set: &VulkanQuerySet,
    slots: Range<u32>,
) -> Result<()> {
    if encoder.raster
        || encoder.compute
        || encoder.active_occlusion.is_some()
        || slots.start >= slots.end
        || slots.end > set.0.count
    {
        return Err(Error::InvalidRange);
    }
    unsafe {
        encoder.device.device.cmd_reset_query_pool(
            encoder.command,
            set.0.pool,
            slots.start,
            slots.end - slots.start,
        );
    }
    encoder.queries.push(set.clone());
    Ok(())
}

pub(super) fn end_occlusion(encoder: &mut VulkanEncoder) -> Result<()> {
    // A dynamic-rendering instance which inherits secondary commands must be
    // populated exclusively by those secondaries.  Reject this before taking
    // the active query so the caller can still abort the raster pass cleanly.
    if encoder.secondary_signature.is_some() {
        return Err(Error::InvalidState);
    }
    let (set, slot) = encoder.active_occlusion.take().ok_or(Error::InvalidState)?;
    unsafe {
        encoder
            .device
            .device
            .cmd_end_query(encoder.command, set.0.pool, slot);
    }
    Ok(())
}

pub(super) fn resolve(
    encoder: &mut VulkanEncoder,
    set: &VulkanQuerySet,
    slots: Range<u32>,
    destination: &VulkanBuffer,
    offset: u64,
) -> Result<()> {
    if encoder.raster
        || encoder.active_occlusion.is_some()
        || slots.start >= slots.end
        || slots.end > set.0.count
        || offset % 8 != 0
        || !destination
            .0
            .usage
            .contains(api::BufferUsage::QUERY_RESOLVE)
        || !destination.0.usage.contains(api::BufferUsage::MAP_READ)
    {
        return Err(Error::InvalidRange);
    }
    let count = slots.end - slots.start;
    let bytes = u64::from(count).checked_mul(8).ok_or(Error::InvalidRange)?;
    if offset.checked_add(bytes).ok_or(Error::InvalidRange)? > destination.0.size {
        return Err(Error::InvalidRange);
    }
    unsafe {
        encoder.device.device.cmd_copy_query_pool_results(
            encoder.command,
            set.0.pool,
            slots.start,
            count,
            destination.0.buffer,
            offset,
            8,
            vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
        );
    }
    encoder.buffers.push(destination.clone());
    encoder.queries.push(set.clone());
    Ok(())
}
