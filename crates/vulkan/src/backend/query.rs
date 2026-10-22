//! Native Vulkan occlusion-query ownership and direct command lowering.
//!
//! A query set owns one `VkQueryPool`. Queries are begun and ended directly on
//! the active command buffer. Result copying and the next reset are encoded
//! after raster rendering, while reset is encoded explicitly before the raster
//! pass, so no CPU-side draw list or query scheduler exists.
//! `active_occlusion` is the sole begin/end state: it prevents nesting and is
//! forcibly balanced by raster abort. The caller supplies all cross-command
//! synchronization; this module only records Vulkan query-pool operations.
//! The submit gate validates slot initialization in submission order. The
//! caller supplies execution dependencies between uses of the same slot on
//! different queues, just as for other GPU resources.

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
    kind: api::QueryKind,
    submitted: std::sync::Mutex<Vec<Option<(u32, bool)>>>,
}

impl Drop for QuerySetState {
    fn drop(&mut self) {
        // SAFETY: the final Arc owns this pool; submitted encoders retain the
        // query set until their completion fence and the device stays alive.
        unsafe { self.device.device.destroy_query_pool(self.pool, None) };
    }
}

pub(super) fn create(device: &VulkanDevice, desc: &api::QueryDesc) -> Result<VulkanQuerySet> {
    if desc.count == 0 {
        return Err(Error::InvalidRange);
    }
    let query_type = match desc.kind {
        api::QueryKind::Occlusion => vk::QueryType::OCCLUSION,
        api::QueryKind::Timestamp
            if device.0.timestamp_bits > 0
                || (device.0.compute_queue.is_some() && device.0.compute_timestamp_bits > 0) =>
        {
            vk::QueryType::TIMESTAMP
        }
        api::QueryKind::PipelineStatistics if device.0.pipeline_statistics => {
            vk::QueryType::PIPELINE_STATISTICS
        }
        _ => return Err(Error::Unsupported("Vulkan query type")),
    };
    let stats = vk::QueryPipelineStatisticFlags::INPUT_ASSEMBLY_VERTICES
        | vk::QueryPipelineStatisticFlags::INPUT_ASSEMBLY_PRIMITIVES
        | vk::QueryPipelineStatisticFlags::VERTEX_SHADER_INVOCATIONS
        | vk::QueryPipelineStatisticFlags::GEOMETRY_SHADER_INVOCATIONS
        | vk::QueryPipelineStatisticFlags::GEOMETRY_SHADER_PRIMITIVES
        | vk::QueryPipelineStatisticFlags::CLIPPING_INVOCATIONS
        | vk::QueryPipelineStatisticFlags::CLIPPING_PRIMITIVES
        | vk::QueryPipelineStatisticFlags::FRAGMENT_SHADER_INVOCATIONS
        | vk::QueryPipelineStatisticFlags::TESSELLATION_CONTROL_SHADER_PATCHES
        | vk::QueryPipelineStatisticFlags::TESSELLATION_EVALUATION_SHADER_INVOCATIONS
        | vk::QueryPipelineStatisticFlags::COMPUTE_SHADER_INVOCATIONS;
    let info = vk::QueryPoolCreateInfo::default()
        .query_type(query_type)
        .query_count(desc.count)
        .pipeline_statistics(if desc.kind == api::QueryKind::PipelineStatistics {
            stats
        } else {
            vk::QueryPipelineStatisticFlags::empty()
        });
    // SAFETY: DeviceState owns the live device; the returned pool is moved into
    // the new QuerySetState that retains it until destruction.
    let pool = unsafe { device.0.device.create_query_pool(&info, None) }.map_err(vk_error)?;
    Ok(VulkanQuerySet(Arc::new(QuerySetState {
        device: device.0.clone(),
        pool,
        count: desc.count,
        kind: desc.kind,
        submitted: std::sync::Mutex::new(vec![None; desc.count as usize]),
    })))
}

pub(super) fn begin_occlusion(
    encoder: &mut VulkanEncoder,
    set: &VulkanQuerySet,
    slot: u32,
) -> Result<()> {
    if set.0.kind != api::QueryKind::Occlusion
        || !encoder.raster
        || encoder.secondary_signature.is_some()
        || slot >= set.0.count
        || encoder.active_occlusion.is_some()
    {
        return Err(Error::InvalidState);
    }
    // SAFETY: raster state proves this is a recording primary command buffer;
    // slot is in the retained pool's range and no query is currently active.
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
    // SAFETY: query reset is outside rendering and the checked slot range is
    // contained by the retained native query pool.
    unsafe {
        encoder.device.device.cmd_reset_query_pool(
            encoder.command,
            set.0.pool,
            slots.start,
            slots.end - slots.start,
        );
    }
    encoder.query_ops.push(QueryOp::Reset(set.clone(), slots));
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
    // SAFETY: `active_occlusion` holds the matching retained pool and slot on
    // this recording command buffer, so the native begin/end pair is balanced.
    unsafe {
        encoder
            .device
            .device
            .cmd_end_query(encoder.command, set.0.pool, slot);
    }
    encoder.query_ops.push(QueryOp::Write(set, slot));
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
    {
        return Err(Error::InvalidRange);
    }
    let count = slots.end - slots.start;
    let stride = set.0.kind.result_stride();
    let bytes = u64::from(count)
        .checked_mul(stride)
        .ok_or(Error::InvalidRange)?;
    if offset.checked_add(bytes).ok_or(Error::InvalidRange)? > destination.0.size {
        return Err(Error::InvalidRange);
    }
    // SAFETY: the command is outside rendering, all source slots and the
    // aligned destination range were checked, and both objects are retained.
    unsafe {
        encoder.device.device.cmd_copy_query_pool_results(
            encoder.command,
            set.0.pool,
            slots.start,
            count,
            destination.0.buffer,
            offset,
            stride,
            vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
        );
    }
    encoder.query_ops.push(QueryOp::Resolve(set.clone(), slots));
    encoder.buffers.push(destination.clone());
    encoder.queries.push(set.clone());
    Ok(())
}

#[derive(Clone)]
pub(super) enum QueryOp {
    Reset(VulkanQuerySet, Range<u32>),
    Write(VulkanQuerySet, u32),
    Resolve(VulkanQuerySet, Range<u32>),
}
impl QueryOp {
    fn set(&self) -> &VulkanQuerySet {
        match self {
            Self::Reset(s, _) | Self::Write(s, _) | Self::Resolve(s, _) => s,
        }
    }
    fn apply(&self, slots: &mut [Option<(u32, bool)>], lane: u32) -> Result<()> {
        match self {
            Self::Reset(_, range) => {
                for slot in range.clone() {
                    slots[slot as usize] = Some((lane, false));
                }
            }
            Self::Write(_, slot) => slots[*slot as usize] = Some((lane, true)),
            Self::Resolve(_, range) => {
                if range
                    .clone()
                    .any(|slot| !slots[slot as usize].is_some_and(|(_, written)| written))
                {
                    return Err(Error::InvalidState);
                }
            }
        }
        Ok(())
    }
}
pub(super) fn validate_ops<'a>(ops: impl Iterator<Item = (u32, &'a QueryOp)>) -> Result<()> {
    let mut sets = std::collections::HashMap::<usize, Vec<Option<(u32, bool)>>>::new();
    for (lane, op) in ops {
        let set = op.set();
        let key = Arc::as_ptr(&set.0) as usize;
        let slots = sets.entry(key).or_insert_with(|| {
            set.0
                .submitted
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .clone()
        });
        op.apply(slots, lane)?;
    }
    Ok(())
}
pub(super) fn commit_ops(ops: &[QueryOp], lane: u32) {
    for op in ops {
        let mut slots = op
            .set()
            .0
            .submitted
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        op.apply(&mut slots, lane).expect("preflighted query state");
    }
}
pub(super) fn invalidate_ops(ops: &[QueryOp], lane: u32) {
    for op in ops {
        let mut slots = op
            .set()
            .0
            .submitted
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let range = match op {
            QueryOp::Reset(_, range) | QueryOp::Resolve(_, range) => range.clone(),
            QueryOp::Write(_, slot) => *slot..*slot + 1,
        };
        for slot in range {
            slots[slot as usize] = Some((lane, false));
        }
    }
}
pub(super) fn timestamp(
    encoder: &mut VulkanEncoder,
    set: &VulkanQuerySet,
    slot: u32,
) -> Result<()> {
    if set.0.kind != api::QueryKind::Timestamp
        || (if encoder.lane.ordinal == 1 {
            encoder.device.compute_timestamp_bits
        } else {
            encoder.device.timestamp_bits
        }) == 0
        || slot >= set.0.count
        || encoder.raster
        || encoder.compute
    {
        return Err(Error::InvalidState);
    }
    // SAFETY: pool and index are valid; the command buffer retains the pool.
    unsafe {
        encoder
            .device
            .device
            .cmd_reset_query_pool(encoder.command, set.0.pool, slot, 1);
        encoder.device.device.cmd_write_timestamp(
            encoder.command,
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            set.0.pool,
            slot,
        );
    }
    encoder.query_ops.push(QueryOp::Write(set.clone(), slot));
    encoder.queries.push(set.clone());
    Ok(())
}

/// Reset pass timestamp slots before entering dynamic rendering, then record
/// the beginning timestamp. The caller has validated all other pass state.
pub(super) fn begin_pass_timestamps(
    encoder: &mut VulkanEncoder,
    writes: Option<&api::PassTimestampWrites<'_, super::Vulkan>>,
) -> Result<()> {
    let Some(writes) = writes else { return Ok(()) };
    // SAFETY: Core validates the set identity and retains the descriptor for
    // this call; the encoder retains its own clone for the native commands.
    let set = unsafe { writes.set.native_ref() };
    if set.0.kind != api::QueryKind::Timestamp
        || encoder.device.timestamp_bits == 0
        || encoder.pass_timestamp_end.is_some()
    {
        return Err(Error::InvalidState);
    }
    for slot in [writes.begin, writes.end].into_iter().flatten() {
        if slot >= set.0.count {
            return Err(Error::InvalidRange);
        }
    }
    // SAFETY: both slots are in the retained pool and reset outside rendering.
    unsafe {
        for slot in [
            writes.begin,
            writes.end.filter(|slot| Some(*slot) != writes.begin),
        ]
        .into_iter()
        .flatten()
        {
            encoder
                .device
                .device
                .cmd_reset_query_pool(encoder.command, set.0.pool, slot, 1);
        }
        if let Some(slot) = writes.begin {
            encoder.device.device.cmd_write_timestamp(
                encoder.command,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                set.0.pool,
                slot,
            );
        }
    }
    if let Some(slot) = writes.begin {
        encoder.query_ops.push(QueryOp::Write(set.clone(), slot));
    }
    encoder.pass_timestamp_end = writes
        .end
        .map(|slot| (set.clone(), slot, writes.begin == Some(slot)));
    encoder.queries.push(set.clone());
    Ok(())
}

/// Record the matching end timestamp after leaving dynamic rendering.
pub(super) fn end_pass_timestamps(encoder: &mut VulkanEncoder) {
    if let Some((set, slot, reset_again)) = encoder.pass_timestamp_end.take() {
        // SAFETY: the slot was reset before this pass and its pool is retained.
        unsafe {
            if reset_again {
                // The beginning sample must finish before its slot is reset.
                encoder.device.device.cmd_pipeline_barrier(
                    encoder.command,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[],
                );
                encoder
                    .device
                    .device
                    .cmd_reset_query_pool(encoder.command, set.0.pool, slot, 1);
            }
            encoder.device.device.cmd_write_timestamp(
                encoder.command,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                set.0.pool,
                slot,
            );
        }
        encoder.query_ops.push(QueryOp::Write(set, slot));
    }
}
pub(super) fn begin_statistics(
    encoder: &mut VulkanEncoder,
    set: &VulkanQuerySet,
    slot: u32,
) -> Result<()> {
    if set.0.kind != api::QueryKind::PipelineStatistics
        || slot >= set.0.count
        || encoder.raster
        || encoder.compute
        || encoder.active_statistics.is_some()
    {
        return Err(Error::InvalidState);
    }
    // SAFETY: begin and reset are outside a pass, and pool/slot are retained.
    unsafe {
        encoder
            .device
            .device
            .cmd_reset_query_pool(encoder.command, set.0.pool, slot, 1);
        encoder.device.device.cmd_begin_query(
            encoder.command,
            set.0.pool,
            slot,
            vk::QueryControlFlags::empty(),
        );
    }
    encoder.active_statistics = Some((set.clone(), slot));
    encoder.queries.push(set.clone());
    Ok(())
}
pub(super) fn end_statistics(encoder: &mut VulkanEncoder) -> Result<()> {
    if encoder.raster || encoder.compute {
        return Err(Error::InvalidState);
    }
    let (set, slot) = encoder
        .active_statistics
        .take()
        .ok_or(Error::InvalidState)?;
    // SAFETY: this exact retained query was begun and is ended exactly once.
    unsafe {
        encoder
            .device
            .device
            .cmd_end_query(encoder.command, set.0.pool, slot);
    }
    encoder.query_ops.push(QueryOp::Write(set, slot));
    Ok(())
}

/// Return an integer frequency only when the binary32 period represents it exactly.
pub(super) fn integer_frequency(period: f32) -> Option<u64> {
    if !period.is_finite() || period <= 0.0 {
        return None;
    }
    let bits = period.to_bits();
    let e = ((bits >> 23) & 255) as i32;
    let mantissa = if e == 0 {
        bits & 0x7f_ffff
    } else {
        (bits & 0x7f_ffff) | (1 << 23)
    };
    let exponent = if e == 0 { -149 } else { e - 150 };
    let (n, d) = if exponent >= 0 {
        (
            1_000_000_000u128,
            u128::from(mantissa).checked_mul(1u128.checked_shl(exponent as u32)?)?,
        )
    } else {
        (
            1_000_000_000u128.checked_mul(1u128.checked_shl((-exponent) as u32)?)?,
            u128::from(mantissa),
        )
    };
    if d == 0 || n % d != 0 {
        return None;
    }
    u64::try_from(n / d).ok().filter(|v| *v != 0)
}
#[cfg(test)]
mod tests {
    #[test]
    fn exact_frequency_only() {
        assert_eq!(super::integer_frequency(10.0), Some(100_000_000));
        assert_eq!(super::integer_frequency(3.0), None);
        assert_eq!(super::integer_frequency(0.0), None);
        assert_eq!(super::integer_frequency(f32::MIN_POSITIVE), None);
    }
}
