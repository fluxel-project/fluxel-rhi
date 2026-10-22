//! D3D12 occlusion-query command lowering.
use super::*;
use crate::backend::{Dx12Buffer, Dx12QuerySet};
use fluxel_rhi_core::api::{BufferUsage, Error, MemoryClass, QueryKind, Result};
use std::ops::Range;

/// Ordered logical effects; recording never changes submitted slot state.
#[derive(Clone)]
pub(crate) enum QueryOp {
    Reset(Dx12QuerySet, Range<u32>),
    Write(Dx12QuerySet, u32),
    Resolve(Dx12QuerySet, Range<u32>),
}

impl QueryOp {
    fn set(&self) -> &Dx12QuerySet {
        match self {
            Self::Reset(s, _) | Self::Write(s, _) | Self::Resolve(s, _) => s,
        }
    }
    fn apply(&self, slots: &mut [bool]) -> Result<()> {
        match self {
            Self::Reset(_, range) => {
                for slot in range.clone() {
                    slots[slot as usize] = false;
                }
            }
            Self::Write(_, slot) => slots[*slot as usize] = true,
            Self::Resolve(_, range) => {
                if range.clone().any(|slot| !slots[slot as usize]) {
                    return Err(Error::InvalidState);
                }
            }
        }
        Ok(())
    }
}

/// Simulate the entire packet without changing submitted state.
pub(crate) fn validate_ops<'a>(ops: impl Iterator<Item = (u32, &'a QueryOp)>) -> Result<()> {
    let mut states = std::collections::HashMap::<usize, Vec<bool>>::new();
    let mut lanes = std::collections::HashMap::<usize, u32>::new();
    for (lane, op) in ops {
        let set = op.set();
        let key = key(set, 0).0;
        if set
            .0
            .submitted_lane
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .is_some_and(|owner| owner != lane)
            || lanes.insert(key, lane).is_some_and(|owner| owner != lane)
        {
            return Err(Error::Unsupported("DX12 query set across queues"));
        }
        let state = states.entry(key).or_insert_with(|| {
            set.0
                .submitted_slots
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .clone()
        });
        op.apply(state)?;
    }
    Ok(())
}

/// Commit only a command list already issued to the native queue.
pub(crate) fn commit_ops(ops: &[QueryOp], lane: u32) {
    for op in ops {
        *op.set()
            .0
            .submitted_lane
            .lock()
            .unwrap_or_else(|p| p.into_inner()) = Some(lane);
        let mut slots = op
            .set()
            .0
            .submitted_slots
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        // Entire packet was validated under the submission gate.
        op.apply(&mut slots).expect("validated query operations");
    }
}

/// A failed handoff or fence cannot establish which issued operations completed.
pub(crate) fn invalidate_ops(ops: &[QueryOp]) {
    for op in ops {
        op.set()
            .0
            .submitted_slots
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .fill(false);
    }
}

fn key(set: &Dx12QuerySet, slot: u32) -> (usize, u32) {
    (std::sync::Arc::as_ptr(&set.0) as usize, slot)
}

pub(crate) fn reset(
    encoder: &mut Dx12Encoder,
    set: &Dx12QuerySet,
    slots: Range<u32>,
) -> Result<()> {
    if encoder.is_compute && set.0.kind != QueryKind::Timestamp {
        return Err(Error::Unsupported("DX12 compute queue query kind"));
    }
    if slots.start >= slots.end || slots.end > set.0.count {
        return Err(Error::InvalidRange);
    }
    for active in [&encoder.active_occlusion, &encoder.active_statistics]
        .into_iter()
        .flatten()
    {
        if std::sync::Arc::ptr_eq(&active.0.0, &set.0) && slots.contains(&active.1) {
            return Err(Error::InvalidState);
        }
    }
    encoder
        .query_ops
        .push(QueryOp::Reset(set.clone(), slots.clone()));
    Ok(())
}

pub(crate) fn timestamp(encoder: &mut Dx12Encoder, set: &Dx12QuerySet, slot: u32) -> Result<()> {
    if encoder.is_compute && set.0.kind != QueryKind::Timestamp {
        return Err(Error::Unsupported("DX12 compute queue query kind"));
    }
    if set.0.kind != QueryKind::Timestamp || slot >= set.0.count {
        return Err(Error::InvalidRange);
    }
    // SAFETY: the query heap is retained and the slot was checked.
    unsafe {
        encoder
            .list
            .EndQuery(&set.0.heap, D3D12_QUERY_TYPE_TIMESTAMP, slot);
    }
    encoder.queries.push(set.clone());
    encoder.query_ops.push(QueryOp::Write(set.clone(), slot));
    Ok(())
}

pub(crate) fn begin_statistics(
    encoder: &mut Dx12Encoder,
    set: &Dx12QuerySet,
    slot: u32,
) -> Result<()> {
    if encoder.is_compute {
        return Err(Error::Unsupported("DX12 compute queue query"));
    }
    if set.0.kind != QueryKind::PipelineStatistics
        || slot >= set.0.count
        || encoder.active_statistics.is_some()
    {
        return Err(Error::InvalidState);
    }
    // SAFETY: the heap is retained, slot checked, and no statistics query is active.
    unsafe {
        encoder
            .list
            .BeginQuery(&set.0.heap, D3D12_QUERY_TYPE_PIPELINE_STATISTICS, slot);
    }
    encoder.active_statistics = Some((set.clone(), slot));
    encoder.queries.push(set.clone());
    Ok(())
}

pub(crate) fn end_statistics(encoder: &mut Dx12Encoder) -> Result<()> {
    let (set, slot) = encoder
        .active_statistics
        .take()
        .ok_or(Error::InvalidState)?;
    // SAFETY: the active record retains the heap and identifies the begun slot.
    unsafe {
        encoder
            .list
            .EndQuery(&set.0.heap, D3D12_QUERY_TYPE_PIPELINE_STATISTICS, slot);
    }
    encoder.query_ops.push(QueryOp::Write(set, slot));
    Ok(())
}

pub(crate) fn begin_occlusion(
    encoder: &mut Dx12Encoder,
    set: &Dx12QuerySet,
    slot: u32,
) -> Result<()> {
    if encoder.is_compute {
        return Err(Error::Unsupported("DX12 compute queue query"));
    }
    if set.0.kind != QueryKind::Occlusion
        || slot >= set.0.count
        || encoder.active_occlusion.is_some()
    {
        return Err(Error::InvalidState);
    }
    // SAFETY: set is same-device and retained by encoder; slot is checked and
    // this open list has no other active occlusion query.
    unsafe {
        encoder
            .list
            .BeginQuery(&set.0.heap, D3D12_QUERY_TYPE_OCCLUSION, slot);
    }
    encoder.active_occlusion = Some((set.clone(), slot));
    encoder.queries.push(set.clone());
    Ok(())
}

pub(crate) fn end_occlusion(encoder: &mut Dx12Encoder) -> Result<()> {
    let (set, slot) = encoder.active_occlusion.take().ok_or(Error::InvalidState)?;
    // SAFETY: `active_occlusion` proves this exact retained query/slot began
    // on this command list and is ended once here.
    unsafe {
        encoder
            .list
            .EndQuery(&set.0.heap, D3D12_QUERY_TYPE_OCCLUSION, slot);
    }
    encoder.query_ops.push(QueryOp::Write(set, slot));
    Ok(())
}

pub(crate) fn resolve(
    encoder: &mut Dx12Encoder,
    set: &Dx12QuerySet,
    slots: Range<u32>,
    dst: &Dx12Buffer,
    offset: u64,
) -> Result<()> {
    if encoder.is_compute && set.0.kind != QueryKind::Timestamp {
        return Err(Error::Unsupported("DX12 compute queue query kind"));
    }
    if encoder.active_occlusion.is_some()
        || encoder.active_statistics.is_some()
        || slots.start >= slots.end
        || slots.end > set.0.count
        || offset % 8 != 0
    {
        return Err(Error::InvalidRange);
    }
    let bytes = u64::from(slots.end - slots.start)
        .checked_mul(set.0.kind.result_stride())
        .ok_or(Error::InvalidRange)?;
    if offset.checked_add(bytes).ok_or(Error::InvalidRange)? > dst.size {
        return Err(Error::InvalidRange);
    }
    if !dst.usage.contains(BufferUsage::QUERY_RESOLVE)
        || (matches!(dst.memory, MemoryClass::Readback)
            && !dst.usage.contains(BufferUsage::MAP_READ))
    {
        return Err(Error::Unsupported("DX12 query resolve destination usage"));
    }
    if !matches!(dst.memory, MemoryClass::Readback) {
        barrier(
            &encoder.list,
            &dst.resource,
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_COPY_DEST,
        );
    }
    // SAFETY: set/destination are retained below, slot and byte range were
    // validated, and the destination is in COPY_DEST for this native command.
    unsafe {
        encoder.list.ResolveQueryData(
            &set.0.heap,
            match set.0.kind {
                QueryKind::Occlusion => D3D12_QUERY_TYPE_OCCLUSION,
                QueryKind::Timestamp => D3D12_QUERY_TYPE_TIMESTAMP,
                QueryKind::PipelineStatistics => D3D12_QUERY_TYPE_PIPELINE_STATISTICS,
            },
            slots.start,
            slots.end - slots.start,
            &dst.resource,
            offset,
        );
    }
    if !matches!(dst.memory, MemoryClass::Readback) {
        barrier(
            &encoder.list,
            &dst.resource,
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_COMMON,
        );
    }
    encoder.buffers.push(dst.clone());
    encoder.queries.push(set.clone());
    encoder.query_ops.push(QueryOp::Resolve(set.clone(), slots));
    Ok(())
}
