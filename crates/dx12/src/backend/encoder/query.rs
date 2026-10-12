//! D3D12 occlusion-query command lowering.
use super::*;
use crate::backend::{Dx12Buffer, Dx12QuerySet};
use fluxel_rhi_core::api::{BufferUsage, Error, MemoryClass, Result};
use std::ops::Range;

pub(crate) fn begin_occlusion(
    encoder: &mut Dx12Encoder,
    set: &Dx12QuerySet,
    slot: u32,
) -> Result<()> {
    if slot >= set.0.count || encoder.active_occlusion.is_some() {
        return Err(Error::InvalidState);
    }
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
    unsafe {
        encoder
            .list
            .EndQuery(&set.0.heap, D3D12_QUERY_TYPE_OCCLUSION, slot);
    }
    Ok(())
}

pub(crate) fn resolve(
    encoder: &mut Dx12Encoder,
    set: &Dx12QuerySet,
    slots: Range<u32>,
    dst: &Dx12Buffer,
    offset: u64,
) -> Result<()> {
    if encoder.active_occlusion.is_some()
        || slots.start >= slots.end
        || slots.end > set.0.count
        || offset % 8 != 0
    {
        return Err(Error::InvalidRange);
    }
    let bytes = u64::from(slots.end - slots.start)
        .checked_mul(8)
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
    unsafe {
        encoder.list.ResolveQueryData(
            &set.0.heap,
            D3D12_QUERY_TYPE_OCCLUSION,
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
    Ok(())
}
