//! Native D3D12 occlusion queries for the command path.
//!
//! Query-set ownership ends with its Rust handle, while individual begin/end
//! commands are encoded by the command module. This layer only creates and
//! validates the native heap; it deliberately does not infer pass boundaries
//! or completion, which remain Core and submission concerns.
//! Timestamp slots can move between queues after their preceding submitted
//! operation has a completed native fence. Pipeline statistics and occlusion remain direct-only.
use super::{Dx12, Dx12Buffer, Dx12Device, Dx12QuerySet};
use fluxel_rhi_core::api::{self, Error, Result};
use std::{ops::Range, sync::Arc};
use windows::Win32::Graphics::Direct3D12::*;

#[derive(Clone)]
pub(crate) struct SlotSubmission {
    pub(crate) lane: u32,
    /// None means work was issued but its completion is not yet proven.
    pub(crate) point: Option<(ID3D12Fence, u64)>,
}

pub(crate) struct QuerySetState {
    pub(crate) heap: ID3D12QueryHeap,
    pub(crate) count: u32,
    pub(crate) kind: api::QueryKind,
    pub(crate) submitted_slots: std::sync::Mutex<Vec<bool>>,
    pub(crate) submitted_lanes: std::sync::Mutex<Vec<Option<SlotSubmission>>>,
}

impl api::QueryBackend for Dx12 {
    fn query_supported(_: &Dx12Device, kind: api::QueryKind) -> bool {
        matches!(
            kind,
            api::QueryKind::Occlusion
                | api::QueryKind::Timestamp
                | api::QueryKind::PipelineStatistics
        )
    }
    fn timestamp_frequency(device: &Dx12Device, lane: api::LaneId) -> Option<u64> {
        let queue = match lane.ordinal {
            0 => &device.0.queue,
            1 => &device.0.multi_queue.queue,
            _ => return None,
        };
        // SAFETY: the selected native queue remains valid for this call.
        let frequency = unsafe { queue.GetTimestampFrequency() }.ok()?;
        (frequency != 0).then_some(frequency)
    }
    fn create_query_set(device: &Dx12Device, desc: &api::QueryDesc) -> Result<Dx12QuerySet> {
        if desc.count == 0 {
            return Err(Error::Unsupported("DX12 query type/count"));
        }
        let native = D3D12_QUERY_HEAP_DESC {
            Type: match desc.kind {
                api::QueryKind::Occlusion => D3D12_QUERY_HEAP_TYPE_OCCLUSION,
                api::QueryKind::Timestamp => D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
                api::QueryKind::PipelineStatistics => D3D12_QUERY_HEAP_TYPE_PIPELINE_STATISTICS,
            },
            Count: desc.count,
            NodeMask: 0,
        };
        let mut heap = None;
        // SAFETY: `native` is fully initialized, `heap` is valid writable
        // storage for the returned COM pointer, and `device` owns the D3D12
        // device for the duration of this synchronous call.
        unsafe { device.0.native.CreateQueryHeap(&native, &mut heap) }.map_err(err)?;
        let heap = heap.ok_or(Error::OutOfMemory)?;
        Ok(Dx12QuerySet(Arc::new(QuerySetState {
            heap,
            count: desc.count,
            kind: desc.kind,
            submitted_slots: std::sync::Mutex::new(vec![false; desc.count as usize]),
            submitted_lanes: std::sync::Mutex::new(vec![None; desc.count as usize]),
        })))
    }
    fn write_timestamp(
        encoder: &mut super::encoder::Dx12Encoder,
        set: &Dx12QuerySet,
        slot: u32,
    ) -> Result<()> {
        super::encoder::query::timestamp(encoder, set, slot)
    }
    fn begin_occlusion(
        encoder: &mut super::encoder::Dx12Encoder,
        set: &Dx12QuerySet,
        slot: u32,
    ) -> Result<()> {
        super::encoder::query::begin_occlusion(encoder, set, slot)
    }
    fn reset_queries(
        encoder: &mut super::encoder::Dx12Encoder,
        set: &Dx12QuerySet,
        slots: Range<u32>,
    ) -> Result<()> {
        super::encoder::query::reset(encoder, set, slots)
    }
    fn end_occlusion(encoder: &mut super::encoder::Dx12Encoder) -> Result<()> {
        super::encoder::query::end_occlusion(encoder)
    }
    fn begin_statistics(
        encoder: &mut super::encoder::Dx12Encoder,
        set: &Dx12QuerySet,
        slot: u32,
    ) -> Result<()> {
        super::encoder::query::begin_statistics(encoder, set, slot)
    }
    fn end_statistics(encoder: &mut super::encoder::Dx12Encoder) -> Result<()> {
        super::encoder::query::end_statistics(encoder)
    }
    fn resolve_queries(
        encoder: &mut super::encoder::Dx12Encoder,
        set: &Dx12QuerySet,
        slots: Range<u32>,
        dst: &Dx12Buffer,
        offset: u64,
    ) -> Result<()> {
        super::encoder::query::resolve(encoder, set, slots, dst, offset)
    }
}
fn err(e: windows::core::Error) -> Error {
    Error::Backend(format!("Direct3D 12: {e}"))
}
