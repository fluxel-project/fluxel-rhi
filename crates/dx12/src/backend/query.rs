//! Native D3D12 occlusion queries for the command path.
use super::{Dx12, Dx12Buffer, Dx12Device, Dx12QuerySet};
use fluxel_rhi_core::api::{self, Error, Result};
use std::{ops::Range, sync::Arc};
use windows::Win32::Graphics::Direct3D12::*;

pub(crate) struct QuerySetState {
    pub(crate) heap: ID3D12QueryHeap,
    pub(crate) count: u32,
}

impl api::QueryBackend for Dx12 {
    fn query_supported(_: &Dx12Device, kind: api::QueryKind) -> bool {
        matches!(kind, api::QueryKind::Occlusion)
    }
    fn create_query_set(device: &Dx12Device, desc: &api::QueryDesc) -> Result<Dx12QuerySet> {
        if !matches!(desc.kind, api::QueryKind::Occlusion) || desc.count == 0 {
            return Err(Error::Unsupported("DX12 query type/count"));
        }
        let native = D3D12_QUERY_HEAP_DESC {
            Type: D3D12_QUERY_HEAP_TYPE_OCCLUSION,
            Count: desc.count,
            NodeMask: 0,
        };
        let mut heap = None;
        unsafe { device.0.native.CreateQueryHeap(&native, &mut heap) }.map_err(err)?;
        let heap = heap.ok_or(Error::OutOfMemory)?;
        Ok(Dx12QuerySet(Arc::new(QuerySetState {
            heap,
            count: desc.count,
        })))
    }
    fn write_timestamp(
        _: &mut super::encoder::Dx12Encoder,
        _: &Dx12QuerySet,
        _: u32,
    ) -> Result<()> {
        Err(Error::Unsupported("DX12 timestamps"))
    }
    fn begin_occlusion(
        encoder: &mut super::encoder::Dx12Encoder,
        set: &Dx12QuerySet,
        slot: u32,
    ) -> Result<()> {
        super::encoder::query::begin_occlusion(encoder, set, slot)
    }
    fn end_occlusion(encoder: &mut super::encoder::Dx12Encoder) -> Result<()> {
        super::encoder::query::end_occlusion(encoder)
    }
    fn begin_statistics(
        _: &mut super::encoder::Dx12Encoder,
        _: &Dx12QuerySet,
        _: u32,
    ) -> Result<()> {
        Err(Error::Unsupported("DX12 pipeline statistics"))
    }
    fn end_statistics(_: &mut super::encoder::Dx12Encoder) -> Result<()> {
        Err(Error::Unsupported("DX12 pipeline statistics"))
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
