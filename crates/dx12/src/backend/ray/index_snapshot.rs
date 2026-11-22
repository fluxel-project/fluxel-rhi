//! Private DXR index snapshots and their one-time native initialization.
//!
//! The caller supplies producer, copy, and consumer ordering. Publication here
//! proves that the initializing list was handed to the driver, not completion.

use std::ops::Range;

use super::{PreparedInputId, PreparedInputOp};
use fluxel_rhi_core::{
    api::{self, Error, Result},
    backend::{CommandBackend, ResourceBackend},
};
use windows::Win32::Graphics::Direct3D12::*;

use crate::backend::{
    Dx12, Dx12Buffer,
    encoder::{Dx12Encoder, barrier},
};

#[derive(Clone)]
pub struct Dx12IndexSnapshot {
    pub(super) storage: Dx12Buffer,
    pub(super) id: PreparedInputId,
}

pub(super) fn record(
    encoder: &mut Dx12Encoder,
    source: &Dx12Buffer,
    range: Range<u64>,
) -> Result<Dx12IndexSnapshot> {
    encoder.device.require_active()?;
    super::ray_device(&encoder.device)?;
    if range.start >= range.end
        || range.end > source.size
        || source.memory == api::MemoryClass::Readback
    {
        return Err(Error::InvalidRange);
    }
    if !source.usage.contains(api::BufferUsage::COPY_SRC) {
        return Err(Error::Unsupported("DX12 index snapshot requires COPY_SRC"));
    }
    let size = range.end - range.start;
    let storage = Dx12::create_buffer(
        &encoder.device,
        &api::BufferDesc {
            label: None,
            size,
            usage: api::BufferUsage::COPY_DST | api::BufferUsage::ACCEL_BUILD_INPUT,
            memory: api::MemoryClass::Device,
        },
    )?;
    let id = PreparedInputId::new();
    barrier(
        &encoder.list,
        &storage.resource,
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_COPY_DEST,
    );
    Dx12::copy_buffer(encoder, source, range.start, &storage, 0, size)?;
    barrier(
        &encoder.list,
        &storage.resource,
        D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
    );
    encoder.ray_inputs.push(PreparedInputOp::Initialize {
        id: id.clone(),
        segment: encoder.closed_segments.len(),
    });
    Ok(Dx12IndexSnapshot { storage, id })
}
