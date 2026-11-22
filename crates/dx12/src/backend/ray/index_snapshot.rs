//! Private DXR index snapshots and their one-time native initialization.
//!
//! The caller supplies producer, copy, and consumer ordering. Publication here
//! proves that the initializing list was handed to the driver, not completion.

use std::{
    collections::HashSet,
    ops::Range,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

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
pub(crate) struct IndexSnapshotId(Arc<AtomicBool>);

impl PartialEq for IndexSnapshotId {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for IndexSnapshotId {}

#[derive(Clone)]
pub struct Dx12IndexSnapshot {
    pub(super) storage: Dx12Buffer,
    pub(super) id: IndexSnapshotId,
}

pub(crate) enum IndexSnapshotOp {
    Copy { id: IndexSnapshotId, segment: usize },
    Read(IndexSnapshotId),
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
    let id = IndexSnapshotId(Arc::new(AtomicBool::new(false)));
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
    encoder.ray_index_snapshots.push(IndexSnapshotOp::Copy {
        id: id.clone(),
        segment: encoder.closed_segments.len(),
    });
    Ok(Dx12IndexSnapshot { storage, id })
}

#[derive(Default)]
pub(super) struct SnapshotAdmission {
    copies: Option<HashSet<*const AtomicBool>>,
}

impl SnapshotAdmission {
    pub(super) fn validate(&mut self, ops: &[IndexSnapshotOp]) -> Result<()> {
        for op in ops {
            match op {
                IndexSnapshotOp::Copy { id, .. } => {
                    if id.0.load(Ordering::Relaxed)
                        || !self
                            .copies
                            .get_or_insert_with(HashSet::new)
                            .insert(Arc::as_ptr(&id.0))
                    {
                        return Err(Error::InvalidState);
                    }
                }
                IndexSnapshotOp::Read(id) => {
                    if !id.0.load(Ordering::Relaxed)
                        && !self
                            .copies
                            .as_ref()
                            .is_some_and(|copies| copies.contains(&Arc::as_ptr(&id.0)))
                    {
                        return Err(Error::InvalidState);
                    }
                }
            }
        }
        Ok(())
    }
}

pub(super) fn publish(ops: &[IndexSnapshotOp], segment: usize) {
    for op in ops {
        if let IndexSnapshotOp::Copy {
            id,
            segment: copy_segment,
        } = op
        {
            if *copy_segment == segment {
                // All admission and publication run under the submission gate.
                id.0.store(true, Ordering::Relaxed);
            }
        }
    }
}
