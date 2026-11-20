//! Exact-sized DXR compaction using a private readback and submitted build identity.

use std::sync::{Arc, Mutex, OnceLock};

use fluxel_rhi_core::{
    api::{self, Error, Result},
    backend::{AccelerationCompactionBackend, ReadbackBackend, ResourceBackend},
};
use windows::{Win32::Graphics::Direct3D12::*, core::Interface};

use super::{
    Dx12, Dx12Acceleration, RecordedAccelerationBuild, allocate_acceleration, build, committed,
};
use crate::backend::{
    Dx12Buffer,
    encoder::{Dx12Completion, Dx12Encoder, barrier},
    native_error,
    readback::{self, Dx12Readback},
};

pub(crate) struct CompactionReceipt {
    pub(super) revision: Arc<()>,
    completion: OnceLock<Dx12Completion>,
}

impl CompactionReceipt {
    pub(super) fn seal(&self, fence: &ID3D12Fence, value: u64) {
        let _ = self.completion.set(Dx12Completion {
            fence: fence.clone(),
            value,
        });
    }
}

pub struct Dx12CompactionTicket {
    source: Dx12Acceleration,
    receipt: Arc<CompactionReceipt>,
    readback: Dx12Readback,
    size: Mutex<Option<u64>>,
}

impl AccelerationCompactionBackend for Dx12 {
    type NativeCompactionTicket = Dx12CompactionTicket;

    fn record_compactable_build(
        encoder: &mut Dx12Encoder,
        source: &Dx12Acceleration,
        scratch: &Dx12Buffer,
        desc: &api::AccelerationDesc<Dx12>,
        refit: bool,
    ) -> Result<Dx12CompactionTicket> {
        let output = Dx12::create_buffer(
            &encoder.device,
            &api::BufferDesc {
                label: None,
                size: 8,
                usage: api::BufferUsage::STORAGE | api::BufferUsage::COPY_SRC,
                memory: api::MemoryClass::Device,
            },
        )?;
        // This private buffer has no caller-visible resource states.
        barrier(
            &encoder.list,
            &output.resource,
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        );
        build(
            encoder,
            source,
            scratch,
            desc,
            refit,
            Some((&output, 0, api::AccelerationSizeKind::Compacted)),
        )?;
        barrier(
            &encoder.list,
            &output.resource,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        );
        let readback = Dx12::encode_readback(encoder, &output, 0..8)?;
        let receipt = Arc::new(CompactionReceipt {
            revision: Arc::new(()),
            completion: OnceLock::new(),
        });
        encoder
            .ray_builds
            .last_mut()
            .expect("the build recorded acceleration metadata")
            .receipt = Some(receipt.clone());
        Ok(Dx12CompactionTicket {
            source: source.clone(),
            receipt,
            readback,
            size: Mutex::new(None),
        })
    }

    fn compacted_size(ticket: &Dx12CompactionTicket) -> Result<Option<u64>> {
        let Some(point) = ticket.receipt.completion.get() else {
            return Ok(None);
        };
        // SAFETY: the receipt owns the native fence signalled after its readback copy.
        let value = unsafe { point.fence.GetCompletedValue() };
        if value == u64::MAX {
            return Err(Error::DeviceLost);
        }
        if value < point.value {
            return Ok(None);
        }
        let mut cached = ticket.size.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(size) = *cached {
            return Ok(Some(size));
        }
        let bytes = readback::map(ticket.readback.clone())?.bytes;
        let size = u64::from_ne_bytes(bytes.try_into().map_err(|_| Error::InvalidState)?);
        if size == 0 || size > ticket.source.0.size {
            return Err(Error::InvalidRange);
        }
        *cached = Some(size);
        Ok(Some(size))
    }

    fn compact_acceleration(
        encoder: &mut Dx12Encoder,
        ticket: &Dx12CompactionTicket,
    ) -> Result<Dx12Acceleration> {
        let size = Self::compacted_size(ticket)?.ok_or(Error::InvalidState)?;
        let source = &ticket.source;
        let (signature, blas) = {
            let _gate = encoder
                .device
                .0
                .query_submission_gate
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            if !source
                .0
                .committed_revision
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .as_ref()
                .is_some_and(|revision| Arc::ptr_eq(revision, &ticket.receipt.revision))
            {
                return Err(Error::InvalidState);
            }
            let signature = source
                .0
                .committed_signature
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .clone();
            (signature, committed(source))
        };
        let destination = allocate_acceleration(&encoder.device, size, source.0.is_top_level)?;
        let list = encoder
            .list
            .cast::<ID3D12GraphicsCommandList4>()
            .map_err(native_error)?;
        // SAFETY: both distinct resources use the acceleration structure state.
        // Submission validates that the measured build still occupies the source.
        unsafe {
            list.CopyRaytracingAccelerationStructure(
                destination.0.resource.GetGPUVirtualAddress(),
                source.0.resource.GetGPUVirtualAddress(),
                D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_COMPACT,
            );
        }
        encoder
            .accelerations
            .extend([source.clone(), destination.clone()]);
        encoder.ray_builds.push(RecordedAccelerationBuild {
            target: destination.clone(),
            blas,
            segment: encoder.closed_segments.len(),
            signature,
            refit: false,
            receipt: None,
            required_source: Some((source.clone(), ticket.receipt.revision.clone())),
        });
        Ok(destination)
    }
}
