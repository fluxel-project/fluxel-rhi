//! Completion-gated mapping of DX12 readback buffers.
//!
//! A private staging ticket retains exactly its copied buffer and byte range.
//! An encoder-owned receipt records the first covering backend queue signal.
//! Shared-source tickets still require external coverage and
//! exclusion from later writes.

use std::future::{Ready, ready};
use std::ops::Range;
use std::sync::{Arc, OnceLock};

use fluxel_rhi_core::api::{self, Error, ReadbackData, Result};

use crate::backend::encoder::{Dx12Completion, Dx12Encoder};
use crate::backend::{Dx12, Dx12Buffer, Dx12Device, mapping};
use fluxel_rhi_core::backend::CompletionBackend;
use windows::Win32::Graphics::Direct3D12::ID3D12Fence;

type CopyCompletion = OnceLock<Result<Dx12Completion>>;

/// Owned by exactly one encoder or its recorded work. A discarded work item
/// closes pending tickets even if the ticket outlives the command list.
pub(crate) struct ReadbackReceipt(Arc<CopyCompletion>);

impl ReadbackReceipt {
    pub(crate) fn seal(&self, fence: &ID3D12Fence, value: u64) {
        let _ = self.0.set(Ok(Dx12Completion {
            fence: fence.clone(),
            value,
        }));
    }

    pub(crate) fn fail(&self, error: &Error) {
        let _ = self.0.set(Err(error.clone()));
    }
}

impl Drop for ReadbackReceipt {
    fn drop(&mut self) {
        self.fail(&Error::InvalidState);
    }
}

/// Native ticket for a raw buffer readback. Texture row layout remains the
/// caller's `TexelCopyBufferLayout`; a buffer has no intrinsic row geometry.
#[derive(Clone)]
pub struct Dx12Readback {
    buffer: Dx12Buffer,
    range: Range<u64>,
    completion: Option<Arc<CopyCompletion>>,
}

impl fluxel_rhi_core::backend::ReadbackBackend for Dx12 {
    type NativeReadback = Dx12Readback;
    type ReadbackFuture<'a>
        = Ready<Result<ReadbackData>>
    where
        Self: 'a;

    fn encode_readback(
        encoder: &mut Dx12Encoder,
        src: &Dx12Buffer,
        range: Range<u64>,
    ) -> Result<Dx12Readback> {
        if range.start > range.end || range.end > src.size {
            return Err(Error::InvalidRange);
        }
        if src.usage.contains(api::BufferUsage::MAP_READ) {
            encoder.buffers.push(src.clone());
            return Ok(Dx12Readback {
                buffer: src.clone(),
                range,
                completion: None,
            });
        }
        if !src.usage.contains(api::BufferUsage::COPY_SRC) {
            return Err(Error::Unsupported("DX12 device readback requires COPY_SRC"));
        }
        let size = range.end - range.start;
        if size == 0 {
            return Err(Error::InvalidRange);
        }
        // The caller explicitly transitions device-local sources to COPY_SRC.
        let staging = <Dx12 as fluxel_rhi_core::backend::ResourceBackend>::create_buffer(
            &encoder.device,
            &api::BufferDesc {
                label: Some("DX12 readback staging".into()),
                size,
                usage: api::BufferUsage::COPY_DST.union(api::BufferUsage::MAP_READ),
                memory: api::MemoryClass::Readback,
            },
        )?;
        // SAFETY: `src` is in COPY_SOURCE, `staging` is a
        // newly allocated READBACK resource in its required COPY_DEST state,
        // and the validated ranges fit both resources. The encoder retains
        // both typed handles through command-list submission.
        unsafe {
            encoder
                .list
                .CopyBufferRegion(&staging.resource, 0, &src.resource, range.start, size);
        }
        encoder.buffers.extend([src.clone(), staging.clone()]);
        let completion = Arc::new(OnceLock::new());
        encoder
            .readback_receipts
            .push(ReadbackReceipt(completion.clone()));
        Ok(Dx12Readback {
            buffer: staging,
            range: 0..size,
            completion: Some(completion),
        })
    }

    fn private_readback_status(
        device: &Dx12Device,
        ticket: &Dx12Readback,
    ) -> Result<api::CompletionState> {
        let completion = ticket
            .completion
            .as_ref()
            .ok_or(Error::Unsupported("DX12 shared readback storage"))?;
        Ok(match completion.get() {
            None => api::CompletionState::Pending,
            Some(Ok(point)) => Self::completion_status(device, point),
            Some(Err(error)) => api::CompletionState::Failed(error.clone()),
        })
    }

    fn map_private_readback<'a>(
        device: &'a Dx12Device,
        ticket: Dx12Readback,
    ) -> impl Future<Output = Result<ReadbackData>> + 'a {
        ready(map_private(device, ticket))
    }

    fn map_readback<'a>(device: &'a Dx12Device, ticket: Dx12Readback) -> Self::ReadbackFuture<'a> {
        ready(if ticket.completion.is_some() {
            map_private(device, ticket)
        } else {
            map(ticket)
        })
    }
}

fn map_private(device: &Dx12Device, ticket: Dx12Readback) -> Result<ReadbackData> {
    use fluxel_rhi_core::backend::ReadbackBackend;
    match Dx12::private_readback_status(device, &ticket)? {
        api::CompletionState::Complete => map(ticket),
        api::CompletionState::Pending => Err(Error::InvalidState),
        api::CompletionState::Failed(error) => Err(error),
    }
}

pub(super) fn map(ticket: Dx12Readback) -> Result<ReadbackData> {
    let size = ticket.range.end - ticket.range.start;
    let bytes = if size == 0 {
        Vec::new()
    } else {
        mapping::open(&ticket.buffer, ticket.range, api::MapMode::Read)?
            .bytes()
            .to_vec()
    };
    // This SPI maps a buffer. Texture geometry is explicit in the preceding
    // copy layout and is supplied by the workload when it interprets bytes.
    Ok(ReadbackData {
        bytes,
        row_pitch: size,
        slice_pitch: size,
    })
}
