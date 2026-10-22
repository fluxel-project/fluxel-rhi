//! Completion-gated mapping of DX12 readback buffers.
//!
//! The public contract requires callers to observe the matching completion
//! point before mapping. This module does not wait or poll for them: mapping
//! early would race GPU writes, and completion ownership belongs to the queue
//! implementation. The ticket retains the buffer so its native resource stays
//! alive through mapping.

use std::future::{Ready, ready};
use std::ops::Range;

use fluxel_rhi_core::api::{self, Error, ReadbackData, Result};
use windows::Win32::Graphics::Direct3D12::D3D12_RANGE;

use crate::backend::encoder::Dx12Encoder;
use crate::backend::{Dx12, Dx12Buffer, Dx12Device};

/// Native ticket for a raw buffer readback. Texture row layout remains the
/// caller's `TexelCopyBufferLayout`; a buffer has no intrinsic row geometry.
pub struct Dx12Readback {
    buffer: Dx12Buffer,
    range: Range<u64>,
}

impl api::ReadbackBackend for Dx12 {
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
        let staging = <Dx12 as api::ResourceBackend>::create_buffer(
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
        Ok(Dx12Readback {
            buffer: staging,
            range: 0..size,
        })
    }

    fn map_readback<'a>(_: &'a Dx12Device, ticket: Dx12Readback) -> Self::ReadbackFuture<'a> {
        ready(map(ticket))
    }
}

fn map(ticket: Dx12Readback) -> Result<ReadbackData> {
    let size = ticket.range.end - ticket.range.start;
    let size_usize = usize::try_from(size).map_err(|_| Error::InvalidRange)?;
    let begin = usize::try_from(ticket.range.start).map_err(|_| Error::InvalidRange)?;
    let end = begin.checked_add(size_usize).ok_or(Error::InvalidRange)?;
    let mut pointer = std::ptr::null_mut();
    // SAFETY: the completion-gated API guarantees the GPU no longer writes this
    // MAP_READ resource; range conversion and bounds checks above make the
    // D3D12 read range valid, and `pointer` is writable out storage.
    unsafe {
        ticket.buffer.resource.Map(
            0,
            Some(&D3D12_RANGE {
                Begin: begin,
                End: end,
            }),
            Some(&mut pointer),
        )
    }
    .map_err(crate::backend::native_error)?;
    if pointer.is_null() {
        return Err(Error::Backend("DX12 readback Map returned null".into()));
    }
    // SAFETY: Map returned a non-null pointer to the retained resource, and
    // the checked [begin, end) range is inside the mapped subresource.
    let bytes =
        unsafe { std::slice::from_raw_parts(pointer.cast::<u8>().add(begin), size_usize).to_vec() };
    // SAFETY: this exactly balances the successful Map above. D3D12 permits
    // the empty written range because CPU only read the resource.
    unsafe {
        ticket
            .buffer
            .resource
            .Unmap(0, Some(&D3D12_RANGE { Begin: 0, End: 0 }))
    };
    // This SPI maps a buffer. Texture geometry is explicit in the preceding
    // copy layout and is supplied by the workload when it interprets bytes.
    Ok(ReadbackData {
        bytes,
        row_pitch: size,
        slice_pitch: size,
    })
}
