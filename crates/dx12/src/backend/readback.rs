//! Completion-gated mapping of DX12 readback buffers.

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
        if !src.usage.contains(api::BufferUsage::MAP_READ) {
            return Err(Error::Unsupported("DX12 readback requires MAP_READ buffer"));
        }
        encoder.buffers.push(src.clone());
        Ok(Dx12Readback {
            buffer: src.clone(),
            range,
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
    let bytes =
        unsafe { std::slice::from_raw_parts(pointer.cast::<u8>().add(begin), size_usize).to_vec() };
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
