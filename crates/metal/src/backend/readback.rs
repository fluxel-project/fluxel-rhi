//! Completion-gated host copies from Metal buffers.
//!
//! Encoding adds one direct blit to the current command buffer. The ticket
//! owns shared storage until Core calls `map_readback` after the supplied
//! completion proof; this module never waits on the GPU or chooses a checkpoint.

use std::{
    future::{Ready, ready},
    ops::Range,
};

use fluxel_rhi_core::api::{self, Error, ReadbackData, Result};

use super::{Metal, MetalBuffer, MetalCommandEncoder, MetalDevice};

pub struct MetalReadback {
    #[cfg(target_vendor = "apple")]
    buffer: objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>,
    #[cfg(target_vendor = "apple")]
    size: u64,
    range: Range<u64>,
}

impl api::ReadbackBackend for Metal {
    type NativeReadback = MetalReadback;
    type ReadbackFuture<'a> = Ready<Result<ReadbackData>>;

    fn encode_readback(
        encoder: &mut MetalCommandEncoder,
        source: &MetalBuffer,
        range: Range<u64>,
    ) -> Result<MetalReadback> {
        if range.start > range.end || range.end > source.size() {
            return Err(Error::InvalidRange);
        }
        let size = range.end - range.start;
        if size == 0 {
            return Ok(MetalReadback {
                #[cfg(target_vendor = "apple")]
                buffer: source.raw.clone(),
                #[cfg(target_vendor = "apple")]
                size: source.size(),
                range: 0..0,
            });
        }
        #[cfg(target_vendor = "apple")]
        {
            use objc2_metal::{
                MTLBlitCommandEncoder as _, MTLCommandBuffer as _, MTLCommandEncoder as _,
                MTLDevice as _, MTLResourceOptions,
            };
            if !source.usage.contains(api::BufferUsage::COPY_SRC) {
                return Err(Error::Unsupported("Metal readback requires COPY_SRC"));
            }
            let size = usize::try_from(size).map_err(|_| Error::InvalidRange)?;
            let native_source_offset =
                usize::try_from(range.start).map_err(|_| Error::InvalidRange)?;
            let staging = encoder
                .command
                .device()
                .newBufferWithLength_options(size, MTLResourceOptions::StorageModeShared)
                .ok_or(Error::OutOfMemory)?;
            let blit = encoder
                .command
                .blitCommandEncoder()
                .ok_or(Error::OutOfMemory)?;
            // SAFETY: the requested range was checked against the source size;
            // staging is newly allocated with exactly `size` bytes. The native
            // command buffer retains both buffers until GPU execution ends.
            unsafe {
                blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                    &source.raw,
                    native_source_offset,
                    &staging,
                    0,
                    size,
                );
            }
            blit.endEncoding();
            Ok(MetalReadback {
                buffer: staging,
                #[cfg(target_vendor = "apple")]
                size: size as u64,
                range: 0..size as u64,
            })
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = (encoder, source, range);
            Err(Error::Unsupported("Metal requires an Apple target"))
        }
    }

    fn map_readback<'a>(_: &'a MetalDevice, ticket: MetalReadback) -> Self::ReadbackFuture<'a> {
        ready(map(ticket))
    }
}

fn map(ticket: MetalReadback) -> Result<ReadbackData> {
    let size = ticket.range.end - ticket.range.start;
    if size == 0 {
        return Ok(ReadbackData {
            bytes: Vec::new(),
            row_pitch: 0,
            slice_pitch: 0,
        });
    }
    #[cfg(target_vendor = "apple")]
    {
        use objc2_metal::MTLBuffer as _;
        let start = usize::try_from(ticket.range.start).map_err(|_| Error::InvalidRange)?;
        let length = usize::try_from(size).map_err(|_| Error::InvalidRange)?;
        let end = start.checked_add(length).ok_or(Error::InvalidRange)?;
        if end > usize::try_from(ticket.size).map_err(|_| Error::InvalidRange)? {
            return Err(Error::InvalidRange);
        }
        // SAFETY: `read_after` only reaches this method after the caller has
        // supplied completion evidence covering the encoded copy. This shared
        // allocation stays alive in the consumed ticket and the range fits it.
        let bytes = unsafe {
            std::slice::from_raw_parts(
                ticket.buffer.contents().cast::<u8>().as_ptr().add(start),
                length,
            )
            .to_vec()
        };
        Ok(ReadbackData {
            bytes,
            row_pitch: size,
            slice_pitch: size,
        })
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        let _ = ticket;
        Err(Error::Unsupported("Metal requires an Apple target"))
    }
}

#[cfg(test)]
mod tests {
    #[cfg(not(target_vendor = "apple"))]
    use super::{MetalReadback, map};

    #[cfg(not(target_vendor = "apple"))]
    #[test]
    fn empty_readback_has_no_row_or_slice_pitch() {
        let data = map(MetalReadback { range: 0..0 }).unwrap();
        assert!(data.bytes.is_empty());
        assert_eq!(data.row_pitch, 0);
        assert_eq!(data.slice_pitch, 0);
    }
}
