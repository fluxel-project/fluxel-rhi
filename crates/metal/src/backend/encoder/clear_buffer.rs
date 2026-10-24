//! Whole-word buffer clears recorded as native Metal commands.
//!
//! Metal's blit fill operation repeats one byte, while the Core operation
//! repeats an arbitrary `u32`. Repeated-byte values use that native fast path;
//! all other values copy a bounded shared upload pattern through Metal blits.
//! This records directly into the current command buffer and retains no CPU
//! command stream or resource-state tracker.

use super::{MetalBuffer, MetalCommandEncoder};
use fluxel_rhi_core::api::{self, Error, Result};
use std::ops::Range;

/// Large enough to avoid per-word native calls while keeping every clear's
/// temporary CPU-visible allocation bounded.
#[cfg(target_vendor = "apple")]
const UPLOAD_BYTES: usize = 64 * 1024;

pub(super) fn clear_buffer(
    encoder: &mut MetalCommandEncoder,
    dst: &MetalBuffer,
    range: Range<u64>,
    value: u32,
) -> Result<()> {
    #[cfg(target_vendor = "apple")]
    {
        use objc2_foundation::NSRange;
        use objc2_metal::{
            MTLBlitCommandEncoder as _, MTLBuffer as _, MTLCommandBuffer as _,
            MTLCommandEncoder as _, MTLDevice as _, MTLResourceOptions,
        };

        if encoder.render.is_some() || encoder.compute.is_some() {
            return Err(Error::InvalidState);
        }
        if !dst.usage.contains(api::BufferUsage::COPY_DST)
            || range.start >= range.end
            || range.end > dst.size()
            || !range.start.is_multiple_of(4)
            || !(range.end - range.start).is_multiple_of(4)
        {
            return Err(Error::InvalidRange);
        }

        let start = usize::try_from(range.start).map_err(|_| Error::InvalidRange)?;
        let bytes = usize::try_from(range.end - range.start).map_err(|_| Error::InvalidRange)?;
        let word = value.to_le_bytes();
        if word.iter().all(|byte| *byte == word[0]) {
            let blit = encoder
                .command
                .blitCommandEncoder()
                .ok_or(Error::OutOfMemory)?;
            // SAFETY: the checked whole-word range is within `dst`; Metal's
            // byte fill is exact when all four bytes of the word match.
            unsafe {
                blit.fillBuffer_range_value(&dst.raw, NSRange::new(start, bytes), word[0]);
            }
            blit.endEncoding();
            return Ok(());
        }

        let upload_len = bytes.min(UPLOAD_BYTES);
        debug_assert!(upload_len.is_multiple_of(4));
        let upload = encoder
            .command
            .device()
            .newBufferWithLength_options(upload_len, MTLResourceOptions::StorageModeShared)
            .ok_or(Error::OutOfMemory)?;
        // SAFETY: the newly allocated shared buffer has exactly `upload_len`
        // writable bytes. Its length is a multiple of four by validation.
        unsafe {
            let pattern =
                std::slice::from_raw_parts_mut(upload.contents().as_ptr().cast::<u8>(), upload_len);
            for chunk in pattern.chunks_exact_mut(4) {
                chunk.copy_from_slice(&word);
            }
        }

        let mut destination = start;
        let end = start.checked_add(bytes).ok_or(Error::InvalidRange)?;
        while destination < end {
            let count = (end - destination).min(upload_len);
            // All values are word-aligned, so this follows the native blit
            // route in `copy_bytes`; the helper remains the single owner of
            // Metal's stricter buffer-copy alignment lowering.
            super::super::copy::copy_bytes(
                &encoder.command,
                &upload,
                0,
                &dst.raw,
                destination,
                count,
            )?;
            destination = destination.checked_add(count).ok_or(Error::InvalidRange)?;
        }
        Ok(())
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        let _ = (encoder, dst, range, value);
        Err(Error::Unsupported("Metal requires an Apple target"))
    }
}
