//! Bounded upload-pattern lowering for explicit whole-word buffer clears.
//!
//! One immutable upload range feeds all native copies. Destination states and
//! placement/aliasing order remain the caller's responsibility.

use super::*;

pub(super) fn record(
    encoder: &mut Dx12Encoder,
    dst: &Dx12Buffer,
    range: std::ops::Range<u64>,
    value: u32,
) -> Result<(), Error> {
    if range.start >= range.end
        || range.end > dst.size
        || range.start % 4 != 0
        || (range.end - range.start) % 4 != 0
        || dst.memory == api::MemoryClass::Upload
        || !dst.usage.contains(api::BufferUsage::COPY_DST)
    {
        return Err(Error::InvalidRange);
    }
    // A bounded upload page is copied repeatedly.  This preserves the
    // caller's explicit COPY_DEST state and avoids a CPU allocation whose
    // size scales with the cleared range.
    let pattern = value.to_le_bytes();
    let chunk = pattern.repeat(16_384);
    let (upload, upload_offset) = encoder
        .upload_arena
        .write(&encoder.device.0.native, &chunk)?;
    let mut offset = range.start;
    while offset < range.end {
        let length = (range.end - offset).min(chunk.len() as u64) as usize;
        // SAFETY: source page and destination are retained by this encoder;
        // the checked aligned range is entirely inside the destination.
        unsafe {
            encoder.list.CopyBufferRegion(
                &dst.resource,
                offset,
                &upload,
                upload_offset,
                length as u64,
            );
        }
        offset += length as u64;
    }
    encoder.buffers.push(dst.clone());
    Ok(())
}
