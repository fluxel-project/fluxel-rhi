//! Staged CPU writes into device-local buffers with fixed boundary states.

use super::*;

pub(super) fn record(
    encoder: &mut Dx12Encoder,
    dst: &super::Dx12Buffer,
    offset: u64,
    bytes: &[u8],
) -> Result<(), Error> {
    if dst.memory != api::MemoryClass::Device || !dst.usage.contains(api::BufferUsage::COPY_DST) {
        return Err(Error::Unsupported(
            "DX12 write_buffer requires a device COPY_DST buffer",
        ));
    }
    if bytes.is_empty() {
        return Ok(());
    }
    let end = offset
        .checked_add(bytes.len() as u64)
        .ok_or(Error::InvalidRange)?;
    if end > dst.size {
        return Err(Error::InvalidRange);
    }
    let (upload, upload_offset) = encoder
        .upload_arena
        .write(&encoder.device.0.native, bytes)?;
    // Default-heap buffers begin and end every upload in COMMON.  This is
    // an explicit stable boundary, so no hidden cross-frame state tracker
    // is needed for the minimal path.
    barrier(
        &encoder.list,
        &dst.resource,
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_COPY_DEST,
    );
    // SAFETY: both resources are retained by the encoder and range checks
    // above ensure the native copy does not address outside either buffer.
    unsafe {
        encoder.list.CopyBufferRegion(
            &dst.resource,
            offset,
            &upload,
            upload_offset,
            bytes.len() as u64,
        );
    }
    barrier(
        &encoder.list,
        &dst.resource,
        D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_RESOURCE_STATE_COMMON,
    );
    encoder.buffers.push(dst.clone());
    Ok(())
}
