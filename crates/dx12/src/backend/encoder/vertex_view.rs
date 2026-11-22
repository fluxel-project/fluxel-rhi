//! DX12 input-assembler vertex views shared by direct lists and bundles.

use super::{Dx12Buffer, Error, api};
use windows::Win32::Graphics::Direct3D12::D3D12_VERTEX_BUFFER_VIEW;

pub(super) fn make(
    buffer: &Dx12Buffer,
    offset: u64,
    size: u64,
    stride: u32,
) -> api::Result<D3D12_VERTEX_BUFFER_VIEW> {
    if !buffer.usage.contains(api::BufferUsage::VERTEX) {
        return Err(Error::Unsupported("DX12 vertex buffer usage"));
    }
    if offset.checked_add(size).is_none_or(|end| end > buffer.size) {
        return Err(Error::InvalidRange);
    }
    let size = u32::try_from(size).map_err(|_| Error::InvalidRange)?;
    // SAFETY: the buffer retains the native resource throughout the query.
    let base = unsafe { buffer.resource.GetGPUVirtualAddress() };
    Ok(D3D12_VERTEX_BUFFER_VIEW {
        BufferLocation: base.checked_add(offset).ok_or(Error::InvalidRange)?,
        SizeInBytes: size,
        StrideInBytes: stride,
    })
}
