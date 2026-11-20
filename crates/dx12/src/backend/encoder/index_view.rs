//! DX12 input-assembler index views shared by direct lists and bundles.

use super::{Dx12Buffer, Error, api};
use windows::Win32::Graphics::{
    Direct3D12::D3D12_INDEX_BUFFER_VIEW,
    Dxgi::Common::{DXGI_FORMAT, DXGI_FORMAT_R16_UINT, DXGI_FORMAT_R32_UINT},
};

pub(in crate::backend) fn validate(
    buffer: &Dx12Buffer,
    offset: u64,
    size: u64,
    format: api::IndexFormat,
) -> api::Result<(u32, DXGI_FORMAT)> {
    if !buffer.usage.contains(api::BufferUsage::INDEX) {
        return Err(Error::Unsupported("DX12 index buffer usage"));
    }
    let (width, native_format) = match format {
        api::IndexFormat::Uint16 => (2, DXGI_FORMAT_R16_UINT),
        api::IndexFormat::Uint32 => (4, DXGI_FORMAT_R32_UINT),
    };
    if offset % width != 0
        || size % width != 0
        || offset.checked_add(size).is_none_or(|end| end > buffer.size)
    {
        return Err(Error::InvalidRange);
    }
    Ok((
        u32::try_from(size).map_err(|_| Error::InvalidRange)?,
        native_format,
    ))
}

pub(super) fn make(
    buffer: &Dx12Buffer,
    offset: u64,
    size: u64,
    format: api::IndexFormat,
) -> api::Result<D3D12_INDEX_BUFFER_VIEW> {
    let (size, format) = validate(buffer, offset, size, format)?;
    // SAFETY: the resource is retained by the buffer throughout the native query.
    let base = unsafe { buffer.resource.GetGPUVirtualAddress() };
    Ok(D3D12_INDEX_BUFFER_VIEW {
        BufferLocation: base.checked_add(offset).ok_or(Error::InvalidRange)?,
        SizeInBytes: size,
        Format: format,
    })
}
