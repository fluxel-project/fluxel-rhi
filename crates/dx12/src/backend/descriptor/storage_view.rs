//! Shared validation and lowering for native storage buffer descriptors.

use super::{Dx12Buffer, Error, Range, Result, api};
use windows::Win32::Graphics::{
    Direct3D12::*,
    Dxgi::Common::{DXGI_FORMAT_R32_TYPELESS, DXGI_FORMAT_UNKNOWN},
};

pub(in crate::backend) struct StorageViewRange {
    first: u64,
    count: u32,
    structure_stride: u32,
}

pub(in crate::backend) fn storage_view_range(
    buffer: &Dx12Buffer,
    range: Range<u64>,
    view: api::StorageBufferView,
    writable: bool,
) -> Result<StorageViewRange> {
    if writable && buffer.memory == api::MemoryClass::Upload {
        return Err(Error::Unsupported(
            "DX12 upload storage buffers are read-only",
        ));
    }
    if !buffer.usage.contains(api::BufferUsage::STORAGE)
        || range.end > buffer.size
        || range.start >= range.end
    {
        return Err(Error::InvalidRange);
    }
    let (width, alignment, structure_stride) = match view {
        api::StorageBufferView::Raw => (4, D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT, 0),
        api::StorageBufferView::Structured { stride }
            if stride > 0
                && stride <= D3D12_REQ_MULTI_ELEMENT_STRUCTURE_SIZE_IN_BYTES
                && stride % 4 == 0 =>
        {
            (stride, stride, stride)
        }
        _ => return Err(Error::InvalidRange),
    };
    let bytes = range.end - range.start;
    if range.start % u64::from(alignment) != 0 || bytes % u64::from(width) != 0 {
        return Err(Error::InvalidRange);
    }
    let count = u32::try_from(bytes / u64::from(width)).map_err(|_| Error::InvalidRange)?;
    if structure_stride != 0 && count > 1 << D3D12_REQ_BUFFER_RESOURCE_TEXEL_COUNT_2_TO_EXP {
        return Err(Error::InvalidRange);
    }
    Ok(StorageViewRange {
        first: range.start / u64::from(width),
        count,
        structure_stride,
    })
}

impl StorageViewRange {
    /// # Safety
    /// The buffer and access must match the validated range. The destination
    /// must be a live view descriptor reserved by the caller.
    pub(super) unsafe fn write(
        &self,
        device: &ID3D12Device,
        buffer: &Dx12Buffer,
        writable: bool,
        cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
    ) {
        let raw = self.structure_stride == 0;
        let format = if raw {
            DXGI_FORMAT_R32_TYPELESS
        } else {
            DXGI_FORMAT_UNKNOWN
        };
        if writable {
            let desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                Format: format,
                ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                    Buffer: D3D12_BUFFER_UAV {
                        FirstElement: self.first,
                        NumElements: self.count,
                        StructureByteStride: self.structure_stride,
                        CounterOffsetInBytes: 0,
                        Flags: if raw {
                            D3D12_BUFFER_UAV_FLAG_RAW
                        } else {
                            D3D12_BUFFER_UAV_FLAG_NONE
                        },
                    },
                },
            };
            // SAFETY: the checked range describes this live buffer and the caller owns the slot.
            unsafe {
                device.CreateUnorderedAccessView(
                    &buffer.resource,
                    None::<&ID3D12Resource>,
                    Some(&desc),
                    cpu,
                )
            };
        } else {
            let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: format,
                ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Buffer: D3D12_BUFFER_SRV {
                        FirstElement: self.first,
                        NumElements: self.count,
                        StructureByteStride: self.structure_stride,
                        Flags: if raw {
                            D3D12_BUFFER_SRV_FLAG_RAW
                        } else {
                            D3D12_BUFFER_SRV_FLAG_NONE
                        },
                    },
                },
            };
            // SAFETY: the checked range describes this live buffer and the caller owns the slot.
            unsafe { device.CreateShaderResourceView(&buffer.resource, Some(&desc), cpu) };
        }
    }
}
