//! Native heap profiles for explicitly placed CPU-visible buffers.
//!
//! Profiles use the upload or readback state contract. Exact cache and pool
//! requests are resolved against node zero, the node used for ordinary work.

use fluxel_rhi_core::api;
use windows::Win32::Graphics::Direct3D12::*;

pub(in crate::backend) struct CpuVisibleHeapProfiles {
    upload: D3D12_HEAP_PROPERTIES,
    readback: D3D12_HEAP_PROPERTIES,
    gpu_upload: Option<D3D12_HEAP_PROPERTIES>,
    unified: bool,
}

impl CpuVisibleHeapProfiles {
    pub(in crate::backend) fn for_device(
        device: &ID3D12Device,
        topology: Option<api::DeviceMemoryTopology>,
        gpu_upload_supported: bool,
    ) -> Self {
        // SAFETY: the device stays alive during the synchronous property read.
        let node_mask = if unsafe { device.GetNodeCount() } > 1 {
            1
        } else {
            0
        };
        let properties = |heap_type| {
            // SAFETY: the mask selects node zero. The caller only supplies
            // regular heap types admitted on this device.
            unsafe { device.GetCustomHeapProperties(node_mask, heap_type) }
        };
        Self {
            upload: properties(D3D12_HEAP_TYPE_UPLOAD),
            readback: properties(D3D12_HEAP_TYPE_READBACK),
            gpu_upload: gpu_upload_supported.then(|| properties(D3D12_HEAP_TYPE_GPU_UPLOAD)),
            unified: matches!(topology, Some(api::DeviceMemoryTopology::Unified { .. })),
        }
    }

    pub(in crate::backend) fn select(
        &self,
        memory: api::MemoryClass,
        cache: api::CpuVisibleBufferCache,
        pool: api::CpuVisibleBufferPool,
    ) -> Result<D3D12_HEAP_PROPERTIES, &'static str> {
        use api::{CpuVisibleBufferPool as Pool, MemoryClass};
        let local = pool == Pool::DeviceLocal && !self.unified;
        let properties = match (memory, local) {
            (MemoryClass::Upload, false) => self.upload,
            (MemoryClass::Readback, false) => self.readback,
            (MemoryClass::Upload, true) => self
                .gpu_upload
                .ok_or("DX12 CPU-visible device-local upload heap")?,
            _ => return Err("DX12 custom buffer memory class and pool"),
        };
        let expected_pool = if local {
            D3D12_MEMORY_POOL_L1
        } else {
            D3D12_MEMORY_POOL_L0
        };
        let expected_page = match cache {
            api::CpuVisibleBufferCache::WriteCombined => D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE,
            api::CpuVisibleBufferCache::WriteBack => D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
        };
        if properties.MemoryPoolPreference != expected_pool
            || properties.CPUPageProperty != expected_page
        {
            return Err("DX12 custom buffer cache policy or pool");
        }
        Ok(D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_CUSTOM,
            CPUPageProperty: properties.CPUPageProperty,
            MemoryPoolPreference: properties.MemoryPoolPreference,
            CreationNodeMask: 1,
            VisibleNodeMask: 1,
        })
    }
}
