//! Native DX12 compute shader atomic admission by memory domain.

use fluxel_rhi_core::api;
use windows::Win32::Graphics::Direct3D12::*;

#[derive(Clone, Copy, Default)]
pub(super) struct Int64AtomicSupport {
    pub(super) buffer: bool,
    pub(super) workgroup: bool,
}

#[derive(Clone, Copy)]
pub(super) enum Memory {
    Buffer,
    Workgroup,
}

impl Int64AtomicSupport {
    pub(super) fn query(device: &ID3D12Device, model: u16, int64: bool) -> Self {
        if model < 66 || !int64 {
            return Self::default();
        }

        let mut options9 = D3D12_FEATURE_DATA_D3D12_OPTIONS9::default();
        let mut options11 = D3D12_FEATURE_DATA_D3D12_OPTIONS11::default();
        // SAFETY: each feature selector matches the initialized output type,
        // which stays writable for the duration of the synchronous query.
        unsafe {
            Self {
                workgroup: device
                    .CheckFeatureSupport(
                        D3D12_FEATURE_D3D12_OPTIONS9,
                        (&mut options9 as *mut D3D12_FEATURE_DATA_D3D12_OPTIONS9).cast(),
                        std::mem::size_of_val(&options9) as u32,
                    )
                    .is_ok()
                    && options9.AtomicInt64OnGroupSharedSupported.as_bool(),
                // Fluxel binds UAV buffers through descriptor tables.
                buffer: device
                    .CheckFeatureSupport(
                        D3D12_FEATURE_D3D12_OPTIONS11,
                        (&mut options11 as *mut D3D12_FEATURE_DATA_D3D12_OPTIONS11).cast(),
                        std::mem::size_of_val(&options11) as u32,
                    )
                    .is_ok()
                    && options11
                        .AtomicInt64OnDescriptorHeapResourceSupported
                        .as_bool(),
            }
        }
    }

    pub(super) fn supports(self, memory: Memory) -> bool {
        match memory {
            Memory::Buffer => self.buffer,
            Memory::Workgroup => self.workgroup,
        }
    }
}

pub(super) fn memory(feature: &str) -> Option<Memory> {
    use api::*;
    match feature {
        SHADER_FEATURE_INT64_ATOMIC_BUFFER_ADD
        | SHADER_FEATURE_INT64_ATOMIC_BUFFER_MIN
        | SHADER_FEATURE_INT64_ATOMIC_BUFFER_MAX
        | SHADER_FEATURE_INT64_ATOMIC_BUFFER_UMIN
        | SHADER_FEATURE_INT64_ATOMIC_BUFFER_UMAX
        | SHADER_FEATURE_INT64_ATOMIC_BUFFER_AND
        | SHADER_FEATURE_INT64_ATOMIC_BUFFER_OR
        | SHADER_FEATURE_INT64_ATOMIC_BUFFER_XOR
        | SHADER_FEATURE_INT64_ATOMIC_BUFFER_EXCHANGE
        | SHADER_FEATURE_INT64_ATOMIC_BUFFER_COMPARE_EXCHANGE => Some(Memory::Buffer),
        SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_ADD
        | SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_MIN
        | SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_MAX
        | SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_UMIN
        | SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_UMAX
        | SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_AND
        | SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_OR
        | SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_XOR
        | SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_EXCHANGE
        | SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_COMPARE_EXCHANGE => Some(Memory::Workgroup),
        _ => None,
    }
}
