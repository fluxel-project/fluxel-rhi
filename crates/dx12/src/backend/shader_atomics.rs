//! Native DX12 compute shader atomic admission by memory domain.

use fluxel_rhi_core::api;
use windows::Win32::Graphics::Direct3D12::*;

#[derive(Clone, Copy, Default)]
pub(super) struct Int64AtomicSupport {
    pub(super) buffer: bool,
    pub(super) workgroup: bool,
    pub(super) r64uint_texture: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Memory {
    Buffer,
    Workgroup,
    R64UintTexture,
}

impl Int64AtomicSupport {
    pub(super) fn query(device: &ID3D12Device, model: u16, int64: bool) -> Self {
        if model < 66 || !int64 {
            return Self::default();
        }

        let mut options9 = D3D12_FEATURE_DATA_D3D12_OPTIONS9::default();
        let mut options11 = D3D12_FEATURE_DATA_D3D12_OPTIONS11::default();
        // SAFETY: the selector matches the initialized writable structure.
        let has_options9 = unsafe {
            device.CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS9,
                (&mut options9 as *mut D3D12_FEATURE_DATA_D3D12_OPTIONS9).cast(),
                std::mem::size_of_val(&options9) as u32,
            )
        }
        .is_ok();
        Self {
            workgroup: has_options9 && options9.AtomicInt64OnGroupSharedSupported.as_bool(),
            r64uint_texture: has_options9
                && options9.AtomicInt64OnTypedResourceSupported.as_bool()
                && super::texture::probe_r64uint_atomic_texture(device),
            // Fluxel binds UAV buffers through descriptor tables.
            // SAFETY: the selector matches the initialized writable structure.
            buffer: unsafe {
                device.CheckFeatureSupport(
                    D3D12_FEATURE_D3D12_OPTIONS11,
                    (&mut options11 as *mut D3D12_FEATURE_DATA_D3D12_OPTIONS11).cast(),
                    std::mem::size_of_val(&options11) as u32,
                )
            }
            .is_ok()
                && options11
                    .AtomicInt64OnDescriptorHeapResourceSupported
                    .as_bool(),
        }
    }

    pub(super) fn supports(self, memory: Memory) -> bool {
        match memory {
            Memory::Buffer => self.buffer,
            Memory::Workgroup => self.workgroup,
            Memory::R64UintTexture => self.r64uint_texture,
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
        SHADER_FEATURE_INT64_ATOMIC_IMAGE_R64UINT_ADD
        | SHADER_FEATURE_INT64_ATOMIC_IMAGE_R64UINT_UMIN
        | SHADER_FEATURE_INT64_ATOMIC_IMAGE_R64UINT_UMAX => Some(Memory::R64UintTexture),
        _ => None,
    }
}
