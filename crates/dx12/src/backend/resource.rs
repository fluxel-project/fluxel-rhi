//! Resource capability checks and native resource creation entry points.
//!
//! Creation delegates views, descriptors and textures to their native owners.
//! This module validates buffer memory classes and sizes without inferring
//! resource uses, scheduling commands, or maintaining resource states.

use super::*;

impl api::ResourceBackend for Dx12 {
    fn buffer_support(
        device: &Dx12Device,
        desc: &api::BufferDesc,
    ) -> api::Support<api::BufferSupport> {
        if desc.size == 0 {
            api::Support::Unsupported("zero-sized buffers")
        } else if desc.size > device.0.max_buffer_bytes {
            api::Support::Unsupported("DX12 buffer size limit")
        } else if desc.usage.contains(api::BufferUsage::UNIFORM)
            && desc
                .size
                .checked_add(255)
                .map(|size| size & !255)
                .is_none_or(|size| size > device.0.max_buffer_bytes)
        {
            api::Support::Unsupported("DX12 padded uniform buffer size limit")
        } else if matches!(desc.memory, api::MemoryClass::Readback) {
            let readback_writes = api::BufferUsage::COPY_DST.0 | api::BufferUsage::QUERY_RESOLVE.0;
            if desc.usage.contains(api::BufferUsage::MAP_READ)
                && desc.usage.0 & readback_writes != 0
                && desc.usage.0 & !(readback_writes | api::BufferUsage::MAP_READ.0) == 0
            {
                api::Support::Supported(api::BufferSupport {
                    alignment: 1,
                    max_size: device.0.max_buffer_bytes,
                })
            } else {
                api::Support::Unsupported("DX12 readback buffer usage")
            }
        } else if matches!(desc.memory, api::MemoryClass::Upload) {
            let allowed = api::BufferUsage::MAP_WRITE.0
                | api::BufferUsage::COPY_SRC.0
                | api::BufferUsage::VERTEX.0
                | api::BufferUsage::INDEX.0
                | api::BufferUsage::UNIFORM.0
                | api::BufferUsage::INDIRECT.0
                | api::BufferUsage::STORAGE.0
                | api::BufferUsage::ACCEL_BUILD_INPUT.0
                | api::BufferUsage::SHADER_TABLE.0;
            if desc.usage.0
                & (api::BufferUsage::ACCEL_BUILD_INPUT.0 | api::BufferUsage::SHADER_TABLE.0)
                != 0
                && device.0.ray_device.is_none()
            {
                api::Support::Unsupported("DX12 ray tracing tier 1.0")
            } else if desc.usage.contains(api::BufferUsage::MAP_WRITE)
                && desc.usage.0 & !allowed == 0
            {
                api::Support::Supported(api::BufferSupport {
                    alignment: 1,
                    max_size: device.0.max_buffer_bytes,
                })
            } else {
                api::Support::Unsupported("DX12 upload buffer usage")
            }
        } else if desc.usage.0
            & (api::BufferUsage::ACCEL_BUILD_INPUT.0
                | api::BufferUsage::ACCEL_SCRATCH.0
                | api::BufferUsage::SHADER_TABLE.0)
            != 0
            && device.0.ray_device.is_none()
        {
            api::Support::Unsupported("DX12 ray tracing tier 1.0")
        } else if desc.usage.0
            & !(api::BufferUsage::VERTEX.0
                | api::BufferUsage::INDEX.0
                | api::BufferUsage::UNIFORM.0
                | api::BufferUsage::STORAGE.0
                | api::BufferUsage::COPY_DST.0
                | api::BufferUsage::COPY_SRC.0
                | api::BufferUsage::INDIRECT.0
                | api::BufferUsage::QUERY_RESOLVE.0
                | api::BufferUsage::ACCEL_BUILD_INPUT.0
                | api::BufferUsage::ACCEL_SCRATCH.0
                | api::BufferUsage::SHADER_TABLE.0)
            != 0
        {
            api::Support::Unsupported("DX12 buffer usage")
        } else {
            api::Support::Supported(api::BufferSupport {
                alignment: 1,
                max_size: device.0.max_buffer_bytes,
            })
        }
    }
    fn texture_support(
        device: &Dx12Device,
        desc: &api::TextureDesc,
    ) -> api::Support<api::TextureSupport> {
        texture::support(device, desc)
    }
    fn create_buffer(device: &Dx12Device, desc: &api::BufferDesc) -> Result<Dx12Buffer> {
        if let api::Support::Unsupported(reason) =
            <Self as api::ResourceBackend>::buffer_support(device, desc)
        {
            return Err(Error::Unsupported(reason));
        }
        let readback = desc.memory == api::MemoryClass::Readback;
        let upload = desc.memory == api::MemoryClass::Upload;
        let allocation_size = if desc.usage.contains(api::BufferUsage::UNIFORM) {
            desc.size.checked_add(255).ok_or(Error::InvalidRange)? & !255
        } else {
            desc.size
        };
        if allocation_size > device.0.max_buffer_bytes {
            return Err(Error::InvalidRange);
        }
        let heap = D3D12_HEAP_PROPERTIES {
            Type: if readback {
                windows::Win32::Graphics::Direct3D12::D3D12_HEAP_TYPE_READBACK
            } else if upload {
                windows::Win32::Graphics::Direct3D12::D3D12_HEAP_TYPE_UPLOAD
            } else {
                D3D12_HEAP_TYPE_DEFAULT
            },
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
            MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
            CreationNodeMask: 1,
            VisibleNodeMask: 1,
        };
        let native = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Alignment: 0,
            Width: allocation_size,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_UNKNOWN,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: if !upload
                && (desc.usage.contains(api::BufferUsage::ACCEL_SCRATCH)
                    || desc.usage.contains(api::BufferUsage::STORAGE))
            {
                windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
            } else {
                D3D12_RESOURCE_FLAG_NONE
            },
        };
        let mut resource = None;
        // SAFETY: heap and buffer descriptors are initialized local values;
        // D3D12 writes the owned resource COM interface into `resource` only.
        unsafe {
            device.0.native.CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &native,
                if readback {
                    windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATE_COPY_DEST
                } else if upload {
                    windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATE_GENERIC_READ
                } else {
                    D3D12_RESOURCE_STATE_COMMON
                },
                None,
                &mut resource,
            )
        }
        .map_err(native_error)?;
        Ok(Dx12Buffer {
            resource: resource.ok_or_else(|| {
                Error::Backend("CreateCommittedResource returned no buffer".into())
            })?,
            size: desc.size,
            allocation_size,
            mapped: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            usage: desc.usage,
            memory: desc.memory,
            indirect_signatures: device.0.indirect_signatures.clone(),
            backing_heap: None,
        })
    }
    fn create_texture(device: &Dx12Device, desc: &api::TextureDesc) -> Result<Dx12Texture> {
        texture::create(device, desc)
    }
    fn create_view(
        device: &Dx12Device,
        texture: &Dx12Texture,
        desc: &api::TextureViewDesc,
    ) -> Result<Dx12TextureView> {
        texture::create_view(device, texture, desc)
    }
    fn create_sampler(device: &Dx12Device, desc: &api::SamplerDesc) -> Result<Dx12Sampler> {
        sampler::create_sampler(device, desc)
    }
    fn create_bind_group_layout(
        device: &Dx12Device,
        desc: &api::BindGroupLayoutDesc,
    ) -> Result<Dx12BindGroupLayout> {
        descriptor::create_bind_group_layout(device, desc)
    }
    fn create_bind_group(
        device: &Dx12Device,
        desc: &api::BindGroupDesc<'_, Self>,
    ) -> Result<Dx12BindGroup> {
        descriptor::create_bind_group(device, desc)
    }
    fn create_pipeline_layout(
        device: &Dx12Device,
        desc: &api::PipelineLayoutDesc<'_, Self>,
    ) -> Result<Dx12PipelineLayout> {
        pipeline::create_pipeline_layout(device, desc)
    }
}
