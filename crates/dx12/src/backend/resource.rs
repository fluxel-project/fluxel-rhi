//! Resource capability checks and native resource creation entry points.
//!
//! Creation delegates views, descriptors and textures to their native owners.
//! This module validates buffer memory classes and sizes without inferring
//! resource uses, scheduling commands, or maintaining resource states.

use super::*;

impl fluxel_rhi_core::backend::ResourceBackend for Dx12 {
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
            <Self as fluxel_rhi_core::backend::ResourceBackend>::buffer_support(device, desc)
        {
            return Err(Error::Unsupported(reason));
        }
        let readback = desc.memory == api::MemoryClass::Readback;
        let upload = desc.memory == api::MemoryClass::Upload;
        let (native, allocation_size, initial_state) = memory::buffer_desc(desc)?;
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
        let class_bits = memory::buffer_class_bits(device, desc.memory);
        let class = (0..64)
            .find(|bit| class_bits & (1u64 << bit) != 0)
            .ok_or(Error::Unsupported("DX12 buffer heap class"))?;
        let requirements = memory::requirements(device, &native, class_bits)?;
        let pooled = device.0.default_resources.allocate(
            class,
            requirements.size,
            requirements.alignment,
        )?;
        let (resource, default_allocation, backing_heap) = if let Some(pool) = pooled {
            let mut resource = None;
            // SAFETY: the pool allocation satisfies the native allocation
            // size/alignment and heap class requirements for this buffer.
            let placed = unsafe {
                device.0.native.CreatePlacedResource(
                    &pool.heap,
                    pool.offset,
                    &native,
                    initial_state,
                    None,
                    &mut resource,
                )
            };
            match placed {
                Ok(()) => (
                    resource.ok_or(Error::OutOfMemory)?,
                    Some(pool.lease),
                    Some(pool.heap),
                ),
                Err(error) if memory::is_out_of_memory(&error) => {
                    drop(pool);
                    let mut resource = None;
                    // SAFETY: the original resource descriptor remains valid;
                    // a committed allocation is the OOM fallback for paging.
                    unsafe {
                        device.0.native.CreateCommittedResource(
                            &heap,
                            D3D12_HEAP_FLAG_NONE,
                            &native,
                            initial_state,
                            None,
                            &mut resource,
                        )
                    }
                    .map_err(native_error)?;
                    (resource.ok_or(Error::OutOfMemory)?, None, None)
                }
                Err(error) => return Err(native_error(error)),
            }
        } else {
            let mut resource = None;
            // SAFETY: heap and buffer descriptors are initialized local values;
            // D3D12 writes the owned resource COM interface into `resource` only.
            unsafe {
                device.0.native.CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &native,
                    initial_state,
                    None,
                    &mut resource,
                )
            }
            .map_err(native_error)?;
            (resource.ok_or(Error::OutOfMemory)?, None, None)
        };
        Ok(Dx12Buffer {
            resource,
            size: desc.size,
            allocation_size,
            mapped: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            usage: desc.usage,
            memory: desc.memory,
            indirect_signatures: device.0.indirect_signatures.clone(),
            backing_heap,
            default_allocation,
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
