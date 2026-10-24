//! Explicit DX12 heap placement without allocation-policy ownership.
//!
//! A `Dx12Heap` owns one D3D12 heap and records its creating device and heap
//! class. Placed resources clone the COM heap into their native state, so a
//! caller may drop `MemoryHeap` after placement without invalidating queued
//! command work. Aliasing order remains an explicit Core barrier obligation.

use super::{Dx12, Dx12Buffer, Dx12Device, Dx12DeviceState, Dx12Texture, texture};
use fluxel_rhi_core::api::{self, Error, MemoryBackend, ResourceBackend, Result, Support};
use std::sync::{Arc, atomic::AtomicBool};
use windows::Win32::Graphics::{
    Direct3D12::*,
    Dxgi::Common::{DXGI_FORMAT_UNKNOWN, DXGI_SAMPLE_DESC},
};

const DEVICE_CLASS: u32 = 0;
const UPLOAD_CLASS: u32 = 1;
const READBACK_CLASS: u32 = 2;
const DEFAULT_BUFFER_CLASS: u32 = 3;
const DEFAULT_NON_RT_DS_TEXTURE_CLASS: u32 = 4;
const DEFAULT_RT_DS_TEXTURE_CLASS: u32 = 5;

/// Native heap retained by explicitly placed DX12 resources.
#[derive(Clone)]
pub struct Dx12Heap {
    heap: ID3D12Heap,
    device: Arc<Dx12DeviceState>,
    size: u64,
    class: u32,
}

fn properties(class: u32) -> Result<D3D12_HEAP_PROPERTIES> {
    let ty = match class {
        DEVICE_CLASS
        | DEFAULT_BUFFER_CLASS
        | DEFAULT_NON_RT_DS_TEXTURE_CLASS
        | DEFAULT_RT_DS_TEXTURE_CLASS => D3D12_HEAP_TYPE_DEFAULT,
        UPLOAD_CLASS => D3D12_HEAP_TYPE_UPLOAD,
        READBACK_CLASS => D3D12_HEAP_TYPE_READBACK,
        _ => return Err(Error::InvalidRange),
    };
    Ok(D3D12_HEAP_PROPERTIES {
        Type: ty,
        CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
        MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
        CreationNodeMask: 1,
        VisibleNodeMask: 1,
    })
}

fn heap_flags(class: u32) -> Result<D3D12_HEAP_FLAGS> {
    match class {
        DEVICE_CLASS => Ok(D3D12_HEAP_FLAG_ALLOW_ALL_BUFFERS_AND_TEXTURES),
        UPLOAD_CLASS | READBACK_CLASS => Ok(D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS),
        DEFAULT_BUFFER_CLASS => Ok(D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS),
        DEFAULT_NON_RT_DS_TEXTURE_CLASS => Ok(D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES),
        DEFAULT_RT_DS_TEXTURE_CLASS => Ok(D3D12_HEAP_FLAG_ALLOW_ONLY_RT_DS_TEXTURES),
        _ => Err(Error::InvalidRange),
    }
}

fn heap_alignment(class: u32) -> Result<u64> {
    match class {
        // These heap classes may contain MSAA resources, which require the
        // 4 MiB native heap alignment. Size need not be a multiple of it.
        DEVICE_CLASS | DEFAULT_RT_DS_TEXTURE_CLASS => Ok(4 * 1024 * 1024),
        UPLOAD_CLASS | READBACK_CLASS | DEFAULT_BUFFER_CLASS | DEFAULT_NON_RT_DS_TEXTURE_CLASS => {
            Ok(64 * 1024)
        }
        _ => Err(Error::InvalidRange),
    }
}

fn default_class_bits(device: &Dx12Device, specific: u32) -> u64 {
    let specific = 1u64 << specific;
    if device.0.resource_heap_tier2 {
        specific | (1u64 << DEVICE_CLASS)
    } else {
        specific
    }
}

fn buffer_class_bits(device: &Dx12Device, memory: api::MemoryClass) -> u64 {
    match memory {
        api::MemoryClass::Device => default_class_bits(device, DEFAULT_BUFFER_CLASS),
        api::MemoryClass::Upload => 1u64 << UPLOAD_CLASS,
        api::MemoryClass::Readback => 1u64 << READBACK_CLASS,
    }
}

fn texture_class_bits(device: &Dx12Device, resource: &D3D12_RESOURCE_DESC) -> u64 {
    let specific = if resource
        .Flags
        .contains(D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET)
        || resource
            .Flags
            .contains(D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL)
    {
        DEFAULT_RT_DS_TEXTURE_CLASS
    } else {
        DEFAULT_NON_RT_DS_TEXTURE_CLASS
    };
    default_class_bits(device, specific)
}

fn buffer_desc(
    desc: &api::BufferDesc,
) -> Result<(D3D12_RESOURCE_DESC, u64, D3D12_RESOURCE_STATES)> {
    let allocation_size = if desc.usage.contains(api::BufferUsage::UNIFORM) {
        desc.size.checked_add(255).ok_or(Error::InvalidRange)? & !255
    } else {
        desc.size
    };
    let upload = desc.memory == api::MemoryClass::Upload;
    let readback = desc.memory == api::MemoryClass::Readback;
    let flags = if !upload
        && (desc.usage.contains(api::BufferUsage::ACCEL_SCRATCH)
            || desc.usage.contains(api::BufferUsage::STORAGE))
    {
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
    } else {
        D3D12_RESOURCE_FLAG_NONE
    };
    Ok((
        D3D12_RESOURCE_DESC {
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
            Flags: flags,
        },
        allocation_size,
        if readback {
            D3D12_RESOURCE_STATE_COPY_DEST
        } else if upload {
            D3D12_RESOURCE_STATE_GENERIC_READ
        } else {
            D3D12_RESOURCE_STATE_COMMON
        },
    ))
}

fn requirements(
    device: &Dx12Device,
    desc: &D3D12_RESOURCE_DESC,
    native_type_bits: u64,
) -> Result<api::MemoryRequirements> {
    // SAFETY: descriptor is initialized and D3D12 copies it synchronously.
    let info = unsafe {
        device
            .0
            .native
            .GetResourceAllocationInfo(0, std::slice::from_ref(desc))
    };
    if info.SizeInBytes == 0 || info.SizeInBytes == u64::MAX || info.Alignment == 0 {
        return Err(Error::Unsupported("DX12 resource allocation requirements"));
    }
    Ok(api::MemoryRequirements {
        size: info.SizeInBytes,
        alignment: info.Alignment,
        native_type_bits,
        dedicated_only: false,
    })
}

fn validate(
    device: &Dx12Device,
    heap: &Dx12Heap,
    offset: u64,
    requirements: api::MemoryRequirements,
) -> Result<()> {
    if !Arc::ptr_eq(&device.0, &heap.device) {
        return Err(Error::WrongDevice);
    }
    if requirements.native_type_bits & (1u64 << heap.class) == 0 {
        return Err(Error::InvalidState);
    }
    if offset % requirements.alignment != 0
        || offset
            .checked_add(requirements.size)
            .is_none_or(|end| end > heap.size)
    {
        return Err(Error::InvalidRange);
    }
    Ok(())
}

impl MemoryBackend for Dx12 {
    type NativeHeap = Dx12Heap;

    fn buffer_requirements(
        device: &Dx12Device,
        desc: &api::BufferDesc,
    ) -> Result<api::MemoryRequirements> {
        if !matches!(Self::buffer_support(device, desc), Support::Supported(_)) {
            return Err(Error::Unsupported("DX12 buffer descriptor"));
        }
        let (native, allocation_size, _) = buffer_desc(desc)?;
        if allocation_size > device.0.max_buffer_bytes {
            return Err(Error::InvalidRange);
        }
        requirements(device, &native, buffer_class_bits(device, desc.memory))
    }

    fn texture_requirements(
        device: &Dx12Device,
        desc: &api::TextureDesc,
    ) -> Result<api::MemoryRequirements> {
        let native = texture::allocation_desc(device, desc)?;
        requirements(
            device,
            &native.resource,
            texture_class_bits(device, &native.resource),
        )
    }

    fn create_heap(device: &Dx12Device, desc: &api::HeapDesc) -> Result<Dx12Heap> {
        if desc.size == 0 {
            return Err(Error::InvalidRange);
        }
        if desc.native_memory_class == DEVICE_CLASS && !device.0.resource_heap_tier2 {
            return Err(Error::Unsupported(
                "DX12 shared heap requires Resource Heap Tier 2",
            ));
        }
        let properties = properties(desc.native_memory_class)?;
        let flags = heap_flags(desc.native_memory_class)?;
        let native = D3D12_HEAP_DESC {
            SizeInBytes: desc.size,
            Properties: properties,
            Alignment: heap_alignment(desc.native_memory_class)?,
            Flags: flags,
        };
        let mut heap = None;
        // SAFETY: descriptor is initialized and D3D12 writes a new owned COM
        // reference to `heap` on success.
        unsafe { device.0.native.CreateHeap(&native, &mut heap) }.map_err(super::native_error)?;
        Ok(Dx12Heap {
            heap: heap.ok_or(Error::OutOfMemory)?,
            device: device.0.clone(),
            size: desc.size,
            class: desc.native_memory_class,
        })
    }

    fn place_buffer(
        device: &Dx12Device,
        heap: &Dx12Heap,
        offset: u64,
        desc: &api::BufferDesc,
    ) -> Result<Dx12Buffer> {
        let requirements = Self::buffer_requirements(device, desc)?;
        validate(device, heap, offset, requirements)?;
        let (native, allocation_size, initial_state) = buffer_desc(desc)?;
        let mut resource = None;
        // SAFETY: heap device identity, resource requirements, offset and
        // range were checked above; D3D12 writes only the owned COM output.
        unsafe {
            device.0.native.CreatePlacedResource(
                &heap.heap,
                offset,
                &native,
                initial_state,
                None,
                &mut resource,
            )
        }
        .map_err(super::native_error)?;
        Ok(Dx12Buffer {
            resource: resource.ok_or(Error::OutOfMemory)?,
            size: desc.size,
            allocation_size,
            mapped: Arc::new(AtomicBool::new(false)),
            usage: desc.usage,
            memory: desc.memory,
            indirect_signatures: device.0.indirect_signatures.clone(),
            backing_heap: Some(heap.heap.clone()),
        })
    }

    fn place_texture(
        device: &Dx12Device,
        heap: &Dx12Heap,
        offset: u64,
        desc: &api::TextureDesc,
    ) -> Result<Dx12Texture> {
        let allocation = texture::allocation_desc(device, desc)?;
        let requirements = requirements(
            device,
            &allocation.resource,
            texture_class_bits(device, &allocation.resource),
        )?;
        validate(device, heap, offset, requirements)?;
        let mut resource = None;
        // SAFETY: heap device identity, allocation metadata and range have
        // been validated; the clear metadata is owned for this call.
        unsafe {
            device.0.native.CreatePlacedResource(
                &heap.heap,
                offset,
                &allocation.resource,
                allocation.initial_state,
                allocation.clear.as_ref().map(|clear| clear as *const _),
                &mut resource,
            )
        }
        .map_err(super::native_error)?;
        Ok(Dx12Texture(Arc::new(texture::TextureState {
            resource: resource.ok_or(Error::OutOfMemory)?,
            backing_heap: Some(heap.heap.clone()),
            desc: desc.clone(),
            sparse: None,
        })))
    }
}
