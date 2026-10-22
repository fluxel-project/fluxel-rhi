//! Explicit Metal placement heaps and native allocation requirements.
//!
//! Aliased resources retain the heap; the renderer supplies ordering and
//! lifetime discipline for overlapping placements. Host mappings share a lock.
use super::{Metal, MetalBuffer, MetalDevice, MetalTexture};
use fluxel_rhi_core::api::{self, Error, MemoryBackend, ResourceBackend, Result, Support};

/// Placement heaps require a GPU family that exposes explicit heap offsets.
#[cfg(target_vendor = "apple")]
pub(super) fn supported(
    device: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>,
) -> bool {
    use objc2::runtime::NSObjectProtocol as _;
    use objc2_metal::{MTLDevice as _, MTLGPUFamily};
    device.respondsToSelector(objc2::sel!(supportsFamily:))
        && (device.supportsFamily(MTLGPUFamily::Apple2)
            || device.supportsFamily(MTLGPUFamily::Mac2))
}

#[derive(Clone)]
pub struct MetalHeap {
    #[cfg(target_vendor = "apple")]
    raw: objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLHeap>>,
    #[cfg(target_vendor = "apple")]
    device: std::sync::Arc<super::DeviceState>,
    #[cfg(target_vendor = "apple")]
    size: u64,
    #[cfg(target_vendor = "apple")]
    class: u32,
    #[cfg(target_vendor = "apple")]
    mapped: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl MemoryBackend for Metal {
    type NativeHeap = MetalHeap;

    fn buffer_requirements(
        device: &MetalDevice,
        desc: &api::BufferDesc,
    ) -> Result<api::MemoryRequirements> {
        #[cfg(target_vendor = "apple")]
        {
            use objc2_metal::MTLDevice as _;
            if !matches!(Self::buffer_support(device, desc), Support::Supported(_)) {
                return Err(Error::Unsupported("Metal buffer descriptor"));
            }
            let size = usize::try_from(desc.size).map_err(|_| Error::InvalidRange)?;
            let result = device
                ._state
                ._device
                .heapBufferSizeAndAlignWithLength_options(size, options(desc.memory));
            return requirements(result, class(desc.memory));
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = (device, desc);
            Err(Error::Unsupported("Metal requires an Apple target"))
        }
    }

    fn texture_requirements(
        device: &MetalDevice,
        desc: &api::TextureDesc,
    ) -> Result<api::MemoryRequirements> {
        #[cfg(target_vendor = "apple")]
        {
            use objc2_metal::MTLDevice as _;
            let native = super::texture::native_descriptor(device, desc, false)?;
            return requirements(
                device
                    ._state
                    ._device
                    .heapTextureSizeAndAlignWithDescriptor(&native),
                0,
            );
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = (device, desc);
            Err(Error::Unsupported("Metal requires an Apple target"))
        }
    }

    fn create_heap(device: &MetalDevice, desc: &api::HeapDesc) -> Result<MetalHeap> {
        #[cfg(target_vendor = "apple")]
        {
            use objc2_metal::{MTLDevice as _, MTLHeapDescriptor, MTLHeapType, MTLStorageMode};
            if desc.size == 0 || desc.native_memory_class > 1 {
                return Err(Error::InvalidRange);
            }
            let size = usize::try_from(desc.size).map_err(|_| Error::InvalidRange)?;
            let descriptor = MTLHeapDescriptor::new();
            descriptor.setType(MTLHeapType::Placement);
            descriptor.setSize(size);
            descriptor.setStorageMode(if desc.native_memory_class == 0 {
                MTLStorageMode::Private
            } else {
                MTLStorageMode::Shared
            });
            let raw = device
                ._state
                ._device
                .newHeapWithDescriptor(&descriptor)
                .ok_or(Error::OutOfMemory)?;
            return Ok(MetalHeap {
                raw,
                device: device._state.clone(),
                size: desc.size,
                class: desc.native_memory_class,
                mapped: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            });
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = (device, desc);
            Err(Error::Unsupported("Metal requires an Apple target"))
        }
    }

    fn place_buffer(
        device: &MetalDevice,
        heap: &MetalHeap,
        offset: u64,
        desc: &api::BufferDesc,
    ) -> Result<MetalBuffer> {
        #[cfg(target_vendor = "apple")]
        {
            use objc2_metal::MTLHeap as _;
            let req = Self::buffer_requirements(device, desc)?;
            validate(device, heap, offset, req)?;
            let offset = usize::try_from(offset).map_err(|_| Error::InvalidRange)?;
            let length = usize::try_from(desc.size).map_err(|_| Error::InvalidRange)?;
            // SAFETY: ownership, class, alignment and the complete native footprint were checked.
            let raw = unsafe {
                heap.raw
                    .newBufferWithLength_options_offset(length, options(desc.memory), offset)
            }
            .ok_or(Error::OutOfMemory)?;
            return Ok(MetalBuffer {
                raw,
                size: desc.size,
                usage: desc.usage,
                memory: desc.memory,
                device: device._state.clone(),
                mapped: heap.mapped.clone(),
                heap: Some(heap.clone()),
            });
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = (device, heap, offset, desc);
            Err(Error::Unsupported("Metal requires an Apple target"))
        }
    }

    fn place_texture(
        device: &MetalDevice,
        heap: &MetalHeap,
        offset: u64,
        desc: &api::TextureDesc,
    ) -> Result<MetalTexture> {
        #[cfg(target_vendor = "apple")]
        {
            use objc2_metal::MTLHeap as _;
            let native = super::texture::native_descriptor(device, desc, false)?;
            let req = Self::texture_requirements(device, desc)?;
            validate(device, heap, offset, req)?;
            let offset = usize::try_from(offset).map_err(|_| Error::InvalidRange)?;
            // SAFETY: ownership, storage mode, alignment and native footprint were checked.
            let raw = unsafe { heap.raw.newTextureWithDescriptor_offset(&native, offset) }
                .ok_or(Error::OutOfMemory)?;
            return Ok(MetalTexture {
                raw,
                desc: desc.clone(),
                sparse: None,
                heap: Some(heap.clone()),
            });
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = (device, heap, offset, desc);
            Err(Error::Unsupported("Metal requires an Apple target"))
        }
    }
}

#[cfg(target_vendor = "apple")]
fn class(memory: api::MemoryClass) -> u32 {
    if memory == api::MemoryClass::Device {
        0
    } else {
        1
    }
}
#[cfg(target_vendor = "apple")]
fn options(memory: api::MemoryClass) -> objc2_metal::MTLResourceOptions {
    if class(memory) == 0 {
        objc2_metal::MTLResourceOptions::StorageModePrivate
    } else {
        objc2_metal::MTLResourceOptions::StorageModeShared
    }
}
#[cfg(target_vendor = "apple")]
fn requirements(
    value: objc2_metal::MTLSizeAndAlign,
    class: u32,
) -> Result<api::MemoryRequirements> {
    if value.size == 0 || value.align == 0 {
        return Err(Error::Unsupported("Metal heap requirements"));
    }
    Ok(api::MemoryRequirements {
        size: value.size as u64,
        alignment: value.align as u64,
        native_type_bits: 1 << class,
        dedicated_only: false,
    })
}
#[cfg(target_vendor = "apple")]
fn validate(
    device: &MetalDevice,
    heap: &MetalHeap,
    offset: u64,
    req: api::MemoryRequirements,
) -> Result<()> {
    if !std::sync::Arc::ptr_eq(&device._state, &heap.device) {
        return Err(Error::WrongDevice);
    }
    if req.native_type_bits & (1 << heap.class) == 0 {
        return Err(Error::InvalidState);
    }
    if offset % req.alignment != 0
        || offset
            .checked_add(req.size)
            .is_none_or(|end| end > heap.size)
    {
        return Err(Error::InvalidRange);
    }
    Ok(())
}
