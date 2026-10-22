//! Import Metal resources and exchange ownership through shared events.
//!
//! The bridge owns native handle transport and producer/consumer ordering. A
//! fresh shared event and value 1 is emitted for every release, so values from
//! unrelated producers cannot accidentally satisfy one another.
use super::{Metal, MetalBuffer, MetalCommandEncoder, MetalDevice, MetalTexture};
use fluxel_rhi_core::api::{self, Error, ExternalBackend, Result};

/// Native payloads accepted by the Metal external bridge.
///
/// Buffer objects must describe precisely the imported region. For host shared
/// memory the bridge may create an MTLBuffer with `newBufferWithBytesNoCopy`;
/// its deallocator must retain the mapped memory until Metal releases it.
///
/// The unsafe importing bridge must ensure that independently imported aliases
/// of a host-mapped allocation are not mapped concurrently. `offset` describes
/// the region in the external allocation; the supplied MTLBuffer must already
/// begin at that region. This also applies to placement-heap buffers. Imported
/// buffer and texture objects must belong to the target Metal device. A shared
/// texture handle is reopened on that device by the backend.
#[cfg(target_vendor = "apple")]
pub enum MetalExternalHandle {
    /// Buffer covering exactly the imported region on the target device.
    Buffer(objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>),
    /// Texture owned by the target device, including an IOSurface-backed texture.
    Texture(objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLTexture>>),
    /// Cross-process Metal shared texture handle.
    SharedTexture(objc2::rc::Retained<objc2_metal::MTLSharedTextureHandle>),
    /// Cross-process shared event and its producer signal value.
    SharedEvent {
        /// Handle recreated for the target device.
        handle: objc2::rc::Retained<objc2_metal::MTLSharedEventHandle>,
        /// Value that the producer will signal.
        value: u64,
    },
}
/// Metal handles cannot be constructed outside Apple targets.
#[cfg(not(target_vendor = "apple"))]
pub enum MetalExternalHandle {}

/// Transferable completion payload for an externally released resource.
pub struct MetalExternalSync {
    #[cfg(target_vendor = "apple")]
    event: objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLSharedEvent>>,
    #[cfg(target_vendor = "apple")]
    value: u64,
}
#[cfg(target_vendor = "apple")]
impl MetalExternalSync {
    /// Produces a handle suitable for transport to another process or device.
    pub fn into_handle(self) -> MetalExternalHandle {
        use objc2_metal::MTLSharedEvent as _;
        MetalExternalHandle::SharedEvent {
            handle: self.event.newSharedEventHandle(),
            value: self.value,
        }
    }
}

impl ExternalBackend for Metal {
    type NativeExternalHandle = MetalExternalHandle;
    type NativeExternalMemory = MetalBuffer;
    type NativeExternalTexture = MetalTexture;
    type NativeExternalSync = MetalExternalSync;

    fn external_buffer(value: &MetalBuffer) -> Result<MetalBuffer> {
        Ok(value.clone())
    }
    fn external_texture(value: &MetalTexture) -> Result<MetalTexture> {
        Ok(value.clone())
    }
    unsafe fn import_external_sync(
        device: &MetalDevice,
        sync: MetalExternalHandle,
    ) -> Result<MetalExternalSync> {
        #[cfg(target_vendor = "apple")]
        {
            use objc2_metal::MTLDevice as _;
            let MetalExternalHandle::SharedEvent { handle, value } = sync else {
                return Err(Error::InvalidState);
            };
            use objc2::runtime::NSObjectProtocol as _;
            if !device
                ._state
                ._device
                .respondsToSelector(objc2::sel!(newSharedEventWithHandle:))
            {
                return Err(Error::Unsupported("Metal shared event handles on this OS"));
            }
            let event = device
                ._state
                ._device
                .newSharedEventWithHandle(&handle)
                .ok_or(Error::InvalidState)?;
            Ok(MetalExternalSync { event, value })
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = (device, sync);
            Err(Error::Unsupported("Metal requires an Apple target"))
        }
    }
    unsafe fn import_memory(
        device: &MetalDevice,
        handle: MetalExternalHandle,
        desc: &api::ExternalResourceDesc,
    ) -> Result<MetalBuffer> {
        #[cfg(target_vendor = "apple")]
        {
            use objc2_metal::{MTLBuffer as _, MTLResource as _, MTLStorageMode};
            let MetalExternalHandle::Buffer(raw) = handle else {
                return Err(Error::InvalidState);
            };
            if !std::ptr::eq(&*raw.device(), &*device._state._device) {
                return Err(Error::WrongDevice);
            }
            if desc.size == 0
                || desc.size != raw.length() as u64
                || desc.offset.checked_add(desc.size).is_none()
            {
                return Err(Error::InvalidRange);
            }
            let mapping = desc.usage.contains(api::BufferUsage::MAP_READ)
                || desc.usage.contains(api::BufferUsage::MAP_WRITE);
            if mapping && raw.storageMode() != MTLStorageMode::Shared {
                return Err(Error::InvalidState);
            }
            let memory = if mapping {
                api::MemoryClass::Upload
            } else {
                api::MemoryClass::Device
            };
            Ok(MetalBuffer {
                raw,
                size: desc.size,
                usage: desc.usage,
                memory,
                device: device._state.clone(),
                mapped: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
                heap: None,
            })
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = (device, handle, desc);
            Err(Error::Unsupported("Metal requires an Apple target"))
        }
    }
    unsafe fn import_texture(
        device: &MetalDevice,
        handle: MetalExternalHandle,
        desc: &api::TextureDesc,
    ) -> Result<MetalTexture> {
        #[cfg(target_vendor = "apple")]
        {
            use objc2_metal::{MTLDevice as _, MTLResource as _, MTLTexture as _, MTLTextureType};
            let raw = match handle {
                MetalExternalHandle::Texture(raw) => raw,
                MetalExternalHandle::SharedTexture(handle) => {
                    use objc2::runtime::NSObjectProtocol as _;
                    if !device
                        ._state
                        ._device
                        .respondsToSelector(objc2::sel!(newSharedTextureWithHandle:))
                    {
                        return Err(Error::Unsupported(
                            "Metal shared texture handles on this OS",
                        ));
                    }
                    if !std::ptr::eq(&*handle.device(), &*device._state._device) {
                        return Err(Error::WrongDevice);
                    }
                    device
                        ._state
                        ._device
                        .newSharedTextureWithHandle(&handle)
                        .ok_or(Error::InvalidState)?
                }
                _ => return Err(Error::InvalidState),
            };
            if !std::ptr::eq(&*raw.device(), &*device._state._device) {
                return Err(Error::WrongDevice);
            }
            let dimension = match raw.textureType() {
                MTLTextureType::Type1D | MTLTextureType::Type1DArray => api::TextureDimension::D1,
                MTLTextureType::Type2D
                | MTLTextureType::Type2DArray
                | MTLTextureType::Type2DMultisample
                | MTLTextureType::Type2DMultisampleArray => api::TextureDimension::D2,
                MTLTextureType::Type3D => api::TextureDimension::D3,
                _ => return Err(Error::InvalidState),
            };
            if raw.isFramebufferOnly()
                && (desc.usage.contains(api::TextureUsage::COPY_SRC)
                    || desc.usage.contains(api::TextureUsage::COPY_DST)
                    || desc.usage.contains(api::TextureUsage::TEXTURE_BINDING)
                    || desc.usage.contains(api::TextureUsage::STORAGE_BINDING))
            {
                return Err(Error::InvalidState);
            }
            if dimension != desc.dimension
                || raw.width() != desc.extent[0] as usize
                || raw.height() != desc.extent[1] as usize
                || raw.depth() != desc.extent[2] as usize
                || raw.arrayLength() != desc.array_layers as usize
                || raw.mipmapLevelCount() != desc.mip_levels as usize
                || raw.sampleCount() != desc.samples as usize
                || Some(raw.pixelFormat()) != super::texture::metal_format(desc.format)
            {
                return Err(Error::InvalidState);
            }
            Ok(MetalTexture {
                raw,
                desc: desc.clone(),
                sparse: None,
                heap: None,
            })
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = (device, handle, desc);
            Err(Error::Unsupported("Metal requires an Apple target"))
        }
    }
    fn acquire_external(
        encoder: &mut MetalCommandEncoder,
        memory: &MetalBuffer,
        sync: MetalExternalSync,
    ) -> Result<()> {
        wait(encoder, sync)?;
        #[cfg(target_vendor = "apple")]
        encoder.external_buffers.push(memory.clone());
        let _ = memory;
        Ok(())
    }
    fn release_external(
        encoder: &mut MetalCommandEncoder,
        memory: &MetalBuffer,
    ) -> Result<MetalExternalSync> {
        let result = signal(encoder)?;
        #[cfg(target_vendor = "apple")]
        encoder.external_buffers.push(memory.clone());
        let _ = memory;
        Ok(result)
    }
    fn acquire_external_texture(
        encoder: &mut MetalCommandEncoder,
        texture: &MetalTexture,
        sync: MetalExternalSync,
        transition: &api::ExternalTextureTransition,
    ) -> Result<()> {
        validate_transition(texture, transition)?;
        wait(encoder, sync)?;
        #[cfg(target_vendor = "apple")]
        encoder.external_textures.push(texture.clone());
        Ok(())
    }
    fn release_external_texture(
        encoder: &mut MetalCommandEncoder,
        texture: &MetalTexture,
        transition: &api::ExternalTextureTransition,
    ) -> Result<MetalExternalSync> {
        validate_transition(texture, transition)?;
        let result = signal(encoder)?;
        #[cfg(target_vendor = "apple")]
        encoder.external_textures.push(texture.clone());
        Ok(result)
    }
}
fn validate_transition(
    texture: &MetalTexture,
    transition: &api::ExternalTextureTransition,
) -> Result<()> {
    let desc = texture.desc();
    let available = match desc.format {
        api::TextureFormat::Depth16Unorm
        | api::TextureFormat::Depth24Plus
        | api::TextureFormat::Depth32Float => api::TextureAspects::DEPTH.0,
        api::TextureFormat::Depth24PlusStencil8 | api::TextureFormat::Depth32FloatStencil8 => {
            api::TextureAspects::DEPTH.0 | api::TextureAspects::STENCIL.0
        }
        api::TextureFormat::Stencil8 => api::TextureAspects::STENCIL.0,
        _ => api::TextureAspects::COLOR.0,
    };
    if transition.mips.start >= transition.mips.end
        || transition.mips.end > desc.mip_levels
        || transition.layers.start >= transition.layers.end
        || transition.layers.end > desc.array_layers
        || transition.aspects.0 == 0
        || transition.aspects.0 & !available != 0
    {
        return Err(Error::InvalidRange);
    }
    Ok(())
}
fn wait(encoder: &mut MetalCommandEncoder, sync: MetalExternalSync) -> Result<()> {
    #[cfg(target_vendor = "apple")]
    {
        use objc2_metal::{MTLCommandBuffer as _, MTLEvent};
        if encoder.render.is_some() || encoder.compute.is_some() {
            return Err(Error::InvalidState);
        }
        let event = objc2::runtime::ProtocolObject::<dyn MTLEvent>::from_ref(&*sync.event);
        encoder.command.encodeWaitForEvent_value(event, sync.value);
        Ok(())
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        let _ = (encoder, sync);
        Err(Error::Unsupported("Metal requires an Apple target"))
    }
}
fn signal(encoder: &mut MetalCommandEncoder) -> Result<MetalExternalSync> {
    #[cfg(target_vendor = "apple")]
    {
        use objc2_metal::{MTLCommandBuffer as _, MTLDevice as _, MTLEvent};
        if encoder.render.is_some() || encoder.compute.is_some() {
            return Err(Error::InvalidState);
        }
        let event = encoder
            .command
            .device()
            .newSharedEvent()
            .ok_or(Error::OutOfMemory)?;
        let native = objc2::runtime::ProtocolObject::<dyn MTLEvent>::from_ref(&*event);
        encoder.command.encodeSignalEvent_value(native, 1);
        Ok(MetalExternalSync { event, value: 1 })
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        let _ = encoder;
        Err(Error::Unsupported("Metal requires an Apple target"))
    }
}
