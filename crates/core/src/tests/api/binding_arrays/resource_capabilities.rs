//! Storage resource capability admission before support and allocation SPI calls.
use super::{BUFFER_CREATE_CALLS, BUFFER_SUPPORT_CALLS, TEXTURE_CALLS, TEXTURE_SUPPORT_CALLS};
use crate::api::tests::type_contract_tests::NativeShared;
use crate::{api::*, backend::*};
use std::sync::Arc;

#[test]
fn storage_resource_usage_rejects_before_support_and_creation_spi() {
    BUFFER_SUPPORT_CALLS.with(|calls| calls.set(0));
    BUFFER_CREATE_CALLS.with(|calls| calls.set(0));
    TEXTURE_SUPPORT_CALLS.with(|calls| calls.set(0));
    TEXTURE_CALLS.with(|calls| calls.set(0));
    let id = DeviceId::allocate().unwrap();
    let device = unsafe {
        Device::<NativeShared>::from_native(id, Arc::new(()), DeviceCapabilities::default())
    };
    let buffer = BufferDesc {
        label: None,
        size: 16,
        usage: BufferUsage::STORAGE,
        memory: MemoryClass::Device,
    };
    let texture = TextureDesc {
        label: None,
        dimension: TextureDimension::D2,
        extent: [16, 16, 1],
        array_layers: 1,
        format: TextureFormat::Rgba8Unorm,
        mip_levels: 1,
        samples: 1,
        usage: TextureUsage::STORAGE_BINDING,
        view_formats: vec![],
    };

    assert!(matches!(
        device.buffer_support(&buffer),
        Support::Unsupported("storage buffers")
    ));
    assert!(matches!(
        device.create_buffer(&buffer),
        Err(Error::Unsupported("storage buffers"))
    ));
    BUFFER_SUPPORT_CALLS.with(|calls| assert_eq!(calls.get(), 0));
    BUFFER_CREATE_CALLS.with(|calls| assert_eq!(calls.get(), 0));

    assert!(matches!(
        device.texture_support(&texture),
        Support::Unsupported("storage textures")
    ));
    assert!(matches!(
        device.create_texture(&texture),
        Err(Error::Unsupported("storage textures"))
    ));
    TEXTURE_SUPPORT_CALLS.with(|calls| assert_eq!(calls.get(), 0));
    TEXTURE_CALLS.with(|calls| assert_eq!(calls.get(), 0));
}
