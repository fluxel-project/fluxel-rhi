//! Native resource creation and the initial buffer ownership path.
//!
//! Metal buffers, a conservative texture/view subset, and samplers are created
//! here or delegated to native owner modules. Bind groups retain native objects
//! through the adjacent binding module; this alone does not record GPU work.

#[cfg(target_vendor = "apple")]
use fluxel_rhi_core::api::MemoryClass;
use fluxel_rhi_core::api::{
    self, BufferDesc, BufferSupport, BufferUsage, SamplerDesc, Support, TextureDesc,
    TextureSupport, TextureViewDesc,
};
use fluxel_rhi_core::backend::ResourceBackend;

use super::{Metal, MetalDevice, sampler, texture};

#[derive(Clone)]
pub struct MetalBuffer {
    #[cfg(target_vendor = "apple")]
    pub(crate) raw: objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>,
    pub(super) size: u64,
    #[cfg(target_vendor = "apple")]
    pub(crate) usage: BufferUsage,
    #[cfg(target_vendor = "apple")]
    pub(crate) memory: MemoryClass,
    #[cfg(target_vendor = "apple")]
    pub(super) device: std::sync::Arc<super::DeviceState>,
    #[cfg(target_vendor = "apple")]
    pub(crate) mapped: std::sync::Arc<std::sync::atomic::AtomicBool>,
    #[cfg(target_vendor = "apple")]
    pub(super) heap: Option<super::memory::MetalHeap>,
}

impl ResourceBackend for Metal {
    fn buffer_support(_device: &MetalDevice, desc: &BufferDesc) -> Support<BufferSupport> {
        #[cfg(target_vendor = "apple")]
        let max_size = {
            use objc2_metal::MTLDevice as _;
            _device._state._device.maxBufferLength() as u64
        };
        #[cfg(not(target_vendor = "apple"))]
        let max_size = 0;
        if unsupported_buffer_usage(desc.usage) {
            return Support::Unsupported("Metal buffer usage is not implemented");
        }
        if (desc.usage.contains(BufferUsage::MAP_READ)
            || desc.usage.contains(BufferUsage::MAP_WRITE))
            && desc.memory == api::MemoryClass::Device
        {
            return Support::Unsupported("Metal mapping requires the matching host memory class");
        }
        if desc.size == 0 || desc.size > max_size {
            return Support::Unsupported("Metal buffer size is outside the supported range");
        }
        #[cfg(target_vendor = "apple")]
        return Support::Supported(BufferSupport {
            alignment: 1,
            max_size,
        });
        #[cfg(not(target_vendor = "apple"))]
        Support::Unsupported("Metal requires an Apple target")
    }

    fn texture_support(device: &MetalDevice, desc: &TextureDesc) -> Support<TextureSupport> {
        texture::texture_support(device, desc)
    }

    fn create_buffer(device: &MetalDevice, desc: &BufferDesc) -> api::Result<MetalBuffer> {
        #[cfg(target_vendor = "apple")]
        {
            use objc2_metal::{MTLDevice as _, MTLResourceOptions};

            let length = usize::try_from(desc.size).map_err(|_| api::Error::OutOfMemory)?;
            let options = match desc.memory {
                MemoryClass::Device => MTLResourceOptions::StorageModePrivate,
                MemoryClass::Upload | MemoryClass::Readback => {
                    MTLResourceOptions::StorageModeShared
                }
            };
            let raw = device
                ._state
                ._device
                .newBufferWithLength_options(length, options)
                .ok_or(api::Error::OutOfMemory)?;
            Ok(MetalBuffer {
                raw,
                size: desc.size,
                usage: desc.usage,
                memory: desc.memory,
                device: device._state.clone(),
                mapped: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
                heap: None,
            })
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = (device, desc);
            Err(api::Error::Unsupported("Metal requires an Apple target"))
        }
    }

    fn create_texture(
        device: &MetalDevice,
        desc: &TextureDesc,
    ) -> api::Result<super::MetalTexture> {
        texture::create_texture(device, desc)
    }

    fn create_view(
        device: &MetalDevice,
        texture: &super::MetalTexture,
        desc: &TextureViewDesc,
    ) -> api::Result<super::MetalTextureView> {
        super::texture::create_view(device, texture, desc)
    }

    fn create_sampler(
        device: &MetalDevice,
        desc: &SamplerDesc,
    ) -> api::Result<super::MetalSampler> {
        sampler::create_sampler(device, desc)
    }

    fn create_bind_group_layout(
        _: &MetalDevice,
        desc: &api::BindGroupLayoutDesc,
    ) -> api::Result<super::MetalBindGroupLayout> {
        super::bindings::create_layout(desc)
    }

    fn create_bind_group(
        _: &MetalDevice,
        desc: &api::BindGroupDesc<'_, Self>,
    ) -> api::Result<super::MetalBindGroup> {
        super::bindings::create_bind_group(desc)
    }

    fn create_pipeline_layout(
        _: &MetalDevice,
        desc: &api::PipelineLayoutDesc<'_, Self>,
    ) -> api::Result<super::MetalPipelineLayout> {
        super::bindings::create_pipeline_layout(desc)
    }
}

fn unsupported_buffer_usage(usage: BufferUsage) -> bool {
    usage.0 == 0 || usage.contains(BufferUsage::SHADER_TABLE)
}

impl MetalBuffer {
    #[allow(dead_code)]
    pub(crate) fn size(&self) -> u64 {
        self.size
    }

    #[cfg(target_vendor = "apple")]
    #[allow(dead_code)]
    pub(crate) fn native(&self) -> &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer> {
        &self.raw
    }
}

#[cfg(test)]
#[path = "../tests/backend/resource.rs"]
mod tests;
