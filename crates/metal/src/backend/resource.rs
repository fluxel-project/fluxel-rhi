//! Native resource creation and the initial buffer ownership path.
//!
//! Metal buffers, a conservative texture/view subset, and samplers are created
//! here or delegated to native owner modules. Bind groups retain native objects
//! through the adjacent binding module; this alone does not record GPU work.

#[cfg(target_vendor = "apple")]
use fluxel_rhi_core::api::MemoryClass;
use fluxel_rhi_core::api::{
    self, BufferDesc, BufferSupport, BufferUsage, ResourceBackend, SamplerDesc, Support,
    TextureDesc, TextureSupport, TextureViewDesc,
};

use super::{Metal, MetalDevice, sampler, texture};

#[derive(Clone)]
pub struct MetalBuffer {
    #[cfg(target_vendor = "apple")]
    pub(crate) raw: objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>,
    size: u64,
    #[cfg(target_vendor = "apple")]
    pub(crate) usage: BufferUsage,
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
        _: &MetalDevice,
        texture: &super::MetalTexture,
        desc: &TextureViewDesc,
    ) -> api::Result<super::MetalTextureView> {
        super::texture::create_view(texture, desc)
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
    usage.0 == 0
        || usage.contains(BufferUsage::MAP_READ)
        || usage.contains(BufferUsage::MAP_WRITE)
        || usage.contains(BufferUsage::ACCEL_BUILD_INPUT)
        || usage.contains(BufferUsage::ACCEL_SCRATCH)
        || usage.contains(BufferUsage::SHADER_TABLE)
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
mod tests {
    use super::unsupported_buffer_usage;
    use fluxel_rhi_core::api::BufferUsage;

    #[test]
    fn buffer_support_does_not_advertise_unimplemented_mapping_or_ray_usage() {
        assert!(unsupported_buffer_usage(BufferUsage(0)));
        assert!(unsupported_buffer_usage(BufferUsage::MAP_READ));
        assert!(unsupported_buffer_usage(BufferUsage::MAP_WRITE));
        assert!(unsupported_buffer_usage(BufferUsage::ACCEL_BUILD_INPUT));
        assert!(unsupported_buffer_usage(BufferUsage::ACCEL_SCRATCH));
        assert!(unsupported_buffer_usage(BufferUsage::SHADER_TABLE));
        assert!(!unsupported_buffer_usage(BufferUsage::COPY_SRC));
        assert!(!unsupported_buffer_usage(BufferUsage::VERTEX));
    }
}
