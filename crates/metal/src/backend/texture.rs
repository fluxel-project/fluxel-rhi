//! Metal texture allocation, format lowering, and bounded view creation.
//!
//! The native path accepts only direct pixel-format mappings and 2D
//! allocations, including render-only Depth32FloatStencil8 attachments. This
//! module validates descriptor dimensions and view ranges; it does not infer
//! or track resource layouts from creation usage.

use fluxel_rhi_core::api::{
    self, Support, TextureDesc, TextureDimension, TextureFormat, TextureSupport, TextureUsage,
    TextureViewDesc,
};
#[cfg(target_vendor = "apple")]
use fluxel_rhi_core::api::{TextureAspects, TextureViewDimension};

use super::MetalDevice;

const MAX_2D_EXTENT: u32 = 8192;
const MAX_ARRAY_LAYERS: u32 = 2048;
const SUPPORTED_USAGE: TextureUsage = TextureUsage(
    TextureUsage::COPY_SRC.0
        | TextureUsage::COPY_DST.0
        | TextureUsage::TEXTURE_BINDING.0
        | TextureUsage::STORAGE_BINDING.0
        | TextureUsage::RENDER_ATTACHMENT.0,
);

#[derive(Clone)]
pub struct MetalTexture {
    #[cfg(target_vendor = "apple")]
    pub(crate) raw:
        objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLTexture>>,
    #[allow(dead_code)]
    desc: TextureDesc,
    pub(super) sparse: Option<std::sync::Arc<super::sparse::SparseState>>,
}

impl MetalTexture {
    pub(crate) fn desc(&self) -> &TextureDesc {
        &self.desc
    }
}

#[derive(Clone)]
pub struct MetalTextureView {
    #[cfg(target_vendor = "apple")]
    #[allow(dead_code)]
    pub(crate) raw:
        objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLTexture>>,
    #[cfg(target_vendor = "apple")]
    pub(crate) format: TextureFormat,
    #[cfg(target_vendor = "apple")]
    pub(crate) dimension: TextureViewDimension,
    #[cfg(target_vendor = "apple")]
    pub(crate) usage: TextureUsage,
}

pub(super) fn texture_support(device: &MetalDevice, desc: &TextureDesc) -> Support<TextureSupport> {
    if !valid_desc(desc) || !format_supported(desc.format) {
        return Support::Unsupported("Metal texture descriptor is outside the implemented subset");
    }
    #[cfg(target_vendor = "apple")]
    {
        use objc2_metal::MTLDevice as _;
        if desc.format == TextureFormat::Depth32FloatStencil8 && !supports_depth32_stencil(device) {
            return Support::Unsupported(
                "Metal device family does not support Depth32FloatStencil8",
            );
        }
        let sample_counts = [1, 2, 4, 8]
            .into_iter()
            .filter(|count| device._state._device.supportsTextureSampleCount(*count))
            .map(|count| count as u32)
            .collect::<Vec<_>>();
        if !sample_counts.contains(&desc.samples) {
            return Support::Unsupported(
                "Metal texture sample count is unsupported by this device",
            );
        }
        Support::Supported(TextureSupport {
            max_extent: [MAX_2D_EXTENT, MAX_2D_EXTENT, 1],
            max_array_layers: MAX_ARRAY_LAYERS,
            max_mips: max_mip_levels(desc.extent),
            sample_counts,
            compatible_view_formats: vec![desc.format],
            filterable: is_filterable(desc.format),
            blendable: is_blendable(desc.format),
            storage: desc.samples == 1 && desc.format == TextureFormat::Rgba8Unorm,
        })
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        let _ = device;
        Support::Unsupported("Metal requires an Apple target")
    }
}

#[cfg(target_vendor = "apple")]
pub(super) fn supports_depth32_stencil(device: &MetalDevice) -> bool {
    use objc2_metal::{MTLDevice as _, MTLGPUFamily};
    device._state._device.supportsFamily(MTLGPUFamily::Apple2)
        || device._state._device.supportsFamily(MTLGPUFamily::Mac2)
}

pub(super) fn create_texture(
    device: &MetalDevice,
    desc: &TextureDesc,
) -> api::Result<MetalTexture> {
    create_texture_inner(device, desc, false)
}

pub(super) fn create_sparse_texture(
    device: &MetalDevice,
    desc: &TextureDesc,
) -> api::Result<MetalTexture> {
    create_texture_inner(device, desc, true)
}

fn create_texture_inner(
    device: &MetalDevice,
    desc: &TextureDesc,
    sparse: bool,
) -> api::Result<MetalTexture> {
    #[cfg(target_vendor = "apple")]
    {
        use objc2_metal::{
            MTLDevice as _, MTLResource as _, MTLStorageMode, MTLTextureDescriptor, MTLTextureType,
            MTLTextureUsage,
        };

        if !matches!(texture_support(device, desc), Support::Supported(_)) {
            return Err(api::Error::Unsupported("Metal texture descriptor"));
        }
        let format =
            metal_format(desc.format).ok_or(api::Error::Unsupported("Metal texture format"))?;
        let descriptor = MTLTextureDescriptor::new();
        if sparse {
            descriptor.setPlacementSparsePageSize(objc2_metal::MTLSparsePageSize::Size64);
        }
        descriptor.setPixelFormat(format);
        let multisampled = desc.samples > 1;
        let arrayed = desc.array_layers > 1;
        descriptor.setTextureType(match (multisampled, arrayed) {
            (false, false) => MTLTextureType::Type2D,
            (false, true) => MTLTextureType::Type2DArray,
            (true, false) => MTLTextureType::Type2DMultisample,
            (true, true) => return Err(api::Error::Unsupported("Metal multisample arrays")),
        });
        // SAFETY: valid_desc bounds nonzero dimensions within MAX_2D_EXTENT.
        unsafe { descriptor.setWidth(desc.extent[0] as usize) };
        // SAFETY: valid_desc bounds the nonzero 2D height within MAX_2D_EXTENT.
        unsafe { descriptor.setHeight(desc.extent[1] as usize) };
        // SAFETY: this implementation accepts only 2D textures, whose depth is one.
        unsafe { descriptor.setDepth(1) };
        // SAFETY: valid_desc checks a nonzero mip count no larger than the full mip chain.
        unsafe { descriptor.setMipmapLevelCount(desc.mip_levels as usize) };
        if arrayed {
            // SAFETY: valid_desc bounds the nonzero array length within MAX_ARRAY_LAYERS.
            unsafe { descriptor.setArrayLength(desc.array_layers as usize) };
        }
        // SAFETY: texture_support confirmed this sample count on the selected device.
        unsafe { descriptor.setSampleCount(desc.samples as usize) };
        descriptor.setStorageMode(MTLStorageMode::Private);
        let mut usage = MTLTextureUsage::Unknown;
        if desc.usage.contains(TextureUsage::TEXTURE_BINDING) {
            usage |= MTLTextureUsage::ShaderRead;
        }
        if desc.usage.contains(TextureUsage::STORAGE_BINDING) {
            usage |= MTLTextureUsage::ShaderWrite;
        }
        if desc.usage.contains(TextureUsage::RENDER_ATTACHMENT) {
            usage |= MTLTextureUsage::RenderTarget;
        }
        descriptor.setUsage(usage);
        let raw = device
            ._state
            ._device
            .newTextureWithDescriptor(&descriptor)
            .ok_or(api::Error::Backend(
                "Metal rejected a supported texture descriptor".into(),
            ))?;
        if let Some(label) = desc.label.as_deref() {
            raw.setLabel(Some(&objc2_foundation::NSString::from_str(label)));
        }
        Ok(MetalTexture {
            raw,
            desc: desc.clone(),
            sparse: None,
        })
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        let _ = (device, desc, sparse);
        Err(api::Error::Unsupported("Metal requires an Apple target"))
    }
}

pub(super) fn create_view(
    texture: &MetalTexture,
    desc: &TextureViewDesc,
) -> api::Result<MetalTextureView> {
    #[cfg(target_vendor = "apple")]
    {
        use objc2_foundation::NSRange;
        use objc2_metal::{MTLTexture as _, MTLTextureType};

        if desc.format != texture.desc.format
            || desc.aspects.0 != aspect_for_format(texture.desc.format)
            || desc.mips.start >= desc.mips.end
            || desc.layers.start >= desc.layers.end
            || desc.mips.end > texture.desc.mip_levels
            || desc.layers.end > texture.desc.array_layers
        {
            return Err(api::Error::InvalidRange);
        }
        let view_type = match desc.dimension {
            TextureViewDimension::D2 if desc.layers.len() == 1 && texture.desc.samples > 1 => {
                if desc.mips.len() != 1 || desc.mips.start != 0 {
                    return Err(api::Error::InvalidRange);
                }
                MTLTextureType::Type2DMultisample
            }
            TextureViewDimension::D2 if desc.layers.len() == 1 => MTLTextureType::Type2D,
            TextureViewDimension::D2Array => MTLTextureType::Type2DArray,
            TextureViewDimension::Cube
                if texture.desc.array_layers == 6
                    && desc.layers.start == 0
                    && desc.layers.len() == 6 =>
            {
                MTLTextureType::TypeCube
            }
            TextureViewDimension::CubeArray
                if desc.layers.start.is_multiple_of(6) && desc.layers.len().is_multiple_of(6) =>
            {
                MTLTextureType::TypeCubeArray
            }
            _ => return Err(api::Error::Unsupported("Metal texture view dimension")),
        };
        if texture.desc.samples > 1 && view_type != MTLTextureType::Type2DMultisample {
            return Err(api::Error::Unsupported("Metal multisample view dimension"));
        }
        // SAFETY: both nonempty ranges were checked against the retained
        // descriptor above, so Metal cannot index outside this texture.
        let raw = unsafe {
            texture
                .raw
                .newTextureViewWithPixelFormat_textureType_levels_slices(
                    metal_format(texture.desc.format)
                        .ok_or(api::Error::Unsupported("Metal texture view format"))?,
                    view_type,
                    NSRange {
                        location: desc.mips.start as usize,
                        length: desc.mips.len(),
                    },
                    NSRange {
                        location: desc.layers.start as usize,
                        length: desc.layers.len(),
                    },
                )
        }
        .ok_or(api::Error::InvalidState)?;
        Ok(MetalTextureView {
            raw,
            format: desc.format,
            dimension: desc.dimension,
            usage: texture.desc.usage,
        })
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        let _ = (texture, desc);
        Err(api::Error::Unsupported("Metal requires an Apple target"))
    }
}

fn valid_desc(desc: &TextureDesc) -> bool {
    desc.dimension == TextureDimension::D2
        && desc.extent[0] != 0
        && desc.extent[1] != 0
        && desc.extent[2] == 1
        && desc.extent[0] <= MAX_2D_EXTENT
        && desc.extent[1] <= MAX_2D_EXTENT
        && desc.array_layers != 0
        && desc.array_layers <= MAX_ARRAY_LAYERS
        && desc.mip_levels != 0
        && desc.mip_levels <= max_mip_levels(desc.extent)
        && matches!(desc.samples, 1 | 2 | 4 | 8)
        && (desc.samples == 1 || (desc.mip_levels == 1 && desc.array_layers == 1))
        && (desc.samples == 1 || desc.usage.0 == TextureUsage::RENDER_ATTACHMENT.0)
        && desc.usage.0 != 0
        && desc.usage.0 & !SUPPORTED_USAGE.0 == 0
        && (!desc.usage.contains(TextureUsage::STORAGE_BINDING)
            || desc.format == TextureFormat::Rgba8Unorm)
        && (desc.format != TextureFormat::Depth32FloatStencil8
            || desc.usage.0 == TextureUsage::RENDER_ATTACHMENT.0)
        && desc
            .view_formats
            .iter()
            .all(|format| *format == desc.format)
}

fn max_mip_levels(extent: [u32; 3]) -> u32 {
    32 - extent[0].max(extent[1]).leading_zeros()
}

#[cfg(target_vendor = "apple")]
fn aspect_for_format(format: TextureFormat) -> u8 {
    match format {
        TextureFormat::Depth32Float => TextureAspects::DEPTH.0,
        TextureFormat::Depth32FloatStencil8 => TextureAspects::DEPTH.0 | TextureAspects::STENCIL.0,
        _ => TextureAspects::COLOR.0,
    }
}

#[cfg(target_vendor = "apple")]
fn is_filterable(format: TextureFormat) -> bool {
    matches!(
        format,
        TextureFormat::R8Unorm
            | TextureFormat::Rg8Unorm
            | TextureFormat::Rgba8Unorm
            | TextureFormat::Rgba8UnormSrgb
            | TextureFormat::Bgra8Unorm
            | TextureFormat::Bgra8UnormSrgb
            | TextureFormat::Rgba16Float
    )
}

#[cfg(target_vendor = "apple")]
fn is_blendable(format: TextureFormat) -> bool {
    matches!(
        format,
        TextureFormat::Rgba8Unorm
            | TextureFormat::Rgba8UnormSrgb
            | TextureFormat::Bgra8Unorm
            | TextureFormat::Bgra8UnormSrgb
            | TextureFormat::Rgba16Float
    )
}

fn format_supported(format: TextureFormat) -> bool {
    matches!(
        format,
        TextureFormat::Rgba8Unorm
            | TextureFormat::Rgba8UnormSrgb
            | TextureFormat::Bgra8Unorm
            | TextureFormat::Bgra8UnormSrgb
            | TextureFormat::Rgba16Float
            | TextureFormat::Depth32Float
            | TextureFormat::Depth32FloatStencil8
    )
}

#[cfg(target_vendor = "apple")]
pub(super) fn metal_format(format: TextureFormat) -> Option<objc2_metal::MTLPixelFormat> {
    use objc2_metal::MTLPixelFormat as Mtl;
    Some(match format {
        TextureFormat::Rgba8Unorm => Mtl::RGBA8Unorm,
        TextureFormat::Rgba8UnormSrgb => Mtl::RGBA8Unorm_sRGB,
        TextureFormat::Bgra8Unorm => Mtl::BGRA8Unorm,
        TextureFormat::Bgra8UnormSrgb => Mtl::BGRA8Unorm_sRGB,
        TextureFormat::Rgba16Float => Mtl::RGBA16Float,
        TextureFormat::Depth32Float => Mtl::Depth32Float,
        TextureFormat::Depth32FloatStencil8 => Mtl::Depth32Float_Stencil8,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::{MAX_2D_EXTENT, MAX_ARRAY_LAYERS, max_mip_levels, valid_desc};
    use fluxel_rhi_core::api::{TextureDesc, TextureDimension, TextureFormat, TextureUsage};

    fn desc() -> TextureDesc {
        TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [512, 256, 1],
            array_layers: 1,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 10,
            samples: 1,
            usage: TextureUsage::TEXTURE_BINDING,
            view_formats: vec![TextureFormat::Rgba8Unorm],
        }
    }

    #[test]
    fn texture_descriptor_bounds_match_the_advertised_subset() {
        let base = desc();
        assert!(valid_desc(&base));
        assert_eq!(max_mip_levels(base.extent), 10);
        let mut invalid = base.clone();
        invalid.extent[0] = MAX_2D_EXTENT + 1;
        assert!(!valid_desc(&invalid));
        let mut invalid = base.clone();
        invalid.array_layers = MAX_ARRAY_LAYERS + 1;
        assert!(!valid_desc(&invalid));
        let mut invalid = base;
        invalid.usage = TextureUsage::STORAGE_BINDING;
        assert!(valid_desc(&invalid));
        invalid.format = TextureFormat::Bgra8Unorm;
        assert!(!valid_desc(&invalid));
        let mut multisampled = desc();
        multisampled.samples = 4;
        multisampled.mip_levels = 1;
        multisampled.usage = TextureUsage::RENDER_ATTACHMENT;
        assert!(valid_desc(&multisampled));
        multisampled.mip_levels = 2;
        assert!(!valid_desc(&multisampled));
        multisampled.mip_levels = 1;
        multisampled.usage = TextureUsage(multisampled.usage.0 | TextureUsage::TEXTURE_BINDING.0);
        assert!(!valid_desc(&multisampled));
        let mut depth_stencil = desc();
        depth_stencil.format = TextureFormat::Depth32FloatStencil8;
        depth_stencil.mip_levels = 1;
        depth_stencil.usage = TextureUsage::RENDER_ATTACHMENT;
        depth_stencil.view_formats = vec![TextureFormat::Depth32FloatStencil8];
        assert!(valid_desc(&depth_stencil));
        depth_stencil.usage =
            TextureUsage(TextureUsage::RENDER_ATTACHMENT.0 | TextureUsage::TEXTURE_BINDING.0);
        assert!(!valid_desc(&depth_stencil));
        let mut invalid = desc();
        invalid.usage = TextureUsage(1 << 63);
        assert!(!valid_desc(&invalid));
    }
}
