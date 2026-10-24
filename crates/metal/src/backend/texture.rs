//! Metal texture allocation, format lowering, and bounded view creation.
//!
//! This module validates device-dependent formats, dimensions, and view ranges.
//! Depth-stencil views retain an attachment view and, when necessary, a distinct
//! stencil sampling view. Resource layouts are managed by the caller.

#[cfg(target_vendor = "apple")]
use fluxel_rhi_core::api::TextureViewDimension;
use fluxel_rhi_core::api::{
    self, Support, TextureDesc, TextureDimension, TextureFormat, TextureSupport, TextureUsage,
    TextureViewDesc,
};

use super::MetalDevice;

mod formats;
pub(in crate::backend) use formats::color_attachment_format;
#[cfg(target_vendor = "apple")]
pub(in crate::backend) use formats::is_blendable;
#[cfg(target_vendor = "apple")]
pub(super) use formats::is_filterable;
#[cfg(target_vendor = "apple")]
pub(super) use formats::metal_format;
#[cfg(target_vendor = "apple")]
use formats::{aspect_for_format, is_32bit_float, needs_32bit_msaa_family};
use formats::{
    compatible_formats, format_supported, is_bc, is_compressed, is_etc_or_astc, max_mip_levels,
    storage_format,
};

const MAX_2D_EXTENT: u32 = 32768;
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
    pub(super) desc: TextureDesc,
    pub(super) sparse: Option<std::sync::Arc<super::sparse::SparseState>>,
    #[cfg(target_vendor = "apple")]
    #[allow(dead_code)] // Retains the backing heap through GPU resource retirement.
    pub(super) heap: Option<super::memory::MetalHeap>,
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
    pub(crate) sampled_raw:
        objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLTexture>>,
    #[cfg(target_vendor = "apple")]
    pub(crate) format: TextureFormat,
    #[cfg(target_vendor = "apple")]
    pub(crate) dimension: TextureViewDimension,
    #[cfg(target_vendor = "apple")]
    pub(crate) usage: TextureUsage,
    #[cfg(target_vendor = "apple")]
    pub(crate) samples: u32,
    #[cfg(target_vendor = "apple")]
    pub(crate) filterable: bool,
    #[cfg(target_vendor = "apple")]
    pub(crate) aspects: api::TextureAspects,
}

pub(super) fn texture_support(device: &MetalDevice, desc: &TextureDesc) -> Support<TextureSupport> {
    if !format_supported(desc.format) {
        return Support::Unsupported("Metal has no native pixel format for this texture format");
    }
    if let Some(reason) = texture_desc_rejection(desc) {
        return Support::Unsupported(reason);
    }
    #[cfg(target_vendor = "apple")]
    {
        use objc2_metal::MTLDevice as _;
        use objc2_metal::MTLGPUFamily;
        if is_etc_or_astc(desc.format) {
            let family = match desc.format {
                TextureFormat::Astc { hdr: true, .. } => MTLGPUFamily::Apple6,
                _ => MTLGPUFamily::Apple2,
            };
            if !device._state._device.supportsFamily(family) {
                return Support::Unsupported(
                    "Metal device does not support this compressed format",
                );
            }
        }
        if is_compressed(desc.format)
            && desc.dimension == TextureDimension::D3
            && !device._state._device.supportsFamily(MTLGPUFamily::Apple3)
            && !device._state._device.supportsFamily(MTLGPUFamily::Mac2)
        {
            return Support::Unsupported(
                "Metal device does not support compressed volume textures",
            );
        }
        if is_bc(desc.format) {
            use objc2::runtime::NSObjectProtocol as _;
            let native = &device._state._device;
            let supported = if native.respondsToSelector(objc2::sel!(supportsBCTextureCompression))
            {
                native.supportsBCTextureCompression()
            } else {
                native.supportsFamily(MTLGPUFamily::Mac2)
            };
            if !supported {
                return Support::Unsupported("Metal device does not support BC compression");
            }
        }
        let max_2d = max_texture_dimension_2d(&device._state._device);
        if desc.dimension != TextureDimension::D3
            && (desc.extent[0] > max_2d || desc.extent[1] > max_2d)
        {
            return Support::Unsupported("Metal texture exceeds device family limits");
        }
        if matches!(
            desc.format,
            TextureFormat::Depth32FloatStencil8 | TextureFormat::Depth24PlusStencil8
        ) && !supports_depth32_stencil(device)
        {
            return Support::Unsupported(
                "Metal device family does not support Depth32FloatStencil8",
            );
        }
        if desc.format == TextureFormat::Stencil8
            && !device._state._device.supportsFamily(MTLGPUFamily::Apple2)
        {
            return Support::Unsupported(
                "Metal standalone stencil format unavailable on this device",
            );
        }
        if desc.samples > 1 && desc.array_layers > 1 {
            use objc2_metal::MTLGPUFamily;
            if !device._state._device.supportsFamily(MTLGPUFamily::Apple7)
                && !device._state._device.supportsFamily(MTLGPUFamily::Mac2)
            {
                return Support::Unsupported(
                    "Metal multisample arrays require a supported GPU family",
                );
            }
        }
        let msaa32 = device._state._device.supportsFamily(MTLGPUFamily::Apple7)
            || device._state._device.supportsFamily(MTLGPUFamily::Mac2);
        use objc2::runtime::NSObjectProtocol as _;
        let float32_filtering = device._state._device.supportsFamily(MTLGPUFamily::Apple9)
            || device._state._device.supportsFamily(MTLGPUFamily::Mac2)
            || (device
                ._state
                ._device
                .respondsToSelector(objc2::sel!(supports32BitFloatFiltering))
                && device._state._device.supports32BitFloatFiltering());
        let storage = storage_format(desc.format)
            || (matches!(
                desc.format,
                TextureFormat::Rgba8UnormSrgb | TextureFormat::Bgra8UnormSrgb
            ) && device._state._device.supportsFamily(MTLGPUFamily::Apple2))
            || (matches!(
                desc.format,
                TextureFormat::Rgb10a2Unorm
                    | TextureFormat::Rgb10a2Uint
                    | TextureFormat::Rg11b10Ufloat
                    | TextureFormat::Rgb9e5Ufloat
            ) && (device._state._device.supportsFamily(MTLGPUFamily::Apple3)
                || device._state._device.supportsFamily(MTLGPUFamily::Mac2)));
        if desc.usage.contains(TextureUsage::STORAGE_BINDING) && !storage {
            return Support::Unsupported(
                "Metal format does not support shader writes on this device",
            );
        }
        let sample_counts = [1, 2, 4, 8]
            .into_iter()
            .filter(|count| {
                (*count == 1
                    || (desc.dimension == TextureDimension::D2
                        && !is_compressed(desc.format)
                        && (!needs_32bit_msaa_family(desc.format) || msaa32)))
                    && device._state._device.supportsTextureSampleCount(*count)
            })
            .map(|count| count as u32)
            .collect::<Vec<_>>();
        if !sample_counts.contains(&desc.samples) {
            return Support::Unsupported(
                "Metal texture sample count is unsupported by this device",
            );
        }
        Support::Supported(TextureSupport {
            max_extent: match desc.dimension {
                TextureDimension::D1 => [max_2d, 1, 1],
                TextureDimension::D2 => [max_2d, max_2d, 1],
                TextureDimension::D3 => [2048, 2048, 2048],
            },
            max_array_layers: MAX_ARRAY_LAYERS,
            max_mips: max_mip_levels(desc.extent),
            sample_counts,
            compatible_view_formats: compatible_formats(desc.format),
            filterable: is_filterable(desc.format)
                || (is_32bit_float(desc.format) && float32_filtering),
            blendable: is_blendable(desc.format),
            storage: desc.samples == 1 && storage,
        })
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        let _ = device;
        Support::Unsupported("Metal requires an Apple target")
    }
}

#[cfg(target_vendor = "apple")]
pub(super) fn max_texture_dimension_2d(
    device: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>,
) -> u32 {
    use objc2_metal::{MTLDevice as _, MTLGPUFamily};
    if device.supportsFamily(MTLGPUFamily::Apple10) {
        32768
    } else if device.supportsFamily(MTLGPUFamily::Apple3)
        || device.supportsFamily(MTLGPUFamily::Mac2)
    {
        16384
    } else {
        8192
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
        use objc2_metal::{MTLDevice as _, MTLResource as _};

        let descriptor = native_descriptor(device, desc, sparse)?;
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
            heap: None,
        })
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        let _ = (device, desc, sparse);
        Err(api::Error::Unsupported("Metal requires an Apple target"))
    }
}

#[cfg(target_vendor = "apple")]
pub(super) fn native_descriptor(
    device: &MetalDevice,
    desc: &TextureDesc,
    sparse: bool,
) -> api::Result<objc2::rc::Retained<objc2_metal::MTLTextureDescriptor>> {
    use objc2_metal::{MTLStorageMode, MTLTextureDescriptor, MTLTextureType, MTLTextureUsage};
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
    descriptor.setTextureType(match (desc.dimension, multisampled, arrayed) {
        (TextureDimension::D1, false, false) => MTLTextureType::Type1D,
        (TextureDimension::D1, false, true) => MTLTextureType::Type1DArray,
        (TextureDimension::D2, false, false) => MTLTextureType::Type2D,
        (TextureDimension::D2, false, true) => MTLTextureType::Type2DArray,
        (TextureDimension::D2, true, false) => MTLTextureType::Type2DMultisample,
        (TextureDimension::D2, true, true) => MTLTextureType::Type2DMultisampleArray,
        (TextureDimension::D3, false, false) => MTLTextureType::Type3D,
        _ => return Err(api::Error::InvalidRange),
    });
    // SAFETY: valid_desc bounds nonzero dimensions within MAX_2D_EXTENT.
    unsafe { descriptor.setWidth(desc.extent[0] as usize) };
    // SAFETY: valid_desc bounds the nonzero 2D height within MAX_2D_EXTENT.
    unsafe { descriptor.setHeight(desc.extent[1] as usize) };
    // SAFETY: descriptor validation bounds the depth for each dimension.
    unsafe { descriptor.setDepth(desc.extent[2] as usize) };
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
    if desc.usage.contains(TextureUsage::RENDER_ATTACHMENT)
        || (desc.samples > 1
            && desc.usage.contains(TextureUsage::COPY_DST)
            && matches!(
                desc.format,
                TextureFormat::Depth24PlusStencil8 | TextureFormat::Depth32FloatStencil8
            ))
    {
        // A per-aspect multisample copy can require a render pass to preserve
        // the other aspect because Metal blits copy the whole pixel format.
        usage |= MTLTextureUsage::RenderTarget;
    }
    if desc
        .view_formats
        .iter()
        .any(|format| *format != desc.format)
    {
        usage |= MTLTextureUsage::PixelFormatView;
    }
    if desc.usage.contains(TextureUsage::TEXTURE_BINDING)
        && matches!(
            desc.format,
            TextureFormat::Depth24PlusStencil8 | TextureFormat::Depth32FloatStencil8
        )
    {
        usage |= MTLTextureUsage::PixelFormatView;
    }
    descriptor.setUsage(usage);
    Ok(descriptor)
}

pub(super) fn create_view(
    device: &MetalDevice,
    texture: &MetalTexture,
    desc: &TextureViewDesc,
) -> api::Result<MetalTextureView> {
    #[cfg(target_vendor = "apple")]
    {
        use objc2_foundation::NSRange;
        use objc2_metal::{MTLTexture as _, MTLTextureType};

        let aspects = aspect_for_format(texture.desc.format);
        if (desc.format != texture.desc.format && !texture.desc.view_formats.contains(&desc.format))
            || desc.aspects.0 == 0
            || desc.aspects.0 & !aspects != 0
            || desc.mips.start >= desc.mips.end
            || desc.layers.start >= desc.layers.end
            || desc.mips.end > texture.desc.mip_levels
            || desc.layers.end > texture.desc.array_layers
        {
            return Err(api::Error::InvalidRange);
        }
        let view_type = match desc.dimension {
            TextureViewDimension::D1
                if texture.desc.dimension == TextureDimension::D1 && desc.layers.len() == 1 =>
            {
                MTLTextureType::Type1D
            }
            TextureViewDimension::D3
                if texture.desc.dimension == TextureDimension::D3 && desc.layers.len() == 1 =>
            {
                MTLTextureType::Type3D
            }
            TextureViewDimension::D2
                if texture.desc.dimension == TextureDimension::D2
                    && desc.layers.len() == 1
                    && texture.desc.samples > 1 =>
            {
                if desc.mips.len() != 1 || desc.mips.start != 0 {
                    return Err(api::Error::InvalidRange);
                }
                MTLTextureType::Type2DMultisample
            }
            TextureViewDimension::D2
                if texture.desc.dimension == TextureDimension::D2 && desc.layers.len() == 1 =>
            {
                MTLTextureType::Type2D
            }
            TextureViewDimension::D2Array
                if texture.desc.dimension == TextureDimension::D2 && texture.desc.samples > 1 =>
            {
                MTLTextureType::Type2DMultisampleArray
            }
            TextureViewDimension::D2Array if texture.desc.dimension == TextureDimension::D2 => {
                MTLTextureType::Type2DArray
            }
            TextureViewDimension::Cube
                if texture.desc.dimension == TextureDimension::D2
                    && texture.desc.extent[0] == texture.desc.extent[1]
                    && desc.layers.start.is_multiple_of(6)
                    && desc.layers.len() == 6 =>
            {
                MTLTextureType::TypeCube
            }
            TextureViewDimension::CubeArray
                if texture.desc.dimension == TextureDimension::D2
                    && texture.desc.extent[0] == texture.desc.extent[1]
                    && desc.layers.start.is_multiple_of(6)
                    && desc.layers.len().is_multiple_of(6) =>
            {
                MTLTextureType::TypeCubeArray
            }
            _ => return Err(api::Error::Unsupported("Metal texture view dimension")),
        };
        if texture.desc.samples > 1
            && view_type != MTLTextureType::Type2DMultisample
            && view_type != MTLTextureType::Type2DMultisampleArray
        {
            return Err(api::Error::Unsupported("Metal multisample view dimension"));
        }
        // SAFETY: both nonempty ranges were checked against the retained
        // descriptor above, so Metal cannot index outside this texture.
        let raw = unsafe {
            texture
                .raw
                .newTextureViewWithPixelFormat_textureType_levels_slices(
                    metal_format(desc.format)
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
        let sampled_raw = if aspects
            == (api::TextureAspects::DEPTH.0 | api::TextureAspects::STENCIL.0)
            && desc.aspects == api::TextureAspects::STENCIL
            && texture.desc.usage.contains(TextureUsage::TEXTURE_BINDING)
        {
            // SAFETY: PixelFormatView is enabled and the mip/slice ranges were checked.
            unsafe {
                texture
                    .raw
                    .newTextureViewWithPixelFormat_textureType_levels_slices(
                        objc2_metal::MTLPixelFormat::X32_Stencil8,
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
            .ok_or(api::Error::InvalidState)?
        } else {
            raw.clone()
        };
        Ok(MetalTextureView {
            raw,
            sampled_raw,
            format: desc.format,
            dimension: desc.dimension,
            usage: texture.desc.usage,
            samples: texture.desc.samples,
            aspects: desc.aspects,
            filterable: match texture_support(device, &texture.desc) {
                Support::Supported(support) => support.filterable,
                Support::Unsupported(_) => false,
            },
        })
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        let _ = (device, texture, desc);
        Err(api::Error::Unsupported("Metal requires an Apple target"))
    }
}

#[cfg(test)]
fn valid_desc(desc: &TextureDesc) -> bool {
    texture_desc_rejection(desc).is_none()
}

fn texture_desc_rejection(desc: &TextureDesc) -> Option<&'static str> {
    if desc.extent.contains(&0) || desc.array_layers == 0 || desc.mip_levels == 0 {
        return Some("Metal texture dimensions, array layers and mip levels must be nonzero");
    }
    if desc.extent[0] > MAX_2D_EXTENT
        || desc.extent[1] > MAX_2D_EXTENT
        || desc.array_layers > MAX_ARRAY_LAYERS
    {
        return Some("Metal texture exceeds the maximum family dimensions or array layers");
    }
    if match desc.dimension {
        TextureDimension::D1 => desc.extent[1] != 1 || desc.extent[2] != 1,
        TextureDimension::D2 => desc.extent[2] != 1,
        TextureDimension::D3 => desc.array_layers != 1 || desc.extent.iter().any(|v| *v > 2048),
    } {
        return Some("Metal texture dimension or volume extent is invalid");
    }
    if desc.mip_levels > max_mip_levels(desc.extent) {
        return Some("Metal texture mip count exceeds its extent");
    }
    if !matches!(desc.samples, 1 | 2 | 4 | 8)
        || (desc.samples > 1
            && (desc.mip_levels != 1
                || desc.dimension != TextureDimension::D2
                || desc.usage.contains(TextureUsage::STORAGE_BINDING)))
    {
        return Some("Metal multisample texture descriptor is invalid");
    }
    if desc.usage.0 == 0 || desc.usage.0 & !SUPPORTED_USAGE.0 != 0 {
        return Some("Metal texture usage is invalid");
    }
    if is_compressed(desc.format)
        && (desc.samples != 1
            || desc.dimension == TextureDimension::D1
            || desc.usage.contains(TextureUsage::STORAGE_BINDING)
            || desc.usage.contains(TextureUsage::RENDER_ATTACHMENT))
    {
        return Some("Metal compressed textures do not support this dimension or usage");
    }
    if desc.usage.contains(TextureUsage::STORAGE_BINDING)
        && !storage_format(desc.format)
        && !matches!(
            desc.format,
            TextureFormat::Rgba8UnormSrgb
                | TextureFormat::Bgra8UnormSrgb
                | TextureFormat::Rgb10a2Unorm
                | TextureFormat::Rgb10a2Uint
                | TextureFormat::Rg11b10Ufloat
                | TextureFormat::Rgb9e5Ufloat
        )
    {
        return Some("Metal pixel format does not support shader writes");
    }
    if desc
        .view_formats
        .iter()
        .any(|format| !compatible_formats(desc.format).contains(format))
    {
        return Some("Metal texture view format is not pixel-format compatible");
    }
    None
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
        invalid.view_formats.clear();
        assert!(valid_desc(&invalid));
        let mut multisampled = desc();
        multisampled.samples = 4;
        multisampled.mip_levels = 1;
        multisampled.usage = TextureUsage::RENDER_ATTACHMENT;
        assert!(valid_desc(&multisampled));
        multisampled.mip_levels = 2;
        assert!(!valid_desc(&multisampled));
        multisampled.mip_levels = 1;
        multisampled.usage = TextureUsage(multisampled.usage.0 | TextureUsage::TEXTURE_BINDING.0);
        assert!(valid_desc(&multisampled));
        let mut depth_stencil = desc();
        depth_stencil.format = TextureFormat::Depth32FloatStencil8;
        depth_stencil.mip_levels = 1;
        depth_stencil.usage = TextureUsage::RENDER_ATTACHMENT;
        depth_stencil.view_formats = vec![TextureFormat::Depth32FloatStencil8];
        assert!(valid_desc(&depth_stencil));
        depth_stencil.usage =
            TextureUsage(TextureUsage::RENDER_ATTACHMENT.0 | TextureUsage::TEXTURE_BINDING.0);
        assert!(valid_desc(&depth_stencil));
        let mut invalid = desc();
        invalid.usage = TextureUsage(1 << 63);
        assert!(!valid_desc(&invalid));
    }

    #[test]
    fn texture_dimensions_and_compatible_views_are_checked() {
        let mut d1 = desc();
        d1.dimension = TextureDimension::D1;
        d1.extent = [512, 1, 1];
        assert!(valid_desc(&d1));
        d1.extent[1] = 2;
        assert!(!valid_desc(&d1));
        let mut d3 = desc();
        d3.dimension = TextureDimension::D3;
        d3.extent = [64, 32, 16];
        d3.mip_levels = 7;
        assert!(valid_desc(&d3));
        d3.array_layers = 2;
        assert!(!valid_desc(&d3));
        let mut view = desc();
        view.view_formats.push(TextureFormat::Rgba8UnormSrgb);
        assert!(valid_desc(&view));
        view.view_formats.push(TextureFormat::Bgra8Unorm);
        assert!(!valid_desc(&view));
    }
}
