//! Metal pixel format mappings and family-independent capability classes.
#[cfg(target_vendor = "apple")]
use fluxel_rhi_core::api::TextureAspects;
use fluxel_rhi_core::api::TextureFormat;

pub(super) fn compatible_formats(format: TextureFormat) -> Vec<TextureFormat> {
    use TextureFormat::*;
    let partner = match format {
        Rgba8Unorm => Some(Rgba8UnormSrgb),
        Rgba8UnormSrgb => Some(Rgba8Unorm),
        Bgra8Unorm => Some(Bgra8UnormSrgb),
        Bgra8UnormSrgb => Some(Bgra8Unorm),
        Bc1RgbaUnorm => Some(Bc1RgbaUnormSrgb),
        Bc1RgbaUnormSrgb => Some(Bc1RgbaUnorm),
        Bc2RgbaUnorm => Some(Bc2RgbaUnormSrgb),
        Bc2RgbaUnormSrgb => Some(Bc2RgbaUnorm),
        Bc3RgbaUnorm => Some(Bc3RgbaUnormSrgb),
        Bc3RgbaUnormSrgb => Some(Bc3RgbaUnorm),
        Bc7RgbaUnorm => Some(Bc7RgbaUnormSrgb),
        Bc7RgbaUnormSrgb => Some(Bc7RgbaUnorm),
        Etc2Rgb8Unorm => Some(Etc2Rgb8UnormSrgb),
        Etc2Rgb8UnormSrgb => Some(Etc2Rgb8Unorm),
        Etc2Rgba8Unorm => Some(Etc2Rgba8UnormSrgb),
        Etc2Rgba8UnormSrgb => Some(Etc2Rgba8Unorm),
        Etc2Rgb8A1Unorm => Some(Etc2Rgb8A1UnormSrgb),
        Etc2Rgb8A1UnormSrgb => Some(Etc2Rgb8A1Unorm),
        Astc {
            block,
            srgb,
            hdr: false,
        } => Some(Astc {
            block,
            srgb: !srgb,
            hdr: false,
        }),
        _ => None,
    };
    let mut result = vec![format];
    if let Some(partner) = partner {
        result.push(partner);
    }
    result
}

pub(super) fn max_mip_levels(extent: [u32; 3]) -> u32 {
    32 - extent[0].max(extent[1]).max(extent[2]).leading_zeros()
}

#[cfg(target_vendor = "apple")]
pub(super) fn aspect_for_format(format: TextureFormat) -> u8 {
    match format {
        TextureFormat::Depth16Unorm | TextureFormat::Depth24Plus | TextureFormat::Depth32Float => {
            TextureAspects::DEPTH.0
        }
        TextureFormat::Depth24PlusStencil8 | TextureFormat::Depth32FloatStencil8 => {
            TextureAspects::DEPTH.0 | TextureAspects::STENCIL.0
        }
        TextureFormat::Stencil8 => TextureAspects::STENCIL.0,
        _ => TextureAspects::COLOR.0,
    }
}

#[cfg(target_vendor = "apple")]
pub(crate) fn is_filterable(format: TextureFormat) -> bool {
    if is_compressed(format) {
        return true;
    }
    matches!(
        format,
        TextureFormat::Depth16Unorm
            | TextureFormat::R16Unorm
            | TextureFormat::R16Snorm
            | TextureFormat::Rg16Unorm
            | TextureFormat::Rg16Snorm
            | TextureFormat::Rgba16Unorm
            | TextureFormat::Rgba16Snorm
            | TextureFormat::Rgb10a2Unorm
            | TextureFormat::Rg11b10Ufloat
            | TextureFormat::Rgb9e5Ufloat
            | TextureFormat::R8Snorm
            | TextureFormat::Rg8Snorm
            | TextureFormat::Rgba8Snorm
            | TextureFormat::R16Float
            | TextureFormat::Rg16Float
            | TextureFormat::R8Unorm
            | TextureFormat::Rg8Unorm
            | TextureFormat::Rgba8Unorm
            | TextureFormat::Rgba8UnormSrgb
            | TextureFormat::Bgra8Unorm
            | TextureFormat::Bgra8UnormSrgb
            | TextureFormat::Rgba16Float
    )
}

#[cfg(target_vendor = "apple")]
pub(in crate::backend) fn is_blendable(format: TextureFormat) -> bool {
    matches!(
        format,
        TextureFormat::R16Unorm
            | TextureFormat::R16Snorm
            | TextureFormat::Rg16Unorm
            | TextureFormat::Rg16Snorm
            | TextureFormat::Rgba16Unorm
            | TextureFormat::Rgba16Snorm
            | TextureFormat::Rgb10a2Unorm
            | TextureFormat::Rg11b10Ufloat
            | TextureFormat::Rgb9e5Ufloat
            | TextureFormat::R32Float
            | TextureFormat::Rg32Float
            | TextureFormat::Rgba32Float
            | TextureFormat::R8Snorm
            | TextureFormat::Rg8Snorm
            | TextureFormat::Rgba8Snorm
            | TextureFormat::R16Float
            | TextureFormat::Rg16Float
            | TextureFormat::R8Unorm
            | TextureFormat::Rg8Unorm
            | TextureFormat::Rgba8Unorm
            | TextureFormat::Rgba8UnormSrgb
            | TextureFormat::Bgra8Unorm
            | TextureFormat::Bgra8UnormSrgb
            | TextureFormat::Rgba16Float
    )
}

pub(in crate::backend) fn color_attachment_format(format: TextureFormat) -> bool {
    format_supported(format)
        && !is_compressed(format)
        && !matches!(
            format,
            TextureFormat::Depth16Unorm
                | TextureFormat::Depth24Plus
                | TextureFormat::Depth24PlusStencil8
                | TextureFormat::Depth32Float
                | TextureFormat::Depth32FloatStencil8
                | TextureFormat::Stencil8
        )
}

pub(super) fn format_supported(format: TextureFormat) -> bool {
    if is_compressed(format) {
        return true;
    }
    matches!(
        format,
        TextureFormat::R16Unorm
            | TextureFormat::R16Snorm
            | TextureFormat::Rg16Unorm
            | TextureFormat::Rg16Snorm
            | TextureFormat::Rgba16Unorm
            | TextureFormat::Rgba16Snorm
            | TextureFormat::R32Uint
            | TextureFormat::R32Sint
            | TextureFormat::R32Float
            | TextureFormat::Rg32Uint
            | TextureFormat::Rg32Sint
            | TextureFormat::Rg32Float
            | TextureFormat::Rgba32Uint
            | TextureFormat::Rgba32Sint
            | TextureFormat::Rgba32Float
            | TextureFormat::Rgb10a2Unorm
            | TextureFormat::Rgb10a2Uint
            | TextureFormat::Rg11b10Ufloat
            | TextureFormat::Rgb9e5Ufloat
            | TextureFormat::R8Snorm
            | TextureFormat::R8Uint
            | TextureFormat::R8Sint
            | TextureFormat::Rg8Snorm
            | TextureFormat::Rg8Uint
            | TextureFormat::Rg8Sint
            | TextureFormat::Rgba8Snorm
            | TextureFormat::Rgba8Uint
            | TextureFormat::Rgba8Sint
            | TextureFormat::R16Float
            | TextureFormat::Rg16Float
            | TextureFormat::R16Uint
            | TextureFormat::R16Sint
            | TextureFormat::Rg16Uint
            | TextureFormat::Rg16Sint
            | TextureFormat::Rgba16Uint
            | TextureFormat::Rgba16Sint
            | TextureFormat::R8Unorm
            | TextureFormat::Rg8Unorm
            | TextureFormat::Rgba8Unorm
            | TextureFormat::Rgba8UnormSrgb
            | TextureFormat::Bgra8Unorm
            | TextureFormat::Bgra8UnormSrgb
            | TextureFormat::Rgba16Float
            | TextureFormat::Depth16Unorm
            | TextureFormat::Depth24Plus
            | TextureFormat::Depth24PlusStencil8
            | TextureFormat::Stencil8
            | TextureFormat::Depth32Float
            | TextureFormat::Depth32FloatStencil8
    )
}

pub(super) fn storage_format(format: TextureFormat) -> bool {
    use TextureFormat::*;
    matches!(
        format,
        R16Unorm
            | R16Snorm
            | Rg16Unorm
            | Rg16Snorm
            | Rgba16Unorm
            | Rgba16Snorm
            | R32Uint
            | R32Sint
            | R32Float
            | Rg32Uint
            | Rg32Sint
            | Rg32Float
            | Rgba32Uint
            | Rgba32Sint
            | Rgba32Float
            | Bgra8Unorm
            | R8Unorm
            | R8Snorm
            | R8Uint
            | R8Sint
            | Rg8Unorm
            | Rg8Snorm
            | Rg8Uint
            | Rg8Sint
            | Rgba8Unorm
            | Rgba8Snorm
            | Rgba8Uint
            | Rgba8Sint
            | R16Float
            | Rg16Float
            | Rgba16Float
            | R16Uint
            | R16Sint
            | Rg16Uint
            | Rg16Sint
            | Rgba16Uint
            | Rgba16Sint
    )
}

#[cfg(target_vendor = "apple")]
pub(super) fn is_32bit_float(format: TextureFormat) -> bool {
    matches!(
        format,
        TextureFormat::R32Float
            | TextureFormat::Rg32Float
            | TextureFormat::Rgba32Float
            | TextureFormat::Depth24Plus
            | TextureFormat::Depth24PlusStencil8
            | TextureFormat::Depth32Float
            | TextureFormat::Depth32FloatStencil8
    )
}

#[cfg(target_vendor = "apple")]
pub(super) fn needs_32bit_msaa_family(format: TextureFormat) -> bool {
    matches!(
        format,
        TextureFormat::R32Uint
            | TextureFormat::R32Sint
            | TextureFormat::Rg32Uint
            | TextureFormat::Rg32Sint
            | TextureFormat::Rg32Float
            | TextureFormat::Rgba32Uint
            | TextureFormat::Rgba32Sint
            | TextureFormat::Rgba32Float
    )
}

pub(super) fn is_compressed(format: TextureFormat) -> bool {
    is_bc(format) || is_etc_or_astc(format)
}

pub(super) fn is_etc_or_astc(format: TextureFormat) -> bool {
    use TextureFormat::*;
    match format {
        Etc2Rgb8Unorm | Etc2Rgb8UnormSrgb | Etc2Rgba8Unorm | Etc2Rgba8UnormSrgb
        | Etc2Rgb8A1Unorm | Etc2Rgb8A1UnormSrgb | EacR11Unorm | EacR11Snorm | EacRg11Unorm
        | EacRg11Snorm => true,
        Astc { block, srgb, hdr } => astc_offset(block).is_some() && !(srgb && hdr),
        _ => false,
    }
}

fn astc_offset(block: [u8; 2]) -> Option<usize> {
    Some(match block {
        [4, 4] => 0,
        [5, 4] => 1,
        [5, 5] => 2,
        [6, 5] => 3,
        [6, 6] => 4,
        [8, 5] => 6,
        [8, 6] => 7,
        [8, 8] => 8,
        [10, 5] => 9,
        [10, 6] => 10,
        [10, 8] => 11,
        [10, 10] => 12,
        [12, 10] => 13,
        [12, 12] => 14,
        _ => return None,
    })
}

pub(super) fn is_bc(format: TextureFormat) -> bool {
    matches!(
        format,
        TextureFormat::Bc1RgbaUnorm
            | TextureFormat::Bc1RgbaUnormSrgb
            | TextureFormat::Bc2RgbaUnorm
            | TextureFormat::Bc2RgbaUnormSrgb
            | TextureFormat::Bc3RgbaUnorm
            | TextureFormat::Bc3RgbaUnormSrgb
            | TextureFormat::Bc4RUnorm
            | TextureFormat::Bc4RSnorm
            | TextureFormat::Bc5RgUnorm
            | TextureFormat::Bc5RgSnorm
            | TextureFormat::Bc6hRgbUfloat
            | TextureFormat::Bc6hRgbFloat
            | TextureFormat::Bc7RgbaUnorm
            | TextureFormat::Bc7RgbaUnormSrgb
    )
}

#[cfg(target_vendor = "apple")]
pub(in crate::backend) fn metal_format(
    format: TextureFormat,
) -> Option<objc2_metal::MTLPixelFormat> {
    use objc2_metal::MTLPixelFormat as Mtl;
    Some(match format {
        TextureFormat::R8Unorm => Mtl::R8Unorm,
        TextureFormat::R8Snorm => Mtl::R8Snorm,
        TextureFormat::R8Uint => Mtl::R8Uint,
        TextureFormat::R8Sint => Mtl::R8Sint,
        TextureFormat::Rg8Unorm => Mtl::RG8Unorm,
        TextureFormat::Rg8Snorm => Mtl::RG8Snorm,
        TextureFormat::Rg8Uint => Mtl::RG8Uint,
        TextureFormat::Rg8Sint => Mtl::RG8Sint,
        TextureFormat::Rgba8Snorm => Mtl::RGBA8Snorm,
        TextureFormat::Rgba8Uint => Mtl::RGBA8Uint,
        TextureFormat::Rgba8Sint => Mtl::RGBA8Sint,
        TextureFormat::R16Float => Mtl::R16Float,
        TextureFormat::Rg16Float => Mtl::RG16Float,
        TextureFormat::R16Uint => Mtl::R16Uint,
        TextureFormat::R16Sint => Mtl::R16Sint,
        TextureFormat::Rg16Uint => Mtl::RG16Uint,
        TextureFormat::Rg16Sint => Mtl::RG16Sint,
        TextureFormat::Rgba16Uint => Mtl::RGBA16Uint,
        TextureFormat::Rgba16Sint => Mtl::RGBA16Sint,
        TextureFormat::R16Unorm => Mtl::R16Unorm,
        TextureFormat::R16Snorm => Mtl::R16Snorm,
        TextureFormat::Rg16Unorm => Mtl::RG16Unorm,
        TextureFormat::Rg16Snorm => Mtl::RG16Snorm,
        TextureFormat::Rgba16Unorm => Mtl::RGBA16Unorm,
        TextureFormat::Rgba16Snorm => Mtl::RGBA16Snorm,
        TextureFormat::R32Uint => Mtl::R32Uint,
        TextureFormat::R32Sint => Mtl::R32Sint,
        TextureFormat::R32Float => Mtl::R32Float,
        TextureFormat::Rg32Uint => Mtl::RG32Uint,
        TextureFormat::Rg32Sint => Mtl::RG32Sint,
        TextureFormat::Rg32Float => Mtl::RG32Float,
        TextureFormat::Rgba32Uint => Mtl::RGBA32Uint,
        TextureFormat::Rgba32Sint => Mtl::RGBA32Sint,
        TextureFormat::Rgba32Float => Mtl::RGBA32Float,
        TextureFormat::Rgb10a2Unorm => Mtl::RGB10A2Unorm,
        TextureFormat::Rgb10a2Uint => Mtl::RGB10A2Uint,
        TextureFormat::Rg11b10Ufloat => Mtl::RG11B10Float,
        TextureFormat::Rgb9e5Ufloat => Mtl::RGB9E5Float,
        TextureFormat::Etc2Rgb8Unorm => Mtl::ETC2_RGB8,
        TextureFormat::Etc2Rgb8UnormSrgb => Mtl::ETC2_RGB8_sRGB,
        TextureFormat::Etc2Rgba8Unorm => Mtl::EAC_RGBA8,
        TextureFormat::Etc2Rgba8UnormSrgb => Mtl::EAC_RGBA8_sRGB,
        TextureFormat::Etc2Rgb8A1Unorm => Mtl::ETC2_RGB8A1,
        TextureFormat::Etc2Rgb8A1UnormSrgb => Mtl::ETC2_RGB8A1_sRGB,
        TextureFormat::EacR11Unorm => Mtl::EAC_R11Unorm,
        TextureFormat::EacR11Snorm => Mtl::EAC_R11Snorm,
        TextureFormat::EacRg11Unorm => Mtl::EAC_RG11Unorm,
        TextureFormat::EacRg11Snorm => Mtl::EAC_RG11Snorm,
        TextureFormat::Astc { block, srgb, hdr } if !(srgb && hdr) => {
            // The Metal ASTC enum ranges have identical block ordering.
            let base = if hdr {
                Mtl::ASTC_4x4_HDR.0
            } else if srgb {
                Mtl::ASTC_4x4_sRGB.0
            } else {
                Mtl::ASTC_4x4_LDR.0
            };
            Mtl(base + astc_offset(block)?)
        }
        TextureFormat::Bc1RgbaUnorm => Mtl::BC1_RGBA,
        TextureFormat::Bc1RgbaUnormSrgb => Mtl::BC1_RGBA_sRGB,
        TextureFormat::Bc2RgbaUnorm => Mtl::BC2_RGBA,
        TextureFormat::Bc2RgbaUnormSrgb => Mtl::BC2_RGBA_sRGB,
        TextureFormat::Bc3RgbaUnorm => Mtl::BC3_RGBA,
        TextureFormat::Bc3RgbaUnormSrgb => Mtl::BC3_RGBA_sRGB,
        TextureFormat::Bc4RUnorm => Mtl::BC4_RUnorm,
        TextureFormat::Bc4RSnorm => Mtl::BC4_RSnorm,
        TextureFormat::Bc5RgUnorm => Mtl::BC5_RGUnorm,
        TextureFormat::Bc5RgSnorm => Mtl::BC5_RGSnorm,
        TextureFormat::Bc6hRgbUfloat => Mtl::BC6H_RGBUfloat,
        TextureFormat::Bc6hRgbFloat => Mtl::BC6H_RGBFloat,
        TextureFormat::Bc7RgbaUnorm => Mtl::BC7_RGBAUnorm,
        TextureFormat::Bc7RgbaUnormSrgb => Mtl::BC7_RGBAUnorm_sRGB,
        TextureFormat::Rgba8Unorm => Mtl::RGBA8Unorm,
        TextureFormat::Rgba8UnormSrgb => Mtl::RGBA8Unorm_sRGB,
        TextureFormat::Bgra8Unorm => Mtl::BGRA8Unorm,
        TextureFormat::Bgra8UnormSrgb => Mtl::BGRA8Unorm_sRGB,
        TextureFormat::Rgba16Float => Mtl::RGBA16Float,
        TextureFormat::Depth32Float => Mtl::Depth32Float,
        TextureFormat::Depth16Unorm => Mtl::Depth16Unorm,
        TextureFormat::Depth24Plus => Mtl::Depth32Float,
        TextureFormat::Depth24PlusStencil8 => Mtl::Depth32Float_Stencil8,
        TextureFormat::Stencil8 => Mtl::Stencil8,
        TextureFormat::Depth32FloatStencil8 => Mtl::Depth32Float_Stencil8,
        _ => return None,
    })
}
