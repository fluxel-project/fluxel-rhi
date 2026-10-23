//! Private GL tokens for native resource allocation and attachment binding.
//!
//! This module contains closed enum-to-token lowering only. Discovery remains
//! the authority for whether a format is supported on the current context;
//! these mappings merely spell an already admitted operation for glow.

#[cfg(any(feature = "native-gl-wgl", feature = "native-gles-egl"))]
pub(crate) const fn native_wrap(mode: crate::backend::gl::api::GlAddressMode) -> i32 {
    match mode {
        crate::backend::gl::api::GlAddressMode::ClampToEdge => 0x812F,
        crate::backend::gl::api::GlAddressMode::Repeat => 0x2901,
        crate::backend::gl::api::GlAddressMode::MirroredRepeat => 0x8370,
    }
}

#[cfg(any(feature = "native-gl-wgl", feature = "native-gles-egl"))]
pub(crate) const fn native_mag(mode: crate::backend::gl::api::GlFilterMode) -> i32 {
    match mode {
        crate::backend::gl::api::GlFilterMode::Nearest => 0x2600,
        crate::backend::gl::api::GlFilterMode::Linear => 0x2601,
    }
}

#[cfg(any(feature = "native-gl-wgl", feature = "native-gles-egl"))]
pub(crate) const fn native_min(
    min: crate::backend::gl::api::GlFilterMode,
    mip: crate::backend::gl::api::GlMipmapFilterMode,
) -> i32 {
    match (min, mip) {
        (
            crate::backend::gl::api::GlFilterMode::Nearest,
            crate::backend::gl::api::GlMipmapFilterMode::Nearest,
        ) => 0x2700,
        (
            crate::backend::gl::api::GlFilterMode::Linear,
            crate::backend::gl::api::GlMipmapFilterMode::Nearest,
        ) => 0x2701,
        (
            crate::backend::gl::api::GlFilterMode::Nearest,
            crate::backend::gl::api::GlMipmapFilterMode::Linear,
        ) => 0x2702,
        (
            crate::backend::gl::api::GlFilterMode::Linear,
            crate::backend::gl::api::GlMipmapFilterMode::Linear,
        ) => 0x2703,
    }
}

#[cfg(any(feature = "native-gl-wgl", feature = "native-gles-egl"))]
pub(crate) const fn native_compare(compare: crate::backend::gl::api::GlCompareFunction) -> i32 {
    match compare {
        crate::backend::gl::api::GlCompareFunction::Never => 0x0200,
        crate::backend::gl::api::GlCompareFunction::Less => 0x0201,
        crate::backend::gl::api::GlCompareFunction::Equal => 0x0202,
        crate::backend::gl::api::GlCompareFunction::LessEqual => 0x0203,
        crate::backend::gl::api::GlCompareFunction::Greater => 0x0204,
        crate::backend::gl::api::GlCompareFunction::NotEqual => 0x0205,
        crate::backend::gl::api::GlCompareFunction::GreaterEqual => 0x0206,
        crate::backend::gl::api::GlCompareFunction::Always => 0x0207,
    }
}

/// The native internal/storage format constant of one discovered `GlFormat`.
///
/// Only formats with a settled mapping are listed; every other format fails
/// closed at its domain's evidence gate even if it maps here.
#[cfg(any(feature = "native-gl-wgl", feature = "native-gles-egl"))]
pub(crate) const fn native_texture_format(
    format: crate::backend::gl::api::GlFormat,
) -> Option<u32> {
    match format {
        crate::backend::gl::api::GlFormat::R8Unorm => Some(glow::R8),
        crate::backend::gl::api::GlFormat::R8Snorm => Some(glow::R8_SNORM),
        crate::backend::gl::api::GlFormat::R8Uint => Some(glow::R8UI),
        crate::backend::gl::api::GlFormat::R8Sint => Some(glow::R8I),
        crate::backend::gl::api::GlFormat::Rg8Unorm => Some(glow::RG8),
        crate::backend::gl::api::GlFormat::Rg8Snorm => Some(glow::RG8_SNORM),
        crate::backend::gl::api::GlFormat::Rg8Uint => Some(glow::RG8UI),
        crate::backend::gl::api::GlFormat::Rg8Sint => Some(glow::RG8I),
        crate::backend::gl::api::GlFormat::Rgba8Snorm => Some(glow::RGBA8_SNORM),
        crate::backend::gl::api::GlFormat::R16Uint => Some(glow::R16UI),
        crate::backend::gl::api::GlFormat::R16Sint => Some(glow::R16I),
        crate::backend::gl::api::GlFormat::R16Float => Some(glow::R16F),
        crate::backend::gl::api::GlFormat::Rg16Uint => Some(glow::RG16UI),
        crate::backend::gl::api::GlFormat::Rg16Sint => Some(glow::RG16I),
        crate::backend::gl::api::GlFormat::Rg16Float => Some(glow::RG16F),
        crate::backend::gl::api::GlFormat::Rgba16Uint => Some(glow::RGBA16UI),
        crate::backend::gl::api::GlFormat::Rgba16Sint => Some(glow::RGBA16I),
        crate::backend::gl::api::GlFormat::R32Uint => Some(glow::R32UI),
        crate::backend::gl::api::GlFormat::R32Sint => Some(glow::R32I),
        crate::backend::gl::api::GlFormat::R32Float => Some(glow::R32F),
        crate::backend::gl::api::GlFormat::Rg32Uint => Some(glow::RG32UI),
        crate::backend::gl::api::GlFormat::Rg32Sint => Some(glow::RG32I),
        crate::backend::gl::api::GlFormat::Rg32Float => Some(glow::RG32F),
        crate::backend::gl::api::GlFormat::Rgba32Uint => Some(glow::RGBA32UI),
        crate::backend::gl::api::GlFormat::Rgba32Sint => Some(glow::RGBA32I),
        crate::backend::gl::api::GlFormat::R16Unorm => Some(glow::R16),
        crate::backend::gl::api::GlFormat::Rg16Unorm => Some(glow::RG16),
        crate::backend::gl::api::GlFormat::Rgba16Unorm => Some(glow::RGBA16),
        crate::backend::gl::api::GlFormat::R16Snorm => Some(glow::R16_SNORM),
        crate::backend::gl::api::GlFormat::Rg16Snorm => Some(glow::RG16_SNORM),
        crate::backend::gl::api::GlFormat::Rgba16Snorm => Some(glow::RGBA16_SNORM),
        crate::backend::gl::api::GlFormat::Rgb9e5Ufloat => Some(glow::RGB9_E5),
        crate::backend::gl::api::GlFormat::Rgb10a2Uint => Some(glow::RGB10_A2UI),
        crate::backend::gl::api::GlFormat::Rgb10a2Unorm => Some(glow::RGB10_A2),
        crate::backend::gl::api::GlFormat::Rg11b10Ufloat => Some(glow::R11F_G11F_B10F),
        crate::backend::gl::api::GlFormat::Rgba8Unorm => Some(glow::RGBA8),
        crate::backend::gl::api::GlFormat::Rgba8Srgb => Some(glow::SRGB8_ALPHA8),
        crate::backend::gl::api::GlFormat::Rgba8Uint => Some(glow::RGBA8UI),
        crate::backend::gl::api::GlFormat::Rgba8Sint => Some(glow::RGBA8I),
        crate::backend::gl::api::GlFormat::Rgba16Float => Some(glow::RGBA16F),
        crate::backend::gl::api::GlFormat::Rgba32Float => Some(glow::RGBA32F),
        crate::backend::gl::api::GlFormat::Depth32Float => Some(glow::DEPTH_COMPONENT32F),
        crate::backend::gl::api::GlFormat::Depth16Unorm => Some(glow::DEPTH_COMPONENT16),
        crate::backend::gl::api::GlFormat::Depth24Unorm => Some(glow::DEPTH_COMPONENT24),
        crate::backend::gl::api::GlFormat::Depth24PlusStencil8 => Some(glow::DEPTH24_STENCIL8),
        crate::backend::gl::api::GlFormat::Depth32FloatStencil8 => Some(glow::DEPTH32F_STENCIL8),
        crate::backend::gl::api::GlFormat::Stencil8 => Some(glow::STENCIL_INDEX8),
        crate::backend::gl::api::GlFormat::Bc1RgbUnorm => Some(0x83F0),
        crate::backend::gl::api::GlFormat::Bc1RgbaUnorm => Some(0x83F1),
        crate::backend::gl::api::GlFormat::Bc2RgbaUnorm => Some(0x83F2),
        crate::backend::gl::api::GlFormat::Bc3RgbaUnorm => Some(0x83F3),
        crate::backend::gl::api::GlFormat::Bc1RgbSrgb => Some(0x8C4C),
        crate::backend::gl::api::GlFormat::Bc1RgbaSrgb => Some(0x8C4D),
        crate::backend::gl::api::GlFormat::Bc2RgbaSrgb => Some(0x8C4E),
        crate::backend::gl::api::GlFormat::Bc3RgbaSrgb => Some(0x8C4F),
        crate::backend::gl::api::GlFormat::Bc4RUnorm => Some(0x8DBB),
        crate::backend::gl::api::GlFormat::Bc4RSnorm => Some(0x8DBC),
        crate::backend::gl::api::GlFormat::Bc5RgUnorm => Some(0x8DBD),
        crate::backend::gl::api::GlFormat::Bc5RgSnorm => Some(0x8DBE),
        crate::backend::gl::api::GlFormat::Bc6hRgbUfloat => Some(0x8E8F),
        crate::backend::gl::api::GlFormat::Bc6hRgbSfloat => Some(0x8E8E),
        crate::backend::gl::api::GlFormat::Bc7RgbaUnorm => Some(0x8E8C),
        crate::backend::gl::api::GlFormat::Bc7RgbaSrgb => Some(0x8E8D),
        crate::backend::gl::api::GlFormat::Etc2Rgb8Unorm => Some(0x9274),
        crate::backend::gl::api::GlFormat::Etc2Rgb8Srgb => Some(0x9275),
        crate::backend::gl::api::GlFormat::Etc2Rgb8A1Unorm => Some(0x9276),
        crate::backend::gl::api::GlFormat::Etc2Rgb8A1Srgb => Some(0x9277),
        crate::backend::gl::api::GlFormat::Etc2Rgba8Unorm => Some(0x9278),
        crate::backend::gl::api::GlFormat::Etc2Rgba8Srgb => Some(0x9279),
        crate::backend::gl::api::GlFormat::EacR11Unorm => Some(0x9270),
        crate::backend::gl::api::GlFormat::EacR11Snorm => Some(0x9271),
        crate::backend::gl::api::GlFormat::EacRg11Unorm => Some(0x9272),
        crate::backend::gl::api::GlFormat::EacRg11Snorm => Some(0x9273),
        crate::backend::gl::api::GlFormat::Astc { block, color_space } => {
            use crate::backend::gl::api::{GlAstcBlock as B, GlCompressedColorSpace as C};
            let index = match block {
                B::B4x4 => 0,
                B::B5x4 => 1,
                B::B5x5 => 2,
                B::B6x5 => 3,
                B::B6x6 => 4,
                B::B8x5 => 5,
                B::B8x6 => 6,
                B::B8x8 => 7,
                B::B10x5 => 8,
                B::B10x6 => 9,
                B::B10x8 => 10,
                B::B10x10 => 11,
                B::B12x10 => 12,
                B::B12x12 => 13,
            };
            // KHR ASTC HDR selects the linear token plus an independently
            // discovered HDR decode profile; it must never be rewritten as
            // the sRGB token.
            Some(match color_space {
                C::Linear | C::Hdr => 0x93B0 + index,
                C::Srgb => 0x93D0 + index,
            })
        }
        _ => None,
    }
}

/// Returns whether the native allocator has an exact storage-token spelling.
///
/// The discovery table is still the admission authority for profile and usage
/// facts. This predicate lets the Core support query avoid advertising a
/// discovery row that a native allocation cannot spell.
#[cfg(any(feature = "native-gl-wgl", feature = "native-gles-egl"))]
pub(crate) const fn native_texture_format_supported(
    format: crate::backend::gl::api::GlFormat,
) -> bool {
    native_texture_format(format).is_some()
}

/// The framebuffer attachment point for a depth/stencil view format.
#[cfg(any(feature = "native-gl-wgl", feature = "native-gles-egl"))]
pub(crate) const fn depth_attachment_point(
    format: crate::backend::gl::api::GlFormat,
) -> Option<u32> {
    match format {
        crate::backend::gl::api::GlFormat::Depth16Unorm
        | crate::backend::gl::api::GlFormat::Depth24Unorm
        | crate::backend::gl::api::GlFormat::Depth32Float => Some(glow::DEPTH_ATTACHMENT),
        crate::backend::gl::api::GlFormat::Depth24PlusStencil8
        | crate::backend::gl::api::GlFormat::Depth32FloatStencil8 => {
            Some(glow::DEPTH_STENCIL_ATTACHMENT)
        }
        crate::backend::gl::api::GlFormat::Stencil8 => Some(glow::STENCIL_ATTACHMENT),
        _ => None,
    }
}

/// Whether a depth/stencil format carries a stencil plane.
#[cfg(any(feature = "native-gl-wgl", feature = "native-gles-egl"))]
pub(crate) const fn has_stencil_plane(format: crate::backend::gl::api::GlFormat) -> bool {
    matches!(
        format,
        crate::backend::gl::api::GlFormat::Depth24PlusStencil8
            | crate::backend::gl::api::GlFormat::Depth32FloatStencil8
            | crate::backend::gl::api::GlFormat::Stencil8
    )
}
