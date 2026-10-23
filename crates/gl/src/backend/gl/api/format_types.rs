//! Private GL format keys and compressed-format calculations.
//!
//! These values describe exact native representations. Context evidence and
//! capability publication stay in the sibling format-table module.

use super::{GlFamilyProfile, GlKnownExtension};

/// Formats with a current common-RHI meaning in the GL-family layer.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) enum GlFormat {
    R8Unorm,
    R8Snorm,
    R8Uint,
    R8Sint,
    Rg8Unorm,
    Rg8Snorm,
    Rg8Uint,
    Rg8Sint,
    /// Eight-bit normalized red, green, blue, and alpha channels.
    Rgba8Unorm,
    /// Eight-bit sRGB red, green, blue and normalized alpha channels.
    Rgba8Srgb,
    Rgba8Snorm,
    Rgba8Uint,
    Rgba8Sint,
    Bgra8Unorm,
    Bgra8Srgb,
    R16Uint,
    R16Sint,
    R16Float,
    R16Unorm,
    R16Snorm,
    Rg16Uint,
    Rg16Sint,
    Rg16Float,
    Rg16Unorm,
    Rg16Snorm,
    Rgba16Uint,
    Rgba16Sint,
    Rgba16Unorm,
    Rgba16Snorm,
    /// Sixteen-bit floating-point RGBA channels.
    Rgba16Float,
    /// Thirty-two-bit floating-point RGBA channels.
    Rgba32Float,
    Rgb9e5Ufloat,
    Rgb10a2Uint,
    Rgb10a2Unorm,
    Rg11b10Ufloat,
    R32Uint,
    R32Sint,
    R32Float,
    R64Uint,
    Rg32Uint,
    Rg32Sint,
    Rg32Float,
    Rgba32Uint,
    Rgba32Sint,
    /// Sixteen-bit normalized depth.
    Depth16Unorm,
    Depth24Unorm,
    /// Twenty-four-bit depth plus eight-bit stencil.
    Depth24PlusStencil8,
    /// Thirty-two-bit floating-point depth.
    Depth32Float,
    Depth32FloatStencil8,
    Stencil8,
    /// BC1/DXT1 RGB blocks.
    Bc1RgbUnorm,
    /// BC1/DXT1 RGBA blocks.
    Bc1RgbaUnorm,
    /// BC1/DXT1 RGBA blocks decoded as sRGB.
    Bc1RgbaSrgb,
    /// BC1/DXT1 RGB blocks decoded as sRGB.
    Bc1RgbSrgb,
    /// BC2/DXT3 RGBA blocks.
    Bc2RgbaUnorm,
    /// BC2/DXT3 RGBA blocks decoded as sRGB.
    Bc2RgbaSrgb,
    /// BC3/DXT5 RGBA blocks.
    Bc3RgbaUnorm,
    /// BC3/DXT5 RGBA blocks decoded as sRGB.
    Bc3RgbaSrgb,
    /// BC4 one-channel unsigned-normalized blocks.
    Bc4RUnorm,
    /// BC4 one-channel signed-normalized blocks.
    Bc4RSnorm,
    /// BC5 two-channel unsigned-normalized blocks.
    Bc5RgUnorm,
    /// BC5 two-channel signed-normalized blocks.
    Bc5RgSnorm,
    /// BC6H unsigned-float RGB blocks.
    Bc6hRgbUfloat,
    /// BC6H signed-float RGB blocks.
    Bc6hRgbSfloat,
    /// BC7 RGBA blocks.
    Bc7RgbaUnorm,
    /// BC7 RGBA blocks decoded as sRGB.
    Bc7RgbaSrgb,
    /// Exact ASTC block dimensions and color space.
    Astc {
        block: GlAstcBlock,
        color_space: GlCompressedColorSpace,
    },
    /// ETC2 RGB blocks.
    Etc2Rgb8Unorm,
    /// ETC2 RGB blocks decoded as sRGB.
    Etc2Rgb8Srgb,
    /// ETC2 RGBA blocks.
    Etc2Rgba8Unorm,
    /// ETC2 RGBA blocks decoded as sRGB.
    Etc2Rgba8Srgb,
    /// ETC2 RGB with one-bit alpha blocks.
    Etc2Rgb8A1Unorm,
    /// ETC2 RGB with one-bit alpha blocks decoded as sRGB.
    Etc2Rgb8A1Srgb,
    /// EAC one-channel unsigned-normalized blocks.
    EacR11Unorm,
    /// EAC two-channel unsigned-normalized blocks.
    EacRg11Unorm,
    /// EAC one-channel signed-normalized blocks.
    EacR11Snorm,
    /// EAC two-channel signed-normalized blocks.
    EacRg11Snorm,
}

impl GlFormat {
    /// Integer framebuffer colors cannot use GL's linear blit filter.
    pub(crate) const fn is_integer_color(self) -> bool {
        matches!(
            self,
            Self::R8Uint
                | Self::R8Sint
                | Self::Rg8Uint
                | Self::Rg8Sint
                | Self::Rgba8Uint
                | Self::Rgba8Sint
                | Self::R16Uint
                | Self::R16Sint
                | Self::Rg16Uint
                | Self::Rg16Sint
                | Self::Rgba16Uint
                | Self::Rgba16Sint
                | Self::Rgb10a2Uint
                | Self::R32Uint
                | Self::R32Sint
                | Self::R64Uint
                | Self::Rg32Uint
                | Self::Rg32Sint
                | Self::Rgba32Uint
                | Self::Rgba32Sint
        )
    }

    pub(crate) const fn is_float_color(self) -> bool {
        matches!(
            self,
            Self::R16Float
                | Self::Rg16Float
                | Self::Rgba16Float
                | Self::R32Float
                | Self::Rg32Float
                | Self::Rgba32Float
                | Self::Rgb9e5Ufloat
                | Self::Rg11b10Ufloat
        )
    }

    pub(crate) fn supports_blit_aspect(
        self,
        aspect: crate::backend::gl::api::GlTextureAspect,
    ) -> bool {
        use crate::backend::gl::api::GlTextureAspect;
        match aspect {
            GlTextureAspect::All => false,
            GlTextureAspect::DepthOnly => matches!(
                self,
                Self::Depth16Unorm
                    | Self::Depth24Unorm
                    | Self::Depth24PlusStencil8
                    | Self::Depth32Float
                    | Self::Depth32FloatStencil8
            ),
            GlTextureAspect::StencilOnly => matches!(
                self,
                Self::Stencil8 | Self::Depth24PlusStencil8 | Self::Depth32FloatStencil8
            ),
            GlTextureAspect::Color => {
                !matches!(
                    self,
                    Self::Depth16Unorm
                        | Self::Depth24Unorm
                        | Self::Depth24PlusStencil8
                        | Self::Depth32Float
                        | Self::Depth32FloatStencil8
                        | Self::Stencil8
                ) && self.compressed_info().is_none()
            }
        }
    }

    /// GL permits color blits within the same numeric class; depth/stencil
    /// formats must match exactly.
    pub(crate) fn blit_compatible(
        self,
        other: Self,
        aspect: crate::backend::gl::api::GlTextureAspect,
    ) -> bool {
        use crate::backend::gl::api::GlTextureAspect;
        if !self.supports_blit_aspect(aspect) || !other.supports_blit_aspect(aspect) {
            return false;
        }
        if aspect != GlTextureAspect::Color {
            return self == other;
        }
        let signed = |format| {
            matches!(
                format,
                Self::R8Sint
                    | Self::Rg8Sint
                    | Self::Rgba8Sint
                    | Self::R16Sint
                    | Self::Rg16Sint
                    | Self::Rgba16Sint
                    | Self::R32Sint
                    | Self::Rg32Sint
                    | Self::Rgba32Sint
            )
        };
        self.is_integer_color() == other.is_integer_color()
            && (!self.is_integer_color() || signed(self) == signed(other))
    }
}

#[cfg(test)]
mod blit_compatibility_tests {
    use super::GlFormat as F;
    use crate::backend::gl::api::GlTextureAspect as A;

    #[test]
    fn blits_obey_color_numeric_class_and_exact_depth_stencil_format() {
        assert!(F::Rgba8Unorm.blit_compatible(F::R16Float, A::Color));
        assert!(F::R8Uint.blit_compatible(F::Rgba16Uint, A::Color));
        assert!(!F::R8Uint.blit_compatible(F::Rgba16Sint, A::Color));
        assert!(!F::R8Uint.blit_compatible(F::Rgba8Unorm, A::Color));
        assert!(F::Depth24PlusStencil8.blit_compatible(F::Depth24PlusStencil8, A::StencilOnly));
        assert!(!F::Depth16Unorm.blit_compatible(F::Depth24Unorm, A::DepthOnly));
        assert!(!F::Depth16Unorm.blit_compatible(F::Depth16Unorm, A::StencilOnly));
    }
}

/// ASTC LDR block dimensions supported by the compressed-format contract.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) enum GlAstcBlock {
    /// 4 by 4 texels.
    B4x4,
    /// 5 by 4 texels.
    B5x4,
    /// 5 by 5 texels.
    B5x5,
    /// 6 by 5 texels.
    B6x5,
    /// 6 by 6 texels.
    B6x6,
    /// 8 by 5 texels.
    B8x5,
    /// 8 by 6 texels.
    B8x6,
    /// 8 by 8 texels.
    B8x8,
    /// 10 by 5 texels.
    B10x5,
    /// 10 by 6 texels.
    B10x6,
    /// 10 by 8 texels.
    B10x8,
    /// 10 by 10 texels.
    B10x10,
    /// 12 by 10 texels.
    B12x10,
    /// 12 by 12 texels.
    B12x12,
}

/// Transfer function encoded by a compressed texture format.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) enum GlCompressedColorSpace {
    /// Linear decoding.
    Linear,
    /// sRGB decoding.
    Srgb,
    /// ASTC HDR decode mode. It is not interchangeable with LDR-linear: the
    /// same block dimensions use a distinct capability extension and must
    /// remain distinguishable in the typed format key.
    Hdr,
}

/// Compression family. This is descriptive metadata, never an enablement bit.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) enum GlCompressedTextureFamily {
    /// Block Compression / S3TC family.
    Bc,
    /// Adaptive Scalable Texture Compression.
    Astc,
    /// Ericsson Texture Compression 2.
    Etc2,
    /// Ericsson Alpha Compression.
    Eac,
}

/// Exact upload layout for a compressed texture format.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GlCompressedFormatInfo {
    /// Descriptive family; a caller must still select one exact `GlFormat`.
    pub family: GlCompressedTextureFamily,
    /// Encoded texels per block horizontally.
    pub block_width: u8,
    /// Encoded texels per block vertically.
    pub block_height: u8,
    /// Bytes in each encoded block.
    pub block_bytes: u8,
    /// Texture decode color space.
    pub color_space: GlCompressedColorSpace,
}

/// Failure to derive an exact compressed upload byte count.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GlCompressedSizeError {
    /// Compressed texture dimensions must both be nonzero.
    ZeroExtent,
    /// The encoded block count or byte count overflowed `u64`.
    Overflow,
}

impl GlCompressedFormatInfo {
    /// Returns the exact encoded byte count for one compressed mip extent.
    ///
    /// Uses ordinary ceil-divided block counts for each exact format.
    pub(crate) fn checked_encoded_size(
        self,
        width: u32,
        height: u32,
    ) -> Result<u64, GlCompressedSizeError> {
        if width == 0 || height == 0 {
            return Err(GlCompressedSizeError::ZeroExtent);
        }
        let blocks = |extent: u32, block: u8| -> Result<u64, GlCompressedSizeError> {
            u64::from(extent)
                .checked_add(u64::from(block) - 1)
                .ok_or(GlCompressedSizeError::Overflow)
                .map(|value| value / u64::from(block))
        };
        let blocks_wide = blocks(width, self.block_width)?;
        let blocks_high = blocks(height, self.block_height)?;
        blocks_wide
            .checked_mul(blocks_high)
            .and_then(|count| count.checked_mul(u64::from(self.block_bytes)))
            .ok_or(GlCompressedSizeError::Overflow)
    }
}

impl GlFormat {
    /// Returns exact compressed upload metadata, if this is a compressed format.
    pub(crate) fn compressed_info(self) -> Option<GlCompressedFormatInfo> {
        use GlCompressedColorSpace::{Linear, Srgb};
        use GlCompressedTextureFamily::{Astc, Bc, Eac, Etc2};
        let info =
            |family, block_width, block_height, block_bytes, color_space| GlCompressedFormatInfo {
                family,
                block_width,
                block_height,
                block_bytes,
                color_space,
            };
        match self {
            Self::Bc1RgbUnorm | Self::Bc1RgbaUnorm => Some(info(Bc, 4, 4, 8, Linear)),
            Self::Bc1RgbaSrgb | Self::Bc1RgbSrgb => Some(info(Bc, 4, 4, 8, Srgb)),
            Self::Bc2RgbaUnorm | Self::Bc3RgbaUnorm => Some(info(Bc, 4, 4, 16, Linear)),
            Self::Bc2RgbaSrgb | Self::Bc3RgbaSrgb => Some(info(Bc, 4, 4, 16, Srgb)),
            Self::Bc4RUnorm | Self::Bc4RSnorm => Some(info(Bc, 4, 4, 8, Linear)),
            Self::Bc5RgUnorm
            | Self::Bc5RgSnorm
            | Self::Bc6hRgbUfloat
            | Self::Bc6hRgbSfloat
            | Self::Bc7RgbaUnorm => Some(info(Bc, 4, 4, 16, Linear)),
            Self::Bc7RgbaSrgb => Some(info(Bc, 4, 4, 16, Srgb)),
            Self::Astc { block, color_space } => {
                let (block_width, block_height) = match block {
                    GlAstcBlock::B4x4 => (4, 4),
                    GlAstcBlock::B5x4 => (5, 4),
                    GlAstcBlock::B5x5 => (5, 5),
                    GlAstcBlock::B6x5 => (6, 5),
                    GlAstcBlock::B6x6 => (6, 6),
                    GlAstcBlock::B8x5 => (8, 5),
                    GlAstcBlock::B8x6 => (8, 6),
                    GlAstcBlock::B8x8 => (8, 8),
                    GlAstcBlock::B10x5 => (10, 5),
                    GlAstcBlock::B10x6 => (10, 6),
                    GlAstcBlock::B10x8 => (10, 8),
                    GlAstcBlock::B10x10 => (10, 10),
                    GlAstcBlock::B12x10 => (12, 10),
                    GlAstcBlock::B12x12 => (12, 12),
                };
                Some(info(Astc, block_width, block_height, 16, color_space))
            }
            Self::Etc2Rgb8Unorm => Some(info(Etc2, 4, 4, 8, Linear)),
            Self::Etc2Rgb8Srgb => Some(info(Etc2, 4, 4, 8, Srgb)),
            Self::Etc2Rgba8Unorm => Some(info(Etc2, 4, 4, 16, Linear)),
            Self::Etc2Rgba8Srgb => Some(info(Etc2, 4, 4, 16, Srgb)),
            Self::Etc2Rgb8A1Unorm => Some(info(Etc2, 4, 4, 8, Linear)),
            Self::Etc2Rgb8A1Srgb => Some(info(Etc2, 4, 4, 8, Srgb)),
            Self::EacR11Unorm | Self::EacR11Snorm => Some(info(Eac, 4, 4, 8, Linear)),
            Self::EacRg11Unorm | Self::EacRg11Snorm => Some(info(Eac, 4, 4, 16, Linear)),
            _ => None,
        }
    }

    /// Whether the selected profile guarantees this exact compressed format.
    ///
    /// GLES3 has ETC2/EAC in core. This deliberately answers per format, not
    /// per compression family, so a positive ETC2 fact cannot imply BC/ASTC.
    pub(crate) const fn is_core_compressed_for(self, profile: GlFamilyProfile) -> bool {
        match profile {
            GlFamilyProfile::Desktop { major, minor } => {
                let bc45 = matches!(
                    self,
                    Self::Bc4RUnorm | Self::Bc4RSnorm | Self::Bc5RgUnorm | Self::Bc5RgSnorm
                );
                let bc67 = matches!(
                    self,
                    Self::Bc6hRgbUfloat
                        | Self::Bc6hRgbSfloat
                        | Self::Bc7RgbaUnorm
                        | Self::Bc7RgbaSrgb
                );
                let etc = matches!(
                    self,
                    Self::Etc2Rgb8Unorm
                        | Self::Etc2Rgb8Srgb
                        | Self::Etc2Rgba8Unorm
                        | Self::Etc2Rgba8Srgb
                        | Self::Etc2Rgb8A1Unorm
                        | Self::Etc2Rgb8A1Srgb
                        | Self::EacR11Unorm
                        | Self::EacRg11Unorm
                        | Self::EacR11Snorm
                        | Self::EacRg11Snorm
                );
                bc45 || (bc67 && (major > 4 || minor >= 2)) || (etc && (major > 4 || minor >= 3))
            }
            GlFamilyProfile::Embedded { major, minor } => {
                let etc = matches!(
                    self,
                    Self::Etc2Rgb8Unorm
                        | Self::Etc2Rgb8Srgb
                        | Self::Etc2Rgba8Unorm
                        | Self::Etc2Rgba8Srgb
                        | Self::Etc2Rgb8A1Unorm
                        | Self::Etc2Rgb8A1Srgb
                        | Self::EacR11Unorm
                        | Self::EacRg11Unorm
                        | Self::EacR11Snorm
                        | Self::EacRg11Snorm
                );
                let astc_ldr = matches!(
                    self,
                    Self::Astc {
                        color_space: GlCompressedColorSpace::Linear | GlCompressedColorSpace::Srgb,
                        ..
                    }
                );
                etc || (astc_ldr && (major > 3 || minor >= 2))
            }
            GlFamilyProfile::WebGl2 => false,
        }
    }

    pub(super) const fn is_non_compressed_core_baseline_for(
        self,
        profile: GlFamilyProfile,
    ) -> bool {
        // Admit only sized storage formats guaranteed by the WebGL2 tables.
        // Availability of storage does not establish every operation: each
        // discovered row still supplies its exact sampling/rendering facts.
        if matches!(profile, GlFamilyProfile::WebGl2) {
            return matches!(
                self,
                Self::Rgba8Unorm
                    | Self::Rgba8Srgb
                    | Self::Rgba8Uint
                    | Self::Rgba8Sint
                    | Self::Depth16Unorm
                    | Self::Depth24Unorm
                    | Self::Depth24PlusStencil8
                    | Self::Depth32Float
                    | Self::Depth32FloatStencil8
            );
        }
        match self {
            // Sized R/RG/RGBA integer, normalized, and float storage is core
            // in both desktop GL 4.x and GLES 3.x.  The table deliberately
            // does not turn this into a claim about filtering/renderability;
            // those remain exact operation facts.
            Self::R8Unorm
            | Self::R8Snorm
            | Self::R8Uint
            | Self::R8Sint
            | Self::Rg8Unorm
            | Self::Rg8Snorm
            | Self::Rg8Uint
            | Self::Rg8Sint
            | Self::Rgba8Unorm
            | Self::Rgba8Srgb
            | Self::Rgba8Snorm
            | Self::Rgba8Uint
            | Self::Rgba8Sint
            | Self::R16Uint
            | Self::R16Sint
            | Self::R16Float
            | Self::R16Unorm
            | Self::R16Snorm
            | Self::Rg16Uint
            | Self::Rg16Sint
            | Self::Rg16Float
            | Self::Rg16Unorm
            | Self::Rg16Snorm
            | Self::Rgba16Uint
            | Self::Rgba16Sint
            | Self::Rgba16Float
            | Self::Rgba16Unorm
            | Self::Rgba16Snorm
            | Self::Rgba32Float
            | Self::R32Uint
            | Self::R32Sint
            | Self::R32Float
            | Self::Rg32Uint
            | Self::Rg32Sint
            | Self::Rg32Float
            | Self::Rgba32Uint
            | Self::Rgba32Sint
            | Self::Rgb9e5Ufloat
            | Self::Rgb10a2Uint
            | Self::Rgb10a2Unorm
            | Self::Rg11b10Ufloat
            | Self::Depth16Unorm
            | Self::Depth24Unorm
            | Self::Depth24PlusStencil8
            | Self::Depth32Float
            | Self::Depth32FloatStencil8
            | Self::Stencil8 => true,
            // These are desktop-core spellings only at the GL 4.x floor. They
            // are intentionally not promoted to GLES3/WebGL2 guarantees.
            Self::R64Uint => {
                matches!(profile, GlFamilyProfile::Desktop { .. })
            }
            Self::Bgra8Unorm | Self::Bgra8Srgb => false,
            _ => false,
        }
    }

    pub(super) const fn is_supplied_by_extension(self, extension: GlKnownExtension) -> bool {
        match extension {
            GlKnownExtension::OesTextureStencil8 | GlKnownExtension::ArbTextureStencil8 => {
                matches!(self, Self::Stencil8)
            }
            GlKnownExtension::ExtTextureNorm16 => matches!(
                self,
                Self::R16Unorm
                    | Self::R16Snorm
                    | Self::Rg16Unorm
                    | Self::Rg16Snorm
                    | Self::Rgba16Unorm
                    | Self::Rgba16Snorm
            ),
            GlKnownExtension::CompressedTextureS3tc => matches!(
                self,
                Self::Bc1RgbUnorm | Self::Bc1RgbaUnorm | Self::Bc2RgbaUnorm | Self::Bc3RgbaUnorm
            ),
            // Desktop GL 4.x has core texture sRGB; combined with EXT S3TC
            // this exposes the corresponding sRGB compressed formats. Native
            // discovery admits this evidence for sRGB only on desktop.
            GlKnownExtension::ExtTextureCompressionS3tc => matches!(
                self,
                Self::Bc1RgbUnorm
                    | Self::Bc1RgbaUnorm
                    | Self::Bc2RgbaUnorm
                    | Self::Bc3RgbaUnorm
                    | Self::Bc1RgbSrgb
                    | Self::Bc1RgbaSrgb
                    | Self::Bc2RgbaSrgb
                    | Self::Bc3RgbaSrgb
            ),
            GlKnownExtension::ExtTextureCompressionDxt1 => {
                matches!(self, Self::Bc1RgbUnorm | Self::Bc1RgbaUnorm)
            }
            GlKnownExtension::AngleTextureCompressionDxt3 => matches!(self, Self::Bc2RgbaUnorm),
            GlKnownExtension::AngleTextureCompressionDxt5 => matches!(self, Self::Bc3RgbaUnorm),
            GlKnownExtension::ExtTextureCompressionS3tcSrgb => matches!(
                self,
                Self::Bc1RgbSrgb | Self::Bc1RgbaSrgb | Self::Bc2RgbaSrgb | Self::Bc3RgbaSrgb
            ),
            GlKnownExtension::CompressedTextureS3tcSrgb => matches!(
                self,
                Self::Bc1RgbSrgb | Self::Bc1RgbaSrgb | Self::Bc2RgbaSrgb | Self::Bc3RgbaSrgb
            ),
            GlKnownExtension::CompressedTextureBptc
            | GlKnownExtension::ArbTextureCompressionBptc => matches!(
                self,
                Self::Bc6hRgbUfloat | Self::Bc6hRgbSfloat | Self::Bc7RgbaUnorm | Self::Bc7RgbaSrgb
            ),
            GlKnownExtension::CompressedTextureRgtc
            | GlKnownExtension::ExtTextureCompressionRgtc => matches!(
                self,
                Self::Bc4RUnorm | Self::Bc4RSnorm | Self::Bc5RgUnorm | Self::Bc5RgSnorm
            ),
            // WebGL's one ASTC extension carries an additional
            // `getSupportedProfiles()` observation for HDR. Browser discovery adds
            // an HDR row only after that observation; native KHR LDR remains
            // restricted to LDR below.
            GlKnownExtension::CompressedTextureAstc => matches!(self, Self::Astc { .. }),
            GlKnownExtension::KhrTextureCompressionAstcLdr => matches!(
                self,
                Self::Astc {
                    color_space: GlCompressedColorSpace::Linear | GlCompressedColorSpace::Srgb,
                    ..
                }
            ),
            GlKnownExtension::KhrTextureCompressionAstcHdr => matches!(
                self,
                Self::Astc {
                    color_space: GlCompressedColorSpace::Hdr
                        | GlCompressedColorSpace::Linear
                        | GlCompressedColorSpace::Srgb,
                    ..
                }
            ),
            GlKnownExtension::OesCompressedEtc2Rgb8Texture => matches!(self, Self::Etc2Rgb8Unorm),
            GlKnownExtension::CompressedTextureEtc | GlKnownExtension::ArbEs3Compatibility => {
                matches!(
                    self,
                    Self::Etc2Rgb8Unorm
                        | Self::Etc2Rgb8Srgb
                        | Self::Etc2Rgba8Unorm
                        | Self::Etc2Rgba8Srgb
                        | Self::Etc2Rgb8A1Unorm
                        | Self::Etc2Rgb8A1Srgb
                        | Self::EacR11Unorm
                        | Self::EacRg11Unorm
                        | Self::EacR11Snorm
                        | Self::EacRg11Snorm
                )
            }
            _ => false,
        }
    }
}

#[cfg(test)]
mod compressed_profile_tests {
    use super::*;
    #[test]
    fn core_and_extension_evidence_is_format_specific() {
        let gl40 = GlFamilyProfile::Desktop { major: 4, minor: 0 };
        let gl42 = GlFamilyProfile::Desktop { major: 4, minor: 2 };
        let es31 = GlFamilyProfile::Embedded { major: 3, minor: 1 };
        let es32 = GlFamilyProfile::Embedded { major: 3, minor: 2 };
        let astc = GlFormat::Astc {
            block: GlAstcBlock::B4x4,
            color_space: GlCompressedColorSpace::Linear,
        };
        assert!(GlFormat::Bc4RUnorm.is_core_compressed_for(gl40));
        assert!(!GlFormat::Bc7RgbaUnorm.is_core_compressed_for(gl40));
        assert!(GlFormat::Bc7RgbaUnorm.is_core_compressed_for(gl42));
        assert!(!astc.is_core_compressed_for(es31));
        assert!(astc.is_core_compressed_for(es32));
        assert!(astc.is_supplied_by_extension(GlKnownExtension::KhrTextureCompressionAstcHdr));
        let hdr = GlFormat::Astc {
            block: GlAstcBlock::B4x4,
            color_space: GlCompressedColorSpace::Hdr,
        };
        assert!(!hdr.is_supplied_by_extension(GlKnownExtension::KhrTextureCompressionAstcLdr));
        assert!(
            GlFormat::Etc2Rgb8Unorm
                .is_supplied_by_extension(GlKnownExtension::OesCompressedEtc2Rgb8Texture)
        );
        assert!(
            !GlFormat::EacR11Unorm
                .is_supplied_by_extension(GlKnownExtension::OesCompressedEtc2Rgb8Texture)
        );
    }
}
