//! Texture and renderbuffer descriptors for private GL resource creation.
//!
//! This module validates object-local shape only; context ownership and usage
//! planning remain with the provider and Core respectively.

use super::*;

/// Texture dimensionality accepted by the common GL-family layer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GlTextureDimension {
    D1,
    D2,
    D3,
    Cube,
    D2Array,
}

/// Immutable creation facts for a renderbuffer.
///
/// Renderbuffers are the GL-family's render-only two-dimensional attachment
/// storage; they are never sampled and never mipmapped, so the descriptor is
/// deliberately narrower than [`GlTextureDesc`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GlRenderBufferDesc {
    pub format: GlFormat,
    pub width: u32,
    pub height: u32,
    pub samples: u32,
}

impl GlRenderBufferDesc {
    pub(crate) fn validate(self) -> Result<(), GlResourceValidationError> {
        if self.width == 0 || self.height == 0 {
            return Err(GlResourceValidationError::ZeroRenderBufferExtent);
        }
        if self.samples == 0 {
            return Err(GlResourceValidationError::ZeroSampleCount);
        }
        Ok(())
    }
}

/// Texture operations permitted for a resource.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GlTextureUsage(u32);
impl GlTextureUsage {
    pub(crate) const COPY_SOURCE: Self = Self(1 << 0);
    pub(crate) const COPY_DESTINATION: Self = Self(1 << 1);
    pub(crate) const SAMPLED: Self = Self(1 << 2);
    pub(crate) const RENDER_ATTACHMENT: Self = Self(1 << 3);
    pub(crate) const STORAGE_BINDING: Self = Self(1 << 4);
    pub(crate) const EMPTY: Self = Self(0);
    pub(crate) const fn contains(self, required: Self) -> bool {
        self.0 & required.0 == required.0
    }
    pub(crate) const fn is_empty(self) -> bool {
        self.0 == 0
    }
}
impl core::ops::BitOr for GlTextureUsage {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

/// Pixel extent.  The depth component is layers for arrays and depth for 3D textures.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GlExtent3d {
    pub width: u32,
    pub height: u32,
    pub depth_or_layers: u32,
}

/// Immutable creation facts for a texture.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GlTextureDesc {
    pub dimension: GlTextureDimension,
    pub extent: GlExtent3d,
    pub mip_level_count: u32,
    pub sample_count: u32,
    pub format: GlFormat,
    pub usage: GlTextureUsage,
}

impl GlTextureDesc {
    pub(crate) fn validate(self) -> Result<(), GlResourceValidationError> {
        if self.extent.width == 0 || self.extent.height == 0 || self.extent.depth_or_layers == 0 {
            return Err(GlResourceValidationError::ZeroTextureExtent);
        }
        if self.mip_level_count == 0 {
            return Err(GlResourceValidationError::ZeroMipLevelCount);
        }
        if self.sample_count == 0 {
            return Err(GlResourceValidationError::ZeroSampleCount);
        }
        if self.usage.is_empty() {
            return Err(GlResourceValidationError::EmptyTextureUsage);
        }
        if self.sample_count > 1 && self.mip_level_count != 1 {
            return Err(GlResourceValidationError::MultisampleMipmapped);
        }
        match self.dimension {
            GlTextureDimension::D1
                if self.extent.height != 1 || self.extent.depth_or_layers != 1 =>
            {
                return Err(GlResourceValidationError::InvalidDimensionExtent);
            }
            GlTextureDimension::D2 if self.extent.depth_or_layers != 1 => {
                return Err(GlResourceValidationError::InvalidDimensionExtent);
            }
            GlTextureDimension::Cube
                if self.extent.width != self.extent.height || self.extent.depth_or_layers != 6 =>
            {
                return Err(GlResourceValidationError::InvalidCubeExtent);
            }
            _ => {}
        }
        if self.sample_count > 1
            && (self.dimension != GlTextureDimension::D2 || self.extent.depth_or_layers != 1)
        {
            return Err(GlResourceValidationError::MultisampleRequiresD2);
        }
        if self.mip_level_count > self.maximum_mip_level_count() {
            return Err(GlResourceValidationError::MipLevelCountOutOfBounds);
        }
        Ok(())
    }
    pub(crate) const fn maximum_mip_level_count(self) -> u32 {
        let mut largest = self.extent.width;
        if self.extent.height > largest {
            largest = self.extent.height;
        }
        if let GlTextureDimension::D3 = self.dimension {
            if self.extent.depth_or_layers > largest {
                largest = self.extent.depth_or_layers;
            }
        }
        32 - largest.leading_zeros()
    }
    pub(crate) const fn mip_extent(self, mip_level: u32) -> Option<GlExtent3d> {
        if mip_level >= self.mip_level_count {
            return None;
        }
        let width = mip_dimension(self.extent.width, mip_level);
        let height = mip_dimension(self.extent.height, mip_level);
        let depth_or_layers = match self.dimension {
            GlTextureDimension::D3 => mip_dimension(self.extent.depth_or_layers, mip_level),
            _ => self.extent.depth_or_layers,
        };
        Some(GlExtent3d {
            width,
            height,
            depth_or_layers,
        })
    }
}

/// Returns a nonzero mip dimension without relying on post-MSRT const methods.
const fn mip_dimension(value: u32, level: u32) -> u32 {
    if level >= u32::BITS {
        1
    } else {
        let shifted = value >> level;
        if shifted == 0 { 1 } else { shifted }
    }
}

/// A color, depth, or stencil plane.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GlTextureAspect {
    All,
    Color,
    DepthOnly,
    StencilOnly,
}

/// One mip and contiguous layer range of a texture.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GlTextureSubresource {
    pub texture: TextureId,
    pub aspect: GlTextureAspect,
    pub mip_level: u32,
    pub base_layer: u32,
    pub layer_count: u32,
}

impl GlTextureSubresource {
    pub(crate) fn validate_for(self, desc: GlTextureDesc) -> Result<(), GlResourceValidationError> {
        if self.layer_count == 0 {
            return Err(GlResourceValidationError::ZeroLayerCount);
        }
        let Some(extent) = desc.mip_extent(self.mip_level) else {
            return Err(GlResourceValidationError::MipLevelOutOfBounds);
        };
        if desc.dimension == GlTextureDimension::D3
            && (self.base_layer != 0 || self.layer_count != 1)
        {
            return Err(GlResourceValidationError::ThreeDimensionalSubresourceLayers);
        }
        if self
            .base_layer
            .checked_add(self.layer_count)
            .is_none_or(|end| end > extent.depth_or_layers)
        {
            return Err(GlResourceValidationError::LayerRangeOutOfBounds);
        }
        Ok(())
    }
}

/// A rectangular region within one validated subresource.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GlTextureRegion {
    pub subresource: GlTextureSubresource,
    pub origin: [u32; 3],
    pub extent: GlExtent3d,
}
impl GlTextureRegion {
    pub(crate) fn validate_for(self, desc: GlTextureDesc) -> Result<(), GlResourceValidationError> {
        self.subresource.validate_for(desc)?;
        if self.extent.width == 0 || self.extent.height == 0 || self.extent.depth_or_layers == 0 {
            return Err(GlResourceValidationError::ZeroCopyExtent);
        }
        let Some(bound) = desc.mip_extent(self.subresource.mip_level) else {
            return Err(GlResourceValidationError::MipLevelOutOfBounds);
        };
        if self.origin[0]
            .checked_add(self.extent.width)
            .is_none_or(|end| end > bound.width)
            || self.origin[1]
                .checked_add(self.extent.height)
                .is_none_or(|end| end > bound.height)
        {
            return Err(GlResourceValidationError::TextureRegionOutOfBounds);
        }
        let maximum_depth = if desc.dimension == GlTextureDimension::D3 {
            bound.depth_or_layers
        } else {
            self.subresource.layer_count
        };
        if self.origin[2]
            .checked_add(self.extent.depth_or_layers)
            .is_none_or(|end| end > maximum_depth)
        {
            return Err(GlResourceValidationError::SubresourceRegionOutOfBounds);
        }
        Ok(())
    }
}

/// Validates an exact texture copy before a provider changes bindings or issues GL.
pub(crate) fn validate_texture_copy(
    source: GlTextureRegion,
    source_desc: GlTextureDesc,
    destination: GlTextureRegion,
    destination_desc: GlTextureDesc,
) -> Result<(), GlResourceValidationError> {
    source.validate_for(source_desc)?;
    destination.validate_for(destination_desc)?;
    if source_desc.format != destination_desc.format {
        return Err(GlResourceValidationError::IncompatibleCopyFormat);
    }
    if source_desc.sample_count != destination_desc.sample_count {
        return Err(GlResourceValidationError::IncompatibleCopySampleCount);
    }
    if source.subresource.aspect != destination.subresource.aspect {
        return Err(GlResourceValidationError::IncompatibleCopyAspect);
    }
    if source.extent != destination.extent {
        return Err(GlResourceValidationError::IncompatibleCopyExtent);
    }
    if let Some(block) = source_desc.format.compressed_info() {
        let source_mip = source_desc
            .mip_extent(source.subresource.mip_level)
            .ok_or(GlResourceValidationError::MipLevelOutOfBounds)?;
        let destination_mip = destination_desc
            .mip_extent(destination.subresource.mip_level)
            .ok_or(GlResourceValidationError::MipLevelOutOfBounds)?;
        let aligned = |origin: u32, extent: u32, full: u32, block: u8| {
            origin % u32::from(block) == 0
                && (extent % u32::from(block) == 0 || origin.saturating_add(extent) == full)
        };
        if !aligned(
            source.origin[0],
            source.extent.width,
            source_mip.width,
            block.block_width,
        ) || !aligned(
            source.origin[1],
            source.extent.height,
            source_mip.height,
            block.block_height,
        ) || !aligned(
            destination.origin[0],
            destination.extent.width,
            destination_mip.width,
            block.block_width,
        ) || !aligned(
            destination.origin[1],
            destination.extent.height,
            destination_mip.height,
            block.block_height,
        ) {
            return Err(GlResourceValidationError::CompressedCopyBlockMisaligned);
        }
    }
    Ok(())
}

/// Validation failures which providers must report before issuing a GL call.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GlResourceValidationError {
    ZeroBufferSize,
    EmptyBufferUsage,
    /// Two usage roles that cannot share one buffer; see [`roles_conflict`].
    ConflictingBufferRoles {
        first: GlBufferRole,
        second: GlBufferRole,
    },
    ZeroRangeSize,
    BufferRangeOutOfBounds,
    ZeroTextureExtent,
    ZeroRenderBufferExtent,
    ZeroMipLevelCount,
    ZeroSampleCount,
    EmptyTextureUsage,
    MultisampleMipmapped,
    MultisampleRequiresD2,
    InvalidDimensionExtent,
    InvalidCubeExtent,
    MipLevelCountOutOfBounds,
    ZeroLayerCount,
    MipLevelOutOfBounds,
    LayerRangeOutOfBounds,
    ZeroCopyExtent,
    TextureRegionOutOfBounds,
    SubresourceRegionOutOfBounds,
    ThreeDimensionalSubresourceLayers,
    IncompatibleCopyFormat,
    IncompatibleCopySampleCount,
    IncompatibleCopyAspect,
    IncompatibleCopyExtent,
    /// A compressed copy starts or ends inside a codec block. GL image copies
    /// transfer encoded blocks, so interior edges must be block-aligned.
    CompressedCopyBlockMisaligned,
}

impl GlResourceValidationError {
    /// A description for the provider's structured error, naming the actual
    /// cause.
    ///
    /// Providers report validation through `GlError::Validation { message }`.
    /// Deriving that message here keeps the offending roles visible to the
    /// caller instead of collapsing every descriptor failure into one generic
    /// string.
    pub(crate) fn message(&self) -> String {
        match self {
            Self::ConflictingBufferRoles { first, second } => format!(
                "buffer usage combines conflicting roles `{}` and `{}`",
                first.name(),
                second.name()
            ),
            other => format!("{other:?}"),
        }
    }
}
