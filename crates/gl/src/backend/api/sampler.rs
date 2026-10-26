//! Platform-neutral sampler descriptions.

use super::{GlDiscoverySnapshot, GlError, GlFamilyApi, GlKnownExtension, SamplerId};

/// Sampling direction along an axis.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GlAddressMode {
    ClampToEdge,
    Repeat,
    MirroredRepeat,
}
/// Minification/magnification filtering mode.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GlFilterMode {
    Nearest,
    Linear,
}
/// Mipmap filtering mode.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GlMipmapFilterMode {
    Nearest,
    Linear,
}
/// Depth comparison used by a comparison sampler.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GlCompareFunction {
    Never,
    Less,
    Equal,
    LessEqual,
    Greater,
    NotEqual,
    GreaterEqual,
    Always,
}

/// Immutable sampler creation facts.  `lod_min` and `lod_max` retain exact finite bits.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GlSamplerDesc {
    pub address_mode_u: GlAddressMode,
    pub address_mode_v: GlAddressMode,
    pub address_mode_w: GlAddressMode,
    pub mag_filter: GlFilterMode,
    pub min_filter: GlFilterMode,
    pub mipmap_filter: GlMipmapFilterMode,
    pub lod_min_bits: u32,
    pub lod_max_bits: u32,
    pub compare: Option<GlCompareFunction>,
    pub max_anisotropy_bits: Option<u32>,
}

impl GlSamplerDesc {
    pub(crate) fn validate(self) -> Result<(), GlSamplerValidationError> {
        let min = f32::from_bits(self.lod_min_bits);
        let max = f32::from_bits(self.lod_max_bits);
        if !min.is_finite() || !max.is_finite() {
            return Err(GlSamplerValidationError::NonFiniteLod);
        }
        if min > max {
            return Err(GlSamplerValidationError::InvertedLodRange);
        }
        if let Some(bits) = self.max_anisotropy_bits {
            let value = f32::from_bits(bits);
            if !value.is_finite() || value < 1.0 {
                return Err(GlSamplerValidationError::InvalidAnisotropy);
            }
        }
        Ok(())
    }

    /// Validates requested anisotropy against evidence bound to this context generation.
    pub(crate) fn validate_for(
        self,
        discovery: &GlDiscoverySnapshot,
    ) -> Result<(), GlSamplerValidationError> {
        self.validate()?;
        let Some(requested_bits) = self.max_anisotropy_bits else {
            return Ok(());
        };
        let requested = f32::from_bits(requested_bits);
        if !discovery
            .extensions()
            .is_acquired(GlKnownExtension::ExtTextureFilterAnisotropic)
        {
            return Err(GlSamplerValidationError::AnisotropyExtensionUnavailable);
        }
        let Some(maximum) = discovery.limits().max_texture_anisotropy else {
            return Err(GlSamplerValidationError::AnisotropyLimitUnavailable);
        };
        if requested > maximum.get() {
            return Err(GlSamplerValidationError::AnisotropyExceedsLimit);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GlSamplerValidationError {
    NonFiniteLod,
    InvertedLodRange,
    InvalidAnisotropy,
    AnisotropyExtensionUnavailable,
    AnisotropyLimitUnavailable,
    AnisotropyExceedsLimit,
}

/// Sampler allocation domain. Providers validate the description before mutation.
pub(crate) trait GlSamplerApi: GlFamilyApi {
    fn create_sampler(&mut self, desc: GlSamplerDesc) -> Result<SamplerId, GlError>;
    fn destroy_sampler(&mut self, sampler: SamplerId) -> Result<(), GlError>;
}

#[cfg(test)]
#[path = "../../tests/backend/api/sampler__tests.rs"]
mod tests;
