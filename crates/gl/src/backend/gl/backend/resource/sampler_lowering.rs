//! Native GL sampler enumeration lowering.
use fluxel_rhi_core::api::{self, Error, Result};

pub(super) fn lower_address(
    value: api::AddressMode,
) -> Result<crate::backend::gl::api::GlAddressMode> {
    match value {
        api::AddressMode::ClampToEdge => Ok(crate::backend::gl::api::GlAddressMode::ClampToEdge),
        api::AddressMode::Repeat => Ok(crate::backend::gl::api::GlAddressMode::Repeat),
        api::AddressMode::MirrorRepeat => {
            Ok(crate::backend::gl::api::GlAddressMode::MirroredRepeat)
        }
        api::AddressMode::ClampToBorder | api::AddressMode::ClampToZero => {
            Err(Error::Unsupported("GL sampler address mode"))
        }
    }
}

pub(super) fn lower_filter(value: api::FilterMode) -> crate::backend::gl::api::GlFilterMode {
    match value {
        api::FilterMode::Nearest => crate::backend::gl::api::GlFilterMode::Nearest,
        api::FilterMode::Linear => crate::backend::gl::api::GlFilterMode::Linear,
    }
}

pub(super) fn lower_mip_filter(
    value: api::FilterMode,
) -> crate::backend::gl::api::GlMipmapFilterMode {
    match value {
        api::FilterMode::Nearest => crate::backend::gl::api::GlMipmapFilterMode::Nearest,
        api::FilterMode::Linear => crate::backend::gl::api::GlMipmapFilterMode::Linear,
    }
}

pub(super) fn lower_compare(
    value: api::CompareFunction,
) -> crate::backend::gl::api::GlCompareFunction {
    match value {
        api::CompareFunction::Never => crate::backend::gl::api::GlCompareFunction::Never,
        api::CompareFunction::Less => crate::backend::gl::api::GlCompareFunction::Less,
        api::CompareFunction::Equal => crate::backend::gl::api::GlCompareFunction::Equal,
        api::CompareFunction::LessEqual => crate::backend::gl::api::GlCompareFunction::LessEqual,
        api::CompareFunction::Greater => crate::backend::gl::api::GlCompareFunction::Greater,
        api::CompareFunction::NotEqual => crate::backend::gl::api::GlCompareFunction::NotEqual,
        api::CompareFunction::GreaterEqual => {
            crate::backend::gl::api::GlCompareFunction::GreaterEqual
        }
        api::CompareFunction::Always => crate::backend::gl::api::GlCompareFunction::Always,
    }
}
