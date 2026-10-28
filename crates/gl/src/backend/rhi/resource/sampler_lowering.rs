//! Native GL sampler enumeration lowering.
use fluxel_rhi_core::api::{self, Error, Result};

pub(super) fn lower_address(value: api::AddressMode) -> Result<crate::backend::api::GlAddressMode> {
    match value {
        api::AddressMode::ClampToEdge => Ok(crate::backend::api::GlAddressMode::ClampToEdge),
        api::AddressMode::Repeat => Ok(crate::backend::api::GlAddressMode::Repeat),
        api::AddressMode::MirrorRepeat => Ok(crate::backend::api::GlAddressMode::MirroredRepeat),
        api::AddressMode::ClampToBorder | api::AddressMode::ClampToZero => {
            Err(Error::Unsupported("GL sampler address mode"))
        }
    }
}

pub(super) fn lower_filter(value: api::FilterMode) -> crate::backend::api::GlFilterMode {
    match value {
        api::FilterMode::Nearest => crate::backend::api::GlFilterMode::Nearest,
        api::FilterMode::Linear => crate::backend::api::GlFilterMode::Linear,
    }
}

pub(super) fn lower_mip_filter(value: api::FilterMode) -> crate::backend::api::GlMipmapFilterMode {
    match value {
        api::FilterMode::Nearest => crate::backend::api::GlMipmapFilterMode::Nearest,
        api::FilterMode::Linear => crate::backend::api::GlMipmapFilterMode::Linear,
    }
}

pub(super) fn lower_compare(value: api::CompareFunction) -> crate::backend::api::GlCompareFunction {
    match value {
        api::CompareFunction::Never => crate::backend::api::GlCompareFunction::Never,
        api::CompareFunction::Less => crate::backend::api::GlCompareFunction::Less,
        api::CompareFunction::Equal => crate::backend::api::GlCompareFunction::Equal,
        api::CompareFunction::LessEqual => crate::backend::api::GlCompareFunction::LessEqual,
        api::CompareFunction::Greater => crate::backend::api::GlCompareFunction::Greater,
        api::CompareFunction::NotEqual => crate::backend::api::GlCompareFunction::NotEqual,
        api::CompareFunction::GreaterEqual => crate::backend::api::GlCompareFunction::GreaterEqual,
        api::CompareFunction::Always => crate::backend::api::GlCompareFunction::Always,
    }
}
