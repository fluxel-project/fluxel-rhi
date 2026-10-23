//! Single-sample ordinary texture capability facts.

use super::*;

pub(super) fn record_ordinary(
    profile: GlFamilyProfile,
    extensions: &GlExtensionSet,
    record: &mut impl FnMut(GlFormatCapabilities) -> Result<(), NativeDiscoveryError>,
) -> Result<(), NativeDiscoveryError> {
    for format in [
        GlFormat::R8Unorm,
        GlFormat::R8Snorm,
        GlFormat::R8Uint,
        GlFormat::R8Sint,
        GlFormat::Rg8Unorm,
        GlFormat::Rg8Snorm,
        GlFormat::Rg8Uint,
        GlFormat::Rg8Sint,
        GlFormat::Rgba8Snorm,
    ] {
        let integer = matches!(
            format,
            GlFormat::R8Uint | GlFormat::R8Sint | GlFormat::Rg8Uint | GlFormat::Rg8Sint
        );
        let renderable = integer || matches!(format, GlFormat::R8Unorm | GlFormat::Rg8Unorm);
        record(GlFormatCapabilities {
            format,
            resource_kind: GlFormatResourceKind::Texture,
            sample_count: 1,
            evidence: GlFormatEvidence::CoreGuaranteed,
            sampled: true,
            filterable: !integer,
            renderable,
            blendable: renderable && !integer,
            storage_read: false,
            storage_write: false,
            copy_source: false,
            copy_destination: true,
        })?;
    }
    for format in [
        GlFormat::R16Uint,
        GlFormat::R16Sint,
        GlFormat::R16Float,
        GlFormat::Rg16Uint,
        GlFormat::Rg16Sint,
        GlFormat::Rg16Float,
        GlFormat::Rgba16Uint,
        GlFormat::Rgba16Sint,
        GlFormat::R32Uint,
        GlFormat::R32Sint,
        GlFormat::R32Float,
        GlFormat::Rg32Uint,
        GlFormat::Rg32Sint,
        GlFormat::Rg32Float,
        GlFormat::Rgba32Uint,
        GlFormat::Rgba32Sint,
        GlFormat::Rgb9e5Ufloat,
        GlFormat::Rgb10a2Uint,
        GlFormat::Rgb10a2Unorm,
        GlFormat::Rg11b10Ufloat,
    ] {
        let integer = matches!(
            format,
            GlFormat::R16Uint
                | GlFormat::R16Sint
                | GlFormat::Rg16Uint
                | GlFormat::Rg16Sint
                | GlFormat::Rgba16Uint
                | GlFormat::Rgba16Sint
                | GlFormat::R32Uint
                | GlFormat::R32Sint
                | GlFormat::Rg32Uint
                | GlFormat::Rg32Sint
                | GlFormat::Rgba32Uint
                | GlFormat::Rgba32Sint
                | GlFormat::Rgb10a2Uint
        );
        let is_32_float = matches!(format, GlFormat::R32Float | GlFormat::Rg32Float);
        let renderable = integer || matches!(format, GlFormat::Rgb10a2Unorm);
        let filterable = !integer
            && (!is_32_float
                || matches!(profile, GlFamilyProfile::Desktop { .. })
                || extensions.is_acquired(GlKnownExtension::OesTextureFloatLinear));
        record(GlFormatCapabilities {
            format,
            resource_kind: GlFormatResourceKind::Texture,
            sample_count: 1,
            evidence: GlFormatEvidence::CoreGuaranteed,
            sampled: true,
            filterable,
            renderable,
            blendable: renderable && !integer,
            storage_read: false,
            storage_write: false,
            copy_source: false,
            copy_destination: true,
        })?;
    }
    if matches!(profile, GlFamilyProfile::Desktop { .. })
        || extensions.is_acquired(GlKnownExtension::ExtTextureNorm16)
    {
        for format in [
            GlFormat::R16Unorm,
            GlFormat::R16Snorm,
            GlFormat::Rg16Unorm,
            GlFormat::Rg16Snorm,
            GlFormat::Rgba16Unorm,
            GlFormat::Rgba16Snorm,
        ] {
            let renderable = matches!(
                format,
                GlFormat::R16Unorm | GlFormat::Rg16Unorm | GlFormat::Rgba16Unorm
            );
            record(GlFormatCapabilities {
                format,
                resource_kind: GlFormatResourceKind::Texture,
                sample_count: 1,
                evidence: if matches!(profile, GlFamilyProfile::Desktop { .. }) {
                    GlFormatEvidence::CoreGuaranteed
                } else {
                    GlFormatEvidence::ExtensionAcquired(GlKnownExtension::ExtTextureNorm16)
                },
                sampled: true,
                filterable: true,
                renderable,
                blendable: renderable,
                storage_read: false,
                storage_write: false,
                copy_source: false,
                copy_destination: true,
            })?;
        }
    }
    Ok(())
}
