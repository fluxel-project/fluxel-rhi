//! Native GL format facts and profile parsing.

use super::*;
use crate::backend::gl::api::GlCompressedTextureFamily;

#[path = "ordinary_formats.rs"]
mod ordinary;

/// Builds the complete native format table: core baselines plus every fact the
/// operation probes actually observed.
///
/// Float facts are probe facts. Neither RGBA16F nor RGBA32F renderability is
/// core-guaranteed on every accepted native profile (ES 3.x requires
/// `EXT_color_buffer_float` and desktop depends on the exact version), and the
/// same holds for float depth rendering, so all of them are recorded only with
/// `OperationProbed` evidence from real framebuffer-completeness probes.
///
/// Every sample count recorded here is bounded by the limit that governs its
/// class, so a caller can read the presence of a fact as the proof that the
/// context both accepts that count and can allocate it.
pub(super) fn native_formats(
    profile: GlFamilyProfile,
    extensions: &GlExtensionSet,
    limits: &GlLimits,
    report: &ProbeReport,
) -> Result<GlFormatTable, NativeDiscoveryError> {
    let mut table = GlFormatTable::default();
    let mut record = |mut facts: GlFormatCapabilities| {
        if facts.resource_kind == GlFormatResourceKind::Texture && facts.sample_count == 1 {
            if let Some(i) = super::super::probes::IMAGE_FORMATS
                .iter()
                .position(|(format, ..)| *format == facts.format)
            {
                let desktop = matches!(profile, GlFamilyProfile::Desktop { .. });
                let snorm = matches!(
                    facts.format,
                    GlFormat::R8Snorm
                        | GlFormat::Rg8Snorm
                        | GlFormat::Rgba8Snorm
                        | GlFormat::R16Snorm
                        | GlFormat::Rg16Snorm
                        | GlFormat::Rgba16Snorm
                );
                let float = matches!(
                    facts.format,
                    GlFormat::R16Float
                        | GlFormat::Rg16Float
                        | GlFormat::R32Float
                        | GlFormat::Rg32Float
                        | GlFormat::Rgba16Float
                        | GlFormat::Rgba32Float
                        | GlFormat::Rg11b10Ufloat
                );
                let snorm_route = desktop
                    || extensions
                        .raw_reported_names()
                        .any(|name| name == "GL_EXT_render_snorm");
                let full_float = extensions
                    .raw_reported_names()
                    .any(|name| name == "GL_EXT_color_buffer_float");
                let half_float = extensions
                    .raw_reported_names()
                    .any(|name| name == "GL_EXT_color_buffer_half_float");
                let half_format = matches!(
                    facts.format,
                    GlFormat::R16Float | GlFormat::Rg16Float | GlFormat::Rgba16Float
                );
                let float_route = desktop || full_float || (half_float && half_format);
                if !facts.renderable
                    && report.ordinary_render[i] == ProbeAnswer::Passed
                    && (!snorm || snorm_route)
                    && (!float || float_route)
                {
                    facts.evidence = GlFormatEvidence::OperationProbed;
                    facts.renderable = true;
                    facts.blendable = if snorm {
                        true
                    } else if float {
                        desktop
                    } else {
                        facts.blendable
                    };
                }
                if float && facts.renderable && float_route {
                    let float32 = matches!(
                        facts.format,
                        GlFormat::R32Float | GlFormat::Rg32Float | GlFormat::Rgba32Float
                    );
                    let float_blend = extensions
                        .raw_reported_names()
                        .any(|name| name == "GL_EXT_float_blend");
                    facts.blendable = desktop || !float32 || float_blend;
                }
            }
        }
        if facts.resource_kind == GlFormatResourceKind::Texture && facts.sample_count == 1 {
            let index = super::super::probes::EXACT_INTEGER_FORMATS
                .iter()
                .position(|format| *format == facts.format);
            if index.is_some_and(|i| report.exact_integer_read[i] == ProbeAnswer::Passed)
                || matches!(facts.format, GlFormat::Rgba8Unorm | GlFormat::Rgba8Srgb)
            {
                table.mark_exact_buffer_readback(facts.format);
                facts.copy_source = true;
            }
            if facts.renderable {
                facts.copy_source = true;
                facts.copy_destination = true;
            }
        }
        if facts.resource_kind == GlFormatResourceKind::Texture
            && facts.sample_count == 1
            && report.storage_image == ProbeAnswer::Passed
        {
            if let Some(index) = super::super::probes::IMAGE_FORMATS
                .iter()
                .position(|(format, ..)| *format == facts.format)
            {
                if report.ordinary_image_storage[index] == ProbeAnswer::Passed {
                    facts.evidence = GlFormatEvidence::OperationProbed;
                    facts.storage_read = true;
                    facts.storage_write = true;
                }
            }
        }
        table.record(facts).map_err(|error| {
            NativeDiscoveryError::Snapshot(GlDiscoveryError::InvalidFormats(error))
        })
    };
    // RGBA8 sRGB baseline is an unconditional guarantee of every accepted
    // profile. `Rgba8Unorm` is recorded below together with its probed
    // storage-image facts so the table never sees two conflicting records.
    record(GlFormatCapabilities {
        format: GlFormat::Rgba8Srgb,
        resource_kind: GlFormatResourceKind::Texture,
        sample_count: 1,
        evidence: GlFormatEvidence::CoreGuaranteed,
        sampled: true,
        filterable: true,
        renderable: true,
        blendable: true,
        storage_read: false,
        storage_write: false,
        copy_source: true,
        copy_destination: true,
    })?;
    // RGBA8 storage-image facts: recorded when the image load/store probe ran,
    // regardless of outcome, so a failed probe is distinguishable from absent
    // evidence and can never silently enable the capability.
    let rgba8_storage = report.rgba8_storage == ProbeAnswer::Passed;
    let mut rgba8_unorm = rgba8_texture_facts();
    if report.rgba8_storage.ran() {
        rgba8_unorm.evidence = GlFormatEvidence::OperationProbed;
        rgba8_unorm.storage_read = rgba8_storage;
        rgba8_unorm.storage_write = rgba8_storage;
    }
    record(rgba8_unorm)?;
    ordinary::record_ordinary(profile, extensions, &mut record)?;
    // These sized depth/stencil and integer color formats are guaranteed
    // texture/renderbuffer storage in the admitted desktop 4.x and ES 3.x
    // profiles. Renderable textures support framebuffer blits; byte transfers
    // validate their own format/aspect route independently.
    let mut attachment_formats = vec![
        GlFormat::Depth16Unorm,
        GlFormat::Depth24Unorm,
        GlFormat::Depth24PlusStencil8,
        GlFormat::Depth32FloatStencil8,
        GlFormat::Rgba8Uint,
        GlFormat::Rgba8Sint,
    ];
    let stencil_texture = match profile {
        GlFamilyProfile::Desktop { major, minor } => major > 4 || (major == 4 && minor >= 4),
        GlFamilyProfile::Embedded { major, minor } => major > 3 || (major == 3 && minor >= 2),
        GlFamilyProfile::WebGl2 => false,
    };
    if stencil_texture {
        attachment_formats.push(GlFormat::Stencil8);
    }
    for format in attachment_formats {
        for resource_kind in [
            GlFormatResourceKind::Texture,
            GlFormatResourceKind::Renderbuffer,
        ] {
            record(GlFormatCapabilities {
                format,
                resource_kind,
                sample_count: 1,
                evidence: GlFormatEvidence::CoreGuaranteed,
                sampled: resource_kind == GlFormatResourceKind::Texture,
                filterable: false,
                renderable: true,
                blendable: false,
                storage_read: false,
                storage_write: false,
                copy_source: resource_kind == GlFormatResourceKind::Texture,
                copy_destination: resource_kind == GlFormatResourceKind::Texture,
            })?;
        }
    }
    // RGBA8 renderbuffer facts: color-renderable storage for every accepted
    // profile, at single sample and at the portable multisample counts.
    for sample_count in [1] {
        if sample_count > limits.max_samples {
            continue;
        }
        for format in [GlFormat::Rgba8Unorm, GlFormat::Rgba8Srgb] {
            record(GlFormatCapabilities {
                format,
                resource_kind: GlFormatResourceKind::Renderbuffer,
                sample_count,
                evidence: GlFormatEvidence::CoreGuaranteed,
                sampled: false,
                filterable: false,
                renderable: true,
                blendable: true,
                storage_read: false,
                storage_write: false,
                copy_source: false,
                copy_destination: false,
            })?;
        }
    }
    // Float depth facts come only from real attachment probes; without probe
    // evidence renderability stays false instead of an optimistic core claim.
    record(depth_fact(
        GlFormatResourceKind::Texture,
        report.depth_texture_attachment,
    )?)?;
    if report.depth_renderbuffer_attachment.ran() {
        record(depth_fact(
            GlFormatResourceKind::Renderbuffer,
            report.depth_renderbuffer_attachment,
        )?)?;
    }
    // RGBA8 storage-image facts: (recorded above with the unorm baseline)
    // Float color facts are recorded only when their attachment probe ran:
    // a failed probe records `renderable: false` with operation evidence, and
    // an absent probe backend leaves the format out of the table entirely
    // (the contract rejects an unsupported core guarantee for float formats).
    let float32_filterable = match profile {
        // Desktop GL 4.x core lists both float formats as texture-filterable.
        GlFamilyProfile::Desktop { .. } => true,
        // ES 3.x requires OES_texture_float_linear for 32F filtering.
        _ => extensions.is_acquired(GlKnownExtension::OesTextureFloatLinear),
    };
    if report.rgba16f_attachment.ran() {
        record(float_fact(
            GlFormat::Rgba16Float,
            // Half-float textures are texture-filterable in every accepted core.
            true,
            report.rgba16f_attachment,
            // ES forbids blending with float attachments until EXT_float_blend;
            // desktop core permits 16F blending wherever rendering is legal.
            matches!(profile, GlFamilyProfile::Desktop { .. }),
        )?)?;
    }
    if report.rgba32f_attachment.ran() {
        record(float_fact(
            GlFormat::Rgba32Float,
            float32_filterable,
            report.rgba32f_attachment,
            // 32F blending additionally requires EXT_float_blend, whose typed
            // route exists only for the browser profile; native ES stays false.
            matches!(profile, GlFamilyProfile::Desktop { .. }),
        )?)?;
    }
    // Compressed textures are exact format facts, never a family-level flag.
    // GLES 3.x supplies ETC2/EAC in core; all other native families below are
    // admitted only when their matching extension was acquired on this same
    // context.  Allocation and compressedTexImage2D share the token mapping
    // in `provider`, so a positive row is an executable upload route rather
    // than a version-string promise.
    for format in native_optional_compressed_formats() {
        let evidence = if format.is_core_compressed_for(profile) {
            GlFormatEvidence::CoreGuaranteed
        } else if let Some(extension) = native_compressed_extension(format, profile, extensions) {
            GlFormatEvidence::ExtensionAcquired(extension)
        } else {
            continue;
        };
        // `glCopyImageSubData` is a real compressed-to-compressed route only
        // on desktop GL 4.3+. GLES 3.0 and WebGL2 deliberately keep these
        // fields false: neither has this v13 route, and framebuffer copying
        // cannot attach compressed images.
        let direct_copy = (compressed_direct_copy(profile) && report.copy_image_symbols[0])
            || match profile {
                GlFamilyProfile::Desktop { .. } => {
                    report.copy_image_symbols[0]
                        && extensions
                            .raw_reported_names()
                            .any(|name| name == "GL_ARB_copy_image")
                }
                GlFamilyProfile::Embedded { .. } => {
                    (report.copy_image_symbols[1]
                        && extensions
                            .raw_reported_names()
                            .any(|name| name == "GL_EXT_copy_image"))
                        || (report.copy_image_symbols[2]
                            && extensions
                                .raw_reported_names()
                                .any(|name| name == "GL_OES_copy_image"))
                }
                _ => false,
            };
        let exact_read = |format| {
            let read = super::super::probes::EXACT_INTEGER_FORMATS
                .iter()
                .position(|f| *f == format)
                .is_some_and(|i| report.exact_integer_read[i] == ProbeAnswer::Passed);
            let render = super::super::probes::IMAGE_FORMATS
                .iter()
                .position(|(f, ..)| *f == format)
                .is_some_and(|i| report.ordinary_render[i] == ProbeAnswer::Passed);
            read && render
        };
        let block = format.compressed_info().unwrap();
        let integer_route = match block.block_bytes {
            8 => exact_read(GlFormat::Rg32Uint) || exact_read(GlFormat::Rgba16Uint),
            16 => exact_read(GlFormat::Rgba32Uint),
            _ => false,
        };
        let extension_copy = (report.copy_image_symbols[1]
            && extensions
                .raw_reported_names()
                .any(|n| n == "GL_EXT_copy_image"))
            || (report.copy_image_symbols[2]
                && extensions
                    .raw_reported_names()
                    .any(|n| n == "GL_OES_copy_image"));
        let core_compatible = matches!(
            block.family,
            GlCompressedTextureFamily::Etc2
                | GlCompressedTextureFamily::Eac
                | GlCompressedTextureFamily::Astc
        );
        let embedded_read = matches!(profile, GlFamilyProfile::Embedded { .. })
            && integer_route
            && (extension_copy
                || (compressed_direct_copy(profile)
                    && report.copy_image_symbols[0]
                    && core_compatible));
        record(GlFormatCapabilities {
            format,
            resource_kind: GlFormatResourceKind::Texture,
            sample_count: 1,
            evidence,
            sampled: true,
            // GL's compressed formats are sampleable and filterable. They are
            // never FBO attachments or storage images in this contract.
            filterable: true,
            renderable: false,
            blendable: false,
            storage_read: false,
            storage_write: false,
            // Core COPY_SRC covers both image copies and texture-to-buffer.
            // The latter currently uses desktop glGetCompressedTexImage.
            copy_source: direct_copy
                && ((matches!(profile, GlFamilyProfile::Desktop { .. })
                    && report.compressed_get_symbol)
                    || embedded_read),
            copy_destination: true,
        })?;
    }
    drop(record);
    for (index, format) in super::super::probes::MSAA_FORMATS
        .iter()
        .copied()
        .enumerate()
    {
        for (kind, samples) in [
            (
                GlFormatResourceKind::Texture,
                &report.multisample_texture[index],
            ),
            (
                GlFormatResourceKind::MultisampleArray,
                &report.multisample_array[index],
            ),
            (
                GlFormatResourceKind::Renderbuffer,
                &report.multisample_renderbuffer[index],
            ),
        ] {
            for &(sample_count, answer) in samples {
                if answer == ProbeAnswer::ContextLost {
                    return Err(NativeDiscoveryError::QueryFailed(
                        "multisample probe context lost or out of memory",
                    ));
                }
                let Some(single) = table.get(format, 1) else {
                    continue;
                };
                if answer != ProbeAnswer::Passed || !single.renderable {
                    continue;
                }
                table
                    .record(GlFormatCapabilities {
                        format,
                        resource_kind: kind,
                        sample_count,
                        evidence: GlFormatEvidence::OperationProbed,
                        sampled: kind != GlFormatResourceKind::Renderbuffer,
                        filterable: false,
                        renderable: true,
                        blendable: single.blendable,
                        storage_read: false,
                        storage_write: false,
                        copy_source: false,
                        copy_destination: false,
                    })
                    .map_err(|error| {
                        NativeDiscoveryError::Snapshot(GlDiscoveryError::InvalidFormats(error))
                    })?;
            }
        }
    }
    Ok(table)
}

/// Exact optional compressed formats understood by the native storage map.
/// Keeping this list next to native discovery makes additions require review
/// of evidence, allocation and upload together.
fn native_optional_compressed_formats() -> Vec<GlFormat> {
    use crate::backend::gl::api::{GlAstcBlock as B, GlCompressedColorSpace as C};
    let mut formats = vec![
        GlFormat::Bc1RgbUnorm,
        GlFormat::Bc1RgbaUnorm,
        GlFormat::Bc2RgbaUnorm,
        GlFormat::Bc3RgbaUnorm,
        GlFormat::Bc1RgbSrgb,
        GlFormat::Bc1RgbaSrgb,
        GlFormat::Bc2RgbaSrgb,
        GlFormat::Bc3RgbaSrgb,
        GlFormat::Bc4RUnorm,
        GlFormat::Bc4RSnorm,
        GlFormat::Bc5RgUnorm,
        GlFormat::Bc5RgSnorm,
        GlFormat::Bc6hRgbUfloat,
        GlFormat::Bc6hRgbSfloat,
        GlFormat::Bc7RgbaUnorm,
        GlFormat::Bc7RgbaSrgb,
    ];
    for block in [
        B::B4x4,
        B::B5x4,
        B::B5x5,
        B::B6x5,
        B::B6x6,
        B::B8x5,
        B::B8x6,
        B::B8x8,
        B::B10x5,
        B::B10x6,
        B::B10x8,
        B::B10x10,
        B::B12x10,
        B::B12x12,
    ] {
        formats.push(GlFormat::Astc {
            block,
            color_space: C::Linear,
        });
        formats.push(GlFormat::Astc {
            block,
            color_space: C::Srgb,
        });
        formats.push(GlFormat::Astc {
            block,
            color_space: C::Hdr,
        });
    }
    formats.extend([
        GlFormat::Etc2Rgb8Unorm,
        GlFormat::Etc2Rgb8Srgb,
        GlFormat::Etc2Rgba8Unorm,
        GlFormat::Etc2Rgba8Srgb,
        GlFormat::Etc2Rgb8A1Unorm,
        GlFormat::Etc2Rgb8A1Srgb,
        GlFormat::EacR11Unorm,
        GlFormat::EacRg11Unorm,
        GlFormat::EacR11Snorm,
        GlFormat::EacRg11Snorm,
    ]);
    formats
}

fn native_compressed_extension(
    format: GlFormat,
    profile: GlFamilyProfile,
    extensions: &GlExtensionSet,
) -> Option<GlKnownExtension> {
    use crate::backend::gl::api::GlCompressedColorSpace as C;
    let candidates: &[GlKnownExtension] = match format {
        GlFormat::Bc1RgbUnorm
        | GlFormat::Bc1RgbaUnorm
        | GlFormat::Bc2RgbaUnorm
        | GlFormat::Bc3RgbaUnorm => &[GlKnownExtension::ExtTextureCompressionS3tc],
        GlFormat::Bc1RgbSrgb
        | GlFormat::Bc1RgbaSrgb
        | GlFormat::Bc2RgbaSrgb
        | GlFormat::Bc3RgbaSrgb
            if matches!(profile, GlFamilyProfile::Desktop { .. }) =>
        {
            &[
                GlKnownExtension::CompressedTextureS3tcSrgb,
                GlKnownExtension::ExtTextureCompressionS3tc,
            ]
        }
        GlFormat::Bc1RgbSrgb
        | GlFormat::Bc1RgbaSrgb
        | GlFormat::Bc2RgbaSrgb
        | GlFormat::Bc3RgbaSrgb => &[GlKnownExtension::CompressedTextureS3tcSrgb],
        GlFormat::Bc4RUnorm | GlFormat::Bc4RSnorm | GlFormat::Bc5RgUnorm | GlFormat::Bc5RgSnorm => {
            &[GlKnownExtension::ExtTextureCompressionRgtc]
        }
        GlFormat::Bc6hRgbUfloat
        | GlFormat::Bc6hRgbSfloat
        | GlFormat::Bc7RgbaUnorm
        | GlFormat::Bc7RgbaSrgb => &[
            GlKnownExtension::ArbTextureCompressionBptc,
            GlKnownExtension::CompressedTextureBptc,
        ],
        GlFormat::Astc {
            color_space: C::Linear | C::Srgb,
            ..
        } => &[
            GlKnownExtension::KhrTextureCompressionAstcLdr,
            GlKnownExtension::KhrTextureCompressionAstcHdr,
        ],
        GlFormat::Astc {
            color_space: C::Hdr,
            ..
        } => &[GlKnownExtension::KhrTextureCompressionAstcHdr],
        GlFormat::Etc2Rgb8Unorm => &[
            GlKnownExtension::ArbEs3Compatibility,
            GlKnownExtension::OesCompressedEtc2Rgb8Texture,
        ],
        GlFormat::Etc2Rgb8Srgb
        | GlFormat::Etc2Rgba8Unorm
        | GlFormat::Etc2Rgba8Srgb
        | GlFormat::Etc2Rgb8A1Unorm
        | GlFormat::Etc2Rgb8A1Srgb
        | GlFormat::EacR11Unorm
        | GlFormat::EacRg11Unorm
        | GlFormat::EacR11Snorm
        | GlFormat::EacRg11Snorm => &[GlKnownExtension::ArbEs3Compatibility],
        _ => &[],
    };
    candidates
        .iter()
        .copied()
        .find(|extension| extensions.is_acquired(*extension))
}

#[cfg(test)]
mod compressed_extension_tests {
    use super::*;

    #[test]
    fn astc_hdr_extension_also_admits_ldr_formats() {
        use crate::backend::gl::api::{GlAstcBlock, GlCompressedColorSpace};
        let mut extensions = GlExtensionSet::default();
        extensions.report_raw("GL_KHR_texture_compression_astc_hdr");
        assert!(extensions.acquire(GlKnownExtension::KhrTextureCompressionAstcHdr));
        for color_space in [
            GlCompressedColorSpace::Linear,
            GlCompressedColorSpace::Srgb,
            GlCompressedColorSpace::Hdr,
        ] {
            let format = GlFormat::Astc {
                block: GlAstcBlock::B4x4,
                color_space,
            };
            assert_eq!(
                native_compressed_extension(
                    format,
                    GlFamilyProfile::Desktop { major: 4, minor: 1 },
                    &extensions
                ),
                Some(GlKnownExtension::KhrTextureCompressionAstcHdr)
            );
        }
    }

    #[test]
    fn s3tc_srgb_needs_desktop_srgb_or_the_dedicated_extension() {
        let mut extensions = GlExtensionSet::default();
        extensions.report_raw("GL_EXT_texture_compression_s3tc");
        assert!(extensions.acquire(GlKnownExtension::ExtTextureCompressionS3tc));
        let desktop = GlFamilyProfile::Desktop { major: 4, minor: 0 };
        let embedded = GlFamilyProfile::Embedded { major: 3, minor: 0 };
        assert_eq!(
            native_compressed_extension(GlFormat::Bc3RgbaSrgb, desktop, &extensions),
            Some(GlKnownExtension::ExtTextureCompressionS3tc)
        );
        assert_eq!(
            native_compressed_extension(GlFormat::Bc3RgbaSrgb, embedded, &extensions),
            None
        );
    }

    #[test]
    fn compressed_copy_source_requires_readback_entry_point() {
        let limits = GlLimits::unavailable();
        let extensions = GlExtensionSet::default();
        let desktop = GlFamilyProfile::Desktop { major: 4, minor: 6 };
        let embedded = GlFamilyProfile::Embedded { major: 3, minor: 2 };
        let mut report = ProbeReport::default();
        report.copy_image_symbols[0] = true;
        let copy_source = |profile, report: &ProbeReport| {
            native_formats(profile, &extensions, &limits, report)
                .unwrap()
                .get(GlFormat::Etc2Rgb8Unorm, 1)
                .unwrap()
                .copy_source
        };
        assert!(!copy_source(desktop, &report));
        report.compressed_get_symbol = true;
        assert!(copy_source(desktop, &report));
        assert!(!copy_source(embedded, &report));
    }

    #[test]
    fn gles_compressed_copy_requires_compatible_integer_readback() {
        let limits = GlLimits::unavailable();
        let extensions = GlExtensionSet::default();
        let profile = GlFamilyProfile::Embedded { major: 3, minor: 2 };
        let mut report = ProbeReport::default();
        report.copy_image_symbols[0] = true;
        let source = |report: &ProbeReport| {
            native_formats(profile, &extensions, &limits, report)
                .unwrap()
                .get(GlFormat::Etc2Rgb8Unorm, 1)
                .unwrap()
                .copy_source
        };
        assert!(!source(&report));
        let format = GlFormat::Rg32Uint;
        let read = super::super::super::probes::EXACT_INTEGER_FORMATS
            .iter()
            .position(|f| *f == format)
            .unwrap();
        let render = super::super::super::probes::IMAGE_FORMATS
            .iter()
            .position(|(f, ..)| *f == format)
            .unwrap();
        report.exact_integer_read[read] = ProbeAnswer::Passed;
        assert!(!source(&report));
        report.ordinary_render[render] = ProbeAnswer::Passed;
        assert!(source(&report));
        let es31 = GlFamilyProfile::Embedded { major: 3, minor: 1 };
        report.copy_image_symbols[1] = true;
        let mut ext = GlExtensionSet::default();
        let has_source = |ext: &GlExtensionSet| {
            native_formats(es31, ext, &limits, &report)
                .unwrap()
                .get(GlFormat::Etc2Rgb8Unorm, 1)
                .unwrap()
                .copy_source
        };
        assert!(!has_source(&ext));
        ext.report_raw("GL_EXT_copy_image");
        assert!(has_source(&ext));
    }

    #[test]
    fn multisample_facts_require_exact_format_count_and_resource_probe() {
        let mut report = ProbeReport::default();
        let formats = &super::super::super::probes::MSAA_FORMATS;
        let rgba = formats
            .iter()
            .position(|&f| f == GlFormat::Rgba8Unorm)
            .unwrap();
        let integer = formats
            .iter()
            .position(|&f| f == GlFormat::Rgba8Uint)
            .unwrap();
        report.multisample_texture[rgba].push((4, ProbeAnswer::Passed));
        report.multisample_texture[rgba].push((3, ProbeAnswer::Passed));
        report.multisample_renderbuffer[integer].push((4, ProbeAnswer::Passed));
        report.multisample_array[integer].push((4, ProbeAnswer::Passed));
        let mut limits = GlLimits::unavailable();
        limits.max_samples = 16;
        let table = native_formats(
            GlFamilyProfile::Desktop { major: 4, minor: 6 },
            &GlExtensionSet::default(),
            &limits,
            &report,
        )
        .unwrap();
        assert!(table.get(GlFormat::Rgba8Unorm, 4).is_some());
        assert!(table.get(GlFormat::Rgba8Unorm, 3).is_some());
        assert!(table.get(GlFormat::Rgba8Unorm, 2).is_none());
        assert!(table.get(GlFormat::Rgba8Uint, 4).is_none());
        assert!(
            table
                .get_for(GlFormatResourceKind::Renderbuffer, GlFormat::Rgba8Uint, 4)
                .is_some()
        );
        assert!(
            table
                .get_for(
                    GlFormatResourceKind::MultisampleArray,
                    GlFormat::Rgba8Uint,
                    4
                )
                .is_some()
        );
        assert!(
            table
                .get_for(
                    GlFormatResourceKind::MultisampleArray,
                    GlFormat::Rgba8Unorm,
                    4
                )
                .is_none()
        );
        assert!(
            table
                .get_for(GlFormatResourceKind::Renderbuffer, GlFormat::Rgba8Unorm, 4)
                .is_none()
        );
    }
}

/// The only GL-family image-copy command accepted for compressed storage.
/// Framebuffer copying has no compressed attachment route, so GLES 3.0/3.1
/// intentionally publish upload-only compressed facts.
const fn compressed_direct_copy(profile: GlFamilyProfile) -> bool {
    match profile {
        GlFamilyProfile::Desktop { major, minor } => major > 4 || (major == 4 && minor >= 3),
        GlFamilyProfile::Embedded { major, minor } => major > 3 || (major == 3 && minor >= 2),
        GlFamilyProfile::WebGl2 => false,
    }
}

fn rgba8_texture_facts() -> GlFormatCapabilities {
    GlFormatCapabilities {
        format: GlFormat::Rgba8Unorm,
        resource_kind: GlFormatResourceKind::Texture,
        sample_count: 1,
        evidence: GlFormatEvidence::CoreGuaranteed,
        sampled: true,
        filterable: true,
        renderable: true,
        blendable: true,
        storage_read: false,
        storage_write: false,
        copy_source: true,
        copy_destination: true,
    }
}

/// Builds the `Depth32Float` fact for one resource kind from its probe.
///
/// The common contract requires a single-sample depth fact in every snapshot;
/// when no probe evidence exists the record stays conservative (`renderable`
/// false) instead of claiming the profile guarantees float depth rendering.
fn depth_fact(
    resource_kind: GlFormatResourceKind,
    probe: ProbeAnswer,
) -> Result<GlFormatCapabilities, NativeDiscoveryError> {
    let renderable = probe == ProbeAnswer::Passed;
    let evidence = if probe.ran() {
        GlFormatEvidence::OperationProbed
    } else {
        GlFormatEvidence::CoreGuaranteed
    };
    Ok(GlFormatCapabilities {
        format: GlFormat::Depth32Float,
        resource_kind,
        sample_count: 1,
        evidence,
        sampled: resource_kind == GlFormatResourceKind::Texture,
        filterable: false,
        renderable,
        blendable: false,
        storage_read: false,
        storage_write: false,
        copy_source: resource_kind == GlFormatResourceKind::Texture && renderable,
        copy_destination: resource_kind == GlFormatResourceKind::Texture && renderable,
    })
}

/// Builds one float color format fact from its attachment probe. Callers only
/// invoke this when the probe ran, so the evidence is always operation based.
fn float_fact(
    format: GlFormat,
    filterable: bool,
    probe: ProbeAnswer,
    blend_when_renderable: bool,
) -> Result<GlFormatCapabilities, NativeDiscoveryError> {
    let renderable = probe == ProbeAnswer::Passed;
    Ok(GlFormatCapabilities {
        format,
        resource_kind: GlFormatResourceKind::Texture,
        sample_count: 1,
        // Float textures are samplable on every accepted core; the record's
        // evidence marks the concrete attachment probe that proved rendering.
        evidence: GlFormatEvidence::OperationProbed,
        sampled: true,
        filterable,
        renderable,
        blendable: renderable && blend_when_renderable,
        storage_read: false,
        storage_write: false,
        // The current pixel-pack route only reads RGBA8. Rendering evidence
        // alone does not prove that this format can be copied back to a buffer.
        copy_source: false,
        copy_destination: true,
    })
}

pub(super) fn nonnegative(
    value: Option<i64>,
    name: &'static str,
) -> Result<u32, NativeDiscoveryError> {
    value
        .ok_or(NativeDiscoveryError::QueryFailed(name))
        .and_then(|value| to_u32(value, name))
}
pub(super) fn to_u32(value: i64, name: &'static str) -> Result<u32, NativeDiscoveryError> {
    u32::try_from(value).map_err(|_| NativeDiscoveryError::QueryFailed(name))
}

/// Parses only Fluxel's supported native profiles: GL 4.x and GLES 3.x.
pub(crate) fn parse_native_profile(version: &str) -> Option<GlFamilyProfile> {
    let embedded = version.strip_prefix("OpenGL ES ");
    let text = embedded.unwrap_or(version);
    let mut digits = text
        .split(|character: char| !character.is_ascii_digit() && character != '.')
        .find(|part| part.contains('.'))?
        .split('.');
    let major = digits.next()?.parse().ok()?;
    let minor = digits.next()?.parse().ok()?;
    match (embedded.is_some(), major) {
        (false, 4) => Some(GlFamilyProfile::Desktop { major, minor }),
        (true, 3) => Some(GlFamilyProfile::Embedded { major, minor }),
        _ => None,
    }
}

pub(crate) mod glow_const {
    pub const VENDOR: u32 = 0x1F00;
    pub const RENDERER: u32 = 0x1F01;
    pub const VERSION: u32 = 0x1F02;
    pub const EXTENSIONS: u32 = 0x1F03;
    pub const SHADING_LANGUAGE_VERSION: u32 = 0x8B8C;
    pub const NUM_EXTENSIONS: u32 = 0x821D;
    pub const MAX_TEXTURE_SIZE: u32 = 0x0D33;
    pub const MAX_3D_TEXTURE_SIZE: u32 = 0x8073;
    pub const MAX_ARRAY_TEXTURE_LAYERS: u32 = 0x88FF;
    pub const MAX_CUBE_MAP_TEXTURE_SIZE: u32 = 0x851C;
    pub const MAX_RENDERBUFFER_SIZE: u32 = 0x84E8;
    pub const MAX_COLOR_ATTACHMENTS: u32 = 0x8CDF;
    pub const MAX_DRAW_BUFFERS: u32 = 0x8824;
    pub const MAX_VERTEX_ATTRIBS: u32 = 0x8869;
    pub const MAX_VIEWPORT_DIMS: u32 = 0x0D3A;
    pub const MAX_VIEWPORTS: u32 = 0x825B;
    pub const MAX_VERTEX_TEXTURE_IMAGE_UNITS: u32 = 0x8B4C;
    pub const MAX_TEXTURE_IMAGE_UNITS: u32 = 0x8872;
    pub const MAX_COMBINED_TEXTURE_IMAGE_UNITS: u32 = 0x8B4D;
    pub const MAX_UNIFORM_BUFFER_BINDINGS: u32 = 0x8A2F;
    pub const MAX_UNIFORM_BLOCK_SIZE: u32 = 0x8A30;
    pub const UNIFORM_BUFFER_OFFSET_ALIGNMENT: u32 = 0x8A34;
    pub const MAX_VERTEX_UNIFORM_BLOCKS: u32 = 0x8A2B;
    pub const MAX_FRAGMENT_UNIFORM_BLOCKS: u32 = 0x8A2D;
    pub const MAX_COMPUTE_UNIFORM_BLOCKS: u32 = 0x91BB;
    pub const MAX_COMBINED_UNIFORM_BLOCKS: u32 = 0x8A2E;
    pub const MAX_SHADER_STORAGE_BUFFER_BINDINGS: u32 = 0x90DD;
    pub const MAX_SHADER_STORAGE_BLOCK_SIZE: u32 = 0x90DE;
    pub const SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT: u32 = 0x90DF;
    pub const MAX_VERTEX_SHADER_STORAGE_BLOCKS: u32 = 0x90D6;
    pub const MAX_FRAGMENT_SHADER_STORAGE_BLOCKS: u32 = 0x90DA;
    pub const MAX_COMPUTE_SHADER_STORAGE_BLOCKS: u32 = 0x90DB;
    pub const MAX_COMBINED_SHADER_STORAGE_BLOCKS: u32 = 0x90DC;
    pub const MAX_IMAGE_UNITS: u32 = 0x8F38;
    pub const MAX_COMBINED_IMAGE_UNIFORMS: u32 = 0x90CF;
    pub const MAX_SAMPLES: u32 = 0x8D57;
    pub const MAX_COLOR_TEXTURE_SAMPLES: u32 = 0x910E;
    pub const MAX_DEPTH_TEXTURE_SAMPLES: u32 = 0x910F;
    pub const MAX_INTEGER_SAMPLES: u32 = 0x9110;
    /// Views one attachment may serve in one pass, as defined by the multiview
    /// extension family (the same registry value WebGL2 exposes).
    pub const MAX_MULTIVIEW_VIEWS: u32 = 0x9631;
    pub const MAX_COMPUTE_WORK_GROUP_COUNT: u32 = 0x91BE;
    pub const MAX_COMPUTE_WORK_GROUP_SIZE: u32 = 0x91BF;
    pub const MAX_COMPUTE_WORK_GROUP_INVOCATIONS: u32 = 0x90EB;
    pub const MAX_TEXTURE_MAX_ANISOTROPY_EXT: u32 = 0x84FF;
    /// The framebuffer bound for drawing, which tells the drawable queries
    /// below whether they would answer about the surface at all.
    pub const DRAW_FRAMEBUFFER_BINDING: u32 = 0x8CA6;
    /// The draw framebuffer itself, as the target an attachment query is asked
    /// about. The same object the binding query above reads the identity of.
    pub const DRAW_FRAMEBUFFER: u32 = 0x8CA9;
    pub const SAMPLE_BUFFERS: u32 = 0x80A8;
    pub const SAMPLES: u32 = 0x80A9;
    /// The drawable's colour buffers and its depth and stencil attachments.
    ///
    /// The component widths are read from an attachment rather than from the
    /// default-framebuffer `GL_*_BITS` queries this table used to carry: those
    /// were deprecated in 3.0 and removed from the core profile in 3.1, so on
    /// the core profile this crate accepts they answer `GL_INVALID_ENUM`, and
    /// `glGetFramebufferAttachmentParameteriv` is the question the profile
    /// still answers. The samples pair above is unaffected -- the core profile
    /// kept it, which is visible on the hardware this was found on, where the
    /// six bit queries failed and these two answered.
    ///
    /// Values cross-checked against two independent bindings, because a wrong
    /// token here is a question asked about nothing: `glow` 0.18 and `web-sys`
    /// 0.3 agree on all six sizes and on the object type, and the two colour
    /// buffers are consistent with the `FRONT`/`BACK` values the latter records
    /// (`GL_FRONT` is `0x0404`, so `GL_FRONT_LEFT`/`GL_BACK_LEFT` are the two
    /// entries below it).
    pub const BACK: u32 = 0x0405;
    pub const BACK_LEFT: u32 = 0x0402;
    pub const FRONT_LEFT: u32 = 0x0400;
    pub const DEPTH: u32 = 0x1801;
    pub const STENCIL: u32 = 0x1802;
    pub const FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE: u32 = 0x8CD0;
    pub const FRAMEBUFFER_ATTACHMENT_RED_SIZE: u32 = 0x8212;
    pub const FRAMEBUFFER_ATTACHMENT_GREEN_SIZE: u32 = 0x8213;
    pub const FRAMEBUFFER_ATTACHMENT_BLUE_SIZE: u32 = 0x8214;
    pub const FRAMEBUFFER_ATTACHMENT_ALPHA_SIZE: u32 = 0x8215;
    pub const FRAMEBUFFER_ATTACHMENT_DEPTH_SIZE: u32 = 0x8216;
    pub const FRAMEBUFFER_ATTACHMENT_STENCIL_SIZE: u32 = 0x8217;
}
