//! Native GL format facts and profile parsing.

use super::*;

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
            let unorm_read = matches!(profile, GlFamilyProfile::Embedded { .. })
                && super::super::probes::EXACT_UNORM_FORMATS
                    .iter()
                    .position(|format| *format == facts.format)
                    .is_some_and(|i| report.exact_unorm_read[i] == ProbeAnswer::Passed);
            let index = super::super::probes::EXACT_INTEGER_FORMATS
                .iter()
                .position(|format| *format == facts.format);
            if index.is_some_and(|i| report.exact_integer_read[i] == ProbeAnswer::Passed)
                || unorm_read
                || matches!(facts.format, GlFormat::Rgba8Unorm | GlFormat::Rgba8Srgb)
            {
                table.mark_exact_buffer_readback(facts.format);
                facts.copy_source = true;
            }
            if facts.format == GlFormat::Rgb9e5Ufloat
                && matches!(profile, GlFamilyProfile::Desktop { .. })
                && (compressed_direct_copy(profile)
                    || extensions
                        .raw_reported_names()
                        .any(|name| name == "GL_ARB_texture_view"))
                && super::super::probes::EXACT_INTEGER_FORMATS
                    .iter()
                    .position(|f| *f == GlFormat::R32Uint)
                    .is_some_and(|i| report.exact_integer_read[i] == ProbeAnswer::Passed)
                && report.copy_image_symbols[0]
                && (compressed_direct_copy(profile)
                    || extensions
                        .raw_reported_names()
                        .any(|name| name == "GL_ARB_copy_image"))
            {
                facts.copy_source = true;
            }
            if ((report.texture_sub_image_read_symbol
                && super::super::exec_texture_sub_image_readback::direct_read_supported(
                    profile, extensions,
                ))
                || (report.tex_image_read_symbol
                    && matches!(profile, GlFamilyProfile::Desktop { .. })))
                && super::super::exec_texture_sub_image_readback::direct_color_format(facts.format)
            {
                facts.copy_source = true;
            }
            if facts.renderable {
                facts.copy_source = true;
                facts.copy_destination = true;
            }
            if gles_exact_alias_route(profile, extensions, report, facts.format) {
                table.mark_exact_alias_readback(facts.format);
                facts.copy_source = true;
            }
            // SNORM's two encodings of -1 require image-class bit copies;
            // ordinary typed pixel reads cannot preserve both endpoints.
            if facts.format != GlFormat::Rgb9e5Ufloat
                && matches!(profile, GlFamilyProfile::Desktop { .. })
            {
                if let Some(alias) =
                    super::super::exec_rgb9e5_readback::exact_integer_alias(facts.format)
                {
                    let alias_probe = super::super::probes::EXACT_INTEGER_FORMATS
                        .iter()
                        .position(|format| *format == alias)
                        .is_some_and(|i| report.exact_integer_read[i] == ProbeAnswer::Passed);
                    let copy_image = report.copy_image_symbols[0]
                        && (compressed_direct_copy(profile)
                            || extensions
                                .raw_reported_names()
                                .any(|name| name == "GL_ARB_copy_image"));
                    let alias_read = alias_probe || report.tex_image_read_symbol
                    || (report.texture_sub_image_read_symbol && super::super::exec_texture_sub_image_readback::direct_read_supported(profile, extensions));
                    facts.copy_source = copy_image && alias_read;
                }
            } else if facts.format != GlFormat::Rgb9e5Ufloat
                && super::super::exec_rgb9e5_readback::exact_integer_alias(facts.format).is_some()
            {
                facts.copy_source =
                    unorm_read || gles_exact_alias_route(profile, extensions, report, facts.format);
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
    let stencil_evidence = match profile {
        GlFamilyProfile::Desktop { major, minor } if major > 4 || (major == 4 && minor >= 4) => {
            Some(GlFormatEvidence::CoreGuaranteed)
        }
        GlFamilyProfile::Embedded { major, minor } if major > 3 || (major == 3 && minor >= 2) => {
            Some(GlFormatEvidence::CoreGuaranteed)
        }
        GlFamilyProfile::Desktop { .. }
            if extensions.is_acquired(GlKnownExtension::ArbTextureStencil8) =>
        {
            Some(GlFormatEvidence::ExtensionAcquired(
                GlKnownExtension::ArbTextureStencil8,
            ))
        }
        GlFamilyProfile::Embedded { major, minor }
            if (major > 3 || (major == 3 && minor >= 1))
                && extensions.is_acquired(GlKnownExtension::OesTextureStencil8) =>
        {
            Some(GlFormatEvidence::ExtensionAcquired(
                GlKnownExtension::OesTextureStencil8,
            ))
        }
        _ => None,
    };
    if stencil_evidence.is_some() {
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
                evidence: if format == GlFormat::Stencil8 {
                    stencil_evidence.expect("stencil fact gated")
                } else {
                    GlFormatEvidence::CoreGuaranteed
                },
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
        // Compressed texture copies require a core or extension image-copy
        // entry for this profile. Framebuffer copying cannot attach these
        // formats; raw block buffer reads have a separate per-format probe.
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
            // The image-copy route remains useful even when this particular
            // format's compressed-to-integer readback probe fails.
            copy_source: direct_copy
                && (matches!(profile, GlFamilyProfile::Embedded { .. })
                    || report.compressed_get_symbol),
            copy_destination: true,
        })?;
    }
    drop(record);
    for (&format, &(alias, answer)) in &report.compressed_alias_read {
        if answer == ProbeAnswer::Passed {
            table.mark_compressed_alias_readback(format, alias);
        }
    }
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
                        // Desktop GL 4.3+ (or ARB_copy_image) can copy a
                        // matching multisample *color* texture image directly.
                        // This remains a narrow image-copy fact: it neither
                        // admits pixel uploads/readback nor applies to
                        // renderbuffers or selected depth/stencil planes.
                        copy_source: format
                            .supports_blit_aspect(crate::backend::gl::api::GlTextureAspect::Color)
                            && kind != GlFormatResourceKind::Renderbuffer
                            && desktop_msaa_copy_image(profile, extensions, report),
                        copy_destination: format
                            .supports_blit_aspect(crate::backend::gl::api::GlTextureAspect::Color)
                            && kind != GlFormatResourceKind::Renderbuffer
                            && desktop_msaa_copy_image(profile, extensions, report),
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
pub(super) fn native_optional_compressed_formats() -> Vec<GlFormat> {
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

pub(super) fn native_compressed_extension(
    format: GlFormat,
    profile: GlFamilyProfile,
    extensions: &GlExtensionSet,
) -> Option<GlKnownExtension> {
    use crate::backend::gl::api::GlCompressedColorSpace as C;
    let candidates: &[GlKnownExtension] = match format {
        GlFormat::Bc1RgbUnorm | GlFormat::Bc1RgbaUnorm => &[
            GlKnownExtension::ExtTextureCompressionS3tc,
            GlKnownExtension::ExtTextureCompressionDxt1,
        ],
        GlFormat::Bc2RgbaUnorm => &[
            GlKnownExtension::ExtTextureCompressionS3tc,
            GlKnownExtension::AngleTextureCompressionDxt3,
        ],
        GlFormat::Bc3RgbaUnorm => &[
            GlKnownExtension::ExtTextureCompressionS3tc,
            GlKnownExtension::AngleTextureCompressionDxt5,
        ],
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
        | GlFormat::Bc3RgbaSrgb => {
            let base = match format {
                GlFormat::Bc1RgbSrgb | GlFormat::Bc1RgbaSrgb => {
                    native_compressed_extension(GlFormat::Bc1RgbUnorm, profile, extensions)
                }
                GlFormat::Bc2RgbaSrgb => {
                    native_compressed_extension(GlFormat::Bc2RgbaUnorm, profile, extensions)
                }
                GlFormat::Bc3RgbaSrgb => {
                    native_compressed_extension(GlFormat::Bc3RgbaUnorm, profile, extensions)
                }
                _ => unreachable!(),
            };
            if base.is_some()
                && extensions.is_acquired(GlKnownExtension::ExtTextureCompressionS3tcSrgb)
            {
                return Some(GlKnownExtension::ExtTextureCompressionS3tcSrgb);
            }
            return None;
        }
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
#[path = "formats/tests.rs"]
mod compressed_extension_tests;

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

/// The Desktop-only image-copy route used for equal-sample MSAA copies.
///
/// GLES has a separately selected core/EXT/OES route; do not accidentally
/// advertise it while this first implementation is limited to Desktop GL.
fn desktop_msaa_copy_image(
    profile: GlFamilyProfile,
    extensions: &GlExtensionSet,
    report: &ProbeReport,
) -> bool {
    let is_desktop = matches!(profile, GlFamilyProfile::Desktop { .. });
    let core = matches!(profile, GlFamilyProfile::Desktop { major, minor } if major > 4 || (major == 4 && minor >= 3));
    let arb = extensions
        .raw_reported_names()
        .any(|name| name == "GL_ARB_copy_image");
    is_desktop && report.copy_image_symbols[0] && (core || arb)
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

#[path = "glow_const.rs"]
pub(crate) mod glow_const;

/// A GLES raw-storage copy requires both a callable image-copy entry point and
/// a successful exact integer readback probe for the compatible image class.
fn gles_exact_alias_route(
    profile: GlFamilyProfile,
    extensions: &GlExtensionSet,
    report: &ProbeReport,
    format: GlFormat,
) -> bool {
    let GlFamilyProfile::Embedded { major, minor } = profile else {
        return false;
    };
    let named = |name| extensions.raw_reported_names().any(|n| n == name);
    let copy = ((major, minor) >= (3, 2) && report.copy_image_symbols[0])
        || (named("GL_EXT_copy_image") && report.copy_image_symbols[1])
        || (named("GL_OES_copy_image") && report.copy_image_symbols[2]);
    // EXT_texture_norm16 amends the EXT compatibility tables, not the OES
    // copy-image table. An advertised but missing EXT entry is not enough.
    let norm16 = matches!(
        format,
        GlFormat::R16Unorm
            | GlFormat::R16Snorm
            | GlFormat::Rg16Unorm
            | GlFormat::Rg16Snorm
            | GlFormat::Rgba16Unorm
            | GlFormat::Rgba16Snorm
    );
    if norm16
        && !named("GL_EXT_texture_view")
        && !(named("GL_EXT_copy_image") && report.copy_image_symbols[1])
    {
        return false;
    }
    let alias = super::super::exec_rgb9e5_readback::gles_exact_integer_alias(format, extensions);
    copy && alias
        .and_then(|a| {
            super::super::probes::EXACT_INTEGER_FORMATS
                .iter()
                .position(|f| *f == a)
        })
        .is_some_and(|i| report.exact_integer_read[i] == ProbeAnswer::Passed)
}
