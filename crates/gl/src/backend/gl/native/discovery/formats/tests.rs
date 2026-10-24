//! Native exact-format discovery and copy-route gate tests.
use super::*;

#[test]
fn rgb9e5_copy_source_needs_compatible_image_copy_and_integer_readback() {
    let limits = GlLimits::unavailable();
    let mut extensions = GlExtensionSet::default();
    let mut report = ProbeReport::default();
    let available = |profile, extensions: &GlExtensionSet, report: &ProbeReport| {
        native_formats(profile, extensions, &limits, report)
            .unwrap()
            .get(GlFormat::Rgb9e5Ufloat, 1)
            .unwrap()
            .copy_source
    };
    let core = GlFamilyProfile::Desktop { major: 4, minor: 6 };
    assert!(!available(core, &extensions, &report));
    report.copy_image_symbols[0] = true;
    assert!(!available(core, &extensions, &report));
    let index = crate::backend::gl::native::probes::EXACT_INTEGER_FORMATS
        .iter()
        .position(|f| *f == GlFormat::R32Uint)
        .unwrap();
    report.exact_integer_read[index] = ProbeAnswer::Passed;
    assert!(available(core, &extensions, &report));
    let old = GlFamilyProfile::Desktop { major: 4, minor: 2 };
    assert!(!available(old, &extensions, &report));
    extensions.report_raw("GL_ARB_copy_image");
    assert!(!available(old, &extensions, &report));
    extensions.report_raw("GL_ARB_texture_view");
    assert!(available(old, &extensions, &report));
    assert!(available(
        GlFamilyProfile::Embedded { major: 3, minor: 2 },
        &extensions,
        &report
    ));
}

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
fn gles_partial_s3tc_extensions_admit_only_their_own_formats() {
    let profile = GlFamilyProfile::Embedded { major: 3, minor: 1 };
    let mut dxt1 = GlExtensionSet::default();
    dxt1.report_raw("GL_EXT_texture_compression_dxt1");
    assert!(dxt1.acquire(GlKnownExtension::ExtTextureCompressionDxt1));
    assert_eq!(
        native_compressed_extension(GlFormat::Bc1RgbaUnorm, profile, &dxt1),
        Some(GlKnownExtension::ExtTextureCompressionDxt1)
    );
    assert_eq!(
        native_compressed_extension(GlFormat::Bc2RgbaUnorm, profile, &dxt1),
        None
    );
    assert_eq!(
        native_compressed_extension(GlFormat::Bc3RgbaUnorm, profile, &dxt1),
        None
    );

    let mut dxt35 = GlExtensionSet::default();
    dxt35.report_raw("GL_ANGLE_texture_compression_dxt3");
    dxt35.report_raw("GL_ANGLE_texture_compression_dxt5");
    assert!(dxt35.acquire(GlKnownExtension::AngleTextureCompressionDxt3));
    assert!(dxt35.acquire(GlKnownExtension::AngleTextureCompressionDxt5));
    assert_eq!(
        native_compressed_extension(GlFormat::Bc2RgbaUnorm, profile, &dxt35),
        Some(GlKnownExtension::AngleTextureCompressionDxt3)
    );
    assert_eq!(
        native_compressed_extension(GlFormat::Bc3RgbaUnorm, profile, &dxt35),
        Some(GlKnownExtension::AngleTextureCompressionDxt5)
    );

    dxt1.report_raw("GL_EXT_texture_compression_s3tc_srgb");
    assert!(dxt1.acquire(GlKnownExtension::ExtTextureCompressionS3tcSrgb));
    assert_eq!(
        native_compressed_extension(GlFormat::Bc1RgbaSrgb, profile, &dxt1),
        Some(GlKnownExtension::ExtTextureCompressionS3tcSrgb)
    );
    assert_eq!(
        native_compressed_extension(GlFormat::Bc2RgbaSrgb, profile, &dxt1),
        None
    );
}

#[test]
fn desktop_compressed_copy_source_requires_readback_entry_point() {
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
    assert!(copy_source(embedded, &report));
}

#[test]
fn gles_compressed_copy_and_readback_have_independent_facts() {
    let limits = GlLimits::unavailable();
    let extensions = GlExtensionSet::default();
    let profile = GlFamilyProfile::Embedded { major: 3, minor: 2 };
    let mut report = ProbeReport::default();
    report.copy_image_symbols[0] = true;
    let table =
        |report: &ProbeReport| native_formats(profile, &extensions, &limits, report).unwrap();
    let format = GlFormat::Etc2Rgb8Unorm;
    assert!(table(&report).get(format, 1).unwrap().copy_source);
    assert_eq!(table(&report).compressed_alias_readback(format), None);
    report
        .compressed_alias_read
        .insert(format, (GlFormat::Rg32Uint, ProbeAnswer::Failed));
    assert!(table(&report).get(format, 1).unwrap().copy_source);
    assert_eq!(table(&report).compressed_alias_readback(format), None);
    report
        .compressed_alias_read
        .insert(format, (GlFormat::Rgba16Uint, ProbeAnswer::Passed));
    assert_eq!(
        table(&report).compressed_alias_readback(format),
        Some(GlFormat::Rgba16Uint)
    );
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

#[test]
fn msaa_color_copy_facts_require_direct_image_and_exact_probes() {
    let limits = GlLimits::unavailable();
    let mut extensions = GlExtensionSet::default();
    let mut report = ProbeReport::default();
    let rgba = super::super::super::probes::MSAA_FORMATS
        .iter()
        .position(|&format| format == GlFormat::Rgba8Unorm)
        .unwrap();
    let table = |profile, extensions: &GlExtensionSet, report: &ProbeReport| {
        native_formats(profile, extensions, &limits, report).unwrap()
    };
    let texture_facts = |profile, extensions: &GlExtensionSet, report: &ProbeReport| {
        table(profile, extensions, report)
            .get_for(GlFormatResourceKind::Texture, GlFormat::Rgba8Unorm, 4)
            .unwrap()
    };

    report.multisample_texture[rgba].push((4, ProbeAnswer::Passed));
    let desktop = GlFamilyProfile::Desktop { major: 4, minor: 3 };
    let facts = texture_facts(desktop, &extensions, &report);
    assert!(!facts.copy_source);
    assert!(!facts.copy_destination);

    report.copy_image_symbols[0] = true;
    let facts = texture_facts(desktop, &extensions, &report);
    assert!(facts.copy_source);
    assert!(facts.copy_destination);

    // A fact for one exact format/resource/count cannot stand in for an
    // unprobed format or multisample-array allocation.
    let facts = table(desktop, &extensions, &report);
    assert!(
        facts
            .get_for(GlFormatResourceKind::Texture, GlFormat::Rgba8Uint, 4)
            .is_none()
    );
    assert!(
        facts
            .get_for(
                GlFormatResourceKind::MultisampleArray,
                GlFormat::Rgba8Unorm,
                4
            )
            .is_none()
    );

    report.multisample_array[rgba].push((4, ProbeAnswer::Passed));
    let array_facts = table(desktop, &extensions, &report)
        .get_for(
            GlFormatResourceKind::MultisampleArray,
            GlFormat::Rgba8Unorm,
            4,
        )
        .unwrap();
    assert!(array_facts.copy_source);
    assert!(array_facts.copy_destination);

    // GLES 3.2 gains the core direct-image route only with the loaded symbol.
    let embedded32 = texture_facts(
        GlFamilyProfile::Embedded { major: 3, minor: 2 },
        &extensions,
        &report,
    );
    assert!(embedded32.copy_source);
    assert!(embedded32.copy_destination);

    // GLES 3.1 must name an extension and load its matching ABI entry point.
    let embedded31 = GlFamilyProfile::Embedded { major: 3, minor: 1 };
    let mut ext = GlExtensionSet::default();
    let facts = texture_facts(embedded31, &ext, &report);
    assert!(!facts.copy_source);
    ext.report_raw("GL_EXT_copy_image");
    let facts = texture_facts(embedded31, &ext, &report);
    assert!(!facts.copy_source);
    report.copy_image_symbols[1] = true;
    let facts = texture_facts(embedded31, &ext, &report);
    assert!(facts.copy_source);
    assert!(facts.copy_destination);

    // OES uses its separate loaded entry and remains unavailable to GLES 3.0.
    let mut oes = GlExtensionSet::default();
    oes.report_raw("GL_OES_copy_image");
    let mut oes_report = report.clone();
    oes_report.copy_image_symbols[1] = false;
    let facts = texture_facts(embedded31, &oes, &oes_report);
    assert!(!facts.copy_source);
    oes_report.copy_image_symbols[2] = true;
    let facts = texture_facts(embedded31, &oes, &oes_report);
    assert!(facts.copy_source);
    assert!(facts.copy_destination);
    let facts = texture_facts(
        GlFamilyProfile::Embedded { major: 3, minor: 0 },
        &oes,
        &oes_report,
    );
    assert!(!facts.copy_source);
    assert!(!facts.copy_destination);

    // WebGL2 exposes no corresponding direct image-copy operation.
    let webgl = texture_facts(GlFamilyProfile::WebGl2, &extensions, &report);
    assert!(!webgl.copy_source);
    assert!(!webgl.copy_destination);

    // Older Desktop contexts need the advertised extension as well as the
    // loaded ABI entry point.
    let old_desktop = GlFamilyProfile::Desktop { major: 4, minor: 2 };
    let facts = texture_facts(old_desktop, &extensions, &report);
    assert!(!facts.copy_source);
    extensions.report_raw("GL_ARB_copy_image");
    let facts = texture_facts(old_desktop, &extensions, &report);
    assert!(facts.copy_source);
    assert!(facts.copy_destination);
}

#[test]
fn direct_color_copy_source_needs_profile_extension_and_loaded_entry() {
    let limits = GlLimits::unavailable();
    let mut extensions = GlExtensionSet::default();
    let mut report = ProbeReport::default();
    let source = |profile, extensions: &GlExtensionSet, report: &ProbeReport| {
        let facts = native_formats(profile, extensions, &limits, report)
            .unwrap()
            .get_for(GlFormatResourceKind::Texture, GlFormat::R16Float, 1)
            .unwrap();
        assert!(!facts.renderable);
        facts.copy_source
    };
    let modern = GlFamilyProfile::Desktop { major: 4, minor: 5 };
    let old = GlFamilyProfile::Desktop { major: 4, minor: 4 };
    let es = GlFamilyProfile::Embedded { major: 3, minor: 2 };
    assert!(!source(modern, &extensions, &report));
    report.texture_sub_image_read_symbol = true;
    assert!(source(modern, &extensions, &report));
    assert!(!source(old, &extensions, &report));
    extensions.report_raw("GL_ARB_get_texture_sub_image");
    assert!(source(old, &extensions, &report));
    assert!(!source(es, &extensions, &report));
    report.texture_sub_image_read_symbol = false;
    report.tex_image_read_symbol = true;
    assert!(source(old, &GlExtensionSet::default(), &report));
    assert!(!source(es, &extensions, &report));
}

#[test]
fn snorm_copy_source_requires_bit_preserving_copy_and_integer_readback() {
    let limits = GlLimits::unavailable();
    let mut report = ProbeReport::default();
    let extensions = GlExtensionSet::default();
    let profile = GlFamilyProfile::Desktop { major: 4, minor: 6 };
    let source = |report: &ProbeReport, format| {
        native_formats(profile, &extensions, &limits, report)
            .unwrap()
            .get_for(GlFormatResourceKind::Texture, format, 1)
            .unwrap()
            .copy_source
    };
    report.tex_image_read_symbol = true;
    for format in [
        GlFormat::R8Snorm,
        GlFormat::Rg8Snorm,
        GlFormat::Rgba8Snorm,
        GlFormat::R16Snorm,
        GlFormat::Rg16Snorm,
        GlFormat::Rgba16Snorm,
    ] {
        assert!(!source(&report, format));
        report.copy_image_symbols[0] = true;
        assert!(source(&report, format));
        report.copy_image_symbols[0] = false;
    }
}

#[test]
fn gles_renderable_snorm_does_not_advertise_unimplemented_exact_readback() {
    let mut extensions = GlExtensionSet::default();
    extensions.report_raw("GL_EXT_render_snorm");
    let mut report = ProbeReport::default();
    let index = crate::backend::gl::native::probes::IMAGE_FORMATS
        .iter()
        .position(|(format, ..)| *format == GlFormat::R8Snorm)
        .unwrap();
    report.ordinary_render[index] = ProbeAnswer::Passed;
    let formats = native_formats(
        GlFamilyProfile::Embedded { major: 3, minor: 1 },
        &extensions,
        &GlLimits::unavailable(),
        &report,
    )
    .unwrap();
    let facts = formats
        .get_for(GlFormatResourceKind::Texture, GlFormat::R8Snorm, 1)
        .unwrap();
    assert!(facts.renderable);
    assert!(!facts.copy_source);
}

#[test]
fn gles_raw_alias_requires_matching_copy_entry_and_integer_probe() {
    let es31 = GlFamilyProfile::Embedded { major: 3, minor: 1 };
    let es32 = GlFamilyProfile::Embedded { major: 3, minor: 2 };
    let mut ext = GlExtensionSet::default();
    let mut report = ProbeReport::default();
    let index = super::super::super::probes::EXACT_INTEGER_FORMATS
        .iter()
        .position(|f| *f == GlFormat::R32Uint)
        .unwrap();
    report.exact_integer_read[index] = ProbeAnswer::Passed;
    assert!(!gles_exact_alias_route(
        es31,
        &ext,
        &report,
        GlFormat::Rgb9e5Ufloat
    ));
    ext.report_raw("GL_EXT_copy_image");
    assert!(!gles_exact_alias_route(
        es31,
        &ext,
        &report,
        GlFormat::Rgb9e5Ufloat
    ));
    report.copy_image_symbols[1] = true;
    assert!(gles_exact_alias_route(
        es31,
        &ext,
        &report,
        GlFormat::Rgb9e5Ufloat
    ));
    report.exact_integer_read[index] = ProbeAnswer::Failed;
    assert!(!gles_exact_alias_route(
        es31,
        &ext,
        &report,
        GlFormat::Rgb9e5Ufloat
    ));
    report.exact_integer_read[index] = ProbeAnswer::Passed;
    let no_ext = GlExtensionSet::default();
    assert!(!gles_exact_alias_route(
        es32,
        &no_ext,
        &report,
        GlFormat::Rgb9e5Ufloat
    ));
    report.copy_image_symbols[0] = true;
    assert!(gles_exact_alias_route(
        es32,
        &no_ext,
        &report,
        GlFormat::Rgb9e5Ufloat
    ));
    let mut oes = GlExtensionSet::default();
    oes.report_raw("GL_OES_copy_image");
    report.copy_image_symbols[2] = true;
    assert!(gles_exact_alias_route(
        es31,
        &oes,
        &report,
        GlFormat::Rgb9e5Ufloat
    ));
}

#[test]
fn stencil_texture_requires_core_version_or_profile_legal_extension() {
    let limits = GlLimits::unavailable();
    let report = ProbeReport::default();
    let es30 = GlFamilyProfile::Embedded { major: 3, minor: 0 };
    let es31 = GlFamilyProfile::Embedded { major: 3, minor: 1 };
    let es32 = GlFamilyProfile::Embedded { major: 3, minor: 2 };
    let gl43 = GlFamilyProfile::Desktop { major: 4, minor: 3 };
    let mut ext = GlExtensionSet::default();
    let available = |profile, ext: &GlExtensionSet| {
        let table = native_formats(profile, ext, &limits, &report).unwrap();
        table.validate_evidence(profile, ext).unwrap();
        table
            .get_for(GlFormatResourceKind::Texture, GlFormat::Stencil8, 1)
            .is_some()
    };
    assert!(!available(es31, &ext));
    assert!(available(es32, &ext));
    assert!(!available(gl43, &ext));
    ext.report_raw("GL_OES_texture_stencil8");
    assert!(ext.acquire(GlKnownExtension::OesTextureStencil8));
    assert!(available(es31, &ext));
    assert!(!available(es30, &ext));
    assert!(!available(gl43, &ext));
    ext.report_raw("GL_ARB_texture_stencil8");
    assert!(ext.acquire(GlKnownExtension::ArbTextureStencil8));
    assert!(available(gl43, &ext));
}
