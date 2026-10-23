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
    assert!(!available(
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
