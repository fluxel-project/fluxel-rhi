//! Native GL format facts and profile parsing.

use super::*;

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
    let mut record = |facts: GlFormatCapabilities| {
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
    // RGBA8 renderbuffer facts: color-renderable storage for every accepted
    // profile, at single sample and at the portable multisample counts.
    for sample_count in [1, 4, 8] {
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
    // Multisample texture facts: the same color and depth formats that are
    // guaranteed renderable single-sample storage, recorded up to the ceiling
    // GL records for their class. A count above that ceiling, or a format
    // outside the three, gets no fact at all, which is what makes an
    // unsupported request fail closed at allocation instead of reaching the
    // driver as a plausible-looking allocation that the driver then refuses.
    if supports_multisample_texture_storage(profile) {
        for (format, class) in [
            (GlFormat::Rgba8Unorm, TextureSampleClass::Color),
            (GlFormat::Rgba8Srgb, TextureSampleClass::Color),
            (GlFormat::Depth32Float, TextureSampleClass::DepthStencil),
        ] {
            let ceiling = texture_sample_ceiling(class, limits);
            for sample_count in [2, 4, 8, 16] {
                if sample_count > ceiling {
                    continue;
                }
                record(GlFormatCapabilities {
                    format,
                    resource_kind: GlFormatResourceKind::Texture,
                    sample_count,
                    evidence: GlFormatEvidence::CoreGuaranteed,
                    // Multisample storage is never read by a sampler and never
                    // filtered: it is read by a resolve, which is a framebuffer
                    // blit between whole attachments. The copy fields stay false
                    // for the same reason -- the shared copy word is
                    // single-sample, and claiming it here would describe an
                    // operation that no path in this layer performs.
                    sampled: false,
                    filterable: false,
                    renderable: true,
                    blendable: class == TextureSampleClass::Color,
                    storage_read: false,
                    storage_write: false,
                    copy_source: false,
                    copy_destination: false,
                })?;
            }
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
    for format in [
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
    ] {
        if format.is_core_compressed_for(profile) {
            let direct_copy = compressed_direct_copy(profile);
            record(GlFormatCapabilities {
                format,
                resource_kind: GlFormatResourceKind::Texture,
                sample_count: 1,
                evidence: GlFormatEvidence::CoreGuaranteed,
                sampled: true,
                filterable: false,
                renderable: false,
                blendable: false,
                storage_read: false,
                storage_write: false,
                copy_source: direct_copy,
                copy_destination: direct_copy,
            })?;
        }
    }
    for format in native_optional_compressed_formats() {
        if format.is_core_compressed_for(profile) {
            continue;
        }
        let Some(extension) = native_compressed_extension(format, extensions) else {
            continue;
        };
        // `glCopyImageSubData` is a real compressed-to-compressed route only
        // on desktop GL 4.3+. GLES 3.0 and WebGL2 deliberately keep these
        // fields false: neither has this v13 route, and framebuffer copying
        // cannot attach compressed images.
        let direct_copy = compressed_direct_copy(profile);
        record(GlFormatCapabilities {
            format,
            resource_kind: GlFormatResourceKind::Texture,
            sample_count: 1,
            evidence: GlFormatEvidence::ExtensionAcquired(extension),
            sampled: true,
            // GL's compressed formats are sampleable and filterable. They are
            // never FBO attachments or storage images in this contract.
            filterable: true,
            renderable: false,
            blendable: false,
            storage_read: false,
            storage_write: false,
            copy_source: direct_copy,
            copy_destination: direct_copy,
        })?;
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
        | GlFormat::Bc3RgbaSrgb => &[GlKnownExtension::CompressedTextureS3tcSrgb],
        GlFormat::Bc4RUnorm | GlFormat::Bc4RSnorm | GlFormat::Bc5RgUnorm | GlFormat::Bc5RgSnorm => {
            &[GlKnownExtension::ExtTextureCompressionRgtc]
        }
        GlFormat::Bc6hRgbUfloat
        | GlFormat::Bc6hRgbSfloat
        | GlFormat::Bc7RgbaUnorm
        | GlFormat::Bc7RgbaSrgb => &[GlKnownExtension::ArbTextureCompressionBptc],
        GlFormat::Astc {
            color_space: C::Linear | C::Srgb,
            ..
        } => &[GlKnownExtension::KhrTextureCompressionAstcLdr],
        GlFormat::Astc {
            color_space: C::Hdr,
            ..
        } => &[GlKnownExtension::KhrTextureCompressionAstcHdr],
        GlFormat::Etc2Rgb8Unorm
        | GlFormat::Etc2Rgb8Srgb
        | GlFormat::Etc2Rgba8Unorm
        | GlFormat::Etc2Rgba8Srgb
        | GlFormat::Etc2Rgb8A1Unorm
        | GlFormat::Etc2Rgb8A1Srgb
        | GlFormat::EacR11Unorm
        | GlFormat::EacRg11Unorm
        | GlFormat::EacR11Snorm
        | GlFormat::EacRg11Snorm => &[
            GlKnownExtension::ArbEs3Compatibility,
            GlKnownExtension::OesCompressedEtc2Rgb8Texture,
        ],
        _ => &[],
    };
    candidates
        .iter()
        .copied()
        .find(|extension| extensions.is_acquired(*extension))
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
        copy_source: false,
        copy_destination: false,
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
        copy_source: renderable,
        copy_destination: renderable,
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
    pub const MAX_MULTIVIEW_VIEWS: u32 = 0x9632;
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
