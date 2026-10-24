//! Native operation probes: the capability-resolution executor.
//!
//! Discovery treats a version string as evidence of an available *route*, never
//! as proof that a capability works. This module runs small, fully owned GL
//! operations (compile/link a trivial program, bind a scratch buffer or image,
//! issue a zero-effect dispatch/draw, attach a scratch texture or renderbuffer
//! to a scratch framebuffer) and records the structured outcome. Every probe
//! uses scratch objects that are unbound and deleted on every return path,
//! consumes the GL error it produced, and never performs wall-clock or
//! heuristic inference: `Unavailable` (the probe could not run), `Failed` (the
//! driver answered or errored), and `Passed` are the only outcomes.
//!
//! Probe sources are minimal GLSL/ESSL programs written here from the public
//! GL/GLES specifications; nothing outside Fluxel contracts and the public
//! registry semantics is consulted.

use crate::backend::gl::api::{GlExtensionSet, GlFamilyProfile, GlKnownExtension};

/// The `glGetQueryiv` entry-point shape glow 0.18 does not bind.
///
/// The pointer is resolved by the same Host loader that built the `glow`
/// context and may only be called while that exact context is current, which
/// discovery guarantees for the whole probe.
pub(crate) type GetQueryivFn = unsafe extern "system" fn(u32, u32, *mut i32);

/// Outcome of one operation probe.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) enum ProbeAnswer {
    /// The probe could not run (entry point absent, scratch allocation failed).
    #[default]
    Unavailable,
    /// The driver executed the operation and answered a structured negative.
    Failed,
    /// The operation completed without error.
    Passed,
    /// The context was lost while probing; discovery must abort.
    ContextLost,
}

impl ProbeAnswer {
    /// Whether the driver actually answered the probe.
    pub(super) const fn ran(self) -> bool {
        !matches!(self, Self::Unavailable)
    }
    /// Converts one probe answer into the discovery ledger's probe record.
    pub(super) const fn to_operation_probe(self) -> crate::backend::gl::api::GlOperationProbe {
        match self {
            Self::Unavailable => crate::backend::gl::api::GlOperationProbe::NotRun,
            Self::Failed | Self::ContextLost => crate::backend::gl::api::GlOperationProbe::Failed,
            Self::Passed => crate::backend::gl::api::GlOperationProbe::Passed,
        }
    }
}

/// Complete set of operation-probe outcomes for one context.
///
/// Every field starts `Unavailable`, so a probe backend that cannot run leaves
/// all optional capabilities fail-closed instead of silently enabled.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ProbeReport {
    pub multiview: ProbeAnswer,
    pub depth_texture_attachment: ProbeAnswer,
    pub depth_renderbuffer_attachment: ProbeAnswer,
    pub rgba16f_attachment: ProbeAnswer,
    pub rgba32f_attachment: ProbeAnswer,
    pub rgba8_storage: ProbeAnswer,
    pub ordinary_image_storage: [ProbeAnswer; IMAGE_FORMATS.len()],
    pub ordinary_render: [ProbeAnswer; IMAGE_FORMATS.len()],
    pub exact_integer_read: [ProbeAnswer; EXACT_INTEGER_FORMATS.len()],
    pub exact_unorm_read: [ProbeAnswer; EXACT_UNORM_FORMATS.len()],
    pub compressed_alias_read: std::collections::BTreeMap<
        crate::backend::gl::api::GlFormat,
        (crate::backend::gl::api::GlFormat, ProbeAnswer),
    >,
    pub multisample_texture: [Vec<(u32, ProbeAnswer)>; MSAA_FORMATS.len()],
    pub multisample_array: [Vec<(u32, ProbeAnswer)>; MSAA_FORMATS.len()],
    pub multisample_renderbuffer: [Vec<(u32, ProbeAnswer)>; MSAA_FORMATS.len()],
    pub compute: ProbeAnswer,
    pub storage_buffer: ProbeAnswer,
    pub storage_image: ProbeAnswer,
    pub indirect_draw: ProbeAnswer,
    pub indirect_dispatch: ProbeAnswer,
    pub multi_draw_indirect: ProbeAnswer,
    /// `glGetQueryiv(GL_TIMESTAMP, GL_QUERY_COUNTER_BITS)`; `None` when the
    /// driver rejected the query or the loader did not supply the entry point.
    pub query_counter_bits: Option<u32>,
    pub copy_image_symbols: [bool; 3],
    pub compressed_get_symbol: bool,
    pub texture_sub_image_read_symbol: bool,
    pub tex_image_read_symbol: bool,
}

impl Default for ProbeReport {
    fn default() -> Self {
        Self {
            multiview: ProbeAnswer::Unavailable,
            depth_texture_attachment: ProbeAnswer::Unavailable,
            depth_renderbuffer_attachment: ProbeAnswer::Unavailable,
            rgba16f_attachment: ProbeAnswer::Unavailable,
            rgba32f_attachment: ProbeAnswer::Unavailable,
            rgba8_storage: ProbeAnswer::Unavailable,
            ordinary_image_storage: [ProbeAnswer::Unavailable; IMAGE_FORMATS.len()],
            ordinary_render: [ProbeAnswer::Unavailable; IMAGE_FORMATS.len()],
            exact_integer_read: [ProbeAnswer::Unavailable; EXACT_INTEGER_FORMATS.len()],
            exact_unorm_read: [ProbeAnswer::Unavailable; EXACT_UNORM_FORMATS.len()],
            compressed_alias_read: Default::default(),
            multisample_texture: std::array::from_fn(|_| Vec::new()),
            multisample_array: std::array::from_fn(|_| Vec::new()),
            multisample_renderbuffer: std::array::from_fn(|_| Vec::new()),
            compute: ProbeAnswer::Unavailable,
            storage_buffer: ProbeAnswer::Unavailable,
            storage_image: ProbeAnswer::Unavailable,
            indirect_draw: ProbeAnswer::Unavailable,
            indirect_dispatch: ProbeAnswer::Unavailable,
            multi_draw_indirect: ProbeAnswer::Unavailable,
            query_counter_bits: None,
            copy_image_symbols: [false; 3],
            compressed_get_symbol: false,
            texture_sub_image_read_symbol: false,
            tex_image_read_symbol: false,
        }
    }
}

pub(super) const EXACT_UNORM_FORMATS: [crate::backend::gl::api::GlFormat; 6] = {
    use crate::backend::gl::api::GlFormat as F;
    [
        F::R8Unorm,
        F::Rg8Unorm,
        F::R16Unorm,
        F::Rg16Unorm,
        F::Rgba16Unorm,
        F::Rgb10a2Unorm,
    ]
};

pub(super) const EXACT_INTEGER_FORMATS: [crate::backend::gl::api::GlFormat; 19] = {
    use crate::backend::gl::api::GlFormat as F;
    [
        F::R8Uint,
        F::R8Sint,
        F::Rg8Uint,
        F::Rg8Sint,
        F::Rgba8Uint,
        F::Rgba8Sint,
        F::R16Uint,
        F::R16Sint,
        F::Rg16Uint,
        F::Rg16Sint,
        F::Rgba16Uint,
        F::Rgba16Sint,
        F::R32Uint,
        F::R32Sint,
        F::Rg32Uint,
        F::Rg32Sint,
        F::Rgba32Uint,
        F::Rgba32Sint,
        F::Rgb10a2Uint,
    ]
};

/// Sized image formats exposed by the common shader contract.
pub(super) const IMAGE_FORMATS: [(crate::backend::gl::api::GlFormat, u32, &str, &str); 38] = {
    use crate::backend::gl::api::GlFormat as F;
    [
        (F::R32Uint, glow::R32UI, "r32ui", "uimage2D"),
        (F::R32Sint, glow::R32I, "r32i", "iimage2D"),
        (F::R32Float, glow::R32F, "r32f", "image2D"),
        (F::Rg32Uint, glow::RG32UI, "rg32ui", "uimage2D"),
        (F::Rg32Sint, glow::RG32I, "rg32i", "iimage2D"),
        (F::Rg32Float, glow::RG32F, "rg32f", "image2D"),
        (F::Rgba32Uint, glow::RGBA32UI, "rgba32ui", "uimage2D"),
        (F::Rgba32Sint, glow::RGBA32I, "rgba32i", "iimage2D"),
        (F::Rgba32Float, glow::RGBA32F, "rgba32f", "image2D"),
        (F::Rgba16Uint, glow::RGBA16UI, "rgba16ui", "uimage2D"),
        (F::Rgba16Sint, glow::RGBA16I, "rgba16i", "iimage2D"),
        (F::Rgba16Float, glow::RGBA16F, "rgba16f", "image2D"),
        (F::Rgba8Uint, glow::RGBA8UI, "rgba8ui", "uimage2D"),
        (F::Rgba8Sint, glow::RGBA8I, "rgba8i", "iimage2D"),
        (F::Rgba8Snorm, glow::RGBA8_SNORM, "rgba8_snorm", "image2D"),
        (F::Rgba16Unorm, glow::RGBA16, "rgba16", "image2D"),
        (
            F::Rgba16Snorm,
            glow::RGBA16_SNORM,
            "rgba16_snorm",
            "image2D",
        ),
        (F::R8Unorm, glow::R8, "r8", "image2D"),
        (F::R8Snorm, glow::R8_SNORM, "r8_snorm", "image2D"),
        (F::R8Uint, glow::R8UI, "r8ui", "uimage2D"),
        (F::R8Sint, glow::R8I, "r8i", "iimage2D"),
        (F::Rg8Unorm, glow::RG8, "rg8", "image2D"),
        (F::Rg8Snorm, glow::RG8_SNORM, "rg8_snorm", "image2D"),
        (F::Rg8Uint, glow::RG8UI, "rg8ui", "uimage2D"),
        (F::Rg8Sint, glow::RG8I, "rg8i", "iimage2D"),
        (F::R16Unorm, glow::R16, "r16", "image2D"),
        (F::R16Snorm, glow::R16_SNORM, "r16_snorm", "image2D"),
        (F::R16Uint, glow::R16UI, "r16ui", "uimage2D"),
        (F::R16Sint, glow::R16I, "r16i", "iimage2D"),
        (F::R16Float, glow::R16F, "r16f", "image2D"),
        (F::Rg16Unorm, glow::RG16, "rg16", "image2D"),
        (F::Rg16Snorm, glow::RG16_SNORM, "rg16_snorm", "image2D"),
        (F::Rg16Uint, glow::RG16UI, "rg16ui", "uimage2D"),
        (F::Rg16Sint, glow::RG16I, "rg16i", "iimage2D"),
        (F::Rg16Float, glow::RG16F, "rg16f", "image2D"),
        (F::Rgb10a2Uint, glow::RGB10_A2UI, "rgb10_a2ui", "uimage2D"),
        (F::Rgb10a2Unorm, glow::RGB10_A2, "rgb10_a2", "image2D"),
        (
            F::Rg11b10Ufloat,
            glow::R11F_G11F_B10F,
            "r11f_g11f_b10f",
            "image2D",
        ),
    ]
};

pub(super) const MSAA_FORMATS: [crate::backend::gl::api::GlFormat; 46] = {
    use crate::backend::gl::api::GlFormat as F;
    let mut result = [F::Rgba8Unorm; 46];
    let mut i = 0;
    while i < IMAGE_FORMATS.len() {
        result[i] = IMAGE_FORMATS[i].0;
        i += 1;
    }
    result[38] = F::Rgba8Unorm;
    result[39] = F::Rgba8Srgb;
    result[40] = F::Depth16Unorm;
    result[41] = F::Depth24Unorm;
    result[42] = F::Depth32Float;
    result[43] = F::Depth24PlusStencil8;
    result[44] = F::Depth32FloatStencil8;
    result[45] = F::Stencil8;
    result
};

impl ProbeReport {
    /// Guards the optional-capability set on a trivial raster program actually
    /// linking: a context that cannot link has no executable shader domains.
    fn with_trivial_link_guard(
        mut self,
        probes: &impl NativeGlProbes,
        stages: &[(u32, &'static str)],
    ) -> Self {
        if probes.links_trivial_program(stages) != ProbeAnswer::Passed {
            self.compute = ProbeAnswer::Unavailable;
            self.storage_buffer = ProbeAnswer::Unavailable;
            self.storage_image = ProbeAnswer::Unavailable;
            self.indirect_draw = ProbeAnswer::Unavailable;
            self.indirect_dispatch = ProbeAnswer::Unavailable;
            self.multi_draw_indirect = ProbeAnswer::Unavailable;
        }
        self
    }
}

/// Scratch-object operation probes a native discovery backend can execute.
///
/// Default implementations are deliberately fail-closed: an implementation
/// that does not provide a probe keeps the related capability `NotRun`.
pub(super) trait NativeGlProbes {
    fn compressed_alias_read(
        &self,
        _format: crate::backend::gl::api::GlFormat,
        _integer: crate::backend::gl::api::GlFormat,
        _entry: usize,
    ) -> ProbeAnswer {
        ProbeAnswer::Unavailable
    }

    fn copy_image_symbols(&self) -> [bool; 3] {
        [false; 3]
    }
    fn tex_image_read_symbol(&self) -> bool {
        false
    }
    fn texture_sub_image_read_symbol(&self) -> bool {
        false
    }
    fn compressed_get_symbol(&self) -> bool {
        false
    }
    fn multiview_attachment(&self) -> ProbeAnswer {
        ProbeAnswer::Unavailable
    }
    /// Reads `GL_QUERY_COUNTER_BITS` for the timestamp query target.
    fn query_counter_bits(&self) -> Option<u32> {
        None
    }
    /// Attaches a freshly stored texture of one internal format to a scratch
    /// framebuffer at the named attachment and reports completeness.
    fn attachment_completes(
        &self,
        internal_format: u32,
        upload_format: u32,
        upload_type: u32,
        attachment: u32,
    ) -> ProbeAnswer {
        let _ = (internal_format, upload_format, upload_type, attachment);
        ProbeAnswer::Unavailable
    }
    /// Allocates renderbuffer storage and reports the completeness of one
    /// framebuffer attachment naming it. `samples == 1` is plain storage.
    fn renderbuffer_attachment_completes(&self, internal_format: u32, samples: u32) -> ProbeAnswer {
        let _ = (internal_format, samples);
        ProbeAnswer::Unavailable
    }
    /// Compiles and links one program from the given `(stage, source)` pairs.
    fn links_trivial_program(&self, stages: &[(u32, &'static str)]) -> ProbeAnswer {
        let _ = stages;
        ProbeAnswer::Unavailable
    }
    /// Links one compute program and issues a single no-op work group.
    fn dispatches_compute(&self, source: &'static str) -> ProbeAnswer {
        let _ = source;
        ProbeAnswer::Unavailable
    }
    /// Compiles a program declaring one SSBO and binds a scratch buffer at
    /// indexed binding point zero.
    fn binds_shader_storage(&self, source: &'static str) -> ProbeAnswer {
        let _ = source;
        ProbeAnswer::Unavailable
    }
    /// Compiles a program with image load/store uniforms and binds scratch
    /// textures at two image units.
    fn binds_image_load_store(&self, source: &'static str) -> ProbeAnswer {
        let _ = source;
        ProbeAnswer::Unavailable
    }
    fn binds_image_format(
        &self,
        _format: u32,
        _qualifier: &str,
        _image_type: &str,
        _desktop: bool,
    ) -> ProbeAnswer {
        ProbeAnswer::Unavailable
    }
    fn reads_integer_format(
        &self,
        _format: u32,
        _external: u32,
        _kind: u32,
        _bytes: u32,
    ) -> ProbeAnswer {
        ProbeAnswer::Unavailable
    }
    fn reads_unorm_format(&self, _format: crate::backend::gl::api::GlFormat) -> ProbeAnswer {
        ProbeAnswer::Unavailable
    }
    fn multisample_texture_attachment(
        &self,
        _format: u32,
        _attachment: u32,
        _samples: u32,
        _immutable: bool,
    ) -> ProbeAnswer {
        ProbeAnswer::Unavailable
    }
    fn multisample_array_attachment(
        &self,
        _format: u32,
        _attachment: u32,
        _samples: u32,
        _profile: GlFamilyProfile,
        _extensions: &GlExtensionSet,
    ) -> ProbeAnswer {
        ProbeAnswer::Unavailable
    }
    fn multisample_renderbuffer_attachment(
        &self,
        _format: u32,
        _attachment: u32,
        _samples: u32,
    ) -> ProbeAnswer {
        ProbeAnswer::Unavailable
    }
    /// Issues one zero-primitive indirect draw through scratch state.
    fn issues_indirect_draw(&self, indexed: bool, stages: &[(u32, &'static str)]) -> ProbeAnswer {
        let _ = (indexed, stages);
        ProbeAnswer::Unavailable
    }
    /// Issues two zero-primitive commands through one typed multi-draw entry
    /// point. The executor uses the same entry point, so a missing loader or a
    /// rejected batch remains unavailable instead of being inferred from the
    /// single-draw route.
    fn issues_multi_draw_indirect(
        &self,
        indexed: bool,
        stages: &[(u32, &'static str)],
    ) -> ProbeAnswer {
        let _ = (indexed, stages);
        ProbeAnswer::Unavailable
    }
    /// Issues one no-op indirect dispatch through scratch state.
    fn issues_indirect_dispatch(&self, compute_source: &'static str) -> ProbeAnswer {
        let _ = compute_source;
        ProbeAnswer::Unavailable
    }
}

/// Minimal GL 4.0 raster pair; later features have their own shader probes.
const DESKTOP_VERTEX: &str = "#version 400\nvoid main() {}\n";
const DESKTOP_FRAGMENT: &str = "#version 400\nvoid main() {}\n";
const DESKTOP_IMAGE_VERTEX: &str = "#version 420\nvoid main() {}\n";
const DESKTOP_STORAGE_VERTEX: &str = "#version 430\nvoid main() {}\n";
/// Minimal `#version 310 es` pair with the same property.
const ES_VERTEX: &str = "#version 310 es\nvoid main() {}\n";
const ES_FRAGMENT: &str = "#version 310 es\nvoid main() {}\n";
/// Compute probes declare one 1x1x1 work group and an empty main.
const DESKTOP_COMPUTE: &str = "#version 430\nlayout(local_size_x = 1) in;\nvoid main() {}\n";
const ES_COMPUTE: &str = "#version 310 es\nlayout(local_size_x = 1) in;\nvoid main() {}\n";
/// SSBO probes reference one `std430` block bound at index 0.
const DESKTOP_STORAGE: &str = concat!(
    "#version 430\n",
    "layout(std430, binding = 0) buffer ProbeBlock { float probe_value; };\n",
    "void main() { probe_value = 0.0; }\n",
);
const ES_STORAGE: &str = concat!(
    "#version 310 es\n",
    "layout(std430, binding = 0) buffer ProbeBlock { mediump float probe_value; };\n",
    "void main() { probe_value = 0.0; }\n",
);
/// Image probes load from one unit and store to another with explicit layout
/// bindings, so the probe never needs uniform locations.
const DESKTOP_IMAGE: &str = concat!(
    "#version 420\n",
    "layout(rgba8, binding = 0) readonly uniform image2D probe_source;\n",
    "layout(rgba8, binding = 1) writeonly uniform image2D probe_target;\n",
    "void main() {\n",
    "  imageStore(probe_target, ivec2(0), imageLoad(probe_source, ivec2(0)));\n",
    "}\n",
);
const ES_IMAGE: &str = concat!(
    "#version 310 es\n",
    "layout(rgba8, binding = 0) readonly uniform highp image2D probe_source;\n",
    "layout(rgba8, binding = 1) writeonly uniform highp image2D probe_target;\n",
    "void main() {\n",
    "  imageStore(probe_target, ivec2(0), imageLoad(probe_source, ivec2(0)));\n",
    "}\n",
);

/// Whether the profile's core version (or an acquired desktop extension)
/// supplies the exact route for one probe.
fn route_supplied(
    profile: GlFamilyProfile,
    extensions: &GlExtensionSet,
    desktop_core: Option<(u8, u8)>,
    embedded_core: Option<(u8, u8)>,
    desktop_extension: Option<GlKnownExtension>,
) -> bool {
    match profile {
        GlFamilyProfile::Desktop { major, minor } => {
            desktop_core.is_some_and(|(required_major, required_minor)| {
                major > required_major || (major == required_major && minor >= required_minor)
            }) || desktop_extension.is_some_and(|extension| extensions.is_acquired(extension))
        }
        GlFamilyProfile::Embedded { major, minor } => {
            embedded_core.is_some_and(|(required_major, required_minor)| {
                major > required_major || (major == required_major && minor >= required_minor)
            })
        }
        GlFamilyProfile::WebGl2 => false,
    }
}

/// Runs every capability-relevant probe the selected route can reach.
///
/// Probes that would call entry points outside the context's core version or
/// acquired extension set stay `Unavailable`: calling a driver export that the
/// context does not promise is undefined behavior, so the route gate is
/// evaluated before each probe instead of trusting the loader.
pub(super) fn run_operation_probes(
    probes: &impl NativeGlProbes,
    profile: GlFamilyProfile,
    extensions: &GlExtensionSet,
    max_samples: u32,
) -> ProbeReport {
    let desktop = matches!(profile, GlFamilyProfile::Desktop { .. });
    let trivial: &[(u32, &'static str)] = if desktop {
        &[
            (glow::VERTEX_SHADER, DESKTOP_VERTEX),
            (glow::FRAGMENT_SHADER, DESKTOP_FRAGMENT),
        ]
    } else {
        &[
            (glow::VERTEX_SHADER, ES_VERTEX),
            (glow::FRAGMENT_SHADER, ES_FRAGMENT),
        ]
    };
    let compute = if desktop { DESKTOP_COMPUTE } else { ES_COMPUTE };
    let storage = if desktop { DESKTOP_STORAGE } else { ES_STORAGE };
    let image = if desktop { DESKTOP_IMAGE } else { ES_IMAGE };

    let image_route = route_supplied(
        profile,
        extensions,
        Some((4, 2)),
        Some((3, 1)),
        Some(GlKnownExtension::ArbShaderImageLoadStore),
    );
    let compute_route = route_supplied(
        profile,
        extensions,
        Some((4, 3)),
        Some((3, 1)),
        Some(GlKnownExtension::ArbComputeShader),
    );
    let storage_route = route_supplied(
        profile,
        extensions,
        Some((4, 3)),
        Some((3, 1)),
        Some(GlKnownExtension::ArbShaderStorageBufferObject),
    );
    let indirect_route = route_supplied(profile, extensions, Some((4, 0)), Some((3, 1)), None);
    let indirect_dispatch_route =
        route_supplied(profile, extensions, Some((4, 3)), Some((3, 1)), None);
    let multi_draw_indirect_route = route_supplied(
        profile,
        extensions,
        Some((4, 3)),
        None,
        Some(GlKnownExtension::ArbMultiDrawIndirect),
    );

    ProbeReport {
        compressed_alias_read: Default::default(),
        multiview: if extensions
            .raw_reported_names()
            .any(|name| name == "GL_OVR_multiview2")
        {
            probes.multiview_attachment()
        } else {
            ProbeAnswer::Unavailable
        },
        // Depth and float attachment facts are answered by real framebuffer
        // completeness in every native profile.
        depth_texture_attachment: probes.attachment_completes(
            glow::DEPTH_COMPONENT32F,
            glow::DEPTH_COMPONENT,
            glow::FLOAT,
            glow::DEPTH_ATTACHMENT,
        ),
        depth_renderbuffer_attachment: probes
            .renderbuffer_attachment_completes(glow::DEPTH_COMPONENT32F, 1),
        rgba16f_attachment: probes.attachment_completes(
            glow::RGBA16F,
            glow::RGBA,
            glow::HALF_FLOAT,
            glow::COLOR_ATTACHMENT0,
        ),
        rgba32f_attachment: probes.attachment_completes(
            glow::RGBA32F,
            glow::RGBA,
            glow::FLOAT,
            glow::COLOR_ATTACHMENT0,
        ),
        rgba8_storage: if image_route {
            probes.binds_image_load_store(image)
        } else {
            ProbeAnswer::Unavailable
        },
        ordinary_image_storage: if image_route {
            IMAGE_FORMATS.map(|(_, format, qualifier, image_type)| {
                probes.binds_image_format(format, qualifier, image_type, desktop)
            })
        } else {
            [ProbeAnswer::Unavailable; IMAGE_FORMATS.len()]
        },
        ordinary_render: IMAGE_FORMATS.map(|(format, internal, ..)| {
            match super::exec_copy::upload_encoding(format) {
                Some((external, kind)) => {
                    probes.attachment_completes(internal, external, kind, glow::COLOR_ATTACHMENT0)
                }
                None => ProbeAnswer::Unavailable,
            }
        }),
        exact_integer_read: EXACT_INTEGER_FORMATS.map(|format| {
            match (
                super::resource::native_texture_format(format),
                super::exec_copy::upload_encoding(format),
                super::exec_copy::upload_pixel_format(format),
            ) {
                (Some(internal), Some((external, kind)), Some(pixel)) => {
                    probes.reads_integer_format(internal, external, kind, pixel.bytes_per_pixel())
                }
                _ => ProbeAnswer::Unavailable,
            }
        }),
        exact_unorm_read: EXACT_UNORM_FORMATS.map(|format| {
            if matches!(profile, GlFamilyProfile::Embedded { .. }) {
                probes.reads_unorm_format(format)
            } else {
                ProbeAnswer::Unavailable
            }
        }),
        multisample_texture: if super::discovery::supports_multisample_texture_storage(profile) {
            MSAA_FORMATS.map(|format| {
                (2..=max_samples)
                    .map(|samples| {
                        let Some(internal) = super::resource::native_texture_format(format) else {
                            return (samples, ProbeAnswer::Unavailable);
                        };
                        let attachment = match format {
                            crate::backend::gl::api::GlFormat::Depth16Unorm
                            | crate::backend::gl::api::GlFormat::Depth24Unorm
                            | crate::backend::gl::api::GlFormat::Depth32Float => {
                                glow::DEPTH_ATTACHMENT
                            }
                            crate::backend::gl::api::GlFormat::Depth24PlusStencil8
                            | crate::backend::gl::api::GlFormat::Depth32FloatStencil8 => {
                                glow::DEPTH_STENCIL_ATTACHMENT
                            }
                            crate::backend::gl::api::GlFormat::Stencil8 => glow::STENCIL_ATTACHMENT,
                            _ => glow::COLOR_ATTACHMENT0,
                        };
                        let immutable = profile.meets(
                            Some(crate::backend::gl::api::GlVersion::new(4, 3)),
                            Some(crate::backend::gl::api::GlVersion::new(3, 1)),
                        );
                        (
                            samples,
                            probes.multisample_texture_attachment(
                                internal, attachment, samples, immutable,
                            ),
                        )
                    })
                    .collect()
            })
        } else {
            std::array::from_fn(|_| Vec::new())
        },
        multisample_array: MSAA_FORMATS.map(|format| {
            (2..=max_samples)
                .map(|samples| {
                    let Some(internal) = super::resource::native_texture_format(format) else {
                        return (samples, ProbeAnswer::Unavailable);
                    };
                    let attachment = match format {
                        crate::backend::gl::api::GlFormat::Depth16Unorm
                        | crate::backend::gl::api::GlFormat::Depth24Unorm
                        | crate::backend::gl::api::GlFormat::Depth32Float => glow::DEPTH_ATTACHMENT,
                        crate::backend::gl::api::GlFormat::Depth24PlusStencil8
                        | crate::backend::gl::api::GlFormat::Depth32FloatStencil8 => {
                            glow::DEPTH_STENCIL_ATTACHMENT
                        }
                        crate::backend::gl::api::GlFormat::Stencil8 => glow::STENCIL_ATTACHMENT,
                        _ => glow::COLOR_ATTACHMENT0,
                    };
                    (
                        samples,
                        probes.multisample_array_attachment(
                            internal, attachment, samples, profile, extensions,
                        ),
                    )
                })
                .collect()
        }),
        multisample_renderbuffer: MSAA_FORMATS.map(|format| {
            (2..=max_samples)
                .map(|samples| {
                    let Some(internal) = super::resource::native_texture_format(format) else {
                        return (samples, ProbeAnswer::Unavailable);
                    };
                    let attachment = match format {
                        crate::backend::gl::api::GlFormat::Depth16Unorm
                        | crate::backend::gl::api::GlFormat::Depth24Unorm
                        | crate::backend::gl::api::GlFormat::Depth32Float => glow::DEPTH_ATTACHMENT,
                        crate::backend::gl::api::GlFormat::Depth24PlusStencil8
                        | crate::backend::gl::api::GlFormat::Depth32FloatStencil8 => {
                            glow::DEPTH_STENCIL_ATTACHMENT
                        }
                        crate::backend::gl::api::GlFormat::Stencil8 => glow::STENCIL_ATTACHMENT,
                        _ => glow::COLOR_ATTACHMENT0,
                    };
                    (
                        samples,
                        probes.multisample_renderbuffer_attachment(internal, attachment, samples),
                    )
                })
                .collect()
        }),
        compute: if compute_route {
            probes.dispatches_compute(compute)
        } else {
            ProbeAnswer::Unavailable
        },
        storage_buffer: if storage_route {
            probes.binds_shader_storage(storage)
        } else {
            ProbeAnswer::Unavailable
        },
        // The image-load/store probe is the storage-image evidence; the route
        // gate is shared with the RGBA8 storage facts.
        storage_image: if image_route {
            probes.binds_image_load_store(image)
        } else {
            ProbeAnswer::Unavailable
        },
        indirect_draw: if indirect_route {
            let non_indexed = probes.issues_indirect_draw(false, trivial);
            if non_indexed != ProbeAnswer::Passed {
                non_indexed
            } else {
                probes.issues_indirect_draw(true, trivial)
            }
        } else {
            ProbeAnswer::Unavailable
        },
        indirect_dispatch: if indirect_dispatch_route {
            probes.issues_indirect_dispatch(compute)
        } else {
            ProbeAnswer::Unavailable
        },
        multi_draw_indirect: if multi_draw_indirect_route {
            let non_indexed = probes.issues_multi_draw_indirect(false, trivial);
            if non_indexed != ProbeAnswer::Passed {
                non_indexed
            } else {
                probes.issues_multi_draw_indirect(true, trivial)
            }
        } else {
            ProbeAnswer::Unavailable
        },
        query_counter_bits: probes.query_counter_bits(),
        copy_image_symbols: probes.copy_image_symbols(),
        compressed_get_symbol: probes.compressed_get_symbol(),
        texture_sub_image_read_symbol: probes.texture_sub_image_read_symbol(),
        tex_image_read_symbol: probes.tex_image_read_symbol(),
    }
    .with_trivial_link_guard(probes, trivial)
}

/// Advances the extension ledger with the extensions whose operation probe
/// really ran on this context and answered success.
///
/// Every pairing below is an extension that exposes exactly the operation its
/// probe performs, so `Probed` here means "the driver executed this operation
/// and no error was raised" rather than "the version string promised it". Two
/// gates keep that claim honest:
///
/// - The extension must already be acquired. A name that only appears in the
///   extension string is not a route, so no probe result can promote it.
/// - The probe must be `Passed`. `Unavailable` and `Failed` leave the entry at
///   whatever acquisition reached: this deliberately does not call `fail()`,
///   because "the probe did not run" is not "the extension was refused", and a
///   missing probe must not erase a real acquisition.
///
/// A probe that ran through the core version rather than the extension route
/// still proves the operation on this context, and resolution prefers the core
/// route anyway, so the entry advances either way.
///
/// Extensions with no command probe are absent by construction: the anisotropic
/// filter name is read as a limit, the timer-query route is proved by the
/// counter-width observation rather than by a command, and multiview has no
/// native command probe on this path.
pub(super) fn record_extension_probes(
    profile: GlFamilyProfile,
    report: &ProbeReport,
    extensions: &mut GlExtensionSet,
) {
    for (extension, answer) in [
        (GlKnownExtension::ArbComputeShader, report.compute),
        (
            GlKnownExtension::ArbShaderStorageBufferObject,
            report.storage_buffer,
        ),
        (
            GlKnownExtension::ArbShaderImageLoadStore,
            report.storage_image,
        ),
        (
            GlKnownExtension::ArbMultiDrawIndirect,
            report.multi_draw_indirect,
        ),
    ] {
        if answer == ProbeAnswer::Passed
            && extension.is_legal_for(profile)
            && extensions.is_acquired(extension)
        {
            let _ = extensions.probe(extension);
        }
    }
}

#[path = "probes/glow.rs"]
mod glow_probes;

pub(crate) use glow_probes::GlowProbes;
