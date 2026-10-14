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
            Self::Failed => crate::backend::gl::api::GlOperationProbe::Failed,
            Self::Passed => crate::backend::gl::api::GlOperationProbe::Passed,
        }
    }
}

/// Complete set of operation-probe outcomes for one context.
///
/// Every field starts `Unavailable`, so a probe backend that cannot run leaves
/// all optional capabilities fail-closed instead of silently enabled.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct ProbeReport {
    pub depth_texture_attachment: ProbeAnswer,
    pub depth_renderbuffer_attachment: ProbeAnswer,
    pub rgba16f_attachment: ProbeAnswer,
    pub rgba32f_attachment: ProbeAnswer,
    pub rgba8_storage: ProbeAnswer,
    pub compute: ProbeAnswer,
    pub storage_buffer: ProbeAnswer,
    pub storage_image: ProbeAnswer,
    pub indirect_draw: ProbeAnswer,
    pub indirect_dispatch: ProbeAnswer,
    pub multi_draw_indirect: ProbeAnswer,
    /// `glGetQueryiv(GL_TIMESTAMP, GL_QUERY_COUNTER_BITS)`; `None` when the
    /// driver rejected the query or the loader did not supply the entry point.
    pub query_counter_bits: Option<u32>,
}

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
    /// Issues one zero-primitive indirect draw through scratch state.
    fn issues_indirect_draw(&self, indexed: bool, stages: &[(u32, &'static str)]) -> ProbeAnswer {
        let _ = (indexed, stages);
        ProbeAnswer::Unavailable
    }
    /// Issues one no-op indirect dispatch through scratch state.
    fn issues_indirect_dispatch(&self, compute_source: &'static str) -> ProbeAnswer {
        let _ = compute_source;
        ProbeAnswer::Unavailable
    }
}

/// Minimal `#version 430` pair: a linked raster program with no side effects.
const DESKTOP_VERTEX: &str = "#version 430\nvoid main() {}\n";
const DESKTOP_FRAGMENT: &str = "#version 430\nvoid main() {}\n";
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
    "#version 430\n",
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

    ProbeReport {
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
        // glow 0.18 does not bind glMultiDrawArraysIndirect, so neither a probe
        // nor an executor can exist; the capability stays fail-closed instead
        // of pretending a portable count limit exists. This is the recorded
        // narrowing rather than a missing probe (plan P2-15): the domain's
        // supported range is the mock route, which is the only one that has the
        // verb and a count limit to enable it with, and this answer is what
        // keeps a real context from advertising a command no provider serves.
        multi_draw_indirect: ProbeAnswer::Unavailable,
        query_counter_bits: probes.query_counter_bits(),
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
/// counter-width observation rather than by a command, and the multiview and
/// batch rows have no probe on this path at all.
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
