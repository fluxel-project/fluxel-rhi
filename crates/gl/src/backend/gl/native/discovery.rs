//! Native GL/GLES discovery over a context made current by the Host.
//!
//! This module never creates a window, display, surface, or context.  Its one
//! unsafe boundary is the `glow` adapter: callers must keep the supplied
//! context current on its owning thread for the entire call.  The resulting
//! snapshot is data only and remains bound to the caller supplied stamp.

use super::probes::{
    GlowProbes, NativeGlProbes, ProbeAnswer, ProbeReport, record_extension_probes,
    run_operation_probes,
};
use super::surface_facts;
use crate::backend::gl::api::{
    ContextStamp, CoreOrExtension, GlCapability, GlContextFlags, GlContextInfo, GlDiscoveryBuilder,
    GlDiscoveryError, GlDiscoverySnapshot, GlExtensionSet, GlFamilyProfile, GlFiniteF32, GlFormat,
    GlFormatCapabilities, GlFormatEvidence, GlFormatResourceKind, GlFormatTable, GlKnownExtension,
    GlLimits, GlOperationProbe, GlVersion,
};
use core::ffi::c_void;

/// Narrows one immutable native discovery record to the v13 routes which the
/// native owner actually executes today.
///
/// This deliberately lives beside discovery, rather than on the platform
/// adapters: WGL and EGL must publish the same answer for the same GL record.
/// In particular, an entry point found by discovery is not a public feature.
/// Compute, indirect execution, queries, barriers, and compressed upload stay
/// closed until `driver.rs` has an end-to-end submission route for them.
#[path = "discovery/formats.rs"]
mod formats;
#[path = "discovery/limits.rs"]
mod limits;
pub(crate) use formats::glow_const;
use formats::*;

#[path = "discovery/error.rs"]
mod error;
pub(crate) use error::NativeDiscoveryError;

#[path = "discovery/query.rs"]
mod query;
pub(crate) use query::NativeGlQuery;

/// The driver identity recorded when no platform layer supplied one.
///
/// A recorded absence rather than a stand-in. The GL version string is already
/// recorded as `version`, and copying it here is exactly what made the previous
/// record claim a driver identity nobody had observed (audit P2-6): a report
/// that reads `4.6.0 NVIDIA` as the driver cannot tell an observed platform
/// string from a repeated GL one. The marker follows the
/// `gl.context-flags-unavailable` style so an absence stays greppable.
pub(crate) const DRIVER_IDENTITY_UNAVAILABLE: &str = "gl.driver-identity-unavailable=true";

/// Normalizes a platform-supplied driver identity.
///
/// The platform layer is the only place a WGL or EGL driver string exists, so it
/// supplies one or supplies nothing. A blank string is nothing: recording it
/// verbatim would show an empty driver where the truth is that none was read.
pub(super) fn normalized_driver_identity(supplied: &str) -> String {
    let supplied = supplied.trim();
    if supplied.is_empty() {
        DRIVER_IDENTITY_UNAVAILABLE.to_owned()
    } else {
        supplied.to_owned()
    }
}

/// Discovers native facts using an already-current `glow` context.
///
/// Uses the same Host loader that built `glow` for the one registry query glow
/// 0.18 does not bind (`glGetQueryiv`).
///
/// # Safety contract
///
/// The Host/RHI provider must have made `context` current on its owner thread,
/// must keep it current throughout this call, and must serialize access to the
/// context.  `glow` forwards to the current native context; violating that
/// contract is outside Rust's type system and can call an unrelated driver.
#[cfg(any(feature = "native-gl-wgl", feature = "native-gles-egl"))]
pub(crate) unsafe fn discover_current_glow(
    context: &glow::Context,
    stamp: ContextStamp,
) -> Result<GlDiscoverySnapshot, NativeDiscoveryError> {
    // SAFETY: the absent loader only keeps `glGetQueryiv` probes closed; no
    // probe calls an entry point the context did not promise.
    unsafe { discover_current_glow_with_loader(context, stamp, |_| core::ptr::null()) }
}

/// [`discover_current_glow`] plus the Host proc loader, used only to resolve
/// `glGetQueryiv` for the timer-query counter width.
///
/// # Safety contract
///
/// Same as [`discover_current_glow`]. The loader must return valid entry
/// points for the currently bound context or null.
#[cfg(any(feature = "native-gl-wgl", feature = "native-gles-egl"))]
pub(crate) unsafe fn discover_current_glow_with_loader(
    context: &glow::Context,
    stamp: ContextStamp,
    proc_loader: impl Fn(&str) -> *const c_void,
) -> Result<GlDiscoverySnapshot, NativeDiscoveryError> {
    // SAFETY: forwarded from this function's documented caller contract.
    unsafe {
        discover_current_glow_identified(context, stamp, proc_loader, DRIVER_IDENTITY_UNAVAILABLE)
    }
}

/// [`discover_current_glow_with_loader`] plus the platform driver identity.
///
/// The driver identity is the one fact discovery cannot read from GL: on this
/// family it belongs to the platform layer (the ICD behind WGL, the EGL driver
/// behind an EGL display), so the provider that owns that layer supplies it.
/// Leaving the string empty records [`DRIVER_IDENTITY_UNAVAILABLE`] instead, so
/// a context whose platform layer did not report an identity is visible as an
/// absence rather than as whatever GL happened to answer.
///
/// # Safety contract
///
/// Same as [`discover_current_glow_with_loader`].
#[cfg(any(feature = "native-gl-wgl", feature = "native-gles-egl"))]
pub(crate) unsafe fn discover_current_glow_identified(
    context: &glow::Context,
    stamp: ContextStamp,
    proc_loader: impl Fn(&str) -> *const c_void,
    driver_identity: &str,
) -> Result<GlDiscoverySnapshot, NativeDiscoveryError> {
    use glow::HasContext as _;

    struct GlowQuery<'a>(&'a glow::Context);
    impl NativeGlQuery for GlowQuery<'_> {
        fn take_error(&self) -> bool {
            // SAFETY: upheld by discover_current_glow's current-context contract.
            unsafe { self.0.get_error() != glow::NO_ERROR }
        }
        fn string(&self, name: u32) -> Option<String> {
            // SAFETY: upheld by discover_current_glow's current-context contract.
            let value = unsafe { self.0.get_parameter_string(name) };
            (!self.take_error() && !value.is_empty()).then_some(value)
        }
        fn integer(&self, name: u32) -> Option<i64> {
            // SAFETY: upheld by discover_current_glow's current-context contract.
            let value = unsafe { self.0.get_parameter_i32(name) };
            // SAFETY: get_error reads the same current context and makes failed
            // optional queries fail closed instead of turning them into support.
            (!self.take_error()).then_some(i64::from(value))
        }
        fn integer_pair(&self, name: u32) -> Option<[i64; 2]> {
            let mut values = [0_i32; 2];
            // SAFETY: upheld by discover_current_glow's current-context contract.
            unsafe { self.0.get_parameter_i32_slice(name, &mut values) };
            // SAFETY: see integer.
            (!self.take_error()).then_some([i64::from(values[0]), i64::from(values[1])])
        }
        fn indexed_integer(&self, name: u32, index: u32) -> Option<i64> {
            // SAFETY: upheld by discover_current_glow's current-context contract.
            let value = unsafe { self.0.get_parameter_indexed_i32(name, index) };
            // SAFETY: see integer.
            (!self.take_error()).then_some(i64::from(value))
        }
        fn float(&self, name: u32) -> Option<f32> {
            // SAFETY: upheld by discover_current_glow's current-context contract.
            let value = unsafe { self.0.get_parameter_f32(name) };
            // SAFETY: see integer.
            (!self.take_error()).then_some(value)
        }
        fn indexed_string(&self, name: u32, index: u32) -> Option<String> {
            // SAFETY: upheld by discover_current_glow's current-context contract.
            let value = unsafe { self.0.get_parameter_indexed_string(name, index) };
            (!self.take_error() && !value.is_empty()).then_some(value)
        }
        fn drawable_attachment(&self, attachment: u32, name: u32) -> Option<i64> {
            // SAFETY: upheld by discover_current_glow's current-context contract.
            let value = unsafe {
                self.0.get_framebuffer_attachment_parameter_i32(
                    glow_const::DRAW_FRAMEBUFFER,
                    attachment,
                    name,
                )
            };
            // SAFETY: see integer. The draw-framebuffer target is named rather
            // than left to a default because there is no default: a query
            // against the wrong target would answer about a framebuffer nobody
            // bound and read as an observation of the drawable.
            (!self.take_error()).then_some(i64::from(value))
        }
    }

    let raw_get_query_iv = proc_loader("glGetQueryiv");
    // SAFETY: `glGetQueryiv` has this exact ABI; the null-pointer case only
    // disables the timer-query counter probe.
    let get_query_iv = (!raw_get_query_iv.is_null()).then(|| unsafe {
        core::mem::transmute::<*const c_void, super::probes::GetQueryivFn>(raw_get_query_iv)
    });

    // SAFETY: forwarded from this function's documented caller contract.
    let probes = GlowProbes::new(context, get_query_iv);
    let query = GlowQuery(context);
    // Both wrappers read the same current context and never run concurrently.
    discover_with_query_pair(&query, &probes, stamp, driver_identity)
}

pub(super) fn discover_with_query(
    query: &(impl NativeGlQuery + NativeGlProbes),
    stamp: ContextStamp,
) -> Result<GlDiscoverySnapshot, NativeDiscoveryError> {
    discover_with_query_identified(query, stamp, DRIVER_IDENTITY_UNAVAILABLE)
}

/// [`discover_with_query`] with a platform driver identity supplied.
pub(super) fn discover_with_query_identified(
    query: &(impl NativeGlQuery + NativeGlProbes),
    stamp: ContextStamp,
    driver_identity: &str,
) -> Result<GlDiscoverySnapshot, NativeDiscoveryError> {
    discover_with_query_pair(&NativeOnly(query), query, stamp, driver_identity)
}

/// Adapts a combined test fake to the query half only.
struct NativeOnly<'a, Q: NativeGlQuery + NativeGlProbes>(&'a Q);
impl<Q: NativeGlQuery + NativeGlProbes> NativeGlQuery for NativeOnly<'_, Q> {
    fn take_error(&self) -> bool {
        self.0.take_error()
    }
    fn string(&self, name: u32) -> Option<String> {
        self.0.string(name)
    }
    fn integer(&self, name: u32) -> Option<i64> {
        self.0.integer(name)
    }
    fn integer_pair(&self, name: u32) -> Option<[i64; 2]> {
        self.0.integer_pair(name)
    }
    fn indexed_integer(&self, name: u32, index: u32) -> Option<i64> {
        self.0.indexed_integer(name, index)
    }
    fn float(&self, name: u32) -> Option<f32> {
        self.0.float(name)
    }
    fn indexed_string(&self, name: u32, index: u32) -> Option<String> {
        self.0.indexed_string(name, index)
    }
    fn drawable_attachment(&self, attachment: u32, name: u32) -> Option<i64> {
        self.0.drawable_attachment(attachment, name)
    }
}

fn discover_with_query_pair(
    query: &impl NativeGlQuery,
    probes: &impl NativeGlProbes,
    stamp: ContextStamp,
    driver_identity: &str,
) -> Result<GlDiscoverySnapshot, NativeDiscoveryError> {
    if query.take_error() {
        return Err(NativeDiscoveryError::PreExistingGlError);
    }
    let version = required_string(query, glow_const::VERSION, "GL_VERSION")?;
    let profile = parse_native_profile(&version)
        .ok_or_else(|| NativeDiscoveryError::UnsupportedProfile(version.clone()))?;
    let glsl = required_string(
        query,
        glow_const::SHADING_LANGUAGE_VERSION,
        "GL_SHADING_LANGUAGE_VERSION",
    )?;
    let vendor = required_string(query, glow_const::VENDOR, "GL_VENDOR")?;
    let renderer = required_string(query, glow_const::RENDERER, "GL_RENDERER")?;
    let mut extensions = extensions(query, profile)?;
    let report = run_operation_probes(probes, profile, &extensions);
    // Acquisition says the entry points were loaded, never that they work, so
    // the ledger advances only for the extensions whose probe really ran and
    // passed. This runs before `limits` and `native_formats`, which both read
    // acquisition, and `is_acquired` covers `Probed`, so neither can regress.
    record_extension_probes(profile, &report, &mut extensions);
    let limits = limits::limits(query, profile, &extensions, &report)?;
    let formats = native_formats(profile, &extensions, &limits, &report)?;
    let mut flags = context_flags(query, profile)?;
    // The drawable's observed format rides the context-flags record, which is
    // the snapshot's only durable free-form fact set. It is an observation about
    // the surface rather than about the context, and it is recorded at all
    // because FBO 0 has no `GlFormatTable` row that could carry it.
    let (surface, surface_strings) = surface_facts::observe(query);
    flags.other.extend(surface_strings);
    // The desktop requirement is recorded next to the observed facts so a report
    // that shows a desktop context can see, in the same record, the GL 4.0
    // family floor (audit P2-12). It is stamped only where it applies: the
    // embedded profile has no such floor, and inventing one there would make the
    // marker describe a requirement that does not exist.
    if matches!(profile, GlFamilyProfile::Desktop { .. }) {
        flags.other.insert(desktop_context_floor_marker());
    }
    let mut builder = GlDiscoveryBuilder::new(
        stamp,
        GlContextInfo::new(
            profile,
            &version,
            glsl,
            vendor,
            renderer,
            // The driver identity is the platform layer's, never a copy of the
            // GL version: the version is already its own field, and a duplicate
            // here reads as an observed driver string to every later report.
            normalized_driver_identity(driver_identity),
            flags,
        ),
        extensions,
        limits,
        formats,
    )
    .map_err(NativeDiscoveryError::Snapshot)?;
    // The typed half of the observation above, recorded beside the strings it
    // came out with rather than read back out of them.
    builder.surface_facts(surface);

    // Capability enablement: a resolved core-or-extension route is necessary
    // but never sufficient. Every optional command domain also records the
    // outcome of its real operation probe from `run_operation_probes`.
    builder.resolve(
        GlCapability::Compute,
        CoreOrExtension {
            desktop_core: Some(GlVersion::new(4, 3)),
            embedded_core: Some(GlVersion::new(3, 1)),
            extension: Some(GlKnownExtension::ArbComputeShader),
            extension_requires_probe: true,
        },
        report.compute.to_operation_probe(),
    );
    builder.resolve(
        GlCapability::StorageBuffer,
        CoreOrExtension {
            desktop_core: Some(GlVersion::new(4, 3)),
            embedded_core: Some(GlVersion::new(3, 1)),
            extension: Some(GlKnownExtension::ArbShaderStorageBufferObject),
            extension_requires_probe: true,
        },
        report.storage_buffer.to_operation_probe(),
    );
    builder.resolve(
        GlCapability::StorageImage,
        CoreOrExtension {
            desktop_core: Some(GlVersion::new(4, 2)),
            embedded_core: Some(GlVersion::new(3, 1)),
            extension: Some(GlKnownExtension::ArbShaderImageLoadStore),
            extension_requires_probe: true,
        },
        report.storage_image.to_operation_probe(),
    );
    builder.resolve(
        GlCapability::IndirectDraw,
        CoreOrExtension {
            desktop_core: Some(GlVersion::new(4, 0)),
            embedded_core: Some(GlVersion::new(3, 1)),
            extension: None,
            extension_requires_probe: false,
        },
        report.indirect_draw.to_operation_probe(),
    );
    builder.resolve(
        GlCapability::IndirectDispatch,
        CoreOrExtension {
            desktop_core: Some(GlVersion::new(4, 3)),
            embedded_core: Some(GlVersion::new(3, 1)),
            extension: None,
            extension_requires_probe: false,
        },
        report.indirect_dispatch.to_operation_probe(),
    );
    // No probe (and no glow entry point) exists for multi-draw-indirect, so
    // the fact stays `NotRun` and the capability can never silently enable.
    builder.resolve(
        GlCapability::MultiDrawIndirect,
        CoreOrExtension {
            desktop_core: Some(GlVersion::new(4, 3)),
            embedded_core: None,
            extension: None,
            extension_requires_probe: false,
        },
        report.multi_draw_indirect.to_operation_probe(),
    );
    // The normalized batch domain has no native route in this contract: no
    // accepted core version supplies a combined per-draw batch command here,
    // and the browser extension that does is WebGL2-only. The row is resolved
    // explicitly so a native context records the absence as a fact instead of
    // leaving it unasked, and every native batch therefore takes the
    // provider's single-draw path (audit P1-8).
    builder.resolve(
        GlCapability::MultiDraw,
        CoreOrExtension {
            desktop_core: None,
            embedded_core: None,
            extension: Some(GlKnownExtension::WebglMultiDraw),
            extension_requires_probe: false,
        },
        GlOperationProbe::NotRun,
    );
    // Multiview has no core route on any accepted profile: the desktop core
    // profile it would need does not exist, and the embedded route is the OVR
    // extension, which is additionally gated on its acquisition. No multiview
    // attachment probe exists, so the fact stays `NotRun` and every profile
    // rejects a multiview pass (audit P1-7).
    builder.resolve(
        GlCapability::Multiview,
        CoreOrExtension {
            desktop_core: None,
            embedded_core: None,
            extension: Some(GlKnownExtension::OvrMultiview2),
            extension_requires_probe: true,
        },
        GlOperationProbe::NotRun,
    );
    builder.resolve(
        GlCapability::TimerQuery,
        CoreOrExtension {
            desktop_core: Some(GlVersion::new(3, 3)),
            embedded_core: Some(GlVersion::new(3, 0)),
            extension: None,
            extension_requires_probe: false,
        },
        // The counter-width observation is a limit read, not a command probe;
        // `limits.query_counter_bits` carries the real `glGetQueryiv` answer.
        GlOperationProbe::NotRequired,
    );
    builder.resolve(
        GlCapability::OcclusionQuery,
        CoreOrExtension {
            desktop_core: Some(GlVersion::new(1, 5)),
            embedded_core: Some(GlVersion::new(3, 0)),
            extension: None,
            extension_requires_probe: false,
        },
        GlOperationProbe::NotRequired,
    );
    Ok(builder.build())
}

/// Records the actual `GL_CONTEXT_FLAGS`/profile mask as evidence context
/// flags, falling back to explicit "unavailable" markers instead of defaults.
fn context_flags(
    query: &impl NativeGlQuery,
    profile: GlFamilyProfile,
) -> Result<GlContextFlags, NativeDiscoveryError> {
    let mut flags = GlContextFlags::default();
    const CONTEXT_FLAGS: u32 = 0x821E;
    const CONTEXT_PROFILE_MASK: u32 = 0x9126;
    const CONTEXT_CORE_PROFILE_BIT: i64 = 0x0000_0001;
    const CONTEXT_COMPATIBILITY_PROFILE_BIT: i64 = 0x0000_0002;
    const CONTEXT_ROBUST_ACCESS: u32 = 0x90F3;
    match profile {
        GlFamilyProfile::Desktop { .. } => {
            if let Some(bits) = query.integer(CONTEXT_FLAGS) {
                flags.other.insert(format!("gl.context-flags=0x{bits:08x}"));
                flags.debug = bits & 0x0000_0002 != 0;
                flags.forward_compatible = bits & 0x0000_0001 != 0;
                flags.no_error = bits & 0x0000_0008 != 0;
            } else {
                flags
                    .other
                    .insert("gl.context-flags-unavailable=true".into());
            }
            if let Some(mask) = query.integer(CONTEXT_PROFILE_MASK) {
                if mask & CONTEXT_CORE_PROFILE_BIT != 0 {
                    flags.other.insert("gl.profile=core".into());
                } else if mask & CONTEXT_COMPATIBILITY_PROFILE_BIT != 0 {
                    flags.other.insert("gl.profile=compatibility".into());
                }
            }
            if let Some(robust) = query.integer(CONTEXT_ROBUST_ACCESS) {
                flags.robust_access = robust != 0;
            }
        }
        GlFamilyProfile::Embedded { .. } | GlFamilyProfile::WebGl2 => {
            // ES and WebGL2 expose no context-flags query to discovery, and
            // robustness context creation is Host-owned; record the absence
            // explicitly instead of pretending defaults were observed.
            flags.other.insert("gl.context-flags=unavailable".into());
        }
    }
    Ok(flags)
}

pub(super) fn required_string(
    query: &impl NativeGlQuery,
    token: u32,
    name: &'static str,
) -> Result<String, NativeDiscoveryError> {
    query
        .string(token)
        .filter(|value| !value.trim().is_empty())
        .ok_or(NativeDiscoveryError::InvalidContextString(name))
}

fn extensions(
    query: &impl NativeGlQuery,
    profile: GlFamilyProfile,
) -> Result<GlExtensionSet, NativeDiscoveryError> {
    let count = nonnegative(
        query.integer(glow_const::NUM_EXTENSIONS),
        "GL_NUM_EXTENSIONS",
    )?;
    let mut result = GlExtensionSet::default();
    for index in 0..count {
        let name = query.indexed_string(glow_const::EXTENSIONS, index).ok_or(
            NativeDiscoveryError::QueryFailed("glGetStringi(GL_EXTENSIONS)"),
        )?;
        result.report_raw(name);
    }
    // Native GL entry points are loaded by the provider before it constructs
    // glow. Record acquisition only for typed, legal names; the operation
    // probes in `run_operation_probes` remain the real enablement evidence.
    //
    // The multiview extension is deliberately absent from this list: the
    // native boundary binds no multiview attachment entry point, so acquisition
    // cannot be proved here and the name stays `Reported`. It is still typed
    // and still recorded, which is what keeps a driver that reports it visible
    // in evidence without letting the reported name enable anything.
    for known in [
        GlKnownExtension::ArbComputeShader,
        GlKnownExtension::ArbShaderStorageBufferObject,
        GlKnownExtension::ArbShaderImageLoadStore,
        GlKnownExtension::ExtTextureFilterAnisotropic,
        GlKnownExtension::OesTextureFloatLinear,
        GlKnownExtension::KhrRobustness,
        GlKnownExtension::KhrDebug,
    ] {
        if known.is_legal_for(profile) && result.provenance(known).is_some() {
            let _ = result.acquire(known);
        }
    }
    Ok(result)
}

/// The desktop GL core version this native family requires, and why.
///
/// This is a recorded policy decision rather than a discovered fact (audit
/// P2-12). The native provider requests the highest available core context and
/// falls back through GL 4.x to 4.0. Individual 4.1+ command domains remain
/// disabled unless their exact core-or-extension route, entry points, limits,
/// and v13 lowering were all observed; accepting GL 4.0 never implies the GL
/// 4.3 feature set.
pub(crate) const REQUIRED_DESKTOP_VERSION: GlVersion = GlVersion::new(4, 0);

/// The recorded marker for [`REQUIRED_DESKTOP_VERSION`].
///
/// It rides the context-flags record so that observed facts travel with the
/// requirement they are read against: a report showing any desktop GL context
/// can see the 4.0 family floor instead of learning it from a second constant.
///
/// The marker builder and the version it is built from are read by the WGL
/// provider too, which asks for this version, enforces it on the actual version
/// string and then requires the snapshot to carry this exact marker.  All three
/// read one value rather than three copies of it (audit P2-12).
pub(crate) fn desktop_context_floor_marker() -> String {
    format!(
        "gl.desktop-context-floor={}.{}",
        REQUIRED_DESKTOP_VERSION.major, REQUIRED_DESKTOP_VERSION.minor
    )
}

/// Whether this profile can allocate multisample texture storage at all.
///
/// Two-dimensional multisample texture storage is a 4.3 / 3.1 core feature.
/// Below that floor the sample-count queries still answer, because they are
/// renderbuffer facts there, so a recorded ceiling is not by itself proof that
/// the allocation exists. The floor is recorded rather than discovered by
/// trying because there is nothing to try: the loader never resolves the entry
/// point below the floor, and the dispatch table calls it through a null
/// pointer, which is not a GL error any caller can be handed.
pub(super) const fn supports_multisample_texture_storage(profile: GlFamilyProfile) -> bool {
    profile.meets(Some(GlVersion::new(4, 3)), Some(GlVersion::new(3, 1)))
}

/// The sample-count class one texture format belongs to.
///
/// GL bounds a multisample texture's sample count by a ceiling chosen from the
/// format's class rather than by one global number, so the class is what turns
/// a recorded limit into the bound that actually applies. `GlFormat` has no
/// integer format, so the integer arm is unreachable today; it is modelled
/// instead of folded into the color arm because folding it would silently
/// accept an integer format at the color ceiling the day one is added.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum TextureSampleClass {
    Color,
    DepthStencil,
    Integer,
}

/// The sample class of one format. Both the record and the allocation gate read
/// this one mapping, so a fact can never be written under one ceiling and
/// checked against another.
pub(super) const fn texture_sample_class(format: GlFormat) -> TextureSampleClass {
    match format {
        GlFormat::Depth16Unorm | GlFormat::Depth24PlusStencil8 | GlFormat::Depth32Float => {
            TextureSampleClass::DepthStencil
        }
        _ => TextureSampleClass::Color,
    }
}

/// The recorded ceiling that governs one sample class.
pub(super) const fn texture_sample_ceiling(class: TextureSampleClass, limits: &GlLimits) -> u32 {
    match class {
        TextureSampleClass::Color => limits.max_color_texture_samples,
        TextureSampleClass::DepthStencil => limits.max_depth_texture_samples,
        TextureSampleClass::Integer => limits.max_integer_samples,
    }
}
