//! WebGL2 discovery fact collection.
//!
//! This module contains browser-query helpers used while opening one immutable
//! context generation. It owns no JavaScript execution state after discovery.

use core::cell::Cell;
use js_sys::{Array, Object, Reflect};
use std::collections::BTreeMap;
use wasm_bindgen::{JsCast, JsValue};
use web_sys::{HtmlCanvasElement, WebGl2RenderingContext, WebGlBuffer, WebGlSampler, WebGlTexture};

use super::super::api::{
    BufferId, ContextStamp, CoreOrExtension, GlBufferDesc, GlCapability, GlContextFlags,
    GlContextInfo, GlContextLifecycle, GlDiscoveryBuilder, GlDiscoveryError, GlDiscoverySnapshot,
    GlError, GlExtensionSet, GlFamilyApi as _, GlFamilyProfile, GlFenceLeaseBook, GlFiniteF32,
    GlKnownExtension, GlLimits, GlOperationProbe, GlPixelStoreState, GlSurfaceFacts,
    GlSurfaceLeaseBook, GlTextureDesc, OwnerThreadIdentity, TextureId, VertexArrayId,
};
use super::exec_multidraw::BrowserMultiDraw;
use super::exec_timer::{self, BrowserTimerQuery};
use super::objects::{
    ActivePass, ActiveRaster, BrowserFramebuffer, BrowserProgram, BrowserQuery,
    BrowserRenderbuffer, BrowserShader, BrowserSync, BrowserVertexArray,
};

/// Immutable WebGL2 discovery evidence plus the callable browser context.
///
/// RHI retains the Host-provided canvas and the context it created. `glow`
/// receives a cloned JS context handle, while `raw` remains the RHI-owned
/// browser API handle for discovery and provider work.
/// It is called only after the fallible WebGL2 version check below succeeds.
/// The remaining assumptions made by glow are WebGL binding invariants (a real
/// WebGL2 context returns a string `VERSION` and an array-or-null extension
/// list); violations are browser defects rather than recoverable application
/// input.  All application-observable queries in this module remain fallible.
pub(super) fn ensure_context_live(
    raw: &WebGl2RenderingContext,
    operation: &'static str,
) -> Result<(), GlError> {
    (!raw.is_context_lost())
        .then_some(())
        .ok_or(GlError::ContextLost { operation })
}

/// The extension objects whose commands must be retained for later calls.
///
/// Both are `Some` exactly when their object exposed every command its domain
/// issues, which is also exactly when that domain's capability could resolve.
pub(super) struct RetainedCommands {
    pub(super) multi_draw: Option<BrowserMultiDraw>,
    pub(super) timer: Option<BrowserTimerQuery>,
    pub(super) astc_hdr: bool,
}

/// Records evidence for every reported extension this contract models.
///
/// Returns the ledger together with the acquired objects whose commands must be
/// retained: those extensions define their commands on the object itself, so no
/// context method can reach them later.
pub(super) fn discover_extensions(
    raw: &WebGl2RenderingContext,
) -> Result<(GlExtensionSet, RetainedCommands), GlError> {
    let listed = raw.get_supported_extensions().ok_or_else(|| {
        driver(
            "getSupportedExtensions",
            "browser returned null for a live WebGL2 context",
        )
    })?;
    let mut extensions = GlExtensionSet::default();
    for value in listed.iter() {
        let name = value.as_string().ok_or_else(|| {
            driver(
                "getSupportedExtensions",
                "browser returned a non-string extension name",
            )
        })?;
        extensions.report_raw(name);
    }

    let mut commands = RetainedCommands {
        multi_draw: None,
        timer: None,
        astc_hdr: false,
    };
    for known in [
        GlKnownExtension::ExtDisjointTimerQueryWebgl2,
        GlKnownExtension::ExtColorBufferFloat,
        GlKnownExtension::ExtFloatBlend,
        GlKnownExtension::OesTextureFloatLinear,
        GlKnownExtension::ExtTextureFilterAnisotropic,
        GlKnownExtension::WebglMultiDraw,
        GlKnownExtension::OvrMultiview2,
        GlKnownExtension::KhrParallelShaderCompile,
        GlKnownExtension::CompressedTextureS3tc,
        GlKnownExtension::CompressedTextureS3tcSrgb,
        GlKnownExtension::CompressedTextureBptc,
        GlKnownExtension::CompressedTextureRgtc,
        GlKnownExtension::CompressedTextureAstc,
        GlKnownExtension::CompressedTextureEtc,
    ] {
        if extensions.provenance(known).is_none() {
            continue;
        }
        let object = match raw.get_extension(acquisition_name(known, &extensions)) {
            Ok(Some(object)) => object,
            Ok(None) | Err(_) => {
                // A reported name without a usable extension object is
                // explicitly failed and can never enable a capability.
                extensions.fail(known);
                continue;
            }
        };
        if known == GlKnownExtension::WebglMultiDraw {
            // Its commands live on the object, so the object is the oracle: an
            // object that does not expose all of them fails the acquisition and
            // leaves the batch domain on its single-draw route (audit P1-8).
            match BrowserMultiDraw::acquire(&object) {
                Some(batch) => {
                    extensions.acquire(known);
                    commands.multi_draw = Some(batch);
                }
                None => {
                    extensions.fail(known);
                }
            }
            continue;
        }
        if known == GlKnownExtension::ExtDisjointTimerQueryWebgl2 {
            // Same oracle shape as the batch domain: the timer commands are
            // defined on the object, so a reported name without all five of them
            // fails the acquisition and leaves every timer operation rejected
            // (audit P1-5).
            match BrowserTimerQuery::acquire(&object) {
                Some(timer) => {
                    extensions.acquire(known);
                    commands.timer = Some(timer);
                }
                None => {
                    extensions.fail(known);
                }
            }
            continue;
        }
        if known == GlKnownExtension::CompressedTextureAstc {
            // The ASTC extension object answers `supportedProfiles`; `hdr` is
            // independent from an LDR object acquisition.  This reads the
            // exact object before publishing the private admission bit.
            commands.astc_hdr = astc_hdr_profile(&object);
        }
        extensions.acquire(known);
    }
    Ok((extensions, commands))
}

pub(super) fn astc_hdr_profile(object: &JsValue) -> bool {
    let Ok(value) = Reflect::get(object, &JsValue::from_str("supportedProfiles")) else {
        return false;
    };
    Array::is_array(&value)
        && Array::from(&value)
            .iter()
            .any(|profile| profile.as_string().is_some_and(|profile| profile == "hdr"))
}

pub(super) fn acquisition_name(
    known: GlKnownExtension,
    extensions: &GlExtensionSet,
) -> &'static str {
    if known == GlKnownExtension::ExtTextureFilterAnisotropic {
        for alias in [
            "EXT_texture_filter_anisotropic",
            "WEBKIT_EXT_texture_filter_anisotropic",
            "MOZ_EXT_texture_filter_anisotropic",
        ] {
            if extensions
                .raw_reported_names()
                .any(|reported| reported == alias)
            {
                return alias;
            }
        }
    }
    known.raw_name()
}

pub(super) fn discover_limits(
    raw: &WebGl2RenderingContext,
    extensions: &GlExtensionSet,
    timer: Option<&BrowserTimerQuery>,
) -> Result<GlLimits, GlError> {
    let max_samples = u32_parameter(raw, WebGl2RenderingContext::MAX_SAMPLES, "MAX_SAMPLES")?;
    Ok(GlLimits {
        max_texture_size: u32_parameter(
            raw,
            WebGl2RenderingContext::MAX_TEXTURE_SIZE,
            "MAX_TEXTURE_SIZE",
        )?,
        max_3d_texture_size: u32_parameter(
            raw,
            WebGl2RenderingContext::MAX_3D_TEXTURE_SIZE,
            "MAX_3D_TEXTURE_SIZE",
        )?,
        max_array_texture_layers: u32_parameter(
            raw,
            WebGl2RenderingContext::MAX_ARRAY_TEXTURE_LAYERS,
            "MAX_ARRAY_TEXTURE_LAYERS",
        )?,
        max_cube_map_texture_size: u32_parameter(
            raw,
            WebGl2RenderingContext::MAX_CUBE_MAP_TEXTURE_SIZE,
            "MAX_CUBE_MAP_TEXTURE_SIZE",
        )?,
        max_renderbuffer_size: u32_parameter(
            raw,
            WebGl2RenderingContext::MAX_RENDERBUFFER_SIZE,
            "MAX_RENDERBUFFER_SIZE",
        )?,
        max_color_attachments: u32_parameter(
            raw,
            WebGl2RenderingContext::MAX_COLOR_ATTACHMENTS,
            "MAX_COLOR_ATTACHMENTS",
        )?,
        max_draw_buffers: u32_parameter(
            raw,
            WebGl2RenderingContext::MAX_DRAW_BUFFERS,
            "MAX_DRAW_BUFFERS",
        )?,
        max_vertex_attributes: u32_parameter(
            raw,
            WebGl2RenderingContext::MAX_VERTEX_ATTRIBS,
            "MAX_VERTEX_ATTRIBS",
        )?,
        max_viewport_dimensions: viewport_dimensions(raw)?,
        max_viewports: 0,
        max_vertex_texture_image_units: u32_parameter(
            raw,
            WebGl2RenderingContext::MAX_VERTEX_TEXTURE_IMAGE_UNITS,
            "MAX_VERTEX_TEXTURE_IMAGE_UNITS",
        )?,
        max_fragment_texture_image_units: u32_parameter(
            raw,
            WebGl2RenderingContext::MAX_TEXTURE_IMAGE_UNITS,
            "MAX_TEXTURE_IMAGE_UNITS",
        )?,
        max_combined_texture_image_units: u32_parameter(
            raw,
            WebGl2RenderingContext::MAX_COMBINED_TEXTURE_IMAGE_UNITS,
            "MAX_COMBINED_TEXTURE_IMAGE_UNITS",
        )?,
        max_uniform_buffer_bindings: u32_parameter(
            raw,
            WebGl2RenderingContext::MAX_UNIFORM_BUFFER_BINDINGS,
            "MAX_UNIFORM_BUFFER_BINDINGS",
        )?,
        max_uniform_block_size: u64_parameter(
            raw,
            WebGl2RenderingContext::MAX_UNIFORM_BLOCK_SIZE,
            "MAX_UNIFORM_BLOCK_SIZE",
        )?,
        uniform_buffer_offset_alignment: u64_parameter(
            raw,
            WebGl2RenderingContext::UNIFORM_BUFFER_OFFSET_ALIGNMENT,
            "UNIFORM_BUFFER_OFFSET_ALIGNMENT",
        )?,
        max_vertex_uniform_blocks: u32_parameter(
            raw,
            WebGl2RenderingContext::MAX_VERTEX_UNIFORM_BLOCKS,
            "MAX_VERTEX_UNIFORM_BLOCKS",
        )?,
        max_fragment_uniform_blocks: u32_parameter(
            raw,
            WebGl2RenderingContext::MAX_FRAGMENT_UNIFORM_BLOCKS,
            "MAX_FRAGMENT_UNIFORM_BLOCKS",
        )?,
        max_compute_uniform_blocks: 0,
        max_combined_uniform_blocks: u32_parameter(
            raw,
            WebGl2RenderingContext::MAX_COMBINED_UNIFORM_BLOCKS,
            "MAX_COMBINED_UNIFORM_BLOCKS",
        )?,
        max_storage_buffer_bindings: 0,
        max_storage_block_size: 0,
        storage_buffer_offset_alignment: 0,
        max_vertex_storage_blocks: 0,
        max_fragment_storage_blocks: 0,
        max_compute_storage_blocks: 0,
        max_combined_storage_blocks: 0,
        max_image_units: 0,
        max_combined_image_units: 0,
        max_samples,
        // WebGL2 has renderbuffer multisampling but no multisample textures.
        // Do not turn MAX_SAMPLES into a texture capability.
        max_color_texture_samples: 0,
        max_depth_texture_samples: 0,
        max_integer_samples: 0,
        max_compute_work_group_count: [0; 3],
        max_compute_work_group_size: [0; 3],
        max_compute_work_group_invocations: 0,
        max_multi_draw_indirect_count: None,
        max_multiview_view_count: multiview_view_limit(raw, extensions),
        // Zero here is the capability row's limit half: it means at least one
        // timer target did not answer a usable width, so the row stays disabled
        // while the ledger keeps recording what the entry-point oracle saw.
        query_counter_bits: exec_timer::recorded_counter_width(raw, timer),
        max_texture_anisotropy: anisotropy_limit(raw, extensions)?,
    })
}

/// The number of views one attachment may serve in one pass.
///
/// The token is defined by the multiview extension family and is invalid until
/// an extension object has been acquired, so a context without it records 0 and
/// can never satisfy the multiview floor. One view is the plain single-view
/// attachment every WebGL2 context already has and is not a multiview proof.
///
/// An acquired object that does not answer the token with a number also records
/// 0 rather than failing discovery: Chrome on an ANGLE/SwiftShader context
/// lists and returns the object while answering the token with `null`, and
/// turning that into a discovery error makes the whole provider unopenable for
/// a capability that is disabled either way (observed while adding the browser
/// tests, not a hypothesis).
pub(super) fn multiview_view_limit(
    raw: &WebGl2RenderingContext,
    extensions: &GlExtensionSet,
) -> u32 {
    const MAX_VIEWS_OVR: u32 = 0x9632;
    if !extensions.is_acquired(GlKnownExtension::OvrMultiview2) {
        return 0;
    }
    u32_parameter(raw, MAX_VIEWS_OVR, "MAX_VIEWS_OVR")
        // An unanswered token may also be answered with a driver error; that
        // error belongs to this question and must not surface as the failure of
        // the next unrelated provider call.
        .inspect_err(|_| {
            let _ = raw.get_error();
        })
        .unwrap_or(0)
}

pub(super) fn anisotropy_limit(
    raw: &WebGl2RenderingContext,
    extensions: &GlExtensionSet,
) -> Result<Option<GlFiniteF32>, GlError> {
    // `MAX_TEXTURE_MAX_ANISOTROPY_EXT` has this registry value for all three
    // spellings. It is invalid until an extension object has been acquired.
    const MAX_TEXTURE_MAX_ANISOTROPY_EXT: u32 = 0x84FF;
    if !extensions.is_acquired(GlKnownExtension::ExtTextureFilterAnisotropic) {
        return Ok(None);
    }
    let value = number_parameter(
        raw,
        MAX_TEXTURE_MAX_ANISOTROPY_EXT,
        "MAX_TEXTURE_MAX_ANISOTROPY_EXT",
    )? as f32;
    GlFiniteF32::new(value)
        .ok_or_else(|| {
            driver(
                "getParameter",
                "MAX_TEXTURE_MAX_ANISOTROPY_EXT was not finite f32",
            )
        })
        .map(Some)
}

/// Reads the live context-attribute answers back into the evidence record.
///
/// Every accepted attribute is recorded as an exact `name=value` entry so the
/// snapshot proves what the browser actually granted, including attributes the
/// requester set but the browser could not honor (audit P1-10).
pub(super) fn discover_context_flags(
    raw: &WebGl2RenderingContext,
) -> Result<GlContextFlags, GlError> {
    let mut flags = GlContextFlags::default();
    let Some(attributes) = raw.get_context_attributes() else {
        // A `None` answer gets an explicit marker instead of pretending
        // defaults were observed.
        flags
            .other
            .insert("webgl.context-attributes-unavailable=true".into());
        return Ok(flags);
    };
    let record = |flags: &mut GlContextFlags, name: &str, value: Option<bool>| {
        if let Some(value) = value {
            flags.other.insert(format!("webgl.{name}={value}"));
        }
    };
    record(&mut flags, "alpha", attributes.get_alpha());
    record(&mut flags, "antialias", attributes.get_antialias());
    record(&mut flags, "depth", attributes.get_depth());
    record(&mut flags, "stencil", attributes.get_stencil());
    record(
        &mut flags,
        "premultiplied-alpha",
        attributes.get_premultiplied_alpha(),
    );
    record(
        &mut flags,
        "preserve-drawing-buffer",
        attributes.get_preserve_drawing_buffer(),
    );
    record(
        &mut flags,
        "fail-if-major-performance-caveat",
        attributes.get_fail_if_major_performance_caveat(),
    );
    Ok(flags)
}

pub(super) fn string_parameter(
    raw: &WebGl2RenderingContext,
    pname: u32,
    name: &'static str,
) -> Result<String, GlError> {
    raw.get_parameter(pname)
        .map_err(|value| js_error("getParameter", value))?
        .as_string()
        .ok_or_else(|| driver("getParameter", &format!("{name} was not a string")))
}

/// Reads the drawing buffer's colour component widths.
///
/// One observation, and the typed half of what the native path reports as
/// `gl.surface-color-bits`. The browser records none of the `gl.surface-*` keys:
/// that channel exists on the native side to name *why* an observation failed,
/// which is a distinction the typed value deliberately does not carry, and a
/// second renderer with no reasons to report would be a reporting channel with
/// nothing in it.
///
/// A parameter that cannot be read answers [`GlSurfaceFacts::Unavailable`] rather
/// than failing discovery, which is the native path's rule for the same
/// observation: a drawable a presenter cannot describe is a missing surface, not
/// a context that failed to be discovered. The depth, stencil, sample-buffer and
/// sample-count parameters the native path also reads are left to it, because
/// they are read for keys this side does not write and for no field of the typed
/// value.
pub(super) fn drawing_buffer_facts(raw: &WebGl2RenderingContext) -> GlSurfaceFacts {
    const PARAMETERS: [(u32, &str); 4] = [
        (WebGl2RenderingContext::RED_BITS, "RED_BITS"),
        (WebGl2RenderingContext::GREEN_BITS, "GREEN_BITS"),
        (WebGl2RenderingContext::BLUE_BITS, "BLUE_BITS"),
        (WebGl2RenderingContext::ALPHA_BITS, "ALPHA_BITS"),
    ];
    let mut color_bits = [0u32; PARAMETERS.len()];
    for (slot, (pname, name)) in color_bits.iter_mut().zip(PARAMETERS) {
        match u32_parameter(raw, pname, name) {
            Ok(value) => *slot = value,
            Err(_) => return GlSurfaceFacts::Unavailable,
        }
    }
    GlSurfaceFacts::Observed { color_bits }
}

pub(super) fn u32_parameter(
    raw: &WebGl2RenderingContext,
    pname: u32,
    name: &'static str,
) -> Result<u32, GlError> {
    let value = number_parameter(raw, pname, name)?;
    if !value.is_finite() || value < 0.0 || value.fract() != 0.0 || value > f64::from(u32::MAX) {
        return Err(driver("getParameter", &format!("{name} was not a u32")));
    }
    Ok(value as u32)
}

pub(super) fn u64_parameter(
    raw: &WebGl2RenderingContext,
    pname: u32,
    name: &'static str,
) -> Result<u64, GlError> {
    let value = number_parameter(raw, pname, name)?;
    if !value.is_finite() || value < 0.0 || value.fract() != 0.0 || value > 9_007_199_254_740_991.0
    {
        return Err(driver(
            "getParameter",
            &format!("{name} was not an exact JavaScript integer"),
        ));
    }
    Ok(value as u64)
}

pub(super) fn number_parameter(
    raw: &WebGl2RenderingContext,
    pname: u32,
    name: &'static str,
) -> Result<f64, GlError> {
    raw.get_parameter(pname)
        .map_err(|value| js_error("getParameter", value))?
        .as_f64()
        .ok_or_else(|| driver("getParameter", &format!("{name} was not a number")))
}

pub(super) fn viewport_dimensions(raw: &WebGl2RenderingContext) -> Result<[u32; 2], GlError> {
    let value = raw
        .get_parameter(WebGl2RenderingContext::MAX_VIEWPORT_DIMS)
        .map_err(|value| js_error("getParameter", value))?;
    let values = Array::from(&value);
    if values.length() != 2 {
        return Err(driver(
            "getParameter",
            "MAX_VIEWPORT_DIMS did not contain two values",
        ));
    }
    let width = values.get(0).as_f64();
    let height = values.get(1).as_f64();
    match (width, height) {
        (Some(width), Some(height))
            if width.is_finite()
                && height.is_finite()
                && width.fract() == 0.0
                && height.fract() == 0.0
                && width >= 0.0
                && height >= 0.0
                && width <= f64::from(u32::MAX)
                && height <= f64::from(u32::MAX) =>
        {
            Ok([width as u32, height as u32])
        }
        _ => Err(driver(
            "getParameter",
            "MAX_VIEWPORT_DIMS was not a u32 pair",
        )),
    }
}

pub(super) fn browser_identity() -> Result<String, GlError> {
    let global = js_sys::global();
    let navigator = Reflect::get(&global, &JsValue::from_str("navigator"))
        .map_err(|value| js_error("navigator", value))?;
    let user_agent = Reflect::get(&navigator, &JsValue::from_str("userAgent"))
        .map_err(|value| js_error("navigator.userAgent", value))?;
    user_agent.as_string().ok_or_else(|| {
        driver(
            "navigator.userAgent",
            "browser returned a non-string identity",
        )
    })
}

/// Accepts exactly the `WebGL 2.x` family and rejects look-alike strings.
///
/// The major component must be the single digit `2` followed by `.` or the
/// end of the string, so `"WebGL 20"` and `"WebGL 2foo"` both fail (audit
/// P2-9); minor and vendor text are retained verbatim elsewhere.
pub(super) fn require_webgl2_version(version: &str) -> Result<(), GlError> {
    let accepted = version.strip_prefix("WebGL ").and_then(|remainder| {
        let (major, rest) = remainder.split_at(1.min(remainder.len()));
        (major == "2" && (rest.is_empty() || rest.starts_with('.'))).then_some(())
    });
    accepted.ok_or_else(|| driver("getParameter(VERSION)", "context did not report WebGL 2"))
}

pub(super) fn discovery_error(error: GlDiscoveryError) -> GlError {
    driver("build WebGL2 discovery", &format!("{error:?}"))
}

pub(super) fn js_error(operation: &'static str, value: JsValue) -> GlError {
    driver(operation, &format!("browser exception: {value:?}"))
}

pub(super) fn driver(operation: &'static str, message: &str) -> GlError {
    GlError::Driver {
        operation,
        message: message.to_owned(),
    }
}
