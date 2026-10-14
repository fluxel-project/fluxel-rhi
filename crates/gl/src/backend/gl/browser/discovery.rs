//! Browser-owned WebGL2 discovery evidence and the RHI-owned context record.
//!
//! The Host/JS bridge owns canvas creation, DOM events, RAF, and context-loss
//! listeners. RHI creates and owns the WebGL2 context associated with that
//! Host-provided canvas, then gathers immutable evidence for its `ContextStamp`.
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

use super::discovery_facts::*;
pub(crate) struct WebGl2BrowserDiscovery {
    pub(super) canvas: HtmlCanvasElement,
    pub(super) raw: WebGl2RenderingContext,
    pub(super) glow: glow::Context,
    pub(super) snapshot: GlDiscoverySnapshot,
    pub(super) owner_thread: OwnerThreadIdentity,
    pub(super) lifecycle: Cell<GlContextLifecycle>,
    pub(super) buffers: BTreeMap<u32, BrowserBuffer>,
    pub(super) textures: BTreeMap<u32, BrowserTexture>,
    pub(super) samplers: BTreeMap<u32, BrowserSampler>,
    pub(super) renderbuffers: BTreeMap<u32, BrowserRenderbuffer>,
    pub(super) shaders: BTreeMap<u32, BrowserShader>,
    pub(super) programs: BTreeMap<u32, BrowserProgram>,
    pub(super) vertex_arrays: BTreeMap<u32, BrowserVertexArray>,
    pub(super) framebuffers: BTreeMap<u32, BrowserFramebuffer>,
    pub(super) queries: BTreeMap<u32, BrowserQuery>,
    pub(super) syncs: BTreeMap<u32, BrowserSync>,
    pub(super) fences: GlFenceLeaseBook,
    pub(super) surface: GlSurfaceLeaseBook,
    pub(super) surface_suspended: bool,
    pub(super) pass: Option<ActivePass>,
    pub(super) raster: Option<ActiveRaster>,
    /// The vertex array the modelled driver holds, or `None` when it holds none.
    ///
    /// GL has one vertex-array binding slot, and this provider models the slot
    /// rather than the intent of each verb: a pipeline install and an input
    /// reconcile both reach it, and the install's choice is the older of the two
    /// by the time a draw runs.  A draw resolves *this* record and not the array
    /// the pipeline install named, because the geometry domain replaces that
    /// array on every request under the uncached execution mode.  The two verbs
    /// that bind a real array write it -- the input domain's `bind_vertex_array`
    /// and a raster pipeline install -- and `prepare_draw` is what reads it.
    pub(super) bound_vertex_array: Option<VertexArrayId>,
    /// The batch commands retained from the acquired batch extension object.
    ///
    /// `Some` exactly when that object exposed every command the batch domain
    /// issues, so it is also exactly when the batch capability resolved (audit
    /// P1-8). Dropped with the rest of the executable state on loss, because the
    /// object belongs to the dead context.
    pub(super) multi_draw: Option<BrowserMultiDraw>,
    /// The timer commands retained from the acquired timer-query object.
    ///
    /// `Some` exactly when that object exposed every command the timer domain
    /// issues, so it is also exactly when the timer capability could resolve
    /// (audit P1-5). Dropped with the rest of the executable state on loss,
    /// because the object belongs to the dead context.
    pub(super) timer: Option<BrowserTimerQuery>,
    /// `WEBGL_compressed_texture_astc` exposes LDR and HDR as profile strings
    /// on the extension object.  HDR must not inherit LDR admission merely
    /// because the object exists.
    pub(super) astc_hdr: bool,
    /// Slot of the query currently recording, if any.
    pub(super) active_query: Option<u32>,
    pub(super) next_buffer_slot: u32,
    pub(super) next_texture_slot: u32,
    pub(super) next_sampler_slot: u32,
    pub(super) next_renderbuffer_slot: u32,
    pub(super) next_shader_slot: u32,
    pub(super) next_program_slot: u32,
    pub(super) next_vertex_array_slot: u32,
    pub(super) next_framebuffer_slot: u32,
    pub(super) next_query_slot: u32,
    pub(super) next_sync_slot: u32,
    pub(super) next_surface_slot: u32,
    pub(super) pixel_store: GlPixelStoreState,
}

pub(super) struct BrowserBuffer {
    pub(super) generation: u32,
    pub(super) raw: WebGlBuffer,
    pub(super) desc: GlBufferDesc,
}
pub(super) struct BrowserTexture {
    pub(super) generation: u32,
    pub(super) raw: WebGlTexture,
    pub(super) desc: GlTextureDesc,
}
pub(super) struct BrowserSampler {
    pub(super) generation: u32,
    pub(super) raw: WebGlSampler,
}

impl WebGl2BrowserDiscovery {
    /// Creates and owns a WebGL2 context for a Host-provided canvas, then
    /// discovers its current generation without issuing a rendering command.
    ///
    /// This deliberately preserves the 0.14 readback/residency-oracle setting:
    /// the default framebuffer remains available after presentation.
    pub(crate) fn open(stamp: ContextStamp, canvas: HtmlCanvasElement) -> Result<Self, GlError> {
        // Context creation is itself a browser side effect, so ownership is
        // captured before configuring or touching the canvas.
        let owner_thread = OwnerThreadIdentity::current();
        let options = Object::new();
        Reflect::set(
            &options,
            &JsValue::from_str("preserveDrawingBuffer"),
            &JsValue::TRUE,
        )
        .map_err(|value| js_error("configure WebGL2 context", value))?;
        // BrowserSurface advertises only CompositeAlphaMode::Opaque. Disable
        // the default canvas alpha channel so canvas compositing cannot apply
        // a second, implicit blend to RHI output.
        Reflect::set(&options, &JsValue::from_str("alpha"), &JsValue::FALSE)
            .map_err(|value| js_error("configure WebGL2 context", value))?;
        // The RHI canvas target is single-sampled; multisampling is explicit
        // through texture attachments and resolves, never a hidden browser
        // default-framebuffer sample count.
        Reflect::set(&options, &JsValue::from_str("antialias"), &JsValue::FALSE)
            .map_err(|value| js_error("configure WebGL2 context", value))?;
        let value = canvas
            .get_context_with_context_options("webgl2", &options)
            .map_err(|value| js_error("create WebGL2 context", value))?
            .ok_or_else(|| driver("create WebGL2 context", "browser did not provide WebGL2"))?;
        let raw = value.dyn_into::<WebGl2RenderingContext>().map_err(|_| {
            driver(
                "create WebGL2 context",
                "browser returned a non-WebGL2 context",
            )
        })?;
        Self::from_owned_context(stamp, canvas, raw, owner_thread)
    }

    /// Internal ownership-transfer seam for RHI context restoration only.
    ///
    /// It is intentionally private: Host never creates or owns the WebGL2
    /// context in the Fluxel boundary model.
    fn from_owned_context(
        stamp: ContextStamp,
        canvas: HtmlCanvasElement,
        raw: WebGl2RenderingContext,
        owner_thread: OwnerThreadIdentity,
    ) -> Result<Self, GlError> {
        ensure_context_live(&raw, "discover WebGL2 context")?;
        let version = string_parameter(&raw, WebGl2RenderingContext::VERSION, "VERSION")?;
        require_webgl2_version(&version)?;

        // Do this after validating the only glow constructor precondition that
        // comes from application-visible browser state.  We deliberately do
        // not use glow for discovery: its browser constructor panics on JS
        // binding invariant violations instead of returning `Result`.
        let glow = glow::Context::from_webgl2_context(raw.clone());
        let (extensions, commands) = discover_extensions(&raw)?;
        // The masked vendor/renderer strings above are the context's own answers
        // and stay verbatim; the unmasked pair, when the context exposes the
        // optional debug route, is recorded beside them as flag markers (audit
        // P2-6) rather than replacing an answer the browser chose to give.
        let mut context_flags = discover_context_flags(&raw)?;
        super::driver_identity::record_markers(
            &mut context_flags,
            super::driver_identity::unmasked_identity(&raw),
        );
        let context = GlContextInfo::new(
            GlFamilyProfile::WebGl2,
            version,
            string_parameter(
                &raw,
                WebGl2RenderingContext::SHADING_LANGUAGE_VERSION,
                "SHADING_LANGUAGE_VERSION",
            )?,
            string_parameter(&raw, WebGl2RenderingContext::VENDOR, "VENDOR")?,
            string_parameter(&raw, WebGl2RenderingContext::RENDERER, "RENDERER")?,
            browser_identity()?,
            context_flags,
        );
        // The timer counter width is not a context limit: it is answered by the
        // acquired timer object, and it is zero whenever that object is absent
        // or does not answer both timer targets.
        let limits = discover_limits(&raw, &extensions, commands.timer.as_ref())?;
        // Float and depth format facts are answered by real framebuffer
        // completeness probes on this exact context (audit P1-6); the probes
        // use scratch objects and leave no state behind.
        let mut formats = super::format_map::webgl2_baseline_formats(&raw, &extensions)?;
        if commands.astc_hdr {
            super::format_map::record_astc_hdr_formats(&mut formats)?;
        }
        // Renderbuffer facts are answered the same way, by allocating scratch
        // storage and reading attachment completeness back, so a renderbuffer
        // allocation is admitted only from an observation on this context.
        super::renderbuffer_facts::record_facts(&raw, &limits, &mut formats)?;
        let mut builder = GlDiscoveryBuilder::new(stamp, context, extensions, limits, formats)
            .map_err(discovery_error)?;
        // The drawing buffer is this family's FBO 0, and it is observed through
        // the same drawing-buffer parameters the native path reads. The answers
        // depend on the context attributes the canvas was created with, so they
        // are read rather than assumed from the specification: a context with no
        // alpha really does report a zero alpha width, and a presenter that
        // assumed RGBA would claim a format this context does not have. A
        // parameter that cannot be read leaves the facts unavailable, which is
        // the answer that rejects work rather than the one that guesses.
        builder.surface_facts(drawing_buffer_facts(&raw));

        // Timer queries are the sole currently normalized browser extension
        // domain, and their oracle is the complete, callable entry-point set
        // acquired below together with a nonzero counter width read from the
        // same object: the extension alone would prove nothing about whether a
        // measurement can ever be reported (audit P1-5).
        builder.resolve(
            GlCapability::TimerQuery,
            CoreOrExtension {
                desktop_core: None,
                embedded_core: None,
                extension: Some(GlKnownExtension::ExtDisjointTimerQueryWebgl2),
                extension_requires_probe: false,
            },
            GlOperationProbe::NotRequired,
        );
        // OVR_multiview2 is acquired when the browser exposes the extension
        // object, but no multiview attachment probe exists yet: the pass state
        // that would have to carry a view count is not modelled, so a pass that
        // claims several views would attach a single layer and render the wrong
        // picture. The row therefore records the real evidence and stays
        // permanently disabled until the probe and the multiview attach path
        // land together (audit P1-7).
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
        // WEBGL_multi_draw is the browser's only per-draw parameter batch. Its
        // oracle is the complete, callable entry-point set acquired below
        // together with the per-draw validation each draw already goes through;
        // a batch therefore enables here once the extension object proves its
        // commands, and `multi_draw` falls back to single draws when it does not
        // (audit P1-8).
        builder.resolve(
            GlCapability::MultiDraw,
            CoreOrExtension {
                desktop_core: None,
                embedded_core: None,
                extension: Some(GlKnownExtension::WebglMultiDraw),
                extension_requires_probe: false,
            },
            GlOperationProbe::NotRequired,
        );
        // A context may be lost between any two browser calls. Do not publish
        // a discovery snapshot across that boundary.
        ensure_context_live(&raw, "finish WebGL2 discovery")?;
        Ok(Self {
            canvas,
            raw,
            glow,
            snapshot: builder.build(),
            owner_thread,
            lifecycle: Cell::new(GlContextLifecycle::Active),
            buffers: BTreeMap::new(),
            textures: BTreeMap::new(),
            samplers: BTreeMap::new(),
            renderbuffers: BTreeMap::new(),
            shaders: BTreeMap::new(),
            programs: BTreeMap::new(),
            vertex_arrays: BTreeMap::new(),
            framebuffers: BTreeMap::new(),
            queries: BTreeMap::new(),
            syncs: BTreeMap::new(),
            fences: GlFenceLeaseBook::default(),
            surface: GlSurfaceLeaseBook::new(),
            surface_suspended: false,
            pass: None,
            raster: None,
            bound_vertex_array: None,
            multi_draw: commands.multi_draw,
            timer: commands.timer,
            astc_hdr: commands.astc_hdr,
            active_query: None,
            next_buffer_slot: 0,
            next_texture_slot: 0,
            next_sampler_slot: 0,
            next_renderbuffer_slot: 0,
            next_shader_slot: 0,
            next_program_slot: 0,
            next_vertex_array_slot: 0,
            next_framebuffer_slot: 0,
            next_query_slot: 0,
            next_sync_slot: 0,
            next_surface_slot: 0,
            pixel_store: GlPixelStoreState::DEFAULT,
        })
    }

    /// Returns the discovery evidence bound to the supplied stamp.
    pub(crate) fn snapshot(&self) -> &GlDiscoverySnapshot {
        &self.snapshot
    }

    /// Preflight every future browser provider call inside this module.
    ///
    /// Raw WebGL and glow handles deliberately have no crate-visible borrowing
    /// method, so an executable seam cannot bypass this owner/loss check.
    pub(super) fn assert_provider_ready(&self, operation: &'static str) -> Result<(), GlError> {
        self.assert_owner_thread(operation)?;
        let lifecycle = self.lifecycle.get();
        if lifecycle != GlContextLifecycle::Active {
            return Err(GlError::InvalidLifecycle {
                operation,
                lifecycle,
            });
        }
        if self.raw.is_context_lost() {
            // Host owns loss events, while RHI owns the execution boundary.
            // The first provider preflight that observes loss durably records
            // it before returning, preventing another browser API call.
            self.lifecycle.set(GlContextLifecycle::Lost);
            Err(GlError::ContextLost { operation })
        } else {
            Ok(())
        }
    }

    pub(super) fn assert_owner_thread(&self, operation: &'static str) -> Result<(), GlError> {
        let actual = OwnerThreadIdentity::current();
        if actual == self.owner_thread {
            Ok(())
        } else {
            Err(GlError::WrongThread {
                operation,
                expected: self.owner_thread,
                actual,
            })
        }
    }

    pub(super) fn allocate_slot(next: &mut u32, operation: &'static str) -> Result<u32, GlError> {
        let slot = *next;
        *next = next.checked_add(1).ok_or_else(|| GlError::Driver {
            operation,
            message: "object allocation slots exhausted".into(),
        })?;
        Ok(slot)
    }

    pub(super) fn validation(operation: &'static str, message: &'static str) -> GlError {
        GlError::Validation {
            operation,
            message: message.into(),
        }
    }

    pub(super) fn driver_error(&self, operation: &'static str) -> Result<(), GlError> {
        let error = self.raw.get_error();
        if error == WebGl2RenderingContext::NO_ERROR {
            Ok(())
        } else {
            Err(GlError::Driver {
                operation,
                message: format!("WebGL error 0x{error:04x}"),
            })
        }
    }

    pub(super) fn buffer(
        &self,
        operation: &'static str,
        id: BufferId,
    ) -> Result<&BrowserBuffer, GlError> {
        self.validate_object_context(operation, id.context)?;
        match self.buffers.get(&id.slot) {
            Some(entry) if entry.generation == id.generation => Ok(entry),
            _ => Err(Self::validation(operation, "buffer allocation is not live")),
        }
    }

    pub(super) fn texture(
        &self,
        operation: &'static str,
        id: TextureId,
    ) -> Result<&BrowserTexture, GlError> {
        self.validate_object_context(operation, id.context)?;
        match self.textures.get(&id.slot) {
            Some(entry) if entry.generation == id.generation => Ok(entry),
            _ => Err(Self::validation(
                operation,
                "texture allocation is not live",
            )),
        }
    }

    /// Maps a JS exception from a catch-typed binding into a driver error.
    pub(super) fn js_failure(operation: &'static str, value: JsValue) -> GlError {
        js_error(operation, value)
    }
}
