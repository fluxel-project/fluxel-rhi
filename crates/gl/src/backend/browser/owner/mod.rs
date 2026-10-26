//! Direct WebGL2 context ownership for the static immediate backend.
//!
//! `BrowserOwner` keeps JavaScript WebGL objects on the browser thread by using
//! `Rc<RefCell<_>>`. It is intentionally local: Core must not wrap it in an
//! owner-thread dispatcher or move it to a worker. Every `BrowserEncoder` verb
//! issues its WebGL call before returning; it never records a Rust command
//! stream for later replay.
//!
//! Context loss is terminal for this owner. A restored browser context creates
//! a fresh owner and Core `DeviceId`; native objects cannot be reused after
//! `mark_context_lost`.
//!
//! Resource wrappers share a small `Rc` lifetime token. The final wrapper
//! release deletes its WebGL name on this owner thread while the context is
//! live. If a reentrant drop overlaps an active `RefCell` borrow, only that
//! native deletion is deferred until the next encoder begins or the final
//! owner handle is dropped; this is not a deferred command stream. Context
//! loss skips deletion because the browser already invalidated those names.

use std::{cell::RefCell, collections::BTreeMap, rc::Rc, task::Waker};

use fluxel_rhi_core::api::{Error, LaneId, Result};
use web_sys::{
    WebGl2RenderingContext as Gl, WebGlBuffer, WebGlFramebuffer, WebGlProgram, WebGlRenderbuffer,
    WebGlSampler, WebGlSync, WebGlTexture, WebGlUniformLocation, WebGlVertexArrayObject,
};

use super::super::api::{GlContextLifecycle, GlDiscoverySnapshot, GlFormat, ProgramId};
use super::super::profile::GlProfile;
use super::discovery::WebGl2BrowserDiscovery;

/// Owner-local executable state for the surface sRGB conversion helper.
///
/// The programs are keyed by transfer direction and fragment output location;
/// they remain valid for this WebGL context generation only.
pub(super) struct SurfaceColorCache {
    pub(super) programs: BTreeMap<(bool, u32), SurfaceColorProgram>,
}

pub(super) struct SurfaceColorProgram {
    pub(super) raw: WebGlProgram,
    pub(super) vao: WebGlVertexArrayObject,
    pub(super) image: WebGlUniformLocation,
    pub(super) height: WebGlUniformLocation,
}

/// Owner-local stencil transfer executables. Readback feedback geometry is
/// supplied as uniforms, so one program remains valid for the context.
pub(super) struct StencilTransferCache {
    pub(super) write: Option<StencilWriteProgram>,
    pub(super) read_raster: Option<WebGlProgram>,
    pub(super) read_feedback: Option<StencilReadFeedbackProgram>,
}

pub(super) struct StencilWriteProgram {
    pub(super) raw: WebGlProgram,
    pub(super) bytes_in: WebGlUniformLocation,
    pub(super) origin: WebGlUniformLocation,
    pub(super) bit_index: WebGlUniformLocation,
}

#[derive(Clone)]
pub(super) struct StencilReadFeedbackProgram {
    pub(super) raw: WebGlProgram,
    pub(super) lo: WebGlUniformLocation,
    pub(super) hi: WebGlUniformLocation,
    pub(super) width: WebGlUniformLocation,
    pub(super) words_per_row: WebGlUniformLocation,
}

pub(super) struct NormalizedReadbackCache {
    pub(super) programs:
        BTreeMap<(GlFormat, NormalizedReadbackTextureKind), NormalizedReadbackProgram>,
    pub(super) highp_supported: bool,
}

/// The sampler shape is a program variant; transfer geometry is supplied as
/// uniforms and never contributes to this owner-local cache key.
#[derive(Clone, Copy, Eq, Ord, PartialEq, PartialOrd)]
pub(super) enum NormalizedReadbackTextureKind {
    D2,
    D2Array,
    D3,
}

pub(super) struct NormalizedReadbackProgram {
    pub(super) raw: WebGlProgram,
    pub(super) image: WebGlUniformLocation,
    pub(super) origin: WebGlUniformLocation,
    pub(super) mip: WebGlUniformLocation,
    pub(super) words_per_row: WebGlUniformLocation,
    pub(super) words_per_layer: Option<WebGlUniformLocation>,
    pub(super) elements_per_row: Option<WebGlUniformLocation>,
}

/// Browser-local owner for one WebGL2 context.
#[derive(Clone)]
pub(crate) struct BrowserOwner {
    pub(super) state: Rc<RefCell<BrowserOwnerState>>,
    /// Final drops can happen while an encoder temporarily mutably borrows
    /// `state`. Queue only the native names, then delete them at the next
    /// owner entry after the current immediate call has returned.
    pending_deletes: Rc<RefCell<Vec<BrowserDelete>>>,
}

enum BrowserDelete {
    Buffer(WebGlBuffer),
    Texture(WebGlTexture),
    Renderbuffer(WebGlRenderbuffer),
    Program { id: ProgramId, raw: WebGlProgram },
    VertexArray(WebGlVertexArrayObject),
    Sampler(WebGlSampler),
}

impl BrowserDelete {
    fn issue(self, state: &mut BrowserOwnerState) {
        match self {
            Self::Buffer(raw) => state.context.raw.delete_buffer(Some(&raw)),
            Self::Texture(raw) => state.context.raw.delete_texture(Some(&raw)),
            Self::Renderbuffer(raw) => state.context.raw.delete_renderbuffer(Some(&raw)),
            Self::Program { id, raw } => {
                if state
                    .context
                    .programs
                    .get(&id.slot)
                    .is_some_and(|entry| entry.generation == id.generation)
                {
                    state.context.programs.remove(&id.slot);
                }
                state.context.raw.delete_program(Some(&raw));
            }
            Self::VertexArray(raw) => state.context.raw.delete_vertex_array(Some(&raw)),
            Self::Sampler(raw) => state.context.raw.delete_sampler(Some(&raw)),
        }
    }
}

/// A context-local encoder scope. Its existence proves that calls are made
/// synchronously while one immediate Core encoder is open.
pub(crate) struct BrowserEncoder {
    pub(super) owner: BrowserOwner,
    pub(super) epoch: u64,
    pub(super) closed: bool,
    pub(super) issued: bool,
    /// The owned color target used only while a surface pass has a Core depth
    /// texture. WebGL cannot attach that depth texture to framebuffer zero.
    pub(super) surface_depth_pass: Option<BrowserSurfaceDepthPass>,
    /// The FBO used by a pass whose color attachment is a Core texture.  The
    /// caller owns both attachments; the encoder owns only this short-lived
    /// framebuffer and retains wrappers to keep the borrowed views live until
    /// the immediate pass closes.
    pub(super) offscreen_pass: Option<BrowserOffscreenPass>,
    pub(super) stencil_state: Option<super::super::api::GlDepthStencilState>,
    /// Last public raster scissor. Private mirror transfers may disable the
    /// test, so they restore this owned fact instead of querying WebGL state.
    pub(super) raster_scissor: Option<[i32; 4]>,
}

pub(super) struct BrowserSurfaceDepthPass {
    pub(super) framebuffer: WebGlFramebuffer,
    pub(super) color: WebGlTexture,
    /// Retaining this wrapper keeps the caller's Depth32 allocation alive
    /// after `RasterPassDesc` stops borrowing its texture view.
    pub(super) depth: BrowserTexture,
    pub(super) extent: [u32; 2],
    pub(super) srgb: bool,
}

/// Owner-local lifetime facts for one single-sample offscreen raster pass.
///
/// This is deliberately pass state, rather than a command packet: WebGL calls
/// have already been issued by the time it is constructed.  `framebuffer` is
/// deleted on either end or abort; `color` and `depth` are only retained to
/// prevent a caller from dropping their Core views while WebGL still uses them.
pub(super) struct BrowserOffscreenPass {
    pub(super) framebuffer: WebGlFramebuffer,
    pub(super) colors: Vec<BrowserTexture>,
    pub(super) depth: Option<BrowserTexture>,
    pub(super) extent: [u32; 2],
    pub(super) discard: Vec<u32>,
    pub(super) resolves: Vec<(
        u32,
        Option<BrowserTexture>,
        u32,
        u32,
        super::super::api::GlFormat,
    )>,
    pub(super) surface_color: Option<WebGlTexture>,
    pub(super) surface_slot: Option<u32>,
    pub(super) surface_srgb: bool,
    pub(super) render_area: fluxel_rhi_core::api::Rect,
}

pub(super) struct BrowserOwnerState {
    pub(super) context: WebGl2BrowserDiscovery,
    pub(super) lost: bool,
    /// A deferred WebGL stream error observed at publication. WebGL error
    /// state is global to the context, so it invalidates every outstanding
    /// completion rather than belonging to whichever fence was polled first.
    pub(super) failed: Option<Error>,
    pub(super) next_encoder_epoch: u64,
    pub(super) active_encoder: Option<u64>,
    pub(super) next_completion: u64,
    pub(super) next_waiter: u64,
    /// Commands or a fence have been issued since the last publication to the
    /// browser GL queue.  A surface handoff normally consumes this at the end
    /// of the frame; a completion observer consumes it only when progress is
    /// explicitly requested without a surface.
    pub(super) pending_flush: bool,
    pub(super) completions: BTreeMap<u64, BrowserFence>,
    /// Fences from issued encoders whose Core session was abandoned before it
    /// could receive evidence. They preserve the native lifetime boundary and
    /// are retired opportunistically; they are never exposed as completion.
    pub(super) quarantined: Vec<WebGlSync>,
    pub(super) raster_extent: Option<[u32; 2]>,
    pub(super) dummy_textures: BTreeMap<(u32, u32), WebGlTexture>,
    pub(super) dummy_samplers: BTreeMap<bool, WebGlSampler>,
    pub(super) dummy_uniform_buffers: BTreeMap<u32, WebGlBuffer>,
    pub(super) dual_source_limit: Option<u32>,
    pub(super) surface_color_cache: SurfaceColorCache,
    pub(super) stencil_transfer_cache: StencilTransferCache,
    pub(super) normalized_readback_cache: NormalizedReadbackCache,
    pub(super) normalized_readback_highp: bool,
}

pub(super) struct BrowserFence {
    pub(super) raw: WebGlSync,
    pub(super) wakers: BTreeMap<u64, Waker>,
    pub(super) poll_scheduled: bool,
}

mod completion;
mod mapping;
mod resources;

pub(crate) use resources::{
    BrowserBuffer, BrowserRasterProgram, BrowserSampler, BrowserTexture, BrowserVertexArray,
};

impl BrowserOwner {
    pub(crate) fn stencil_texturing_supported(&self) -> Result<bool> {
        let mut state = self.state.borrow_mut();
        state.ensure_active()?;
        state
            .context
            .raw
            .get_extension("WEBGL_stencil_texturing")
            .map(|extension| extension.is_some())
            .map_err(|_| Error::Backend("WebGL2 stencil extension query failed".into()))
    }
    pub(crate) fn dual_source_limit(&self) -> Result<u32> {
        let mut state = self.state.borrow_mut();
        state.ensure_active()?;
        if let Some(limit) = state.dual_source_limit {
            return Ok(limit);
        }
        let raw = &state.context.raw;
        if raw
            .get_extension("WEBGL_blend_func_extended")
            .map_err(|_| Error::Backend("WebGL dual-source extension query failed".into()))?
            .is_none()
        {
            state.dual_source_limit = Some(0);
            return Ok(0);
        }
        let value = raw
            .get_parameter(0x88FC)
            .map_err(|_| Error::Backend("WebGL dual-source limit query failed".into()))?;
        let limit = value.as_f64().unwrap_or(0.).max(0.) as u32;
        state.dual_source_limit = Some(limit);
        Ok(limit)
    }
    /// Takes ownership of an already-created WebGL2 context on its browser
    /// thread. The `Rc` is the thread-affinity proof: this handle is not Send.
    pub(crate) fn new(context: WebGl2BrowserDiscovery) -> Self {
        let normalized_readback_highp = context.normalized_readback_highp;
        Self {
            state: Rc::new(RefCell::new(BrowserOwnerState {
                context,
                lost: false,
                failed: None,
                next_encoder_epoch: 1,
                active_encoder: None,
                next_completion: 1,
                next_waiter: 1,
                pending_flush: false,
                completions: BTreeMap::new(),
                quarantined: Vec::new(),
                raster_extent: None,
                dummy_textures: BTreeMap::new(),
                dummy_samplers: BTreeMap::new(),
                dummy_uniform_buffers: BTreeMap::new(),
                dual_source_limit: None,
                surface_color_cache: SurfaceColorCache {
                    programs: BTreeMap::new(),
                },
                stencil_transfer_cache: StencilTransferCache {
                    write: None,
                    read_raster: None,
                    read_feedback: None,
                },
                normalized_readback_cache: NormalizedReadbackCache {
                    programs: BTreeMap::new(),
                    highp_supported: false,
                },
                normalized_readback_highp,
            })),
            pending_deletes: Rc::new(RefCell::new(Vec::new())),
        }
    }

    /// Returns immutable discovery evidence captured for this exact context
    /// generation. The clone contains no JavaScript handles and is safe for the
    /// static backend to inspect while selecting capabilities.
    pub(crate) fn discovery_snapshot(&self) -> GlDiscoverySnapshot {
        self.state.borrow().context.snapshot().clone()
    }

    /// Returns the profile selected during discovery for capability routing.
    pub(crate) fn profile(&self) -> GlProfile {
        self.state.borrow().context.snapshot().context().profile()
    }

    /// Returns the immutable context generation that owns every browser-native
    /// object created through this owner.
    pub(crate) fn context_stamp(&self) -> super::super::api::ContextStamp {
        self.state.borrow().context.snapshot().context_stamp()
    }

    /// Opens one direct-issue scope for the sole graphics lane.
    pub(crate) fn begin_encoder(&self, lane: LaneId) -> Result<BrowserEncoder> {
        self.drain_pending_deletes()?;
        let mut state = self.state.borrow_mut();
        state.ensure_active()?;
        state.retire_quarantined();
        // The static GL backend currently exposes one graphics lane at ordinal
        // zero. `LaneId` carries identity rather than its class; Core publishes
        // that class in `DeviceCapabilities` during device creation.
        if lane.ordinal != 0 {
            return Err(Error::WrongLane);
        }
        if state.active_encoder.is_some() {
            return Err(Error::InvalidState);
        }
        let epoch = state.allocate_encoder_epoch()?;
        state.active_encoder = Some(epoch);
        Ok(BrowserEncoder {
            owner: self.clone(),
            epoch,
            closed: false,
            issued: false,
            surface_depth_pass: None,
            offscreen_pass: None,
            stencil_state: None,
            raster_scissor: None,
        })
    }

    /// Marks this context permanently unusable and wakes all waiters. No WebGL
    /// deletion call is made because context loss invalidates every JS GL object.
    pub(crate) fn mark_context_lost(&self) {
        self.state.borrow_mut().mark_lost();
        self.pending_deletes.borrow_mut().clear();
    }

    /// Makes passive GL queries follow the same terminal path as host loss
    /// notifications, including waking completion observers.
    pub(crate) fn ensure_active(&self) -> Result<()> {
        self.state.borrow_mut().ensure_active()
    }

    /// Reports whether this context can still back surface images and work.
    pub(crate) fn is_active(&self) -> bool {
        self.ensure_active().is_ok()
    }

    /// Reads a browser buffer only when Core calls `map_readback`, which is
    /// reached after the caller has established the ticket's completion point.
    pub(crate) fn read_buffer(
        &self,
        source: &BrowserBuffer,
        offset: u64,
        size: u64,
    ) -> Result<Vec<u8>> {
        self.drain_pending_deletes()?;
        if !Rc::ptr_eq(&self.state, &source.owner.state) {
            return Err(Error::WrongDevice);
        }
        if offset.checked_add(size).is_none_or(|end| end > source.size) {
            return Err(Error::InvalidRange);
        }
        const MAX_EXACT_WEBGL_OFFSET: u64 = (1_u64 << 53) - 1;
        if offset > MAX_EXACT_WEBGL_OFFSET
            || size > MAX_EXACT_WEBGL_OFFSET
            || offset
                .checked_add(size)
                .is_none_or(|end| end > MAX_EXACT_WEBGL_OFFSET)
        {
            return Err(Error::InvalidRange);
        }
        let mut bytes = vec![0; usize::try_from(size).map_err(|_| Error::OutOfMemory)?];
        let mut state = self.state.borrow_mut();
        state.ensure_active()?;
        let raw = &state.context.raw;
        raw.bind_buffer(Gl::COPY_READ_BUFFER, Some(&source.raw));
        raw.get_buffer_sub_data_with_f64_and_u8_array(
            Gl::COPY_READ_BUFFER,
            offset as f64,
            &mut bytes,
        );
        let lost = raw.is_context_lost();
        let error = raw.get_error();
        // Readback is explicit and completion-gated, but it must not add a
        // second synchronous state query just to restore an RHI-owned target.
        raw.bind_buffer(Gl::COPY_READ_BUFFER, None);
        if lost {
            state.mark_lost();
            return Err(Error::DeviceLost);
        }
        if error != Gl::NO_ERROR {
            return Err(Error::Backend("WebGL2 buffer readback failed".into()));
        }
        Ok(bytes)
    }

    fn delete_if_active(&self, delete: BrowserDelete) {
        let Ok(mut state) = self.state.try_borrow_mut() else {
            // A wrapper can be released from a re-entrant browser callback.
            // Defer the name so it is not leaked and do not mutate GL state
            // until the current immediate owner borrow has ended.
            self.pending_deletes.borrow_mut().push(delete);
            return;
        };
        if state.ensure_active().is_ok() {
            self.issue_pending_deletes(&mut state);
            delete.issue(&mut state);
        }
    }

    pub(super) fn drain_pending_deletes(&self) -> Result<()> {
        let mut state = self
            .state
            .try_borrow_mut()
            .map_err(|_| Error::InvalidState)?;
        state.ensure_active()?;
        self.issue_pending_deletes(&mut state);
        if state.context.raw.is_context_lost() {
            state.mark_lost();
            return Err(Error::DeviceLost);
        }
        Ok(())
    }

    fn issue_pending_deletes(&self, state: &mut BrowserOwnerState) {
        let pending = std::mem::take(&mut *self.pending_deletes.borrow_mut());
        for delete in pending {
            delete.issue(state);
        }
    }

    fn delete_buffer(&self, raw: &WebGlBuffer) {
        self.delete_if_active(BrowserDelete::Buffer(raw.clone()));
    }

    fn delete_texture(&self, raw: &WebGlTexture) {
        self.delete_if_active(BrowserDelete::Texture(raw.clone()));
    }

    fn delete_renderbuffer(&self, raw: &WebGlRenderbuffer) {
        self.delete_if_active(BrowserDelete::Renderbuffer(raw.clone()));
    }

    fn delete_program(&self, id: ProgramId, raw: &WebGlProgram) {
        self.delete_if_active(BrowserDelete::Program {
            id,
            raw: raw.clone(),
        });
    }

    fn delete_vertex_array(&self, raw: &WebGlVertexArrayObject) {
        self.delete_if_active(BrowserDelete::VertexArray(raw.clone()));
    }

    fn delete_sampler(&self, raw: &WebGlSampler) {
        self.delete_if_active(BrowserDelete::Sampler(raw.clone()));
    }

    /// Configures the canvas backing store and returns the browser-selected
    /// drawing-buffer extent.  CSS layout and DOM resize notifications remain
    /// Host concerns; this changes only the RHI rendering backing store.
    pub(crate) fn resize_drawing_buffer(&self, extent: [u32; 2]) -> Result<[u32; 2]> {
        self.drain_pending_deletes()?;
        let mut state = self.state.borrow_mut();
        state.ensure_active()?;
        state.context.canvas.set_width(extent[0]);
        state.context.canvas.set_height(extent[1]);
        let actual = [
            state.context.raw.drawing_buffer_width().max(0) as u32,
            state.context.raw.drawing_buffer_height().max(0) as u32,
        ];
        if actual != extent {
            return Err(Error::SurfaceOutdated);
        }
        Ok(actual)
    }

    /// Returns the actual canvas drawing-buffer extent.  A zero dimension is
    /// treated by the surface layer as suspended rather than as an image lease.
    pub(crate) fn drawing_buffer_extent(&self) -> Result<[u32; 2]> {
        self.drain_pending_deletes()?;
        let mut state = self.state.borrow_mut();
        state.ensure_active()?;
        Ok([
            state.context.raw.drawing_buffer_width().max(0) as u32,
            state.context.raw.drawing_buffer_height().max(0) as u32,
        ])
    }

    /// Publishes commands already issued to the browser's GL queue without
    /// claiming GPU completion. Completion is established only by a fence.
    pub(crate) fn flush(&self) -> Result<()> {
        self.drain_pending_deletes()?;
        let mut state = self.state.borrow_mut();
        state.ensure_active()?;
        state.context.raw.flush();
        state.pending_flush = false;
        if state.context.raw.is_context_lost() {
            state.mark_lost();
            Err(Error::DeviceLost)
        } else if state.context.raw.get_error() == Gl::NO_ERROR {
            Ok(())
        } else {
            let error = Error::Backend("WebGL2 flush failed".into());
            let wakers = state.mark_failed(error.clone());
            drop(state);
            for waker in wakers {
                waker.wake();
            }
            Err(error)
        }
    }
}

impl Drop for BrowserOwner {
    fn drop(&mut self) {
        // A final resource can be released reentrantly while an encoder holds
        // the state borrow. In that case no owner entry may follow, so retry
        // queued native deletes when the last owner handle leaves scope. The
        // encoder itself owns a clone and therefore keeps this from running
        // until its immediate GL borrow has ended.
        if Rc::strong_count(&self.state) != 1 {
            return;
        }
        let Ok(mut state) = self.state.try_borrow_mut() else {
            return;
        };
        if state.ensure_active().is_ok() {
            self.issue_pending_deletes(&mut state);
            for (_, texture) in std::mem::take(&mut state.dummy_textures) {
                state.context.raw.delete_texture(Some(&texture));
            }
            for (_, sampler) in std::mem::take(&mut state.dummy_samplers) {
                state.context.raw.delete_sampler(Some(&sampler));
            }
            for (_, buffer) in std::mem::take(&mut state.dummy_uniform_buffers) {
                state.context.raw.delete_buffer(Some(&buffer));
            }
            for (_, helper) in std::mem::take(&mut state.surface_color_cache.programs) {
                state.context.raw.delete_vertex_array(Some(&helper.vao));
                state.context.raw.delete_program(Some(&helper.raw));
            }
            if let Some(helper) = state.stencil_transfer_cache.write.take() {
                state.context.raw.delete_program(Some(&helper.raw));
            }
            if let Some(program) = state.stencil_transfer_cache.read_raster.take() {
                state.context.raw.delete_program(Some(&program));
            }
            if let Some(helper) = state.stencil_transfer_cache.read_feedback.take() {
                state.context.raw.delete_program(Some(&helper.raw));
            }
            for (_, helper) in std::mem::take(&mut state.normalized_readback_cache.programs) {
                state.context.raw.delete_program(Some(&helper.raw));
            }
        }
    }
}

impl BrowserOwnerState {
    pub(super) fn normalized_readback_highp(&mut self) -> bool {
        self.normalized_readback_highp
    }
    pub(super) fn ensure_active(&mut self) -> Result<()> {
        if let Some(error) = &self.failed {
            return Err(error.clone());
        }
        if self.lost || self.context.raw.is_context_lost() {
            self.mark_lost();
            return Err(Error::DeviceLost);
        }
        Ok(())
    }
    pub(super) fn ensure_encoder(&self, epoch: u64) -> Result<()> {
        if self.active_encoder == Some(epoch) {
            Ok(())
        } else {
            Err(Error::InvalidState)
        }
    }
    fn allocate_encoder_epoch(&mut self) -> Result<u64> {
        let value = self.next_encoder_epoch;
        self.next_encoder_epoch = value.checked_add(1).ok_or(Error::OutOfMemory)?;
        Ok(value)
    }
    pub(super) fn allocate_completion(&mut self) -> Result<u64> {
        let value = self.next_completion;
        self.next_completion = value.checked_add(1).ok_or(Error::OutOfMemory)?;
        Ok(value)
    }
    fn allocate_waiter(&mut self) -> Result<u64> {
        let value = self.next_waiter;
        self.next_waiter = value.checked_add(1).ok_or(Error::OutOfMemory)?;
        Ok(value)
    }
    /// Reclaims only internal abandonment fences. User-visible completion
    /// points stay in `completions` until their owner observes them.
    fn retire_quarantined(&mut self) {
        let mut keep = Vec::new();
        for raw in std::mem::take(&mut self.quarantined) {
            let status = self.context.raw.client_wait_sync_with_f64(&raw, 0, 0.0);
            if self.context.raw.is_context_lost() {
                // The local `raw` and remaining iterator values are dropped
                // without deleteSync after loss invalidates every sync handle.
                self.mark_lost();
                return;
            }
            match status {
                Gl::TIMEOUT_EXPIRED => keep.push(raw),
                _ => self.context.raw.delete_sync(Some(&raw)),
            }
        }
        self.quarantined = keep;
    }
    pub(super) fn mark_lost(&mut self) {
        if self.lost {
            return;
        }
        self.lost = true;
        self.context.lifecycle.set(GlContextLifecycle::Lost);
        // WebGL invalidated these JS names already. Drop the retained program,
        // shader, query and timer handles without issuing deletes on the lost
        // context, just as the discovery provider does for a loss event.
        self.context.clear_executable_state();
        self.surface_color_cache.programs.clear();
        self.stencil_transfer_cache.write = None;
        self.stencil_transfer_cache.read_raster = None;
        self.stencil_transfer_cache.read_feedback = None;
        self.normalized_readback_cache.programs.clear();
        self.dummy_textures.clear();
        self.dummy_samplers.clear();
        self.dummy_uniform_buffers.clear();
        self.active_encoder = None;
        self.raster_extent = None;
        // Context loss invalidates WebGLSync objects. Do not call deleteSync
        // on that dead context; dropping the JS handles is the required
        // cleanup. Wakers are still released so all waiters observe loss.
        self.quarantined.clear();
        for (_, fence) in std::mem::take(&mut self.completions) {
            for (_, waker) in fence.wakers {
                waker.wake();
            }
        }
    }

    pub(super) fn mark_failed(&mut self, error: Error) -> Vec<Waker> {
        if self.failed.is_some() {
            return Vec::new();
        }
        self.failed = Some(error);
        self.active_encoder = None;
        self.raster_extent = None;
        std::mem::take(&mut self.completions)
            .into_values()
            .flat_map(|fence| fence.wakers.into_values())
            .collect()
    }
}
