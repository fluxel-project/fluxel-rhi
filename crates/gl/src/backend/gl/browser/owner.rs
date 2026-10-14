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

use fluxel_rhi_core::api::{CompletionState, Error, LaneId, Result};
use wasm_bindgen::JsCast;
use web_sys::{
    WebGl2RenderingContext as Gl, WebGlBuffer, WebGlFramebuffer, WebGlProgram, WebGlSampler,
    WebGlSync, WebGlTexture, WebGlVertexArrayObject,
};

use super::super::api::{GlDiscoverySnapshot, GlTextureDesc};
use super::super::profile::GlProfile;
use super::completion::{BrowserCompletion, BrowserCompletionFuture};
use super::discovery::WebGl2BrowserDiscovery;

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
    Program(WebGlProgram),
    VertexArray(WebGlVertexArrayObject),
    Sampler(WebGlSampler),
}

impl BrowserDelete {
    fn issue(self, gl: &Gl) {
        match self {
            Self::Buffer(raw) => gl.delete_buffer(Some(&raw)),
            Self::Texture(raw) => gl.delete_texture(Some(&raw)),
            Self::Program(raw) => gl.delete_program(Some(&raw)),
            Self::VertexArray(raw) => gl.delete_vertex_array(Some(&raw)),
            Self::Sampler(raw) => gl.delete_sampler(Some(&raw)),
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
}

pub(super) struct BrowserSurfaceDepthPass {
    pub(super) framebuffer: WebGlFramebuffer,
    pub(super) color: WebGlTexture,
    /// Retaining this wrapper keeps the caller's Depth32 allocation alive
    /// after `RasterPassDesc` stops borrowing its texture view.
    pub(super) depth: BrowserTexture,
    pub(super) extent: [u32; 2],
}

/// Owner-local lifetime facts for one single-sample offscreen raster pass.
///
/// This is deliberately pass state, rather than a command packet: WebGL calls
/// have already been issued by the time it is constructed.  `framebuffer` is
/// deleted on either end or abort; `color` and `depth` are only retained to
/// prevent a caller from dropping their Core views while WebGL still uses them.
pub(super) struct BrowserOffscreenPass {
    pub(super) framebuffer: WebGlFramebuffer,
    pub(super) color: BrowserTexture,
    pub(super) depth: Option<BrowserTexture>,
    pub(super) extent: [u32; 2],
}

/// A WebGL buffer tied to exactly one browser owner.
#[derive(Clone)]
pub(crate) struct BrowserBuffer {
    pub(super) owner: BrowserOwner,
    pub(super) raw: WebGlBuffer,
    pub(super) size: u64,
    pub(super) lifetime: Rc<()>,
}

/// RGBA8 texture storage tied to exactly one browser owner.
#[derive(Clone)]
pub(crate) struct BrowserTexture {
    pub(super) owner: BrowserOwner,
    pub(super) raw: WebGlTexture,
    pub(super) desc: GlTextureDesc,
    pub(super) lifetime: Rc<()>,
}

/// Linked ESSL 3.00 raster program owned by one browser context.
#[derive(Clone)]
pub(crate) struct BrowserRasterProgram {
    pub(super) owner: BrowserOwner,
    pub(super) raw: WebGlProgram,
    pub(super) lifetime: Rc<()>,
}

/// Vertex array state owned by one browser context.
#[derive(Clone)]
pub(crate) struct BrowserVertexArray {
    pub(super) owner: BrowserOwner,
    pub(super) raw: WebGlVertexArrayObject,
    pub(super) lifetime: Rc<()>,
}

/// Sampler state tied to one browser context.
#[derive(Clone)]
pub(crate) struct BrowserSampler {
    pub(super) owner: BrowserOwner,
    pub(super) raw: WebGlSampler,
    pub(super) lifetime: Rc<()>,
}

pub(super) struct BrowserOwnerState {
    pub(super) context: WebGl2BrowserDiscovery,
    pub(super) lost: bool,
    pub(super) next_encoder_epoch: u64,
    pub(super) active_encoder: Option<u64>,
    pub(super) next_completion: u64,
    pub(super) next_waiter: u64,
    pub(super) completions: BTreeMap<u64, BrowserFence>,
    /// Fences from issued encoders whose Core session was abandoned before it
    /// could receive evidence. They preserve the native lifetime boundary and
    /// are retired opportunistically; they are never exposed as completion.
    pub(super) quarantined: Vec<WebGlSync>,
    pub(super) raster_extent: Option<[u32; 2]>,
}

pub(super) struct BrowserFence {
    pub(super) raw: WebGlSync,
    pub(super) wakers: BTreeMap<u64, Waker>,
    pub(super) poll_scheduled: bool,
}

impl BrowserOwner {
    /// Takes ownership of an already-created WebGL2 context on its browser
    /// thread. The `Rc` is the thread-affinity proof: this handle is not Send.
    pub(crate) fn new(context: WebGl2BrowserDiscovery) -> Self {
        Self {
            state: Rc::new(RefCell::new(BrowserOwnerState {
                context,
                lost: false,
                next_encoder_epoch: 1,
                active_encoder: None,
                next_completion: 1,
                next_waiter: 1,
                completions: BTreeMap::new(),
                quarantined: Vec::new(),
                raster_extent: None,
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
        })
    }

    /// Marks this context permanently unusable and wakes all waiters. No WebGL
    /// deletion call is made because context loss invalidates every JS GL object.
    pub(crate) fn mark_context_lost(&self) {
        self.state.borrow_mut().mark_lost();
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
        let mut bytes = vec![0; usize::try_from(size).map_err(|_| Error::OutOfMemory)?];
        let mut state = self.state.borrow_mut();
        state.ensure_active()?;
        state
            .context
            .raw
            .bind_buffer(Gl::COPY_READ_BUFFER, Some(&source.raw));
        state.context.raw.get_buffer_sub_data_with_f64_and_u8_array(
            Gl::COPY_READ_BUFFER,
            offset as f64,
            &mut bytes,
        );
        if state.context.raw.is_context_lost() {
            state.mark_lost();
            return Err(Error::DeviceLost);
        }
        if state.context.raw.get_error() != Gl::NO_ERROR {
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
            delete.issue(&state.context.raw);
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
            delete.issue(&state.context.raw);
        }
    }

    fn delete_buffer(&self, raw: &WebGlBuffer) {
        self.delete_if_active(BrowserDelete::Buffer(raw.clone()));
    }

    fn delete_texture(&self, raw: &WebGlTexture) {
        self.delete_if_active(BrowserDelete::Texture(raw.clone()));
    }

    fn delete_program(&self, raw: &WebGlProgram) {
        self.delete_if_active(BrowserDelete::Program(raw.clone()));
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
        if state.context.raw.is_context_lost() {
            state.mark_lost();
            Err(Error::DeviceLost)
        } else if state.context.raw.get_error() == Gl::NO_ERROR {
            Ok(())
        } else {
            Err(Error::Backend("WebGL2 flush failed".into()))
        }
    }

    /// Places a completion fence after commands already issued to this WebGL
    /// context. This is host-observation evidence only; it neither closes an
    /// immediate encoder nor invents a queue wait. An open encoder is refused
    /// because its caller still owns the exact issued-prefix contract.
    pub(crate) fn completion_after_current_commands(&self) -> Result<BrowserCompletion> {
        self.drain_pending_deletes()?;
        let mut state = self.state.borrow_mut();
        state.ensure_active()?;
        if state.active_encoder.is_some() {
            return Err(Error::InvalidState);
        }
        let Some(raw) = state
            .context
            .raw
            .fence_sync(Gl::SYNC_GPU_COMMANDS_COMPLETE, 0)
        else {
            state.mark_lost();
            return Err(Error::DeviceLost);
        };
        state.context.raw.flush();
        if state.context.raw.is_context_lost() {
            // A lost context invalidates this new sync; only drop its JS handle.
            drop(raw);
            state.mark_lost();
            return Err(Error::DeviceLost);
        }
        if state.context.raw.get_error() != Gl::NO_ERROR {
            state.context.raw.delete_sync(Some(&raw));
            state.mark_lost();
            return Err(Error::DeviceLost);
        }
        let serial = state.allocate_completion()?;
        state.completions.insert(
            serial,
            BrowserFence {
                raw,
                wakers: BTreeMap::new(),
                poll_scheduled: false,
            },
        );
        Ok(BrowserCompletion::pending(self.clone(), serial))
    }

    /// Polls a browser fence without waiting. A terminal result belongs to
    /// the completion handle, rather than the GLsync map, so cloned points can
    /// be observed repeatedly after the native sync is reclaimed.
    pub(crate) fn completion_status(&self, completion: &BrowserCompletion) -> CompletionState {
        if let Some(status) = completion.terminal_status() {
            return status;
        }
        if !Rc::ptr_eq(&self.state, &completion.owner().state) {
            return CompletionState::Failed(Error::WrongDevice);
        }
        if let Err(error) = self.drain_pending_deletes() {
            return completion.store_terminal(Err(error));
        }
        let mut state = self.state.borrow_mut();
        if state.ensure_active().is_err() {
            return completion.store_terminal(Err(Error::DeviceLost));
        }
        let serial = completion.serial();
        let Some(fence) = state.completions.get(&serial) else {
            return completion.store_terminal(Err(Error::InvalidState));
        };
        match state
            .context
            .raw
            .client_wait_sync_with_f64(&fence.raw, 0, 0.0)
        {
            Gl::ALREADY_SIGNALED | Gl::CONDITION_SATISFIED => {
                let fence = state.completions.remove(&serial).expect("live fence");
                state.context.raw.delete_sync(Some(&fence.raw));
                let wakers = fence.wakers;
                // Waking can synchronously poll this same completion. Release
                // the owner state and publish its terminal result first, so
                // re-entrant poll cannot observe a missing fence as InvalidState.
                drop(state);
                completion.store_terminal(Ok(()));
                for (_, waker) in wakers {
                    waker.wake();
                }
                CompletionState::Complete
            }
            Gl::TIMEOUT_EXPIRED => CompletionState::Pending,
            _ => {
                if state.context.raw.is_context_lost() {
                    let wakers = std::mem::take(&mut state.completions)
                        .into_values()
                        .flat_map(|fence| fence.wakers.into_values())
                        .collect::<Vec<_>>();
                    // Context loss invalidates GLsync objects. Drop their JS
                    // handles without deleteSync and publish one terminal
                    // device-loss fact to every observer.
                    state.mark_lost();
                    drop(state);
                    completion.store_terminal(Err(Error::DeviceLost));
                    for waker in wakers {
                        waker.wake();
                    }
                    return CompletionState::Failed(Error::DeviceLost);
                }
                let wakers = if let Some(fence) = state.completions.remove(&serial) {
                    state.context.raw.delete_sync(Some(&fence.raw));
                    fence.wakers
                } else {
                    BTreeMap::new()
                };
                // See the signalled branch above: a task wake may poll this
                // owner immediately, so it cannot run while state is borrowed.
                drop(state);
                completion.store_terminal(Err(Error::Backend(
                    "WebGL2 clientWaitSync returned an unknown result".into(),
                )));
                for (_, waker) in wakers {
                    waker.wake();
                }
                CompletionState::Failed(Error::Backend(
                    "WebGL2 clientWaitSync returned an unknown result".into(),
                ))
            }
        }
    }

    /// Returns a cancellable, nonblocking wait for this owner's fence.
    pub(crate) fn wait_completion(
        &self,
        completion: &BrowserCompletion,
    ) -> BrowserCompletionFuture {
        BrowserCompletionFuture {
            completion: completion.clone(),
            registration: None,
        }
    }

    pub(super) fn register_completion_waker(
        &self,
        completion: &BrowserCompletion,
        registration: &mut Option<u64>,
        waker: &Waker,
    ) -> CompletionState {
        let status = self.completion_status(completion);
        if matches!(status, CompletionState::Pending) {
            {
                let mut state = self.state.borrow_mut();
                let key = match registration {
                    Some(key) => *key,
                    None => match state.allocate_waiter() {
                        Ok(key) => {
                            *registration = Some(key);
                            key
                        }
                        Err(error) => return CompletionState::Failed(error),
                    },
                };
                let Some(fence) = state.completions.get_mut(&completion.serial()) else {
                    return CompletionState::Failed(Error::InvalidState);
                };
                let update = fence
                    .wakers
                    .get(&key)
                    .is_none_or(|existing| !existing.will_wake(waker));
                if update {
                    fence.wakers.insert(key, waker.clone());
                }
            }
            // Scheduling immediately polls the owner in some browser
            // runtimes, so the registration borrow must have ended first.
            self.schedule_completion_poll(completion.clone());
        }
        status
    }

    pub(super) fn unregister_completion_waker(
        &self,
        completion: &BrowserCompletion,
        registration: u64,
    ) {
        let _ = self.drain_pending_deletes();
        if let Some(fence) = self
            .state
            .borrow_mut()
            .completions
            .get_mut(&completion.serial())
        {
            fence.wakers.remove(&registration);
        }
    }

    /// Retains a fence for work whose issuing encoder was abandoned before it
    /// could return completion evidence to Core. Failure to create that fence
    /// makes the owner terminal: continuing would permit a later resource
    /// release to race an unprovable issued prefix.
    pub(super) fn quarantine_issued(&self) {
        if self.drain_pending_deletes().is_err() {
            return;
        }
        let mut state = self.state.borrow_mut();
        if state.ensure_active().is_err() {
            return;
        }
        let Some(raw) = state
            .context
            .raw
            .fence_sync(Gl::SYNC_GPU_COMMANDS_COMPLETE, 0)
        else {
            state.mark_lost();
            return;
        };
        state.context.raw.flush();
        if state.context.raw.is_context_lost() {
            drop(raw);
            state.mark_lost();
            return;
        }
        if state.context.raw.get_error() != Gl::NO_ERROR {
            state.context.raw.delete_sync(Some(&raw));
            state.mark_lost();
            return;
        }
        state.quarantined.push(raw);
    }

    /// Schedules one bounded browser-task observation for a pending fence.
    /// The task owns a completion clone only until it runs. If all callers
    /// cancel their waits, it finds no waker and does not schedule another.
    fn schedule_completion_poll(&self, completion: BrowserCompletion) {
        let serial = completion.serial();
        let should_schedule = {
            let mut state = self.state.borrow_mut();
            let Some(fence) = state.completions.get_mut(&serial) else {
                return;
            };
            if fence.poll_scheduled {
                false
            } else {
                fence.poll_scheduled = true;
                true
            }
        };
        if !should_schedule {
            return;
        }
        let Some(window) = web_sys::window() else {
            self.fail_completion_schedule(
                &completion,
                "WebGL2 completion polling requires a browser Window",
            );
            return;
        };
        let owner = self.clone();
        let callback_completion = completion.clone();
        let callback = wasm_bindgen::closure::Closure::once_into_js(move || {
            if let Some(fence) = owner.state.borrow_mut().completions.get_mut(&serial) {
                fence.poll_scheduled = false;
                if fence.wakers.is_empty() {
                    return;
                }
            } else {
                return;
            }
            if matches!(
                owner.completion_status(&callback_completion),
                CompletionState::Pending
            ) {
                owner.schedule_completion_poll(callback_completion);
            }
        });
        // `Closure::once_into_js` transfers the callback to the browser timer;
        // `setTimeout` retains it until invocation. The conversion only borrows
        // that live JS function for this synchronous registration.
        let callback: &js_sys::Function = callback.unchecked_ref();
        if window
            .set_timeout_with_callback_and_timeout_and_arguments_0(callback, 16)
            .is_err()
        {
            self.fail_completion_schedule(
                &completion,
                "scheduling WebGL2 completion polling failed",
            );
        }
    }

    /// Turns an unavailable browser timer into a terminal result. The sync is
    /// quarantined, so this error cannot release resources still in flight.
    fn fail_completion_schedule(&self, completion: &BrowserCompletion, message: &'static str) {
        let wakers = {
            let mut state = self.state.borrow_mut();
            if let Some(fence) = state.completions.remove(&completion.serial()) {
                state.quarantined.push(fence.raw);
                fence.wakers.into_values().collect::<Vec<_>>()
            } else {
                Vec::new()
            }
        };
        completion.store_terminal(Err(Error::Backend(message.into())));
        for waker in wakers {
            waker.wake();
        }
    }

    /// Moves an unobserved completion fence to the owner-local quarantine.
    /// This runs only when its final public handle is dropped; a later direct
    /// issue scope polls and deletes the sync without retaining user state.
    pub(super) fn orphan_completion(&self, serial: u64) {
        let mut state = self.state.borrow_mut();
        if let Some(fence) = state.completions.remove(&serial) {
            state.quarantined.push(fence.raw);
        }
    }
}

impl Drop for BrowserBuffer {
    fn drop(&mut self) {
        if Rc::strong_count(&self.lifetime) == 1 {
            self.owner.delete_buffer(&self.raw);
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
        }
    }
}

impl Drop for BrowserTexture {
    fn drop(&mut self) {
        if Rc::strong_count(&self.lifetime) == 1 {
            self.owner.delete_texture(&self.raw);
        }
    }
}

impl Drop for BrowserRasterProgram {
    fn drop(&mut self) {
        if Rc::strong_count(&self.lifetime) == 1 {
            self.owner.delete_program(&self.raw);
        }
    }
}

impl Drop for BrowserVertexArray {
    fn drop(&mut self) {
        if Rc::strong_count(&self.lifetime) == 1 {
            self.owner.delete_vertex_array(&self.raw);
        }
    }
}

impl Drop for BrowserSampler {
    fn drop(&mut self) {
        if Rc::strong_count(&self.lifetime) == 1 {
            self.owner.delete_sampler(&self.raw);
        }
    }
}

impl BrowserOwnerState {
    pub(super) fn ensure_active(&mut self) -> Result<()> {
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
}
