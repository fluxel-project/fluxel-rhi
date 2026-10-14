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
}

pub(super) struct BrowserSurfaceDepthPass {
    pub(super) framebuffer: WebGlFramebuffer,
    pub(super) color: WebGlTexture,
    /// Retaining this wrapper keeps the caller's Depth32 allocation alive
    /// after `RasterPassDesc` stops borrowing its texture view.
    pub(super) depth: BrowserTexture,
    pub(super) extent: [u32; 2],
}

/// A WebGL buffer tied to exactly one browser owner.
#[derive(Clone)]
pub(crate) struct BrowserBuffer {
    pub(super) owner: BrowserOwner,
    pub(super) raw: WebGlBuffer,
    pub(super) size: u64,
}

/// RGBA8 texture storage tied to exactly one browser owner.
#[derive(Clone)]
pub(crate) struct BrowserTexture {
    pub(super) owner: BrowserOwner,
    pub(super) raw: WebGlTexture,
    pub(super) desc: GlTextureDesc,
}

/// Linked ESSL 3.00 raster program owned by one browser context.
#[derive(Clone)]
pub(crate) struct BrowserRasterProgram {
    pub(super) owner: BrowserOwner,
    pub(super) raw: WebGlProgram,
}

/// Vertex array state owned by one browser context.
#[derive(Clone)]
pub(crate) struct BrowserVertexArray {
    pub(super) owner: BrowserOwner,
    pub(super) raw: WebGlVertexArrayObject,
}

/// Sampler state tied to one browser context.
#[derive(Clone)]
pub(crate) struct BrowserSampler {
    pub(super) owner: BrowserOwner,
    pub(super) raw: WebGlSampler,
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
        })
    }

    /// Marks this context permanently unusable and wakes all waiters. No WebGL
    /// deletion call is made because context loss invalidates every JS GL object.
    pub(crate) fn mark_context_lost(&self) {
        self.state.borrow_mut().mark_lost();
    }

    /// Configures the canvas backing store and returns the browser-selected
    /// drawing-buffer extent.  CSS layout and DOM resize notifications remain
    /// Host concerns; this changes only the RHI rendering backing store.
    pub(crate) fn resize_drawing_buffer(&self, extent: [u32; 2]) -> Result<[u32; 2]> {
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
        if state.context.raw.is_context_lost() || state.context.raw.get_error() != Gl::NO_ERROR {
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
                for (_, waker) in fence.wakers {
                    waker.wake();
                }
                completion.store_terminal(Ok(()))
            }
            Gl::TIMEOUT_EXPIRED => CompletionState::Pending,
            _ => {
                if let Some(fence) = state.completions.remove(&serial) {
                    state.context.raw.delete_sync(Some(&fence.raw));
                    for (_, waker) in fence.wakers {
                        waker.wake();
                    }
                }
                completion.store_terminal(Err(Error::Backend(
                    "WebGL2 clientWaitSync returned an unknown result".into(),
                )))
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
            self.schedule_completion_poll(completion.clone());
        }
        status
    }

    pub(super) fn unregister_completion_waker(
        &self,
        completion: &BrowserCompletion,
        registration: u64,
    ) {
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
        if state.context.raw.is_context_lost() || state.context.raw.get_error() != Gl::NO_ERROR {
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
            if let Some(fence) = self.state.borrow_mut().completions.get_mut(&serial) {
                fence.poll_scheduled = false;
            }
            return;
        };
        let owner = self.clone();
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
                owner.completion_status(&completion),
                CompletionState::Pending
            ) {
                owner.schedule_completion_poll(completion);
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
            if let Some(fence) = self.state.borrow_mut().completions.get_mut(&serial) {
                fence.poll_scheduled = false;
            }
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
            match self.context.raw.client_wait_sync_with_f64(&raw, 0, 0.0) {
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
