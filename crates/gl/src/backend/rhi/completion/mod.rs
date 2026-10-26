//! Host-visible completion evidence for the immediate GL backend.
//!
//! A completion owns one context-local fence and never moves GL work to a
//! worker. Native polling runs only when the owner-thread executor polls the
//! future, using a zero-time fence query. A bounded timer wakes that executor
//! later without calling GL itself, so a pending fence neither blocks the
//! runtime nor self-wakes in a busy loop. Browser completion delegates to its
//! owner, which follows the same owner-thread rule with browser tasks.
//!
//! Fence deletion is performed only after a terminal owner-thread observation.
//! Dropping an unobserved completion intentionally leaves its native fence to
//! context/provider teardown: `Drop` cannot safely make an arbitrary GL
//! context current, and premature deletion would invalidate completion
//! evidence still shared by a cloned Core point.

use super::types::{Gl, GlDevice, GlDeviceState};
use fluxel_rhi_core::api::{CompletionState, Error, Result};
use fluxel_rhi_core::backend::CompletionBackend;
use std::{
    future::Future,
    marker::PhantomData,
    pin::Pin,
    rc::Rc,
    task::{Context, Poll},
};

#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
use {
    crate::backend::api::{GlFenceLease, GlFenceStatus, GlSyncApi as _},
    futures_timer::Delay,
    std::{cell::RefCell, rc::Weak, time::Duration},
};

#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
struct OrphanFence {
    state: Weak<GlDeviceState>,
    fence: GlFenceLease,
}

// The table is deliberately thread-local. `Rc<GlDeviceState>` makes GL
// completion points !Send, so their final release and every later collection
// occur on the same context-owning thread. Weak state avoids keeping a dead
// platform context alive merely because a completion was abandoned.
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
thread_local! {
    static ORPHAN_FENCES: RefCell<Vec<OrphanFence>> = const { RefCell::new(Vec::new()) };
}

/// Context-local completion evidence used by Core completion points.
///
/// The `Rc` retains the exact device state that created the evidence. Its
/// pointer identity is checked before every observation, preventing a fence
/// from one adopted GL context from being queried through another context.
#[derive(Clone)]
pub struct GlCompletion {
    state: Rc<GlDeviceState>,
    evidence: GlCompletionEvidence,
}

#[derive(Clone)]
enum GlCompletionEvidence {
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    Native(Rc<RefCell<Option<GlFenceLease>>>),
    #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
    Browser(crate::backend::browser::BrowserCompletion),
    /// Kept only in featureless builds so the backend type remains well formed
    /// while it truthfully reports that no platform completion exists.
    Unsupported,
}

impl GlCompletion {
    /// Captures completion evidence after direct commands have been issued.
    ///
    /// GL orders all earlier commands before the fence. Creating a checkpoint
    /// does not flush: presentation submits the frame, or the first explicit
    /// observation submits pending work when no presentation has done so.
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    pub(crate) fn after_issued(state: Rc<GlDeviceState>) -> Result<Self> {
        // Orphan collection polls fences and may publish pending commands.
        // Keep it at explicit observation, never checkpoint creation, so
        // dropping an older point cannot turn later checkpoints into flushes.
        let lease = state
            .with_native("create GL completion fence", |provider| {
                provider.create_fence()
            })
            .map_err(native_error)?;
        Ok(Self {
            state,
            evidence: GlCompletionEvidence::Native(Rc::new(RefCell::new(Some(lease)))),
        })
    }

    /// Wraps browser-owned completion evidence without exposing DOM handles to
    /// Core. The caller must use the `GlDeviceState` which owns this browser
    /// completion; `completion_status` checks that relation again.
    #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
    pub(crate) fn from_browser(
        state: Rc<GlDeviceState>,
        completion: crate::backend::browser::BrowserCompletion,
    ) -> Self {
        Self {
            state,
            evidence: GlCompletionEvidence::Browser(completion),
        }
    }
}

#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
impl Drop for GlCompletion {
    fn drop(&mut self) {
        let GlCompletionEvidence::Native(fence) = &self.evidence else {
            return;
        };
        // Only the final Core clone may remove the shared lease. Earlier
        // clones still make this fence observable, so quarantining it would
        // create a second owner and invalidate their completion point.
        if Rc::strong_count(fence) != 1 {
            return;
        }
        let Some(lease) = fence.borrow_mut().take() else {
            return;
        };
        ORPHAN_FENCES.with(|orphans| {
            orphans.borrow_mut().push(OrphanFence {
                state: Rc::downgrade(&self.state),
                fence: lease,
            });
        });
    }
}

/// Future returned by the GL completion SPI.
///
/// It deliberately carries the device borrow in `PhantomData`: a native wait
/// owns a clone of completion evidence, but Core's trait promises that the
/// wait cannot outlive the borrowed device supplied at construction.
pub struct GlWaitFuture<'a> {
    inner: GlWaitFutureInner,
    _device: PhantomData<&'a GlDevice>,
}

enum GlWaitFutureInner {
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    Native(NativeWaitFuture),
    #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
    Browser(crate::backend::browser::BrowserCompletionFuture),
    Ready(Option<Result<()>>),
}

impl<'a> GlWaitFuture<'a> {
    fn ready(result: Result<()>) -> Self {
        Self {
            inner: GlWaitFutureInner::Ready(Some(result)),
            _device: PhantomData,
        }
    }

    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    fn native(completion: GlCompletion) -> Self {
        Self {
            inner: GlWaitFutureInner::Native(NativeWaitFuture::new(completion)),
            _device: PhantomData,
        }
    }

    #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
    fn browser(wait: crate::backend::browser::BrowserCompletionFuture) -> Self {
        Self {
            inner: GlWaitFutureInner::Browser(wait),
            _device: PhantomData,
        }
    }
}

impl Future for GlWaitFuture<'_> {
    type Output = Result<()>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.as_mut().get_mut();
        match &mut this.inner {
            #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
            GlWaitFutureInner::Native(wait) => Pin::new(wait).poll(cx),
            #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
            GlWaitFutureInner::Browser(wait) => Pin::new(wait).poll(cx),
            GlWaitFutureInner::Ready(result) => {
                Poll::Ready(result.take().unwrap_or_else(|| Err(Error::InvalidState)))
            }
        }
    }
}

#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
/// Owner-thread wait primitive shared by Core completion points and private
/// native staging operations.  It owns only completion evidence and a timer;
/// polling it is the sole place that observes a GL fence.
pub(crate) struct NativeWaitFuture {
    completion: GlCompletion,
    retry: Option<Delay>,
    next_retry: Duration,
    #[cfg(test)]
    observations: usize,
}

#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
impl NativeWaitFuture {
    pub(crate) fn new(completion: GlCompletion) -> Self {
        Self {
            completion,
            retry: None,
            next_retry: Duration::from_millis(1),
            #[cfg(test)]
            observations: 0,
        }
    }
}

#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
impl Future for NativeWaitFuture {
    type Output = Result<()>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if let Some(retry) = self.retry.as_mut() {
            // A spurious executor poll must stay in Delay rather than querying
            // GL again. Once Delay resolves, this poll performs exactly one
            // zero-time observation below and installs a new delay if needed.
            if Pin::new(retry).poll(cx).is_pending() {
                return Poll::Pending;
            }
            self.retry = None;
        }
        #[cfg(test)]
        {
            self.observations += 1;
        }
        match native_completion_status(&self.completion.state, &self.completion) {
            CompletionState::Complete => Poll::Ready(Ok(())),
            CompletionState::Failed(error) => Poll::Ready(Err(error)),
            CompletionState::Pending => {
                let duration = self.next_retry;
                self.next_retry = next_retry_interval(duration);
                let retry = self.retry.insert(retry_delay(duration));
                let _ = Pin::new(retry).poll(cx);
                Poll::Pending
            }
        }
    }
}

/// A bounded delay supplied by `futures-timer` for a pending native fence.
///
/// `Delay` uses the calling thread's shared timer service and registers the
/// executor Waker through its normal `Future` contract. Dropping it cancels
/// that registration, so cancellation retains neither a GL handle nor an
/// executor reference. The delay service only wakes this future; it never
/// captures a provider or queries GL from another thread.
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
fn retry_delay(duration: Duration) -> Delay {
    Delay::new(duration)
}

/// Short critical-path observation, then bounded backoff for long GPU work.
/// A timer firing admits one query; extra executor polls never query GL.
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
fn next_retry_interval(previous: Duration) -> Duration {
    previous.saturating_mul(2).min(Duration::from_millis(8))
}
impl CompletionBackend for Gl {
    type WaitFuture<'a> = GlWaitFuture<'a>;

    fn completion_status(device: &GlDevice, point: &GlCompletion) -> CompletionState {
        if !Rc::ptr_eq(&device.0, &point.state) {
            return CompletionState::Failed(Error::WrongDevice);
        }
        completion_status(point)
    }

    fn wait_completion<'a>(device: &'a GlDevice, point: &'a GlCompletion) -> Self::WaitFuture<'a> {
        if !Rc::ptr_eq(&device.0, &point.state) {
            return GlWaitFuture::ready(Err(Error::WrongDevice));
        }
        #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
        if matches!(&point.evidence, GlCompletionEvidence::Native(_)) {
            return GlWaitFuture::native(point.clone());
        }
        #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
        if let GlCompletionEvidence::Browser(completion) = &point.evidence {
            return GlWaitFuture::browser(completion.wait());
        }
        GlWaitFuture::ready(Err(Error::Unsupported("GL platform completion")))
    }

    fn wait_idle<'a>(device: &'a GlDevice) -> Self::WaitFuture<'a> {
        #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
        {
            return match GlCompletion::after_issued(device.0.clone()) {
                Ok(completion) => GlWaitFuture::native(completion),
                Err(error) => GlWaitFuture::ready(Err(error)),
            };
        }
        #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
        {
            // BrowserOwner inserts a GLsync after previously issued calls.
            // Its cancellable owner-thread wait publishes pending commands
            // on first observation, then polls without repeated submission.
            return match device.0.owner.completion_after_current_commands() {
                Ok(completion) => GlWaitFuture::browser(completion.wait()),
                Err(error) => GlWaitFuture::ready(Err(error)),
            };
        }
        #[cfg(not(any(
            all(windows, feature = "native-gl-wgl"),
            feature = "native-gles-egl",
            all(target_arch = "wasm32", feature = "webgl2")
        )))]
        {
            let _ = device;
            GlWaitFuture::ready(Err(Error::Unsupported("GL platform completion")))
        }
    }
}

fn completion_status(point: &GlCompletion) -> CompletionState {
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    if matches!(&point.evidence, GlCompletionEvidence::Native(_)) {
        return native_completion_status(&point.state, point);
    }
    #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
    if let GlCompletionEvidence::Browser(completion) = &point.evidence {
        return completion.status();
    }
    CompletionState::Failed(Error::Unsupported("GL platform completion"))
}

#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
fn retire_orphans(current: &Rc<GlDeviceState>) {
    let pending = ORPHAN_FENCES.with(|orphans| std::mem::take(&mut *orphans.borrow_mut()));
    let mut keep = Vec::with_capacity(pending.len());
    for orphan in pending {
        let Some(state) = orphan.state.upgrade() else {
            continue;
        };
        if !Rc::ptr_eq(&state, current) {
            keep.push(orphan);
            continue;
        }
        let retired = state
            .with_native("retire abandoned GL completion fence", |provider| {
                match provider.poll_fence(orphan.fence)? {
                    GlFenceStatus::Complete => {
                        // A successful poll is the completion fact. Deletion
                        // diagnostics cannot make an already-complete fence
                        // pending again, and the provider owns any residue.
                        let _ = provider.destroy_fence(orphan.fence);
                        Ok(true)
                    }
                    GlFenceStatus::Pending | GlFenceStatus::Unknown | GlFenceStatus::Failed => {
                        Ok(false)
                    }
                }
            })
            .unwrap_or(false);
        if !retired {
            keep.push(orphan);
        }
    }
    ORPHAN_FENCES.with(|orphans| orphans.borrow_mut().extend(keep));
}

#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
fn native_completion_status(state: &Rc<GlDeviceState>, point: &GlCompletion) -> CompletionState {
    retire_orphans(state);
    let GlCompletionEvidence::Native(fence) = &point.evidence else {
        return CompletionState::Failed(Error::InvalidState);
    };
    let mut lease = fence.borrow_mut();
    let Some(current) = *lease else {
        return CompletionState::Complete;
    };
    match state.with_native("poll GL completion fence", |provider| {
        match provider.poll_fence(current)? {
            GlFenceStatus::Complete => {
                // Completion is established by the successful zero-time poll.
                // A later delete diagnostic cannot turn that fact into failure;
                // retaining the lease would invite a duplicate delete on the
                // next observation. The provider/context teardown owns any
                // driver-side residue from a failed deletion.
                let _ = provider.destroy_fence(current);
                Ok(GlFenceStatus::Complete)
            }
            status => Ok(status),
        }
    }) {
        Ok(GlFenceStatus::Complete) => {
            *lease = None;
            CompletionState::Complete
        }
        Ok(GlFenceStatus::Pending | GlFenceStatus::Unknown) => CompletionState::Pending,
        Ok(GlFenceStatus::Failed) => {
            CompletionState::Failed(Error::Backend("GL fence polling reported failure".into()))
        }
        Err(error) => CompletionState::Failed(native_error(error)),
    }
}

#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
fn native_error(error: crate::backend::api::GlError) -> Error {
    match error {
        crate::backend::api::GlError::ContextLost { .. } => Error::DeviceLost,
        crate::backend::api::GlError::OutOfMemory { .. } => Error::OutOfMemory,
        crate::backend::api::GlError::Shader { log, .. }
        | crate::backend::api::GlError::Program { log, .. } => Error::Shader(log),
        other => Error::Backend(format!("{other:?}")),
    }
}

#[cfg(all(test, windows, feature = "native-gl-wgl"))]
#[path = "../../../tests/backend/rhi/completion/tests.rs"]
mod tests;
