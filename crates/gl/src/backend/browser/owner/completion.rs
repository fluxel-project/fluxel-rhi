//! Browser-owner completion fence creation, observation, and polling.
//!
//! Completion polling is a browser timer callback on the context thread. It
//! never dispatches GL work to a worker or retains a CPU command stream.

use super::super::completion::{BrowserCompletion, BrowserCompletionFuture};
use super::*;
use fluxel_rhi_core::api::CompletionState;
use wasm_bindgen::JsCast;

impl BrowserOwner {
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
        if state.context.raw.is_context_lost() {
            // A lost context invalidates this new sync; only drop its JS handle.
            drop(raw);
            state.mark_lost();
            return Err(Error::DeviceLost);
        }
        // Do not force a queue submit for every checkpoint. Surface handoff
        // publishes this fence with the frame's one flush; headless callers
        // get the same progress publication on their first observation.
        state.pending_flush = true;
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
        if let Err(error) = state.ensure_active() {
            return completion.store_terminal(Err(error));
        }
        if state.pending_flush {
            // A headless completion has no present boundary. This is the
            // explicit progress request exception, and is coalesced for all
            // fences currently queued on this owner.
            state.context.raw.flush();
            state.pending_flush = false;
            if state.context.raw.is_context_lost() {
                state.mark_lost();
                return completion.store_terminal(Err(Error::DeviceLost));
            }
            if state.context.raw.get_error() != Gl::NO_ERROR {
                let error = Error::Backend("WebGL2 progress flush failed".into());
                let wakers = state.mark_failed(error.clone());
                drop(state);
                let status = completion.store_terminal(Err(error));
                for waker in wakers {
                    waker.wake();
                }
                return status;
            }
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
                let mut wakers = if let Some(fence) = state.completions.remove(&serial) {
                    state.context.raw.delete_sync(Some(&fence.raw));
                    fence.wakers.into_values().collect::<Vec<_>>()
                } else {
                    Vec::new()
                };
                // `clientWaitSync` reports a context queue failure rather
                // than a failure scoped to this one GLsync. Do not permit a
                // later completion to accept the same failed stream.
                let error =
                    Error::Backend("WebGL2 clientWaitSync returned an unknown result".into());
                wakers.extend(state.mark_failed(error.clone()));
                // See the signalled branch above: a task wake may poll this
                // owner immediately, so it cannot run while state is borrowed.
                drop(state);
                let status = completion.store_terminal(Err(error));
                for waker in wakers {
                    waker.wake();
                }
                status
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

    pub(in crate::backend::browser) fn register_completion_waker(
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

    pub(in crate::backend::browser) fn unregister_completion_waker(
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
    pub(in crate::backend::browser) fn quarantine_issued(&self) {
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
        if state.context.raw.is_context_lost() {
            drop(raw);
            state.mark_lost();
            return;
        }
        state.pending_flush = true;
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
    pub(in crate::backend::browser) fn orphan_completion(&self, serial: u64) {
        let mut state = self.state.borrow_mut();
        if let Some(fence) = state.completions.remove(&serial) {
            state.quarantined.push(fence.raw);
        }
    }
}

#[cfg(all(test, target_arch = "wasm32", feature = "webgl2"))]
#[path = "../../../tests/backend/browser/completion__tests.rs"]
mod tests;
