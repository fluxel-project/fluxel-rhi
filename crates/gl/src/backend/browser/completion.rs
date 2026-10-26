//! Shared WebGL completion evidence and cancellable host waits.
//!
//! A completion owns no command stream. It only shares the terminal result of
//! one already-issued GLsync so cloned Core observations remain stable after
//! the owner deletes the native fence.

use std::{
    cell::RefCell,
    future::Future,
    pin::Pin,
    rc::Rc,
    task::{Context, Poll},
};

use fluxel_rhi_core::api::{CompletionState, Result};

use super::owner::BrowserOwner;

/// Completion evidence inserted after commands already reached this context.
///
/// Clones share the terminal observation. Once a GLsync has signalled, every
/// clone reports the same result even though the owner has deleted that sync.
#[derive(Clone)]
pub(crate) struct BrowserCompletion {
    shared: Rc<BrowserCompletionShared>,
}

struct BrowserCompletionShared {
    owner: BrowserOwner,
    serial: u64,
    terminal: RefCell<Option<Result<()>>>,
}

/// A cancellable host wait for one browser fence.
///
/// This future never blocks its polling thread.  It installs one browser-task
/// observation and unregisters its waker when dropped, so abandoned async
/// tasks do not accumulate in the owner until unrelated GPU work completes.
pub(crate) struct BrowserCompletionFuture {
    pub(super) completion: BrowserCompletion,
    pub(super) registration: Option<u64>,
}

impl BrowserCompletion {
    pub(super) fn pending(owner: BrowserOwner, serial: u64) -> Self {
        Self {
            shared: Rc::new(BrowserCompletionShared {
                owner,
                serial,
                terminal: RefCell::new(None),
            }),
        }
    }

    pub(super) fn owner(&self) -> &BrowserOwner {
        &self.shared.owner
    }

    pub(super) fn serial(&self) -> u64 {
        self.shared.serial
    }

    pub(super) fn terminal_status(&self) -> Option<CompletionState> {
        self.shared
            .terminal
            .borrow()
            .as_ref()
            .map(|result| match result {
                Ok(()) => CompletionState::Complete,
                Err(error) => CompletionState::Failed(error.clone()),
            })
    }

    pub(super) fn store_terminal(&self, result: Result<()>) -> CompletionState {
        let status = match &result {
            Ok(()) => CompletionState::Complete,
            Err(error) => CompletionState::Failed(error.clone()),
        };
        *self.shared.terminal.borrow_mut() = Some(result);
        status
    }

    /// Polls this completion point without blocking the browser thread.
    pub(crate) fn status(&self) -> CompletionState {
        self.owner().completion_status(self)
    }

    /// Creates the backend's cancellable host wait future.
    pub(crate) fn wait(&self) -> BrowserCompletionFuture {
        self.owner().wait_completion(self)
    }
}

impl Drop for BrowserCompletion {
    fn drop(&mut self) {
        if Rc::strong_count(&self.shared) == 1 && self.shared.terminal.borrow().is_none() {
            self.shared.owner.orphan_completion(self.shared.serial);
        }
    }
}

impl Future for BrowserCompletionFuture {
    type Output = Result<()>;

    fn poll(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let completion = self.completion.clone();
        let owner = completion.owner().clone();
        let status =
            owner.register_completion_waker(&completion, &mut self.registration, context.waker());
        match status {
            CompletionState::Pending => Poll::Pending,
            CompletionState::Complete => {
                self.registration = None;
                Poll::Ready(Ok(()))
            }
            CompletionState::Failed(error) => {
                self.registration = None;
                Poll::Ready(Err(error))
            }
        }
    }
}

impl Drop for BrowserCompletionFuture {
    fn drop(&mut self) {
        if let Some(registration) = self.registration.take() {
            self.completion
                .owner()
                .unregister_completion_waker(&self.completion, registration);
        }
    }
}
