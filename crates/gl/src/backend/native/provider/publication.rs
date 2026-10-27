//! Deferred command-stream publication and diagnostic boundaries.
//!
//! Ordinary lowering records the first operation that needs attribution but
//! never samples GL's sticky error flag. Presentation, explicit flushes, and
//! the first observed headless fence publish the stream and consume that one
//! diagnostic. Any deferred failure invalidates the complete provider state:
//! optimistic caches and earlier fences must not outlive a rejected command.

use super::NativeGlProvider;
use crate::backend::api::{GlContextLifecycle, GlError};

impl NativeGlProvider {
    /// Immediately observes a cold result-bearing or transactional operation.
    pub(crate) fn driver_error(&mut self, operation: &'static str) -> Result<(), GlError> {
        if self.deferred_error_operation.get().is_some() {
            self.take_deferred_driver_error(operation)?;
        }
        use glow::HasContext as _;
        // SAFETY: upheld by NativeGlProvider::from_current.
        let error = unsafe { self.gl.get_error() };
        match error {
            glow::NO_ERROR => Ok(()),
            glow::CONTEXT_LOST => Err(GlError::ContextLost { operation }),
            glow::OUT_OF_MEMORY => Err(GlError::OutOfMemory { operation }),
            error => Err(GlError::Driver {
                operation,
                message: format!("GL error 0x{error:04x}"),
            }),
        }
    }

    /// Records normal issued work for a later publication boundary.
    pub(crate) fn defer_driver_error(&self, operation: &'static str) -> Result<(), GlError> {
        self.commands_pending_submission.set(true);
        self.defer_readonly_error(operation)
    }

    /// Records diagnostic provenance without marking the command stream dirty.
    pub(crate) fn defer_readonly_error(&self, operation: &'static str) -> Result<(), GlError> {
        if self.deferred_error_operation.get().is_none() {
            self.deferred_error_operation.set(Some(operation));
        }
        Ok(())
    }

    /// Flushes newly issued work only when an explicit completion observation
    /// requires publication. Returns whether a flush was actually issued.
    pub(crate) fn submit_pending_commands(&self, operation: &'static str) -> bool {
        use glow::HasContext as _;
        if self.commands_pending_submission.replace(false) {
            // SAFETY: upheld by NativeGlProvider::from_current.
            unsafe { self.gl.flush() };
            let _ = self.defer_readonly_error(operation);
            true
        } else {
            false
        }
    }

    /// Consumes the single accumulated GL diagnostic and quarantines every
    /// optimistic execution fact if the stream has failed.
    pub(crate) fn take_deferred_driver_error(
        &mut self,
        boundary_operation: &'static str,
    ) -> Result<(), GlError> {
        use glow::HasContext as _;
        // SAFETY: upheld by NativeGlProvider::from_current.
        let error = unsafe { self.gl.get_error() };
        let operation = self
            .deferred_error_operation
            .replace(None)
            .unwrap_or(boundary_operation);
        match error {
            glow::NO_ERROR => Ok(()),
            glow::CONTEXT_LOST => {
                self.lifecycle = GlContextLifecycle::Lost;
                self.reset_executable_state();
                Err(GlError::ContextLost { operation })
            }
            glow::OUT_OF_MEMORY => {
                self.lifecycle = GlContextLifecycle::Lost;
                self.reset_executable_state();
                Err(GlError::OutOfMemory { operation })
            }
            error => {
                self.lifecycle = GlContextLifecycle::Lost;
                self.reset_executable_state();
                Err(GlError::Driver {
                    operation,
                    message: format!("GL error 0x{error:04x}"),
                })
            }
        }
    }
}

#[cfg(all(test, feature = "native-gl-wgl"))]
#[path = "../../../tests/backend/native/publication_call_counts.rs"]
mod publication_call_counts;
