//! Context-local reusable buffers for the GLES indirect-count lowering.
//!
//! A lease is returned only after the transaction has restored every binding
//! that could name it.  The provider is `!Send` and serializes native calls,
//! so GL command ordering makes it safe for the following lowering to rewrite
//! a returned buffer without a CPU wait or readback.

use std::{cell::RefCell, rc::Rc};

use crate::backend::api::GlError;
use glow::HasContext as _;

/// Retain at most one three-buffer set.  Larger requests remain correct, but
/// are transient so an unusually large indirect packet cannot pin device
/// memory for the life of the context.
const RETAINED_BYTES_LIMIT: u64 = 16 * 1024 * 1024;

struct ScratchSet {
    buffers: [glow::NativeBuffer; 3],
}

#[derive(Default)]
struct PoolState {
    available: Option<ScratchSet>,
}

/// Small, context-affine pool held by `NativeGlProvider`.
#[derive(Clone, Default)]
pub(crate) struct IndirectCountScratchPool {
    state: Rc<RefCell<PoolState>>,
}

/// Exclusive scratch names checked out for a single lowering transaction.
///
/// It deliberately owns an `Rc` to the pool rather than borrowing it: the
/// provider must continue issuing GL calls while the transaction owns its
/// scratch names.  The provider's owner-thread and current-context contracts
/// make that interior mutability single-threaded.
pub(super) struct IndirectCountScratchLease {
    gl: Rc<glow::Context>,
    state: Rc<RefCell<PoolState>>,
    set: Option<ScratchSet>,
    retain: bool,
}

impl IndirectCountScratchPool {
    /// # Safety
    ///
    /// `gl` is the current context belonging to this pool's provider.
    pub(super) unsafe fn lease(
        &self,
        gl: Rc<glow::Context>,
        sizes: [u64; 3],
        operation: &'static str,
    ) -> Result<IndirectCountScratchLease, GlError> {
        let bytes = sizes
            .into_iter()
            .try_fold(0_u64, u64::checked_add)
            .ok_or_else(|| GlError::Validation {
                operation,
                message: "indirect-count scratch size overflow".into(),
            })?;
        let retain = bytes <= RETAINED_BYTES_LIMIT;
        let reusable = retain
            .then(|| self.state.borrow_mut().available.take())
            .flatten();
        let set = match reusable {
            Some(set) => set,
            None => {
                let mut created = Vec::with_capacity(3);
                for _ in 0..3 {
                    match unsafe { gl.create_buffer() } {
                        Ok(buffer) => created.push(buffer),
                        Err(_) => {
                            for buffer in created {
                                unsafe { gl.delete_buffer(buffer) };
                            }
                            return Err(GlError::OutOfMemory { operation });
                        }
                    }
                }
                ScratchSet {
                    buffers: [created.remove(0), created.remove(0), created.remove(0)],
                }
            }
        };
        Ok(IndirectCountScratchLease {
            gl,
            state: self.state.clone(),
            set: Some(set),
            retain,
        })
    }

    /// Context loss makes all retained names invalid; no deletion is legal.
    pub(crate) fn discard_lost(&self) {
        self.state.borrow_mut().available.take();
    }

    /// # Safety
    ///
    /// The provider's exact context is current.
    pub(crate) unsafe fn clear(&self, gl: &glow::Context) {
        if let Some(set) = self.state.borrow_mut().available.take() {
            for buffer in set.buffers {
                unsafe { gl.delete_buffer(buffer) };
            }
        }
    }
}

impl IndirectCountScratchLease {
    pub(super) fn buffers(&self) -> [glow::NativeBuffer; 3] {
        self.set
            .as_ref()
            .expect("scratch lease remains live until transaction rollback")
            .buffers
    }

    /// Returns a bounded set after the enclosing transaction has restored its
    /// bindings.  Oversized sets are deleted instead of being retained.
    pub(super) fn release(mut self, bytes: u64) {
        let Some(set) = self.set.take() else {
            return;
        };
        if self.retain && bytes <= RETAINED_BYTES_LIMIT {
            let previous = self.state.borrow_mut().available.replace(set);
            if let Some(previous) = previous {
                for buffer in previous.buffers {
                    // SAFETY: transaction restoration precedes this release.
                    unsafe { self.gl.delete_buffer(buffer) };
                }
            }
        } else {
            for buffer in set.buffers {
                // SAFETY: transaction restoration precedes this release.
                unsafe { self.gl.delete_buffer(buffer) };
            }
        }
    }
}

impl Drop for IndirectCountScratchLease {
    fn drop(&mut self) {
        if let Some(set) = self.set.take() {
            // This is the exceptional construction/unwind path.  The only
            // normal owner is `IndirectCountTransaction`, whose Drop restores
            // bindings before it calls `release`.
            for buffer in set.buffers {
                // SAFETY: a lease can only be created while its provider's
                // native context is current.
                unsafe { self.gl.delete_buffer(buffer) };
            }
        }
    }
}

#[cfg(test)]
#[path = "../../../tests/backend/native/indirect_count_emulated/scratch__tests.rs"]
mod tests;
