//! Rollback ownership for one indirect-count compute/draw transaction.
//!
//! The utility changes global program and buffer bindings while it owns three
//! unpublished scratch buffers. Every fallible operation after the first
//! allocation is covered by this guard, so failures restore caller state before
//! recycling bounded scratch or deleting transient names, without an ad-hoc
//! error-path ladder.

use std::rc::Rc;

use crate::backend::{
    api::GlError,
    native::{buffer_abi::NativeBufferFunctions, provider::NativeStorageBinding},
};
use glow::HasContext as _;

use super::scratch::IndirectCountScratchLease;

#[derive(Clone, Copy)]
struct IndexedSsbo {
    buffer: Option<glow::NativeBuffer>,
    offset: i64,
    size: i64,
}

pub(super) struct IndirectCountTransaction {
    gl: Rc<glow::Context>,
    wide: Option<NativeBufferFunctions>,
    generic_ssbo: Option<glow::NativeBuffer>,
    copy_read: Option<glow::NativeBuffer>,
    copy_write: Option<glow::NativeBuffer>,
    indirect: Option<glow::NativeBuffer>,
    program: Option<glow::NativeProgram>,
    indexed_ssbo: [IndexedSsbo; 2],
    scratch: Option<(IndirectCountScratchLease, u64)>,
    temporary: Vec<glow::NativeBuffer>,
}

impl IndirectCountTransaction {
    /// # Safety
    ///
    /// The caller holds the provider's native context current.
    pub(super) unsafe fn capture(
        gl: Rc<glow::Context>,
        wide: Option<NativeBufferFunctions>,
        storage_bindings: [Option<NativeStorageBinding>; 2],
        generic_ssbo: Option<glow::NativeBuffer>,
        program: Option<glow::NativeProgram>,
        copy_read: Option<glow::NativeBuffer>,
        copy_write: Option<glow::NativeBuffer>,
        indirect: Option<glow::NativeBuffer>,
        operation: &'static str,
    ) -> Result<Self, GlError> {
        let indexed_ssbo = storage_bindings.map(|binding| match binding {
            Some(binding) => IndexedSsbo {
                buffer: binding.buffer,
                offset: i64::try_from(binding.offset).unwrap_or(i64::MAX),
                size: i64::try_from(binding.size).unwrap_or(i64::MAX),
            },
            None => IndexedSsbo {
                buffer: None,
                offset: 0,
                size: 0,
            },
        });
        for binding in indexed_ssbo {
            if binding.buffer.is_some()
                && (binding.offset < 0
                    || binding.size < 0
                    || (wide.is_none()
                        && (binding.offset > i64::from(i32::MAX)
                            || binding.size > i64::from(i32::MAX))))
            {
                return Err(GlError::Unsupported {
                    operation,
                    reason: "existing SSBO binding cannot be restored by the loaded GL ABI",
                });
            }
        }
        Ok(Self {
            generic_ssbo,
            // Production lowering supplies the canonical empty baseline for
            // utility-private copy/indirect slots. Test and interop callers
            // can instead supply their CPU-known foreign bindings, retaining
            // exact rollback without a GL state query.
            copy_read,
            copy_write,
            indirect,
            program,
            gl,
            wide,
            indexed_ssbo,
            scratch: None,
            temporary: Vec::with_capacity(3),
        })
    }

    pub(super) fn retain_temporary(&mut self, buffer: glow::NativeBuffer) {
        self.temporary.push(buffer);
    }

    pub(super) fn retain_scratch(&mut self, scratch: IndirectCountScratchLease, bytes: u64) {
        debug_assert!(self.scratch.is_none(), "one lowering owns one scratch set");
        self.scratch = Some((scratch, bytes));
    }

    pub(super) fn scratch_buffers(&self) -> [glow::NativeBuffer; 3] {
        self.scratch
            .as_ref()
            .expect("scratch must be retained before it is used")
            .0
            .buffers()
    }

    /// Restores the raster-visible part of state between compute and drawing.
    ///
    /// # Safety
    ///
    /// The current context is the one used by `capture`.
    pub(super) unsafe fn restore_draw_state(&self) {
        unsafe { self.restore_ssbo_and_program() };
    }

    unsafe fn restore_ssbo_and_program(&self) {
        for (index, binding) in self.indexed_ssbo.iter().copied().enumerate() {
            if let Some(buffer) = binding.buffer {
                if binding.size > 0 {
                    if let Some(wide) = self.wide {
                        unsafe {
                            (wide.bind_buffer_range)(
                                glow::SHADER_STORAGE_BUFFER,
                                index as u32,
                                buffer.0.get(),
                                binding.offset as isize,
                                binding.size as isize,
                            )
                        };
                    } else {
                        unsafe {
                            self.gl.bind_buffer_range(
                                glow::SHADER_STORAGE_BUFFER,
                                index as u32,
                                Some(buffer),
                                binding.offset as i32,
                                binding.size as i32,
                            )
                        };
                    }
                    continue;
                }
            }
            unsafe {
                self.gl
                    .bind_buffer_base(glow::SHADER_STORAGE_BUFFER, index as u32, binding.buffer)
            };
        }
        unsafe {
            self.gl
                .bind_buffer(glow::SHADER_STORAGE_BUFFER, self.generic_ssbo);
            self.gl.use_program(self.program);
        }
    }
}

impl Drop for IndirectCountTransaction {
    fn drop(&mut self) {
        // Drop cannot surface a driver error. It deliberately restores first so
        // no live binding refers to a scratch name when that name is deleted.
        unsafe {
            self.restore_ssbo_and_program();
            self.gl.bind_buffer(glow::COPY_READ_BUFFER, self.copy_read);
            self.gl
                .bind_buffer(glow::COPY_WRITE_BUFFER, self.copy_write);
            self.gl
                .bind_buffer(glow::DRAW_INDIRECT_BUFFER, self.indirect);
            if let Some((scratch, bytes)) = self.scratch.take() {
                scratch.release(bytes);
            }
            for buffer in self.temporary.drain(..) {
                self.gl.delete_buffer(buffer);
            }
        }
    }
}

#[cfg(test)]
#[path = "../../../tests/backend/native/indirect_count_emulated/transaction__tests.rs"]
mod tests;
