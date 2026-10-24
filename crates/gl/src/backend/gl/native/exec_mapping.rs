//! Native GL buffer mapping primitives used by Core's host mapping lease.
//!
//! COPY_WRITE_BUFFER is restored after each operation; the allocation remains
//! owned by its Core buffer wrapper for the entire mapping lifetime.
use super::provider::NativeGlProvider;
use crate::backend::gl::api::{BufferId, GlBufferUsage, GlError, GlFamilyApi as _};
use crate::backend::gl::native::buffer_abi::pointer_sized;
use glow::HasContext as _;

impl NativeGlProvider {
    pub(crate) fn map_buffer_lease(
        &mut self,
        buffer: BufferId,
        offset: u64,
        length: u64,
        write: bool,
    ) -> Result<*mut u8, GlError> {
        const OP: &str = "map-buffer";
        self.assert_ready(OP)?;
        let (raw, desc) = self.buffer(OP, buffer)?;
        let role = if write {
            GlBufferUsage::MAP_WRITE
        } else {
            GlBufferUsage::MAP_READ
        };
        if !desc.usage.contains(role) {
            return Err(GlError::Unsupported {
                operation: OP,
                reason: "buffer lacks requested mapping usage",
            });
        }
        if length == 0 || offset.checked_add(length).is_none_or(|end| end > desc.size) {
            return Err(Self::validation(OP, "invalid mapping range"));
        }
        let offset = pointer_sized(offset)
            .ok_or_else(|| Self::validation(OP, "mapping offset exceeds native GL range"))?;
        let length = pointer_sized(length)
            .ok_or_else(|| Self::validation(OP, "mapping length exceeds native GL range"))?;
        // SAFETY: current context and live validated allocation; prior binding is restored on every path.
        unsafe {
            let prior = self.gl.get_parameter_i32(glow::COPY_WRITE_BUFFER_BINDING) as u32;
            self.gl.bind_buffer(glow::COPY_WRITE_BUFFER, Some(raw));
            let mapped = self
                .gl
                .get_buffer_parameter_i32(glow::COPY_WRITE_BUFFER, glow::BUFFER_MAPPED)
                != 0;
            let ptr = if mapped {
                core::ptr::null_mut()
            } else {
                let flags = glow::MAP_READ_BIT;
                if let Some(functions) = self.buffer_functions {
                    (functions.map_buffer_range)(glow::COPY_WRITE_BUFFER, offset, length, flags)
                        .cast::<u8>()
                } else {
                    let offset = i32::try_from(offset).map_err(|_| {
                        Self::validation(OP, "GL buffer ABI entry points unavailable")
                    })?;
                    let length = i32::try_from(length).map_err(|_| {
                        Self::validation(OP, "GL buffer ABI entry points unavailable")
                    })?;
                    self.gl
                        .map_buffer_range(glow::COPY_WRITE_BUFFER, offset, length, flags)
                }
            };
            self.gl.bind_buffer(
                glow::COPY_WRITE_BUFFER,
                std::num::NonZeroU32::new(prior).map(glow::NativeBuffer),
            );
            if let Err(error) = self.mapping_driver_error(OP) {
                if !ptr.is_null() {
                    self.gl.bind_buffer(glow::COPY_WRITE_BUFFER, Some(raw));
                    (self.unmap_buffer)(glow::COPY_WRITE_BUFFER);
                    self.gl.bind_buffer(
                        glow::COPY_WRITE_BUFFER,
                        std::num::NonZeroU32::new(prior).map(glow::NativeBuffer),
                    );
                }
                return Err(error);
            }
            if mapped {
                return Err(Self::validation(OP, "buffer already mapped"));
            }
            if ptr.is_null() {
                return Err(GlError::Driver {
                    operation: OP,
                    message: "GL map returned null".into(),
                });
            }
            Ok(ptr)
        }
    }
    /// Copies a temporary driver mapping into CPU-owned lease storage.
    pub(crate) fn snapshot_buffer_lease(
        &mut self,
        buffer: BufferId,
        offset: u64,
        length: u64,
        write: bool,
    ) -> Result<Vec<u8>, GlError> {
        let len = usize::try_from(length)
            .map_err(|_| Self::validation("map-buffer", "mapping size overflow"))?;
        let mut bytes = vec![0; len];
        let ptr = self.map_buffer_lease(buffer, offset, length, write)?;
        // SAFETY: map_buffer_lease returned a live readable range of `len` bytes;
        // no reference to driver memory escapes the current-context call.
        unsafe {
            std::ptr::copy_nonoverlapping(ptr, bytes.as_mut_ptr(), len);
        }
        let (raw, _) = self.buffer("unmap-buffer", buffer)?;
        // SAFETY: this balances the successful map and restores the binding.
        let intact = unsafe {
            let prior = self.gl.get_parameter_i32(glow::COPY_WRITE_BUFFER_BINDING) as u32;
            self.gl.bind_buffer(glow::COPY_WRITE_BUFFER, Some(raw));
            let intact = (self.unmap_buffer)(glow::COPY_WRITE_BUFFER) != 0;
            self.gl.bind_buffer(
                glow::COPY_WRITE_BUFFER,
                std::num::NonZeroU32::new(prior).map(glow::NativeBuffer),
            );
            intact
        };
        self.mapping_driver_error("unmap-buffer")?;
        if !intact {
            return Err(GlError::Driver {
                operation: "unmap-buffer",
                message: "GL unmap reported corrupt contents".into(),
            });
        }
        Ok(bytes)
    }

    fn mapping_driver_error(&self, operation: &'static str) -> Result<(), GlError> {
        // SAFETY: every caller has established current-context ownership.
        match unsafe { self.gl.get_error() } {
            glow::NO_ERROR => Ok(()),
            glow::CONTEXT_LOST => Err(GlError::ContextLost { operation }),
            glow::OUT_OF_MEMORY => Err(GlError::OutOfMemory { operation }),
            error => Err(GlError::Driver {
                operation,
                message: format!("GL error 0x{error:04x}"),
            }),
        }
    }
}
