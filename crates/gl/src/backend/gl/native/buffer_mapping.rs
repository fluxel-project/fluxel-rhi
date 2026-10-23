//! Persistent mappings for desktop GL buffer storage.
//!
//! Storage is mapped once before publication, and retained until buffer
//! retirement. Leases borrow the mapping without remapping or waiting.
use super::provider::NativeGlProvider;
use crate::backend::gl::api::{BufferId, GlBufferUsage, GlError};
use glow::HasContext as _;

impl NativeGlProvider {
    pub(super) unsafe fn allocate_persistent_buffer(
        &mut self,
        size: i32,
    ) -> Result<*mut u8, GlError> {
        const OP: &str = "create-persistent-buffer";
        let flags = glow::MAP_READ_BIT
            | glow::MAP_WRITE_BIT
            | glow::MAP_PERSISTENT_BIT
            | glow::MAP_COHERENT_BIT
            | glow::DYNAMIC_STORAGE_BIT;
        // SAFETY: caller bound the newly created buffer in the current context;
        // the entry was loaded only with matching desktop feature evidence.
        unsafe {
            (self.buffer_storage.expect("checked storage entry"))(
                glow::COPY_WRITE_BUFFER,
                size as isize,
                core::ptr::null(),
                flags,
            )
        };
        self.driver_error(OP)?;
        // SAFETY: the immutable allocation above allows each requested bit.
        let ptr = unsafe {
            self.gl.map_buffer_range(
                glow::COPY_WRITE_BUFFER,
                0,
                size,
                flags & !glow::DYNAMIC_STORAGE_BIT,
            )
        };
        if let Err(error) = self.driver_error(OP) {
            if !ptr.is_null() {
                // SAFETY: a non-null map belongs to the currently bound buffer.
                unsafe {
                    (self.unmap_buffer)(glow::COPY_WRITE_BUFFER);
                }
            }
            return Err(error);
        }
        if ptr.is_null() {
            return Err(GlError::Driver {
                operation: OP,
                message: "GL map returned null".into(),
            });
        }
        // The public lease yields Rust byte slices, which require initialized
        // bytes even before the first GPU write. No GPU work has seen this
        // allocation yet, and this coherent mapping covers the entire store.
        unsafe { core::ptr::write_bytes(ptr, 0, size as usize) };
        Ok(ptr)
    }

    pub(crate) fn persistent_buffer_lease(
        &self,
        id: BufferId,
        write: bool,
    ) -> Result<Option<*mut u8>, GlError> {
        let (_, desc) = self.buffer("map-buffer", id)?;
        let required = if write {
            GlBufferUsage::MAP_WRITE
        } else {
            GlBufferUsage::MAP_READ
        };
        if !desc.usage.contains(required) {
            return Err(GlError::Unsupported {
                operation: "map-buffer",
                reason: "buffer lacks requested mapping usage",
            });
        }
        Ok(self.persistent_buffers.get(&id).copied())
    }
}
