//! Persistent mappings for desktop GL and GLES buffer storage extensions.
//!
//! Storage is mapped once before publication, and retained until buffer
//! retirement. Leases borrow the mapping without remapping or waiting.
use super::provider::NativeGlProvider;
use crate::backend::api::{BufferId, GlBufferUsage, GlError, GlFamilyProfile};
use glow::HasContext as _;

/// Select only the entry whose profile, version and extension evidence match.
pub(super) fn storage_entry(
    profile: GlFamilyProfile,
    arb: bool,
    ext: bool,
    load: impl Fn(&str) -> *const core::ffi::c_void,
) -> Option<*const core::ffi::c_void> {
    let name = match profile {
        GlFamilyProfile::Desktop { major, minor } if (major, minor) >= (4, 4) || arb => {
            "glBufferStorage"
        }
        GlFamilyProfile::Embedded { major, minor } if (major, minor) >= (3, 1) && ext => {
            "glBufferStorageEXT"
        }
        _ => return None,
    };
    let entry = load(name);
    (!entry.is_null()).then_some(entry)
}

impl NativeGlProvider {
    pub(super) unsafe fn allocate_persistent_buffer(
        &mut self,
        size: isize,
    ) -> Result<*mut u8, GlError> {
        const OP: &str = "create-persistent-buffer";
        let flags = glow::MAP_READ_BIT
            | glow::MAP_WRITE_BIT
            | glow::MAP_PERSISTENT_BIT
            | glow::MAP_COHERENT_BIT
            | glow::DYNAMIC_STORAGE_BIT;
        // SAFETY: caller bound the newly created buffer in the current context;
        // the entry was loaded only with matching profile and feature evidence.
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
            if let Some(functions) = self.buffer_functions {
                (functions.map_buffer_range)(
                    glow::COPY_WRITE_BUFFER,
                    0,
                    size,
                    flags & !glow::DYNAMIC_STORAGE_BIT,
                )
                .cast::<u8>()
            } else {
                let size = i32::try_from(size)
                    .map_err(|_| Self::validation(OP, "GL buffer ABI entry points unavailable"))?;
                self.gl.map_buffer_range(
                    glow::COPY_WRITE_BUFFER,
                    0,
                    size,
                    flags & !glow::DYNAMIC_STORAGE_BIT,
                )
            }
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
        unsafe {
            core::ptr::write_bytes(
                ptr,
                0,
                usize::try_from(size).expect("validated positive GL size"),
            )
        };
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

#[cfg(test)]
#[path = "../../tests/backend/native/buffer_mapping__tests.rs"]
mod tests;
