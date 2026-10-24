//! Native buffer allocation and retirement, including persistent mappings.
//!
//! Buffer names and mappings remain owned by the provider until the last
//! resource reference is retired with its context current.
use super::NativeGlProvider;
use crate::backend::gl::api::{self, BufferId, GlBufferDesc, GlError, GlFamilyApi as _};
use crate::backend::gl::native::buffer_abi::pointer_sized;

const BUFFER_ALLOCATION_USAGE: u32 = glow::STATIC_DRAW;

impl NativeGlProvider {
    /// The largest buffer this provider can address through its loaded ABI.
    /// This is an addressing fact, not an allocation promise.
    pub(crate) const fn advertised_max_buffer_bytes(&self) -> u64 {
        super::super::buffer_abi::advertised_native_max_buffer_bytes(
            self.buffer_functions.is_some(),
        )
    }

    pub(super) fn create_native_buffer(&mut self, desc: GlBufferDesc) -> Result<BufferId, GlError> {
        use glow::HasContext as _;
        self.assert_ready("create-buffer")?;
        desc.validate().map_err(|error| GlError::Validation {
            operation: "create-buffer",
            message: error.message(),
        })?;
        let size = pointer_sized(desc.size).ok_or_else(|| {
            Self::validation("create-buffer", "buffer exceeds native GL pointer range")
        })?;
        let slot = self.slot("create-buffer")?;
        // SAFETY: current-context contract; all validation completed before GL mutation.
        let name = unsafe { self.gl.create_buffer() }.map_err(|message| GlError::Driver {
            operation: "create-buffer",
            message,
        })?;
        // COPY_WRITE_BUFFER is Layer 1-private scratch: bound immediately before
        // the allocation, not restored on return (`GlCopyDomainApi` documents
        // why).
        // SAFETY: see above.
        let persistent = self.buffer_storage.is_some()
            && (desc.usage.contains(api::GlBufferUsage::MAP_READ)
                || desc.usage.contains(api::GlBufferUsage::MAP_WRITE));
        let mapping = unsafe {
            self.gl.bind_buffer(glow::COPY_WRITE_BUFFER, Some(name));
            if persistent {
                self.allocate_persistent_buffer(size).map(Some)
            } else {
                if let Some(functions) = self.buffer_functions {
                    (functions.buffer_data)(
                        glow::COPY_WRITE_BUFFER,
                        size,
                        core::ptr::null(),
                        BUFFER_ALLOCATION_USAGE,
                    );
                } else {
                    let size = i32::try_from(size).map_err(|_| {
                        Self::validation("create-buffer", "GL buffer ABI entry points unavailable")
                    })?;
                    self.gl.buffer_data_size(
                        glow::COPY_WRITE_BUFFER,
                        size,
                        BUFFER_ALLOCATION_USAGE,
                    );
                }
                self.driver_error("create-buffer").map(|()| None)
            }
        };
        let mapping = match mapping {
            Ok(mapping) => mapping,
            Err(error) => {
                // SAFETY: rollback deletes only the name allocated by this current
                // context above; it was never published to the resource table.
                unsafe { self.gl.delete_buffer(name) };
                return Err(error);
            }
        };
        let id = BufferId::new(self.context_stamp(), slot, 0);
        self.buffers.insert(id, (name, desc));
        if let Some(ptr) = mapping {
            self.persistent_buffers.insert(id, ptr);
        }
        Ok(id)
    }
    pub(super) fn destroy_native_buffer(&mut self, id: BufferId) -> Result<(), GlError> {
        use glow::HasContext as _;
        self.assert_ready("destroy-buffer")?;
        let (name, _) = self.buffer("destroy-buffer", id)?;
        let mut unmap_failure = None;
        if self.persistent_buffers.remove(&id).is_some() {
            // SAFETY: no lease can remain when the owning buffer is retired.
            let intact = unsafe {
                self.gl.bind_buffer(glow::COPY_WRITE_BUFFER, Some(name));
                (self.unmap_buffer)(glow::COPY_WRITE_BUFFER)
            };
            unmap_failure = self.driver_error("unmap-buffer").err();
            if intact == 0 && unmap_failure.is_none() {
                unmap_failure = Some(GlError::Driver {
                    operation: "unmap-buffer",
                    message: "GL unmap reported corrupt contents".into(),
                });
            }
        }
        // SAFETY: current-context contract; liveness was checked before GL mutation.
        unsafe { self.gl.delete_buffer(name) };
        let deletion = self.driver_error("destroy-buffer");
        self.buffers.remove(&id);
        if let Some(error) = unmap_failure {
            return Err(error);
        }
        deletion
    }
}
