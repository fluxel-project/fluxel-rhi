//! Native program-binary cache operations on the current context.

use super::provider::{NativeGlProvider, NativeProgram};
use crate::backend::gl::api::{
    GlError, GlFamilyApi as _, GlProgramDescriptor, GlProgramReflection, GlShaderApi as _,
    ProgramId,
};
use crate::backend::gl::backend::cache::CachedBinary;
use glow::HasContext as _;

#[derive(Clone, Copy)]
pub(super) struct ProgramBinaryFns {
    get: unsafe extern "system" fn(u32, i32, *mut i32, *mut u32, *mut core::ffi::c_void),
}
impl ProgramBinaryFns {
    /// # Safety
    /// The loader must return symbols from the same current context.
    pub(super) unsafe fn load(loader: &impl Fn(&str) -> *const core::ffi::c_void) -> Option<Self> {
        let get = loader("glGetProgramBinary");
        if get.is_null()
            || loader("glProgramBinary").is_null()
            || loader("glProgramParameteri").is_null()
        {
            return None;
        }
        // SAFETY: GL's ABI specifies this signature for glGetProgramBinary.
        Some(Self {
            get: unsafe { core::mem::transmute(get) },
        })
    }
}

impl NativeGlProvider {
    fn binary_driver_error(&self, operation: &'static str) -> Result<(), GlError> {
        // SAFETY: this provider owns the current context.
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
    pub(super) fn program_binary_supported(&self) -> bool {
        let core = self.profile().has_core(
            crate::backend::gl::profile::GlVersion::new(4, 1),
            crate::backend::gl::profile::GlVersion::new(3, 0),
        );
        if self.program_binary_entries.is_none()
            || (!core
                && !self
                    .gl
                    .supported_extensions()
                    .contains("GL_ARB_get_program_binary"))
        {
            return false;
        }
        // SAFETY: current context; the query exists on the proven route.
        unsafe { self.gl.get_parameter_i32(glow::NUM_PROGRAM_BINARY_FORMATS) > 0 }
    }

    pub(crate) fn program_cache_identity(&self) -> Result<Vec<u8>, GlError> {
        self.assert_ready("program-cache")?;
        let supported = self.program_binary_supported();
        self.binary_driver_error("program-cache")?;
        if !supported {
            return Err(GlError::Unsupported {
                operation: "program-cache",
                reason: "no program binary formats",
            });
        }
        // SAFETY: a live, current context owns these driver strings.
        let values = unsafe {
            [
                self.gl.get_parameter_string(glow::VENDOR),
                self.gl.get_parameter_string(glow::RENDERER),
                self.gl.get_parameter_string(glow::VERSION),
                self.gl.get_parameter_string(glow::SHADING_LANGUAGE_VERSION),
            ]
        };
        self.binary_driver_error("program-cache")?;
        let mut identity = Vec::new();
        for value in values {
            identity.extend_from_slice(&(value.len() as u64).to_le_bytes());
            identity.extend_from_slice(value.as_bytes());
        }
        Ok(identity)
    }

    pub(crate) fn cached_program(
        &mut self,
        descriptor: &GlProgramDescriptor,
        binary: Option<&CachedBinary>,
    ) -> Result<(ProgramId, GlProgramReflection, Option<CachedBinary>, bool), GlError> {
        const OP: &str = "cached-program";
        self.assert_ready(OP)?;
        if let Some(binary) = binary {
            // SAFETY: the context is current and program lifetime is local until publication.
            unsafe {
                let raw = self
                    .gl
                    .create_program()
                    .map_err(|message| GlError::Driver {
                        operation: OP,
                        message,
                    })?;
                self.gl.program_binary(
                    raw,
                    &glow::ProgramBinary {
                        format: binary.format,
                        buffer: binary.bytes.clone(),
                    },
                );
                let accepted = self.gl.get_program_link_status(raw);
                let error = self.gl.get_error();
                if accepted && error == glow::NO_ERROR {
                    self.gl.use_program(Some(raw));
                    let reflected = self.reflect_and_assign(OP, raw, descriptor);
                    self.gl.use_program(None);
                    self.current_program = None;
                    self.context_state.program_unbound();
                    let driver = self.binary_driver_error(OP);
                    let reflection = match driver.and(reflected) {
                        Ok(value) => value,
                        Err(error) => {
                            self.gl.delete_program(raw);
                            return Err(error);
                        }
                    };
                    let slot = match self.slot(OP) {
                        Ok(slot) => slot,
                        Err(error) => {
                            self.gl.delete_program(raw);
                            return Err(error);
                        }
                    };
                    let id = ProgramId::new(self.context_stamp(), slot, 0);
                    self.programs.insert(
                        id,
                        NativeProgram {
                            generation: id.generation,
                            raw,
                            descriptor: descriptor.clone(),
                            reflection: reflection.clone(),
                        },
                    );
                    return Ok((id, reflection, None, true));
                }
                self.gl.delete_program(raw);
                // A stale or rejected binary is only a cache miss. Context-loss
                // errors are never swallowed as a compilation fallback.
                if error == glow::CONTEXT_LOST {
                    return Err(GlError::ContextLost { operation: OP });
                }
                if error == glow::OUT_OF_MEMORY {
                    return Err(GlError::OutOfMemory { operation: OP });
                }
                if error != glow::NO_ERROR
                    && error != glow::INVALID_ENUM
                    && error != glow::INVALID_VALUE
                    && error != glow::INVALID_OPERATION
                {
                    return Err(GlError::Driver {
                        operation: OP,
                        message: format!("GL error 0x{error:04x}"),
                    });
                }
            }
        }
        self.cache_retrievable_hint = true;
        let linked = self.create_program(descriptor);
        self.cache_retrievable_hint = false;
        let (id, reflection) = linked?;
        let raw = self.program(OP, id)?.raw;
        // SAFETY: the linked name is live on the current context.
        let length = unsafe {
            self.gl
                .get_program_parameter_i32(raw, glow::PROGRAM_BINARY_LENGTH)
        };
        if let Err(error) = self.binary_driver_error(OP) {
            let _ = self.destroy_program(id);
            return Err(error);
        }
        let binary = if length > 0 {
            let mut bytes = Vec::new();
            if bytes.try_reserve_exact(length as usize).is_err() {
                let _ = self.destroy_program(id);
                return Err(GlError::OutOfMemory { operation: OP });
            }
            bytes.resize(length as usize, 0);
            let mut written = 0;
            let mut format = 0;
            let get = self
                .program_binary_entries
                .expect("cache admission checked")
                .get;
            // SAFETY: function was resolved from this context and the output
            // allocation is writable for exactly length bytes.
            unsafe {
                get(
                    raw.0.get(),
                    length,
                    &mut written,
                    &mut format,
                    bytes.as_mut_ptr().cast(),
                )
            };
            if let Err(error) = self.binary_driver_error(OP) {
                let _ = self.destroy_program(id);
                return Err(error);
            }
            if written < 0 || written > length {
                let _ = self.destroy_program(id);
                return Err(GlError::Driver {
                    operation: OP,
                    message: "invalid program binary length".into(),
                });
            }
            bytes.truncate(written as usize);
            (!bytes.is_empty()).then_some(CachedBinary { format, bytes })
        } else {
            None
        };
        Ok((id, reflection, binary, false))
    }
}
