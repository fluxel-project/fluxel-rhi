//! Pointer-width buffer entry points for native GL and GLES.
//!
//! `glow` deliberately exposes a portable `i32` surface for several buffer
//! commands.  Native GL's `GLsizeiptr` and `GLintptr` are pointer-width signed
//! integers, however.  This module keeps the wider ABI at the native boundary
//! rather than leaking platform pointers into the common GL API or pretending
//! that WebGL has the same range.

use core::ffi::c_void;

/// The subset of core buffer commands whose `glow` signatures narrow a native
/// pointer-sized offset or length to `i32`.
///
/// All entries are loaded from the context which owns the provider.  The
/// caller must retain that context current for every call.
#[derive(Clone, Copy)]
pub(crate) struct NativeBufferFunctions {
    pub(crate) buffer_data: unsafe extern "system" fn(u32, isize, *const c_void, u32),
    pub(crate) buffer_sub_data: unsafe extern "system" fn(u32, isize, isize, *const c_void),
    pub(crate) copy_buffer_sub_data: unsafe extern "system" fn(u32, u32, isize, isize, isize),
    pub(crate) map_buffer_range: unsafe extern "system" fn(u32, isize, isize, u32) -> *mut c_void,
    pub(crate) bind_buffer_range: unsafe extern "system" fn(u32, u32, u32, isize, isize),
    pub(crate) get_buffer_sub_data:
        Option<unsafe extern "system" fn(u32, isize, isize, *mut c_void)>,
    pub(crate) tex_sub_image_1d:
        Option<unsafe extern "system" fn(u32, i32, i32, i32, u32, u32, *const c_void)>,
    pub(crate) tex_sub_image_2d:
        Option<unsafe extern "system" fn(u32, i32, i32, i32, i32, i32, u32, u32, *const c_void)>,
    pub(crate) tex_sub_image_3d: Option<
        unsafe extern "system" fn(u32, i32, i32, i32, i32, i32, i32, i32, u32, u32, *const c_void),
    >,
    pub(crate) read_pixels:
        Option<unsafe extern "system" fn(i32, i32, i32, i32, u32, u32, *mut c_void)>,
}

/// Native vertex commands whose buffer offset is a `GLintptr` but whose
/// portable glow facade narrows it to `i32`.
#[derive(Clone, Copy)]
pub(crate) struct NativeVertexFunctions {
    pub(crate) attrib_pointer: unsafe extern "system" fn(u32, i32, u32, u8, i32, *const c_void),
    pub(crate) attrib_i_pointer: unsafe extern "system" fn(u32, i32, u32, i32, *const c_void),
    pub(crate) attrib_l_pointer:
        Option<unsafe extern "system" fn(u32, i32, u32, i32, *const c_void)>,
    pub(crate) bind_vertex_buffer: Option<unsafe extern "system" fn(u32, u32, isize, i32)>,
    pub(crate) attrib_l_format: Option<unsafe extern "system" fn(u32, i32, u32, u32)>,
}

impl NativeVertexFunctions {
    /// # Safety
    ///
    /// `load` resolves symbols for the provider's current native context.
    pub(crate) unsafe fn load(mut load: impl FnMut(&str) -> *const c_void) -> Option<Self> {
        macro_rules! required {
            ($name:literal, $ty:ty) => {{
                let pointer = load($name);
                if pointer.is_null() {
                    return None;
                }
                unsafe { core::mem::transmute::<*const c_void, $ty>(pointer) }
            }};
        }
        macro_rules! optional {
            ($name:literal, $ty:ty) => {{
                let pointer = load($name);
                (!pointer.is_null())
                    .then(|| unsafe { core::mem::transmute::<*const c_void, $ty>(pointer) })
            }};
        }
        Some(Self {
            attrib_pointer: required!(
                "glVertexAttribPointer",
                unsafe extern "system" fn(u32, i32, u32, u8, i32, *const c_void)
            ),
            attrib_i_pointer: required!(
                "glVertexAttribIPointer",
                unsafe extern "system" fn(u32, i32, u32, i32, *const c_void)
            ),
            attrib_l_pointer: optional!(
                "glVertexAttribLPointer",
                unsafe extern "system" fn(u32, i32, u32, i32, *const c_void)
            ),
            bind_vertex_buffer: optional!(
                "glBindVertexBuffer",
                unsafe extern "system" fn(u32, u32, isize, i32)
            ),
            attrib_l_format: optional!(
                "glVertexAttribLFormat",
                unsafe extern "system" fn(u32, i32, u32, u32)
            ),
        })
    }

    /// Legacy double attributes only require `glVertexAttribLPointer`.
    /// `glVertexAttribLFormat` belongs to the separate-attrib-binding route
    /// and is not core until later desktop GL versions.
    pub(crate) const fn supports_f64(self) -> bool {
        self.attrib_l_pointer.is_some()
    }

    /// A zero-stride double attribute uses the separate binding API, which
    /// needs both the pointer-width binding offset and the L-format entry.
    pub(crate) const fn supports_zero_stride_f64(self) -> bool {
        self.bind_vertex_buffer.is_some() && self.attrib_l_format.is_some()
    }
}

impl NativeBufferFunctions {
    /// Loads the core GL ABI as one coherent set.  A missing symbol is not
    /// guessed from a version string; callers retain the safe `glow` route for
    /// ranges that fit that route.
    ///
    /// # Safety
    ///
    /// `load` must resolve symbols for the current native GL or GLES context.
    pub(crate) unsafe fn load(mut load: impl FnMut(&str) -> *const c_void) -> Option<Self> {
        macro_rules! entry {
            ($name:literal, $ty:ty) => {{
                let pointer = load($name);
                if pointer.is_null() {
                    return None;
                }
                // SAFETY: the caller guarantees the registered GL ABI.
                unsafe { core::mem::transmute::<*const c_void, $ty>(pointer) }
            }};
        }
        macro_rules! optional_entry {
            ($name:literal, $ty:ty) => {{
                let pointer = load($name);
                (!pointer.is_null())
                    .then(|| unsafe { core::mem::transmute::<*const c_void, $ty>(pointer) })
            }};
        }
        Some(Self {
            buffer_data: entry!(
                "glBufferData",
                unsafe extern "system" fn(u32, isize, *const c_void, u32)
            ),
            buffer_sub_data: entry!(
                "glBufferSubData",
                unsafe extern "system" fn(u32, isize, isize, *const c_void)
            ),
            copy_buffer_sub_data: entry!(
                "glCopyBufferSubData",
                unsafe extern "system" fn(u32, u32, isize, isize, isize)
            ),
            map_buffer_range: entry!(
                "glMapBufferRange",
                unsafe extern "system" fn(u32, isize, isize, u32) -> *mut c_void
            ),
            bind_buffer_range: entry!(
                "glBindBufferRange",
                unsafe extern "system" fn(u32, u32, u32, isize, isize)
            ),
            get_buffer_sub_data: {
                let pointer = load("glGetBufferSubData");
                (!pointer.is_null()).then(|| unsafe {
                    core::mem::transmute::<
                        *const c_void,
                        unsafe extern "system" fn(u32, isize, isize, *mut c_void),
                    >(pointer)
                })
            },
            tex_sub_image_1d: optional_entry!(
                "glTexSubImage1D",
                unsafe extern "system" fn(u32, i32, i32, i32, u32, u32, *const c_void)
            ),
            tex_sub_image_2d: optional_entry!(
                "glTexSubImage2D",
                unsafe extern "system" fn(u32, i32, i32, i32, i32, i32, u32, u32, *const c_void)
            ),
            tex_sub_image_3d: optional_entry!(
                "glTexSubImage3D",
                unsafe extern "system" fn(
                    u32,
                    i32,
                    i32,
                    i32,
                    i32,
                    i32,
                    i32,
                    i32,
                    u32,
                    u32,
                    *const c_void,
                )
            ),
            read_pixels: optional_entry!(
                "glReadPixels",
                unsafe extern "system" fn(i32, i32, i32, i32, u32, u32, *mut c_void)
            ),
        })
    }
}

/// Converts a Core byte count or byte offset to the native GL pointer-sized
/// signed type without truncation.
pub(crate) fn pointer_sized(value: u64) -> Option<isize> {
    isize::try_from(value).ok()
}

/// The largest allocation the native ABI can name.  This is an addressing
/// limit only; allocation can still fail with `GL_OUT_OF_MEMORY`.
pub(crate) const fn native_max_buffer_bytes() -> u64 {
    isize::MAX as u64
}

/// The truthful native capacity for a context: without the loaded ABI table,
/// only glow's portable `i32` route is available.
pub(crate) const fn advertised_native_max_buffer_bytes(wide_abi_loaded: bool) -> u64 {
    if wide_abi_loaded {
        native_max_buffer_bytes()
    } else {
        i32::MAX as u64
    }
}

impl super::provider::NativeGlProvider {
    /// Allocates an uninitialized buffer store without reducing its size to
    /// glow's portable `i32` ABI.
    ///
    /// # Safety
    ///
    /// The target must already be bound in the current native context.
    pub(super) unsafe fn buffer_data_size_wide(
        &self,
        target: u32,
        size: u64,
        usage: u32,
        operation: &'static str,
    ) -> Result<(), crate::backend::api::GlError> {
        use glow::HasContext as _;
        let size = pointer_sized(size)
            .ok_or_else(|| Self::validation(operation, "buffer size exceeds native GLsizeiptr"))?;
        if let Some(functions) = self.buffer_functions {
            // SAFETY: forwarded from this method's contract.
            unsafe { (functions.buffer_data)(target, size, core::ptr::null(), usage) };
        } else {
            let size = i32::try_from(size).map_err(|_| {
                Self::validation(operation, "GL buffer ABI entry points unavailable")
            })?;
            // SAFETY: forwarded from this method's contract.
            unsafe { self.gl.buffer_data_size(target, size, usage) };
        }
        Ok(())
    }

    /// Issues `glCopyBufferSubData` with pointer-width offsets and size.
    pub(super) unsafe fn copy_buffer_sub_data_wide(
        &self,
        read_target: u32,
        write_target: u32,
        read_offset: u64,
        write_offset: u64,
        size: u64,
        operation: &'static str,
    ) -> Result<(), crate::backend::api::GlError> {
        use glow::HasContext as _;
        let read_offset = pointer_sized(read_offset)
            .ok_or_else(|| Self::validation(operation, "copy source exceeds native GLintptr"))?;
        let write_offset = pointer_sized(write_offset).ok_or_else(|| {
            Self::validation(operation, "copy destination exceeds native GLintptr")
        })?;
        let size = pointer_sized(size)
            .ok_or_else(|| Self::validation(operation, "copy size exceeds native GLsizeiptr"))?;
        if let Some(functions) = self.buffer_functions {
            unsafe {
                (functions.copy_buffer_sub_data)(
                    read_target,
                    write_target,
                    read_offset,
                    write_offset,
                    size,
                )
            };
        } else {
            unsafe {
                self.gl.copy_buffer_sub_data(
                    read_target,
                    write_target,
                    i32::try_from(read_offset).map_err(|_| {
                        Self::validation(operation, "GL buffer ABI entry points unavailable")
                    })?,
                    i32::try_from(write_offset).map_err(|_| {
                        Self::validation(operation, "GL buffer ABI entry points unavailable")
                    })?,
                    i32::try_from(size).map_err(|_| {
                        Self::validation(operation, "GL buffer ABI entry points unavailable")
                    })?,
                );
            }
        }
        Ok(())
    }

    /// Binds a native buffer range without narrowing the byte range to glow's
    /// portable `i32` representation.
    ///
    /// # Safety
    ///
    /// The caller validates the target, index, object ownership and range
    /// according to the command's GL contract.
    pub(super) unsafe fn bind_buffer_range_wide(
        &self,
        target: u32,
        index: u32,
        buffer: Option<glow::NativeBuffer>,
        offset: u64,
        size: u64,
        operation: &'static str,
    ) -> Result<(), crate::backend::api::GlError> {
        use glow::HasContext as _;
        let offset = pointer_sized(offset).ok_or_else(|| {
            Self::validation(operation, "buffer range offset exceeds native GLintptr")
        })?;
        let size = pointer_sized(size).ok_or_else(|| {
            Self::validation(operation, "buffer range size exceeds native GLsizeiptr")
        })?;
        if let Some(functions) = self.buffer_functions {
            let name = buffer.map_or(0, |buffer| buffer.0.get());
            // SAFETY: forwarded from this method's contract.
            unsafe { (functions.bind_buffer_range)(target, index, name, offset, size) };
        } else {
            let offset = i32::try_from(offset).map_err(|_| {
                Self::validation(operation, "GL buffer ABI entry points unavailable")
            })?;
            let size = i32::try_from(size).map_err(|_| {
                Self::validation(operation, "GL buffer ABI entry points unavailable")
            })?;
            // SAFETY: forwarded from this method's contract.
            unsafe {
                self.gl
                    .bind_buffer_range(target, index, buffer, offset, size)
            };
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "../../tests/backend/native/buffer_abi__tests.rs"]
mod tests;
