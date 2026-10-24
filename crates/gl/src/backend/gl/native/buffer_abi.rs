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
    ) -> Result<(), crate::backend::gl::api::GlError> {
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
    ) -> Result<(), crate::backend::gl::api::GlError> {
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
    ) -> Result<(), crate::backend::gl::api::GlError> {
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
mod tests {
    use super::{
        NativeBufferFunctions, advertised_native_max_buffer_bytes, native_max_buffer_bytes,
        pointer_sized,
    };
    use core::ffi::c_void;
    #[cfg(target_pointer_width = "64")]
    use core::sync::atomic::{AtomicI64, AtomicU64, Ordering};

    #[test]
    fn pointer_sized_accepts_the_native_addressing_limit() {
        assert_eq!(pointer_sized(native_max_buffer_bytes()), Some(isize::MAX));
    }

    #[test]
    fn pointer_sized_rejects_overflow_without_truncation() {
        assert_eq!(
            pointer_sized(native_max_buffer_bytes().saturating_add(1)),
            None
        );
    }

    #[test]
    fn get_buffer_sub_data_is_optional_for_gles() {
        // SAFETY: sentinel pointers are inspected but never called.
        let functions = unsafe {
            super::NativeBufferFunctions::load(|name| {
                if name == "glGetBufferSubData" {
                    core::ptr::null()
                } else {
                    core::ptr::dangling()
                }
            })
        };
        assert!(functions.is_some());
        assert!(functions.unwrap().get_buffer_sub_data.is_none());
    }

    #[test]
    fn missing_loader_table_does_not_advertise_wide_buffers() {
        assert_eq!(advertised_native_max_buffer_bytes(false), i32::MAX as u64);
        assert_eq!(
            advertised_native_max_buffer_bytes(true),
            native_max_buffer_bytes()
        );
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn native_64_bit_abi_is_not_the_glow_i32_ceiling() {
        assert!(pointer_sized(i32::MAX as u64 + 1).is_some());
    }

    #[cfg(target_pointer_width = "64")]
    static BUFFER_SIZE: AtomicI64 = AtomicI64::new(0);
    #[cfg(target_pointer_width = "64")]
    static BUFFER_OFFSET: AtomicI64 = AtomicI64::new(0);
    #[cfg(target_pointer_width = "64")]
    static TEX_OFFSET: AtomicU64 = AtomicU64::new(0);
    #[cfg(target_pointer_width = "64")]
    static READ_OFFSET: AtomicU64 = AtomicU64::new(0);
    #[cfg(target_pointer_width = "64")]
    static VERTEX_OFFSET: AtomicU64 = AtomicU64::new(0);
    #[cfg(target_pointer_width = "64")]
    static VERTEX_STRIDE: AtomicI64 = AtomicI64::new(0);

    #[cfg(target_pointer_width = "64")]
    unsafe extern "system" fn capture_buffer_data(_: u32, size: isize, _: *const c_void, _: u32) {
        BUFFER_SIZE.store(size as i64, Ordering::Relaxed);
    }
    #[cfg(target_pointer_width = "64")]
    unsafe extern "system" fn capture_buffer_sub_data(
        _: u32,
        offset: isize,
        size: isize,
        _: *const c_void,
    ) {
        BUFFER_OFFSET.store(offset as i64, Ordering::Relaxed);
        BUFFER_SIZE.store(size as i64, Ordering::Relaxed);
    }
    #[cfg(target_pointer_width = "64")]
    unsafe extern "system" fn capture_tex_sub_image_2d(
        _: u32,
        _: i32,
        _: i32,
        _: i32,
        _: i32,
        _: i32,
        _: u32,
        _: u32,
        offset: *const c_void,
    ) {
        TEX_OFFSET.store(offset as usize as u64, Ordering::Relaxed);
    }
    #[cfg(target_pointer_width = "64")]
    unsafe extern "system" fn capture_read_pixels(
        _: i32,
        _: i32,
        _: i32,
        _: i32,
        _: u32,
        _: u32,
        offset: *mut c_void,
    ) {
        READ_OFFSET.store(offset as usize as u64, Ordering::Relaxed);
    }
    #[cfg(target_pointer_width = "64")]
    unsafe extern "system" fn noop_copy(_: u32, _: u32, _: isize, _: isize, _: isize) {}
    #[cfg(target_pointer_width = "64")]
    unsafe extern "system" fn noop_map(_: u32, _: isize, _: isize, _: u32) -> *mut c_void {
        core::ptr::null_mut()
    }
    #[cfg(target_pointer_width = "64")]
    unsafe extern "system" fn noop_bind(_: u32, _: u32, _: u32, _: isize, _: isize) {}
    #[cfg(target_pointer_width = "64")]
    unsafe extern "system" fn capture_vertex_pointer(
        _: u32,
        _: i32,
        _: u32,
        _: u8,
        stride: i32,
        offset: *const c_void,
    ) {
        VERTEX_STRIDE.store(i64::from(stride), Ordering::Relaxed);
        VERTEX_OFFSET.store(offset as usize as u64, Ordering::Relaxed);
    }
    #[cfg(target_pointer_width = "64")]
    unsafe extern "system" fn capture_vertex_i_pointer(
        _: u32,
        _: i32,
        _: u32,
        stride: i32,
        offset: *const c_void,
    ) {
        VERTEX_STRIDE.store(i64::from(stride), Ordering::Relaxed);
        VERTEX_OFFSET.store(offset as usize as u64, Ordering::Relaxed);
    }
    #[cfg(target_pointer_width = "64")]
    unsafe extern "system" fn capture_vertex_l_pointer(
        _: u32,
        _: i32,
        _: u32,
        stride: i32,
        offset: *const c_void,
    ) {
        VERTEX_STRIDE.store(i64::from(stride), Ordering::Relaxed);
        VERTEX_OFFSET.store(offset as usize as u64, Ordering::Relaxed);
    }
    #[cfg(target_pointer_width = "64")]
    unsafe extern "system" fn capture_bind_vertex_buffer(
        _: u32,
        _: u32,
        offset: isize,
        stride: i32,
    ) {
        VERTEX_STRIDE.store(i64::from(stride), Ordering::Relaxed);
        VERTEX_OFFSET.store(offset as u64, Ordering::Relaxed);
    }
    #[cfg(target_pointer_width = "64")]
    unsafe extern "system" fn noop_l_format(_: u32, _: i32, _: u32, _: u32) {}

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn raw_abi_captures_offsets_and_sizes_above_four_gib() {
        let functions = NativeBufferFunctions {
            buffer_data: capture_buffer_data,
            buffer_sub_data: capture_buffer_sub_data,
            copy_buffer_sub_data: noop_copy,
            map_buffer_range: noop_map,
            bind_buffer_range: noop_bind,
            get_buffer_sub_data: None,
            tex_sub_image_1d: None,
            tex_sub_image_2d: Some(capture_tex_sub_image_2d),
            tex_sub_image_3d: None,
            read_pixels: Some(capture_read_pixels),
        };
        let value = 4_u64 * 1024 * 1024 * 1024 + 37;
        let wide = isize::try_from(value).expect("64-bit ABI represents capture value");
        // SAFETY: these are local capture mocks, never GPU functions.
        unsafe {
            (functions.buffer_data)(0, wide, core::ptr::null(), 0);
            (functions.buffer_sub_data)(0, wide, wide - 1, core::ptr::null());
            (functions.tex_sub_image_2d.expect("test mock"))(
                0,
                0,
                0,
                0,
                1,
                1,
                0,
                0,
                value as usize as *const c_void,
            );
            (functions.read_pixels.expect("test mock"))(
                0,
                0,
                1,
                1,
                0,
                0,
                value as usize as *mut c_void,
            );
        }
        assert_eq!(BUFFER_OFFSET.load(Ordering::Relaxed), wide as i64);
        assert_eq!(BUFFER_SIZE.load(Ordering::Relaxed), wide as i64 - 1);
        assert_eq!(TEX_OFFSET.load(Ordering::Relaxed), value);
        assert_eq!(READ_OFFSET.load(Ordering::Relaxed), value);
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn raw_vertex_abi_captures_offsets_above_four_gib() {
        let functions = super::NativeVertexFunctions {
            attrib_pointer: capture_vertex_pointer,
            attrib_i_pointer: capture_vertex_i_pointer,
            attrib_l_pointer: Some(capture_vertex_l_pointer),
            bind_vertex_buffer: Some(capture_bind_vertex_buffer),
            attrib_l_format: Some(noop_l_format),
        };
        let offset = 4_u64 * 1024 * 1024 * 1024 + 53;
        // SAFETY: all calls target local capture mocks.
        unsafe {
            (functions.attrib_pointer)(0, 4, 0, 0, 28, offset as usize as *const c_void);
        }
        assert_eq!(VERTEX_OFFSET.load(Ordering::Relaxed), offset);
        assert_eq!(VERTEX_STRIDE.load(Ordering::Relaxed), 28);
        // SAFETY: all calls target local capture mocks.
        unsafe {
            (functions.attrib_i_pointer)(1, 4, 0, 32, offset as usize as *const c_void);
        }
        assert_eq!(VERTEX_OFFSET.load(Ordering::Relaxed), offset);
        assert_eq!(VERTEX_STRIDE.load(Ordering::Relaxed), 32);
        // SAFETY: all calls target local capture mocks.
        unsafe {
            (functions.attrib_l_pointer.expect("test mock"))(
                2,
                4,
                0,
                40,
                offset as usize as *const c_void,
            );
        }
        assert_eq!(VERTEX_OFFSET.load(Ordering::Relaxed), offset);
        assert_eq!(VERTEX_STRIDE.load(Ordering::Relaxed), 40);
        // SAFETY: all calls target local capture mocks.
        unsafe { (functions.bind_vertex_buffer.expect("test mock"))(0, 7, offset as isize, 0) };
        assert_eq!(VERTEX_OFFSET.load(Ordering::Relaxed), offset);
        assert_eq!(VERTEX_STRIDE.load(Ordering::Relaxed), 0);
        assert!(functions.supports_f64());
        assert!(functions.supports_zero_stride_f64());
    }
}
