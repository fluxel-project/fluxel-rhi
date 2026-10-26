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
