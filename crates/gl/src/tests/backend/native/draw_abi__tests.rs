
use super::NativeDrawFunctions;
use core::ffi::c_void;

#[cfg(target_pointer_width = "64")]
use core::sync::atomic::{AtomicI64, AtomicU64, Ordering};

#[cfg(target_pointer_width = "64")]
static POINTER_OFFSET: AtomicU64 = AtomicU64::new(0);
#[cfg(target_pointer_width = "64")]
static INTEGER_OFFSET: AtomicI64 = AtomicI64::new(0);

#[cfg(target_pointer_width = "64")]
unsafe extern "system" fn capture_elements(_: u32, _: i32, _: u32, pointer: *const c_void) {
    POINTER_OFFSET.store(pointer as usize as u64, Ordering::Relaxed);
}
#[cfg(target_pointer_width = "64")]
unsafe extern "system" fn capture_elements_instanced(
    _: u32,
    _: i32,
    _: u32,
    pointer: *const c_void,
    _: i32,
) {
    POINTER_OFFSET.store(pointer as usize as u64, Ordering::Relaxed);
}
#[cfg(target_pointer_width = "64")]
unsafe extern "system" fn capture_arrays_indirect(_: u32, pointer: *const c_void) {
    POINTER_OFFSET.store(pointer as usize as u64, Ordering::Relaxed);
}
#[cfg(target_pointer_width = "64")]
unsafe extern "system" fn capture_elements_indirect(_: u32, _: u32, pointer: *const c_void) {
    POINTER_OFFSET.store(pointer as usize as u64, Ordering::Relaxed);
}
#[cfg(target_pointer_width = "64")]
unsafe extern "system" fn capture_dispatch_indirect(offset: isize) {
    INTEGER_OFFSET.store(offset as i64, Ordering::Relaxed);
}

#[cfg(target_pointer_width = "64")]
#[test]
fn raw_draw_abi_captures_offsets_above_four_gib() {
    let functions = NativeDrawFunctions {
        elements: Some(capture_elements),
        elements_instanced: Some(capture_elements_instanced),
        arrays_indirect: Some(capture_arrays_indirect),
        elements_indirect: Some(capture_elements_indirect),
        dispatch_indirect: Some(capture_dispatch_indirect),
    };
    let offset = 4_u64 * 1024 * 1024 * 1024 + 71;
    let pointer = offset as usize as *const c_void;
    // SAFETY: each target is a local capture mock.
    unsafe {
        (functions.elements.expect("test mock"))(0, 1, 0, pointer);
        assert_eq!(POINTER_OFFSET.load(Ordering::Relaxed), offset);
        (functions.elements_instanced.expect("test mock"))(0, 1, 0, pointer, 1);
        assert_eq!(POINTER_OFFSET.load(Ordering::Relaxed), offset);
        (functions.arrays_indirect.expect("test mock"))(0, pointer);
        assert_eq!(POINTER_OFFSET.load(Ordering::Relaxed), offset);
        (functions.elements_indirect.expect("test mock"))(0, 0, pointer);
        assert_eq!(POINTER_OFFSET.load(Ordering::Relaxed), offset);
        (functions.dispatch_indirect.expect("test mock"))(offset as isize);
    }
    assert_eq!(INTEGER_OFFSET.load(Ordering::Relaxed), offset as i64);
}
