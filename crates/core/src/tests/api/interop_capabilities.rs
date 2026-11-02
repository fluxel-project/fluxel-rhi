//! Native interop capability admission before backend acquisition.
use super::{support::NativeShared, *};
use std::{cell::Cell, sync::Arc};

thread_local! {
    static ENCODER_INTEROP_CALLS: Cell<usize> = const { Cell::new(0) };
}

impl NativeInteropBackend for NativeShared {
    type NativeInteropGuard = ();

    fn acquire_native(_: &Arc<()>, _: NativeInteropRequest) -> Result<Self::NativeInteropGuard> {
        Ok(())
    }

    fn acquire_encoder(_: &mut ()) -> Result<Self::NativeInteropGuard> {
        ENCODER_INTEROP_CALLS.with(|calls| calls.set(calls.get() + 1));
        Ok(())
    }

    unsafe fn raw_pointer(_: &Self::NativeInteropGuard) -> *mut std::ffi::c_void {
        std::ptr::null_mut()
    }
}

#[test]
fn encoder_native_interop_requires_capability_before_backend_acquisition() {
    ENCODER_INTEROP_CALLS.with(|calls| calls.set(0));
    let id = DeviceId::allocate().unwrap();
    let device = unsafe {
        Device::<NativeShared>::from_native(id, Arc::new(()), DeviceCapabilities::default())
    };
    let lane = LaneId {
        device: id,
        ordinal: 0,
    };
    let mut encoder = CommandEncoder::new(&device, lane).unwrap();

    assert!(matches!(
        unsafe { encoder.native_interop() },
        Err(Error::Unsupported("native interop"))
    ));
    ENCODER_INTEROP_CALLS.with(|calls| assert_eq!(calls.get(), 0));

    let id = DeviceId::allocate().unwrap();
    let mut caps = DeviceCapabilities::default();
    caps.native_interop = true;
    let device = unsafe { Device::<NativeShared>::from_native(id, Arc::new(()), caps) };
    let lane = LaneId {
        device: id,
        ordinal: 0,
    };
    let mut encoder = CommandEncoder::new(&device, lane).unwrap();
    let guard = unsafe { encoder.native_interop() }.unwrap();
    drop(guard);
    ENCODER_INTEROP_CALLS.with(|calls| assert_eq!(calls.get(), 1));
}
