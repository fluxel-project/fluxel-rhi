//! Borrowed access to the actual Metal device and recording command buffer.
//!
//! Core ties encoder guards to an exclusive encoder borrow. Native callers
//! preserve resource lifetimes, restore any modified state, and supply GPU
//! dependencies; the guard does not schedule or replay native work.

use super::{Metal, MetalCommandEncoder, MetalDevice};
use fluxel_rhi_core::api::{Error, NativeInteropBackend, NativeInteropRequest, Result};

pub struct MetalInteropGuard {
    #[cfg(target_vendor = "apple")]
    owner: Owner,
}

#[cfg(target_vendor = "apple")]
enum Owner {
    Device(MetalDevice),
    Command(objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandBuffer>>),
}

impl NativeInteropBackend for Metal {
    type NativeInteropGuard = MetalInteropGuard;

    fn acquire_native(
        device: &MetalDevice,
        request: NativeInteropRequest,
    ) -> Result<MetalInteropGuard> {
        if matches!(request, NativeInteropRequest::Encoder) {
            return Err(Error::InvalidState);
        }
        #[cfg(target_vendor = "apple")]
        {
            Ok(MetalInteropGuard {
                owner: Owner::Device(device.clone()),
            })
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = device;
            Err(Error::Unsupported("Metal requires an Apple target"))
        }
    }

    fn acquire_encoder(encoder: &mut MetalCommandEncoder) -> Result<MetalInteropGuard> {
        #[cfg(target_vendor = "apple")]
        {
            if encoder.render.is_some() || encoder.compute.is_some() {
                return Err(Error::InvalidState);
            }
            Ok(MetalInteropGuard {
                owner: Owner::Command(encoder.command.clone()),
            })
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = encoder;
            Err(Error::Unsupported("Metal requires an Apple target"))
        }
    }

    // SAFETY: the guard owns the Objective-C receiver. Core retains an exclusive
    // encoder borrow for command access; the caller obeys the native API rules.
    unsafe fn raw_pointer(guard: &MetalInteropGuard) -> *mut std::ffi::c_void {
        #[cfg(target_vendor = "apple")]
        {
            match &guard.owner {
                Owner::Device(device) => objc2::rc::Retained::as_ptr(&device._state._device)
                    .cast_mut()
                    .cast(),
                Owner::Command(command) => objc2::rc::Retained::as_ptr(command).cast_mut().cast(),
            }
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = guard;
            std::ptr::null_mut()
        }
    }
}
