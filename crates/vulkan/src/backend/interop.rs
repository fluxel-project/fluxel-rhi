//! Native Vulkan device and recording command-buffer access.
//!
//! Device guards retain the device; encoder guards are additionally bounded by
//! the Core encoder borrow. Native callers own all synchronization and retain
//! resources referenced by commands they record.
//! An RHI external acquire or release may end the current native command
//! buffer to place its queue semaphore at that exact boundary. Callers that
//! resume raw Vulkan recording after such an RHI operation must rebind their
//! own Vulkan state; only RHI-owned native state is restored.
//! The raw pointer is the dispatchable `VkDevice` or `VkCommandBuffer` handle
//! selected by the request, not a pointer to an `ash` wrapper.

use std::{ffi::c_void, sync::Arc};

use ash::vk::Handle;
use fluxel_rhi_core::api::{self, Error, Result};

use super::{DeviceState, Vulkan, VulkanDevice, VulkanEncoder};

/// Retains the Vulkan device while exposing a dispatchable Vulkan handle.
pub struct VulkanInteropGuard {
    _device: Arc<DeviceState>,
    handle: u64,
}

impl fluxel_rhi_core::backend::NativeInteropBackend for Vulkan {
    type NativeInteropGuard = VulkanInteropGuard;

    fn acquire_native(
        device: &VulkanDevice,
        request: api::NativeInteropRequest,
    ) -> Result<Self::NativeInteropGuard> {
        match request {
            api::NativeInteropRequest::Device => Ok(VulkanInteropGuard {
                _device: device.0.clone(),
                handle: device.0.device.handle().as_raw(),
            }),
            api::NativeInteropRequest::Encoder => Err(Error::InvalidState),
        }
    }

    fn acquire_encoder(encoder: &mut VulkanEncoder) -> Result<Self::NativeInteropGuard> {
        Ok(VulkanInteropGuard {
            _device: encoder.device.clone(),
            handle: encoder.command.as_raw(),
        })
    }

    unsafe fn raw_pointer(guard: &Self::NativeInteropGuard) -> *mut c_void {
        guard.handle as usize as *mut c_void
    }
}

#[cfg(all(test, windows, feature = "hardware-tests"))]
#[path = "../tests/backend/interop.rs"]
mod tests;
