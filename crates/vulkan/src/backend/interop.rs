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

impl api::NativeInteropBackend for Vulkan {
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
mod tests {
    use super::*;
    use ash::vk;
    use fluxel_rhi_core::api::*;
    use std::{
        future::Future,
        sync::Arc,
        task::{Context, Poll, Wake, Waker},
    };

    struct WakeThread(std::thread::Thread);
    impl Wake for WakeThread {
        fn wake(self: Arc<Self>) {
            self.0.unpark();
        }
        fn wake_by_ref(self: &Arc<Self>) {
            self.0.unpark();
        }
    }
    fn block_on<F: Future>(f: F) -> F::Output {
        let waker = Waker::from(Arc::new(WakeThread(std::thread::current())));
        let mut cx = Context::from_waker(&waker);
        let mut f = std::pin::pin!(f);
        loop {
            match f.as_mut().poll(&mut cx) {
                Poll::Ready(v) => return v,
                Poll::Pending => std::thread::park(),
            }
        }
    }

    #[test]
    fn native_fill_and_device_guard_lifetime() {
        let instance = super::super::VulkanInstance::new().unwrap();
        let device = block_on(instance.open_default()).unwrap();
        let lane = device.lanes()[0].id;
        let domain = device.native_interop().unwrap();
        // SAFETY: the test only uses the handle while its guard is alive.
        let device_guard = unsafe { domain.acquire(NativeInteropRequest::Device) }.unwrap();
        // SAFETY: the guard retains the logical device.
        assert!(!unsafe { device_guard.raw_pointer() }.is_null());
        // SAFETY: requesting an encoder from a device domain cannot yield one.
        assert!(matches!(
            unsafe { domain.acquire(NativeInteropRequest::Encoder) },
            Err(Error::InvalidState)
        ));
        let buffer = device
            .create_buffer(&BufferDesc {
                label: None,
                size: 16,
                usage: BufferUsage::COPY_SRC.union(BufferUsage::COPY_DST),
                memory: MemoryClass::Device,
            })
            .unwrap();
        // SAFETY: this native reference remains borrowed from the live buffer.
        let native_buffer = unsafe { buffer.native_ref() }.0.buffer;
        // SAFETY: this reference remains borrowed from the live device.
        let native_device = unsafe { device.native_ref() };
        let mut encoder = device.encoder(lane).unwrap();
        // SAFETY: the test retains the buffer through completion, and the
        // readback operation establishes transfer visibility for its copy.
        let guard = unsafe { encoder.native_interop() }.unwrap();
        // SAFETY: the guard borrows the active recording command buffer and
        // the native buffer belongs to its device with TRANSFER_DST usage.
        unsafe {
            let command = vk::CommandBuffer::from_raw(guard.raw_pointer() as usize as u64);
            native_device
                .0
                .device
                .cmd_fill_buffer(command, native_buffer, 0, 16, 0x12345678);
        }
        drop(guard);
        let ticket = encoder.readback_buffer(&buffer, 0..16).unwrap();
        let work = encoder.finish().unwrap();
        let id = PlanId {
            device: device.id(),
            ordinal: 1,
        };
        // SAFETY: the sole step and payload both use this device and lane.
        let plan = unsafe {
            SubmissionPlan::from_compiled(
                id,
                Arc::from([StepDesc {
                    lane,
                    kind: StepKind::Commands,
                    waits: vec![],
                }]),
            )
        };
        let step = plan.step(0);
        // SAFETY: the payload matches the plan's lane and step.
        let receipt = match unsafe {
            device.queue().submit(SubmitPacket {
                plan,
                steps: vec![BoundStep {
                    step,
                    payload: StepPayload::Commands(vec![work]),
                }],
                completion: CompletionRequest::End,
            })
        } {
            SubmitOutcome::Accepted(r) => r,
            _ => panic!("submission failed"),
        };
        let point = &receipt.checkpoints[0].point;
        block_on(device.wait_completion(point)).unwrap();
        // SAFETY: the checkpoint covers the readback.
        let bytes = block_on(unsafe { device.read_after(ticket, point) })
            .unwrap()
            .bytes;
        assert_eq!(bytes, 0x12345678u32.to_ne_bytes().repeat(4));
        drop(receipt);
        drop(buffer);
        drop(device);
        drop(instance);
        // SAFETY: the device guard itself retains the native device.
        assert!(!unsafe { device_guard.raw_pointer() }.is_null());
        drop(device_guard);
        // Use a fresh, never-submitted device to isolate the guard's ownership
        // from the submission system's in-flight references.
        let instance = super::super::VulkanInstance::new().unwrap();
        let device = block_on(instance.open_default()).unwrap();
        // SAFETY: this reference is used only to observe ownership.
        let weak = Arc::downgrade(&unsafe { device.native_ref() }.0);
        // SAFETY: the guard owns the native device lifetime.
        let guard = unsafe {
            device
                .native_interop()
                .unwrap()
                .acquire(NativeInteropRequest::Device)
        }
        .unwrap();
        drop(device);
        drop(instance);
        assert!(weak.upgrade().is_some());
        drop(guard);
        assert!(weak.upgrade().is_none());
    }
}
