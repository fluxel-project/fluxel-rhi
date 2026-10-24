//! Bounded Vulkan image acquisition and completion of the surface lease.
//!
//! The worker retries only bounded native acquires.  Polling an AcquireFuture
//! merely stores a waker; cancellation is observed between retries.  A lease
//! produced after cancellation is dropped immediately, which follows the
//! parent surface's abandoned-lease path rather than exposing an unowned image.

use std::{
    future::Future,
    marker::PhantomData,
    pin::Pin,
    sync::{
        Arc, Mutex, Weak,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    task::{Context, Poll, Waker},
};

use ash::vk;
use fluxel_rhi_core::api::{Error, Result};

use super::{SurfaceState, VulkanSurface, VulkanSurfaceImage, vk_error};

pub(super) struct AcquireWait {
    result: Mutex<(Option<Result<VulkanSurfaceImage>>, Option<Waker>)>,
    cancelled: AtomicBool,
}

pub struct AcquireFuture<'a> {
    wait: Arc<AcquireWait>,
    _surface: PhantomData<&'a mut VulkanSurface>,
}
impl<'a> AcquireFuture<'a> {
    pub(super) fn new(surface: &'a mut VulkanSurface) -> Self {
        let wait = Arc::new(AcquireWait {
            result: Mutex::new((None, None)),
            cancelled: AtomicBool::new(false),
        });
        if surface.0.acquire_sender.send(wait.clone()).is_err() {
            wait.result.lock().unwrap_or_else(|e| e.into_inner()).0 = Some(Err(Error::DeviceLost));
        }
        Self {
            wait,
            _surface: PhantomData,
        }
    }
}
impl Future for AcquireFuture<'_> {
    type Output = Result<VulkanSurfaceImage>;
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut state = self.wait.result.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(result) = state.0.take() {
            Poll::Ready(result)
        } else {
            state.1 = Some(cx.waker().clone());
            Poll::Pending
        }
    }
}
impl Drop for AcquireFuture<'_> {
    fn drop(&mut self) {
        self.wait.cancelled.store(true, Ordering::Release);
    }
}

pub(super) fn start_worker(
    state: Weak<SurfaceState>,
    requests: mpsc::Receiver<Arc<AcquireWait>>,
) -> Result<()> {
    std::thread::Builder::new()
        .name("fluxel-vulkan-acquire".into())
        .spawn(move || acquire_worker(state, requests))
        .map_err(|error| Error::Backend(format!("Vulkan acquire worker: {error}")))?;
    Ok(())
}

fn acquire_worker(state: Weak<SurfaceState>, requests: mpsc::Receiver<Arc<AcquireWait>>) {
    while let Ok(wait) = requests.recv() {
        let Some(state) = state.upgrade() else { break };
        let result = loop {
            if wait.cancelled.load(Ordering::Acquire) {
                break Err(Error::SurfaceOutdated);
            }
            match acquire_once(&state) {
                Ok(None) => {
                    std::thread::yield_now();
                    continue;
                }
                Ok(Some(image)) => break Ok(image),
                Err(error) => break Err(error),
            }
        };
        let waker = {
            let mut pending = wait.result.lock().unwrap_or_else(|e| e.into_inner());
            if wait.cancelled.load(Ordering::Acquire) {
                drop(result);
                None
            } else {
                pending.0 = Some(result);
                pending.1.take()
            }
        };
        if let Some(waker) = waker {
            waker.wake();
        }
    }
}

fn acquire_once(state: &Arc<SurfaceState>) -> Result<Option<VulkanSurfaceImage>> {
    let mut inner = state.inner.lock().unwrap_or_else(|e| e.into_inner());
    if inner.acquired || inner.swapchain == vk::SwapchainKHR::null() {
        return Err(Error::InvalidState);
    }
    if inner.abandoned {
        return Err(Error::SurfaceOutdated);
    }
    // The surface worker owns this bounded wait. Future::poll only observes
    // the result and can be cancelled without blocking the executor thread.
    // SAFETY: SurfaceState retains the live device; this worker exclusively
    // owns the short-lived fence until the bounded acquire has completed.
    let fence = unsafe {
        state
            .device
            .0
            .device
            .create_fence(&vk::FenceCreateInfo::default(), None)
    }
    .map_err(vk_error)?;
    // SAFETY: `inner` holds the configured swapchain and no other acquire can
    // run while it is locked; the fence is valid on the same logical device.
    let acquired = unsafe {
        state.swapchain_loader.acquire_next_image(
            inner.swapchain,
            16_000_000,
            vk::Semaphore::null(),
            fence,
        )
    };
    let result = match acquired {
        Ok((index, _)) => {
            // SAFETY: successful acquire signals this locally owned fence, so
            // waiting here proves the image and its prior present semaphore use.
            unsafe {
                state
                    .device
                    .0
                    .device
                    .wait_for_fences(&[fence], true, u64::MAX)
            }
            .map(|_| index)
        }
        Err(error) => Err(error),
    };
    // SAFETY: the wait above completed on success; timeout/error means acquire
    // did not enqueue work using this local fence.
    unsafe { state.device.0.device.destroy_fence(fence, None) };
    let index = match result {
        Err(vk::Result::TIMEOUT | vk::Result::NOT_READY) => return Ok(None),
        result => result,
    }
    .map_err(|error| match error {
        vk::Result::ERROR_OUT_OF_DATE_KHR => Error::SurfaceOutdated,
        vk::Result::ERROR_SURFACE_LOST_KHR => Error::SurfaceLost,
        _ => vk_error(error),
    })?;
    if let Some(semaphore) = inner.present_waits[index as usize].take() {
        // Waiting for this acquire fence proves the preceding presentation of
        // the same image has consumed its wait semaphore.
        // SAFETY: the acquire fence proves present consumed this semaphore.
        unsafe { state.device.0.device.destroy_semaphore(semaphore, None) };
        for retired in inner.retired.drain(..) {
            // SAFETY: reacquiring a replacement-chain image with a present
            // wait proves its previous presentation consumed that wait. The
            // replacement has therefore presented, retiring the old chain's
            // outstanding presents; its first acquire alone is not proof.
            // SurfaceState owns all retired objects.
            unsafe {
                for view in retired.views {
                    state.device.0.device.destroy_image_view(view, None);
                }
                for semaphore in retired.wait_semaphores {
                    state.device.0.device.destroy_semaphore(semaphore, None);
                }
                state
                    .swapchain_loader
                    .destroy_swapchain(retired.chain, None);
            }
        }
    }
    if !inner.initialized[index as usize] {
        if let Err(error) = initialize_image(state, inner.images[index as usize]) {
            inner.abandoned = true;
            return Err(error);
        }
        inner.initialized[index as usize] = true;
    }
    inner.acquired = true;
    Ok(Some(VulkanSurfaceImage {
        state: state.clone(),
        image_index: index,
        image: inner.images[index as usize],
        view: inner.views[index as usize],
        extent: inner.extent,
        format: inner.format,
        transfer_dst: inner.transfer_dst,
        generation: inner.generation,
        handed: false,
        present_wait: Mutex::new(None),
    }))
}

// Establish a known initial PRESENT layout once, after WSI hands this image
// to us. Later passes can preserve contents across command buffers and leases.
fn initialize_image(state: &Arc<SurfaceState>, image: vk::Image) -> Result<()> {
    let device = &state.device.0.device;
    // SAFETY: the pool belongs to the live graphics family and remains local.
    let pool = unsafe {
        device.create_command_pool(
            &vk::CommandPoolCreateInfo::default()
                .queue_family_index(state.device.0.graphics_family),
            None,
        )
    }
    .map_err(vk_error)?;
    let mut issued = false;
    let result = (|| -> Result<()> {
        // SAFETY: pool is live and exclusively used by this worker.
        let command = unsafe {
            device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
        }
        .map_err(vk_error)?[0];
        let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }
            .map_err(vk_error)?;
        let result = (|| -> Result<()> {
            unsafe {
                device.begin_command_buffer(
                    command,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
            }
            .map_err(vk_error)?;
            let range = vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .level_count(1)
                .layer_count(1);
            let barrier = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(range);
            // SAFETY: WSI acquisition has completed, and this command owns no
            // references beyond the surface, pool, and fence retained here.
            unsafe {
                device.cmd_pipeline_barrier(
                    command,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[barrier],
                );
                device.end_command_buffer(command).map_err(vk_error)?;
            }
            let commands = [command];
            let info = vk::SubmitInfo::default().command_buffers(&commands);
            let submitted = {
                let _queue = state
                    .device
                    .0
                    .queue_gate
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                // SAFETY: queue_gate serializes use of the graphics queue.
                unsafe { device.queue_submit(state.device.0.graphics_queue, &[info], fence) }
            };
            issued = true;
            submitted.map_err(vk_error)?;
            // SAFETY: this worker owns the fence and waits before releasing the pool.
            unsafe { device.wait_for_fences(&[fence], true, u64::MAX) }.map_err(vk_error)?;
            issued = false;
            Ok(())
        })();
        if !issued {
            unsafe { device.destroy_fence(fence, None) };
        }
        result
    })();
    if issued {
        // An uncertain queue call may still reference both the image and pool.
        // Retain their owner on this exceptional path instead of early destroy.
        std::mem::forget(state.clone());
    } else {
        unsafe { device.destroy_command_pool(pool, None) };
    }
    result
}
