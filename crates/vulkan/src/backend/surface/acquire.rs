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
            // SAFETY: the first acquire on the replacement chain proves these
            // retired present waits are consumed; SurfaceState owns all objects.
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
    inner.acquired = true;
    Ok(Some(VulkanSurfaceImage {
        state: state.clone(),
        image_index: index,
        image: inner.images[index as usize],
        view: inner.views[index as usize],
        extent: inner.extent,
        format: inner.format,
        generation: inner.generation,
        handed: false,
        present_wait: Mutex::new(None),
    }))
}
