//! Vulkan Win32 and Android surface and swapchain ownership.
//!
//! A swapchain image stays an exclusive `SurfaceImage` lease.  It is never
//! manufactured as a portable TextureView, so presentation lifetime remains
//! backend-private and the raster encoder receives it only through the Core
//! attachment vocabulary.
//!
//! Configuration replaces swapchains atomically under `inner`.  An acquired
//! image is represented only by `VulkanSurfaceImage`; dropping an unhanded
//! lease marks the chain abandoned rather than pretending the image was
//! released.  Retired chains are destroyed only after the documented acquire
//! evidence shows their presentation wait semaphores are no longer in use.
//! The acquire worker owns bounded CPU waits; `Future::poll` only registers a
//! waker and never blocks an executor thread.

use std::{
    future::{Ready, ready},
    sync::{Arc, Mutex, mpsc},
};

use ash::vk;
use fluxel_rhi_core::api::{self, Error, Result};
#[cfg(target_os = "android")]
use raw_window_handle::AndroidNdkWindowHandle;
#[cfg(windows)]
use raw_window_handle::Win32WindowHandle;
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};

use super::{Vulkan, VulkanDevice, vk_error};
mod acquire;
mod configure;
use acquire::{AcquireFuture, AcquireWait};
use configure::configure;

#[derive(Clone)]
pub struct VulkanSurface(pub(crate) Arc<SurfaceState>);
pub(crate) struct SurfaceState {
    device: VulkanDevice,
    surface_loader: ash::khr::surface::Instance,
    swapchain_loader: ash::khr::swapchain::Device,
    surface: vk::SurfaceKHR,
    inner: Mutex<Inner>,
    acquire_sender: mpsc::Sender<Arc<AcquireWait>>,
}
struct Inner {
    swapchain: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    views: Vec<vk::ImageView>,
    // A present wait semaphore cannot be reused until this image is acquired
    // again and that acquire fence has completed.
    present_waits: Vec<Option<vk::Semaphore>>,
    orphaned_waits: Vec<vk::Semaphore>,
    retired: Vec<RetiredSwapchain>,
    extent: vk::Extent2D,
    format: vk::Format,
    generation: u64,
    acquired: bool,
    abandoned: bool,
    presented: bool,
}
struct RetiredSwapchain {
    chain: vk::SwapchainKHR,
    views: Vec<vk::ImageView>,
    wait_semaphores: Vec<vk::Semaphore>,
}

pub struct VulkanSurfaceImage {
    pub(crate) state: Arc<SurfaceState>,
    pub(crate) image_index: u32,
    pub(crate) image: vk::Image,
    pub(crate) view: vk::ImageView,
    extent: vk::Extent2D,
    format: vk::Format,
    generation: u64,
    handed: bool,
    present_wait: Mutex<Option<vk::Semaphore>>,
}

// SAFETY: callers preserve the raw window/display handles until every surface
// image lease is gone; SurfaceState then owns all Vulkan objects it creates.
pub(crate) unsafe fn create_surface(
    device: &api::Device<Vulkan>,
    window: RawWindowHandle,
    display: RawDisplayHandle,
) -> Result<api::Surface<Vulkan>> {
    // SAFETY: the public caller promises that these borrowed raw handles stay
    // valid for the surface and every originating image lease.  This function
    // copies only their numeric Win32 values into Vulkan's create structure.
    // SAFETY: this backend bridge reads the native VulkanDevice belonging to
    // `device`; cloning it retains the dispatch owners for the new surface.
    let native = unsafe { device.native_ref() }.clone();
    let surface = match (window, display) {
        #[cfg(windows)]
        (
            RawWindowHandle::Win32(Win32WindowHandle {
                hwnd, hinstance, ..
            }),
            RawDisplayHandle::Windows(_),
        ) => {
            let win32 = ash::khr::win32_surface::Instance::new(
                &native.0.instance.entry,
                &native.0.instance.instance,
            );
            let info = vk::Win32SurfaceCreateInfoKHR::default()
                .hwnd(hwnd.get() as vk::HWND)
                .hinstance(hinstance.map_or(0, |v| v.get()) as vk::HINSTANCE);
            // SAFETY: the caller retains this validated live HWND/HINSTANCE pair.
            unsafe { win32.create_win32_surface(&info, None) }.map_err(vk_error)?
        }
        #[cfg(target_os = "android")]
        (
            RawWindowHandle::AndroidNdk(AndroidNdkWindowHandle {
                a_native_window, ..
            }),
            RawDisplayHandle::Android(_),
        ) => {
            let android = ash::khr::android_surface::Instance::new(
                &native.0.instance.entry,
                &native.0.instance.instance,
            );
            let info =
                vk::AndroidSurfaceCreateInfoKHR::default().window(a_native_window.as_ptr().cast());
            // SAFETY: the caller retains the ANativeWindow until SurfaceState is dropped.
            unsafe { android.create_android_surface(&info, None) }.map_err(vk_error)?
        }
        _ => {
            return Err(Error::Unsupported(
                "a Vulkan window handle for this platform",
            ));
        }
    };
    let surface_loader =
        ash::khr::surface::Instance::new(&native.0.instance.entry, &native.0.instance.instance);
    // SAFETY: physical device, queue family, and surface all belong to the
    // retained instance and have not been destroyed.
    let supported = unsafe {
        surface_loader.get_physical_device_surface_support(
            native.0.physical,
            native.0.graphics_family,
            surface,
        )
    }
    .map_err(vk_error)?;
    if !supported {
        // SAFETY: this locally owned surface was never published to SurfaceState.
        unsafe { surface_loader.destroy_surface(surface, None) };
        return Err(Error::Unsupported(
            "Vulkan graphics queue presentation to this HWND",
        ));
    }
    let swapchain_loader =
        ash::khr::swapchain::Device::new(&native.0.instance.instance, &native.0.device);
    let (acquire_sender, acquire_receiver) = mpsc::channel();
    let state = Arc::new(SurfaceState {
        device: native,
        surface_loader,
        swapchain_loader,
        surface,
        acquire_sender,
        inner: Mutex::new(Inner {
            swapchain: vk::SwapchainKHR::null(),
            images: Vec::new(),
            views: Vec::new(),
            present_waits: Vec::new(),
            orphaned_waits: Vec::new(),
            retired: Vec::new(),
            extent: vk::Extent2D::default(),
            format: vk::Format::UNDEFINED,
            generation: 0,
            acquired: false,
            abandoned: false,
            presented: false,
        }),
    });
    acquire::start_worker(Arc::downgrade(&state), acquire_receiver)?;
    // SAFETY: SurfaceState is newly allocated for this Core device and owns
    // every native WSI object it will expose through VulkanSurface.
    Ok(unsafe {
        api::Surface::from_native(
            device.id(),
            api::SurfaceId::allocate()?,
            VulkanSurface(state),
        )
    })
}

impl Drop for SurfaceState {
    fn drop(&mut self) {
        let inner = self.inner.get_mut().unwrap_or_else(|e| e.into_inner());
        // SAFETY: SurfaceState owns every listed WSI object. queue_gate
        // serializes the externally synchronized queue idle call; idle proves
        // no queue operation still refers to the views, semaphores or chains.
        unsafe {
            let _queue = self
                .device
                .0
                .queue_gate
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            let _ = self.device.0.device.device_wait_idle();
            for view in inner.views.drain(..) {
                self.device.0.device.destroy_image_view(view, None);
            }
            for semaphore in inner.present_waits.drain(..).flatten() {
                self.device.0.device.destroy_semaphore(semaphore, None);
            }
            for semaphore in inner.orphaned_waits.drain(..) {
                self.device.0.device.destroy_semaphore(semaphore, None);
            }
            for retired in inner.retired.drain(..) {
                for view in retired.views {
                    self.device.0.device.destroy_image_view(view, None);
                }
                for semaphore in retired.wait_semaphores {
                    self.device.0.device.destroy_semaphore(semaphore, None);
                }
                self.swapchain_loader.destroy_swapchain(retired.chain, None);
            }
            if inner.swapchain != vk::SwapchainKHR::null() {
                self.swapchain_loader
                    .destroy_swapchain(inner.swapchain, None);
            }
            self.surface_loader.destroy_surface(self.surface, None);
        }
    }
}
impl VulkanSurfaceImage {
    pub(super) fn attachment(&self) -> (vk::ImageView, vk::Image, vk::Extent2D, vk::Format) {
        (self.view, self.image, self.extent, self.format)
    }

    pub(super) fn set_present_wait(&self, semaphore: vk::Semaphore) -> Result<()> {
        let mut pending = self.present_wait.lock().unwrap_or_else(|e| e.into_inner());
        if pending.is_some() {
            return Err(Error::InvalidState);
        }
        *pending = Some(semaphore);
        Ok(())
    }
}

impl Drop for VulkanSurfaceImage {
    fn drop(&mut self) {
        if let Some(semaphore) = self
            .present_wait
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
        {
            // Submission may have signaled this semaphore even if presentation
            // was rejected. Retain it until the surface's idle teardown.
            self.state
                .inner
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .orphaned_waits
                .push(semaphore);
        }
        if !self.handed {
            let mut inner = self.state.inner.lock().unwrap_or_else(|e| e.into_inner());
            if inner.generation == self.generation {
                inner.acquired = false;
                // Core permits dropping a lease. Without swapchain maintenance1,
                // Vulkan cannot release that acquired image for reuse. A new
                // swapchain is required before another acquire can proceed.
                inner.abandoned = true;
            }
        }
    }
}
// SAFETY: `inner` serializes generation, acquisition, abandonment, and native
// present handoff, while the lease retains SurfaceState for its entire lifetime.
unsafe impl api::NativeImageLease for VulkanSurfaceImage {
    fn try_handoff(&mut self, generation: u64) -> Result<()> {
        if self.handed || generation != self.generation {
            return Err(Error::SurfaceOutdated);
        }
        let mut inner = self.state.inner.lock().unwrap_or_else(|e| e.into_inner());
        if !inner.acquired || inner.generation != generation {
            return Err(Error::SurfaceOutdated);
        }
        let slot = inner
            .present_waits
            .get_mut(self.image_index as usize)
            .ok_or(Error::InvalidState)?;
        if slot.is_some() {
            return Err(Error::InvalidState);
        }
        let pending = self
            .present_wait
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
            .ok_or(Error::InvalidState)?;
        *slot = Some(pending);
        let chains = [inner.swapchain];
        let indices = [self.image_index];
        let waits = [pending];
        let info = vk::PresentInfoKHR::default()
            .swapchains(&chains)
            .image_indices(&indices)
            .wait_semaphores(&waits);
        let present = {
            let _guard = self
                .state
                .device
                .0
                .queue_gate
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            // SAFETY: lease handoff has moved the semaphore into the current
            // swapchain image slot; queue_gate serializes external queue access.
            unsafe {
                self.state
                    .swapchain_loader
                    .queue_present(self.state.device.0.graphics_queue, &info)
            }
        };
        match present {
            Ok(_) => {
                inner.acquired = false;
                inner.presented = true;
                self.handed = true;
                Ok(())
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => Err(Error::SurfaceOutdated),
            Err(vk::Result::ERROR_SURFACE_LOST_KHR) => Err(Error::SurfaceLost),
            Err(error) => Err(vk_error(error)),
        }
    }
    fn is_current(&self, generation: u64) -> bool {
        let inner = self.state.inner.lock().unwrap_or_else(|e| e.into_inner());
        !self.handed && inner.acquired && inner.generation == generation
    }
}

impl api::SurfaceBackend for Vulkan {
    type ConfigureFuture<'a> = Ready<Result<()>>;
    type AcquireFuture<'a> = AcquireFuture<'a>;
    fn generation(surface: &VulkanSurface) -> u64 {
        surface
            .0
            .inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .generation
    }
    fn capabilities(surface: &VulkanSurface) -> Result<api::SurfaceCapabilities> {
        capabilities(&surface.0)
    }
    fn supports_config(surface: &VulkanSurface, config: &api::SurfaceConfig) -> api::Support<()> {
        match validate_config(&surface.0, config) {
            Ok(()) => api::Support::Supported(()),
            Err(reason) => api::Support::Unsupported(reason),
        }
    }
    fn configure<'a>(
        surface: &'a mut VulkanSurface,
        config: &'a api::SurfaceConfig,
    ) -> Self::ConfigureFuture<'a> {
        ready(configure(&surface.0, config))
    }
    fn acquire<'a>(surface: &'a mut VulkanSurface) -> Self::AcquireFuture<'a> {
        AcquireFuture::new(surface)
    }
}

fn capabilities(state: &SurfaceState) -> Result<api::SurfaceCapabilities> {
    // SAFETY: SurfaceState retains the instance, physical device, and surface;
    // these read-only WSI queries need no queue synchronization.
    let caps = unsafe {
        state
            .surface_loader
            .get_physical_device_surface_capabilities(state.device.0.physical, state.surface)
    }
    .map_err(vk_error)?;
    // SAFETY: same retained physical device/surface pair as the capability query.
    let formats = unsafe {
        state
            .surface_loader
            .get_physical_device_surface_formats(state.device.0.physical, state.surface)
    }
    .map_err(vk_error)?;
    // SAFETY: same retained physical device/surface pair as the capability query.
    let modes = unsafe {
        state
            .surface_loader
            .get_physical_device_surface_present_modes(state.device.0.physical, state.surface)
    }
    .map_err(vk_error)?;
    let mut public_formats: Vec<_> = formats
        .into_iter()
        .filter(|v| v.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR)
        .filter_map(|v| from_format(v.format))
        .collect();
    let mut seen = Vec::new();
    public_formats.retain(|format| {
        if seen.contains(format) {
            false
        } else {
            seen.push(*format);
            true
        }
    });
    let srgb = !public_formats.is_empty();
    Ok(api::SurfaceCapabilities {
        formats: public_formats,
        present_modes: if modes.contains(&vk::PresentModeKHR::FIFO) {
            vec![api::PresentMode::Fifo]
        } else {
            vec![]
        },
        alpha_modes: [
            (
                vk::CompositeAlphaFlagsKHR::OPAQUE,
                api::CompositeAlphaMode::Opaque,
            ),
            (
                vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED,
                api::CompositeAlphaMode::Premultiplied,
            ),
            (
                vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED,
                api::CompositeAlphaMode::Postmultiplied,
            ),
            (
                vk::CompositeAlphaFlagsKHR::INHERIT,
                api::CompositeAlphaMode::Inherit,
            ),
        ]
        .into_iter()
        .filter_map(|(flag, mode)| {
            caps.supported_composite_alpha
                .contains(flag)
                .then_some(mode)
        })
        .collect(),
        color_spaces: if srgb {
            vec![api::ColorSpace::Srgb]
        } else {
            vec![]
        },
        usages: api::SurfaceUsage(
            if caps
                .supported_usage_flags
                .contains(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            {
                api::SurfaceUsage::RENDER_ATTACHMENT.0
            } else {
                0
            },
        ),
        min_extent: if caps.current_extent.width != u32::MAX {
            [caps.current_extent.width, caps.current_extent.height]
        } else {
            [caps.min_image_extent.width, caps.min_image_extent.height]
        },
        max_extent: if caps.current_extent.width != u32::MAX {
            [caps.current_extent.width, caps.current_extent.height]
        } else {
            [caps.max_image_extent.width, caps.max_image_extent.height]
        },
    })
}
fn validate_config(
    state: &SurfaceState,
    config: &api::SurfaceConfig,
) -> std::result::Result<(), &'static str> {
    if config.extent.contains(&0) {
        return Err("zero-sized Vulkan surface");
    }
    if config.present_mode != api::PresentMode::Fifo
        || config.color_space != api::ColorSpace::Srgb
        || config.usage != api::SurfaceUsage::RENDER_ATTACHMENT
        || config.maximum_frame_latency == 0
    {
        return Err("Vulkan surface configuration mode/usage");
    }
    let Some(format) = super::resource::format(config.format) else {
        return Err("Vulkan surface format");
    };
    // SAFETY: validation only queries the SurfaceState-owned WSI pair.
    let caps = unsafe {
        state
            .surface_loader
            .get_physical_device_surface_capabilities(state.device.0.physical, state.surface)
    }
    .map_err(|_| "Vulkan surface capabilities")?;
    if !composite_alpha(config.alpha_mode)
        .is_some_and(|flag| caps.supported_composite_alpha.contains(flag))
        || !caps
            .supported_usage_flags
            .contains(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        || (caps.max_image_count != 0 && caps.min_image_count.max(2) > caps.max_image_count)
    {
        return Err("Vulkan surface attachment/alpha support");
    }
    let requested = vk::Extent2D {
        width: config.extent[0],
        height: config.extent[1],
    };
    if caps.current_extent.width != u32::MAX {
        if requested != caps.current_extent {
            return Err("Vulkan surface current extent");
        }
    } else if requested.width < caps.min_image_extent.width
        || requested.height < caps.min_image_extent.height
        || requested.width > caps.max_image_extent.width
        || requested.height > caps.max_image_extent.height
    {
        return Err("Vulkan surface extent range");
    }
    // SAFETY: validation only queries the SurfaceState-owned WSI pair.
    let formats = unsafe {
        state
            .surface_loader
            .get_physical_device_surface_formats(state.device.0.physical, state.surface)
    }
    .map_err(|_| "Vulkan surface formats")?;
    if !formats
        .iter()
        .any(|f| f.format == format && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR)
    {
        return Err("Vulkan surface format/color space");
    }
    // SAFETY: validation only queries the SurfaceState-owned WSI pair.
    let modes = unsafe {
        state
            .surface_loader
            .get_physical_device_surface_present_modes(state.device.0.physical, state.surface)
    }
    .map_err(|_| "Vulkan surface present modes")?;
    if !modes.contains(&vk::PresentModeKHR::FIFO) {
        return Err("Vulkan FIFO presentation");
    }
    Ok(())
}

pub(super) fn composite_alpha(mode: api::CompositeAlphaMode) -> Option<vk::CompositeAlphaFlagsKHR> {
    Some(match mode {
        api::CompositeAlphaMode::Opaque => vk::CompositeAlphaFlagsKHR::OPAQUE,
        api::CompositeAlphaMode::Premultiplied => vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED,
        api::CompositeAlphaMode::Postmultiplied => vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED,
        api::CompositeAlphaMode::Inherit => vk::CompositeAlphaFlagsKHR::INHERIT,
    })
}
fn from_format(format: vk::Format) -> Option<api::TextureFormat> {
    match format {
        vk::Format::B8G8R8A8_UNORM => Some(api::TextureFormat::Bgra8Unorm),
        vk::Format::B8G8R8A8_SRGB => Some(api::TextureFormat::Bgra8UnormSrgb),
        vk::Format::R8G8B8A8_UNORM => Some(api::TextureFormat::Rgba8Unorm),
        vk::Format::R8G8B8A8_SRGB => Some(api::TextureFormat::Rgba8UnormSrgb),
        _ => None,
    }
}
