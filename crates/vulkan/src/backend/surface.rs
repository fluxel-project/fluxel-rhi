//! Vulkan Win32 surface and swapchain ownership.
//!
//! A swapchain image stays an exclusive `SurfaceImage` lease.  It is never
//! manufactured as a portable TextureView, so presentation lifetime remains
//! backend-private and the raster encoder receives it only through the Core
//! attachment vocabulary.

use std::{
    future::{Ready, ready},
    sync::{Arc, Mutex, mpsc},
};

use ash::vk;
use fluxel_rhi_core::api::{self, Error, Result};
use raw_window_handle::{RawDisplayHandle, RawWindowHandle, Win32WindowHandle};

use super::{Vulkan, VulkanDevice, vk_error};
mod acquire;
use acquire::{AcquireFuture, AcquireWait};

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

pub(crate) unsafe fn create_surface(
    device: &api::Device<Vulkan>,
    window: RawWindowHandle,
    display: RawDisplayHandle,
) -> Result<api::Surface<Vulkan>> {
    let RawWindowHandle::Win32(Win32WindowHandle {
        hwnd, hinstance, ..
    }) = window
    else {
        return Err(Error::Unsupported("a Win32 Vulkan window handle"));
    };
    if !matches!(display, RawDisplayHandle::Windows(_)) {
        return Err(Error::Unsupported("a Windows display handle"));
    }
    let native = unsafe { device.native_ref() }.clone();
    let win32 = ash::khr::win32_surface::Instance::new(
        &native.0.instance.entry,
        &native.0.instance.instance,
    );
    let info = vk::Win32SurfaceCreateInfoKHR::default()
        .hwnd(hwnd.get() as vk::HWND)
        .hinstance(hinstance.map_or(0, |v| v.get()) as vk::HINSTANCE);
    let surface = unsafe { win32.create_win32_surface(&info, None) }.map_err(vk_error)?;
    let surface_loader =
        ash::khr::surface::Instance::new(&native.0.instance.entry, &native.0.instance.instance);
    let supported = unsafe {
        surface_loader.get_physical_device_surface_support(
            native.0.physical,
            native.0.graphics_family,
            surface,
        )
    }
    .map_err(vk_error)?;
    if !supported {
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
    let caps = unsafe {
        state
            .surface_loader
            .get_physical_device_surface_capabilities(state.device.0.physical, state.surface)
    }
    .map_err(vk_error)?;
    let formats = unsafe {
        state
            .surface_loader
            .get_physical_device_surface_formats(state.device.0.physical, state.surface)
    }
    .map_err(vk_error)?;
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
        alpha_modes: if caps
            .supported_composite_alpha
            .contains(vk::CompositeAlphaFlagsKHR::OPAQUE)
        {
            vec![api::CompositeAlphaMode::Opaque]
        } else {
            vec![]
        },
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
        || config.alpha_mode != api::CompositeAlphaMode::Opaque
        || config.color_space != api::ColorSpace::Srgb
        || config.usage != api::SurfaceUsage::RENDER_ATTACHMENT
        || config.maximum_frame_latency == 0
    {
        return Err("Vulkan surface configuration mode/usage");
    }
    let Some(format) = super::resource::format(config.format) else {
        return Err("Vulkan surface format");
    };
    let caps = unsafe {
        state
            .surface_loader
            .get_physical_device_surface_capabilities(state.device.0.physical, state.surface)
    }
    .map_err(|_| "Vulkan surface capabilities")?;
    if !caps
        .supported_composite_alpha
        .contains(vk::CompositeAlphaFlagsKHR::OPAQUE)
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
fn configure(state: &SurfaceState, config: &api::SurfaceConfig) -> Result<()> {
    validate_config(state, config).map_err(Error::Unsupported)?;
    let caps = unsafe {
        state
            .surface_loader
            .get_physical_device_surface_capabilities(state.device.0.physical, state.surface)
    }
    .map_err(vk_error)?;
    let formats = unsafe {
        state
            .surface_loader
            .get_physical_device_surface_formats(state.device.0.physical, state.surface)
    }
    .map_err(vk_error)?;
    let format = super::resource::format(config.format)
        .ok_or(Error::Unsupported("Vulkan surface format"))?;
    let surface_format = formats
        .into_iter()
        .find(|f| f.format == format && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR)
        .ok_or(Error::Unsupported("Vulkan surface format"))?;
    let extent = vk::Extent2D {
        width: config.extent[0],
        height: config.extent[1],
    };
    let mut inner = state.inner.lock().unwrap_or_else(|e| e.into_inner());
    if inner.acquired {
        return Err(Error::InvalidState);
    }
    let next_generation = inner.generation.checked_add(1).ok_or(Error::OutOfMemory)?;
    let old = inner.swapchain;
    let mut usage = vk::ImageUsageFlags::COLOR_ATTACHMENT;
    if std::env::var_os("FLUXEL_VULKAN_CAPTURE").is_some() {
        if !caps
            .supported_usage_flags
            .contains(vk::ImageUsageFlags::TRANSFER_SRC)
        {
            return Err(Error::Unsupported(
                "Vulkan swapchain transfer source capture",
            ));
        }
        usage |= vk::ImageUsageFlags::TRANSFER_SRC;
    }
    let info = vk::SwapchainCreateInfoKHR::default()
        .surface(state.surface)
        .min_image_count(caps.min_image_count.max(2))
        .image_format(surface_format.format)
        .image_color_space(surface_format.color_space)
        .image_extent(extent)
        .image_array_layers(1)
        .image_usage(usage)
        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
        .pre_transform(caps.current_transform)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .present_mode(vk::PresentModeKHR::FIFO)
        .clipped(true)
        .old_swapchain(old);
    let created = unsafe { state.swapchain_loader.create_swapchain(&info, None) };
    if old != vk::SwapchainKHR::null() {
        // The old chain is retired by the create attempt even if creation
        // fails. It must never remain active or be passed as oldSwapchain
        // again. Keep its views and present waits until new presentation
        // provides safe retirement evidence.
        let mut waits: Vec<_> = inner.present_waits.drain(..).flatten().collect();
        waits.extend(inner.orphaned_waits.drain(..));
        let old_views = std::mem::take(&mut inner.views);
        let retired = RetiredSwapchain {
            chain: old,
            views: old_views,
            wait_semaphores: waits,
        };
        if inner.presented {
            // Without a present fence, keep this chain until presentation of
            // the new chain is proved finished by reacquiring its image.
            inner.retired.push(retired);
        } else {
            // A chain never handed to presentation only has queue work. The
            // cold resize path may wait for that work and reclaim it now.
            let idle = {
                let _queue = state
                    .device
                    .0
                    .queue_gate
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                unsafe {
                    state
                        .device
                        .0
                        .device
                        .queue_wait_idle(state.device.0.graphics_queue)
                }
            };
            if idle.is_ok() {
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
            } else {
                inner.retired.push(retired);
            }
        }
        inner.swapchain = vk::SwapchainKHR::null();
        inner.images.clear();
    }
    let chain = created.map_err(vk_error)?;
    let images =
        unsafe { state.swapchain_loader.get_swapchain_images(chain) }.map_err(|error| {
            unsafe { state.swapchain_loader.destroy_swapchain(chain, None) };
            vk_error(error)
        })?;
    let mut views = Vec::with_capacity(images.len());
    for image in &images {
        let range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };
        let view_info = vk::ImageViewCreateInfo::default()
            .image(*image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(range);
        match unsafe { state.device.0.device.create_image_view(&view_info, None) } {
            Ok(view) => views.push(view),
            Err(error) => {
                unsafe {
                    for view in views {
                        state.device.0.device.destroy_image_view(view, None);
                    }
                    state.swapchain_loader.destroy_swapchain(chain, None);
                }
                return Err(vk_error(error));
            }
        }
    }
    inner.swapchain = chain;
    inner.images = images;
    inner.views = views;
    inner.present_waits = vec![None; inner.images.len()];
    inner.extent = extent;
    inner.format = format;
    inner.generation = next_generation;
    inner.abandoned = false;
    inner.presented = false;
    Ok(())
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
