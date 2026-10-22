//! Win32 window and event loop for statically typed native examples.

use std::sync::{Arc, Mutex};

use fluxel_host::{HostApplication, HostContext, HostRuntime, WindowConfig, WindowEvent};
use fluxel_rhi::api::{
    CompletionBackend, Device, Error, Result, Surface, SurfaceBackend, SurfaceConfig, SurfaceImage,
};
#[cfg(feature = "dx12")]
use fluxel_rhi::dx12::{Dx12, Dx12Instance};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle};

use super::block_on;
#[cfg(any(feature = "dx12", feature = "vulkan"))]
use super::{ExampleHost, ExampleInputHost};

#[cfg(feature = "dx12")]
pub struct Dx12Host;
#[cfg(feature = "vulkan")]
pub struct VulkanHost;

pub enum SelectedBackend {
    #[cfg(feature = "dx12")]
    Dx12,
    #[cfg(feature = "vulkan")]
    Vulkan,
}

pub fn selected_backend() -> Result<SelectedBackend> {
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--backend" {
            return match args.next().as_deref() {
                #[cfg(feature = "dx12")]
                Some("dx12") => Ok(SelectedBackend::Dx12),
                #[cfg(feature = "vulkan")]
                Some("vulkan") => Ok(SelectedBackend::Vulkan),
                _ => Err(Error::Unsupported("requested example backend")),
            };
        }
        if arg == "--frames" {
            args.next().ok_or(Error::InvalidRange)?;
        }
    }
    #[cfg(feature = "dx12")]
    {
        Ok(SelectedBackend::Dx12)
    }
    #[cfg(all(not(feature = "dx12"), feature = "vulkan"))]
    {
        Ok(SelectedBackend::Vulkan)
    }
    #[cfg(all(not(feature = "dx12"), not(feature = "vulkan")))]
    {
        Err(Error::Unsupported(
            "no deferred native example backend is enabled",
        ))
    }
}

trait WindowsBackend: SurfaceBackend + CompletionBackend + Sized + 'static {
    fn open() -> Result<Device<Self>>;
    /// # Safety
    ///
    /// `window` and `display` must originate from a live host window and that
    /// window must outlive the returned surface. `Application` stores the
    /// surface only between matching host lifecycle events, which maintains
    /// that handle lifetime.
    unsafe fn surface(
        device: &Device<Self>,
        window: RawWindowHandle,
        display: RawDisplayHandle,
    ) -> Result<Surface<Self>>;
}

#[cfg(feature = "dx12")]
impl WindowsBackend for Dx12 {
    fn open() -> Result<Device<Self>> {
        block_on(Dx12Instance::new()?.open_default())
    }

    // SAFETY: the trait contract keeps the host window alive for the returned
    // surface; this implementation forwards the same borrowed Win32 handle.
    unsafe fn surface(
        device: &Device<Self>,
        window: RawWindowHandle,
        _: RawDisplayHandle,
    ) -> Result<Surface<Self>> {
        // SAFETY: `Application::open` obtained this raw handle from the live
        // host window and retains that window through the surface's lifetime.
        unsafe { Dx12::surface_from_raw_window_handle(device, window) }
    }
}

#[cfg(feature = "vulkan")]
impl WindowsBackend for fluxel_rhi::vulkan::Vulkan {
    fn open() -> Result<Device<Self>> {
        block_on(fluxel_rhi::vulkan::VulkanInstance::new()?.open_default())
    }

    // SAFETY: both raw handles belong to the live host window whose lifetime
    // the caller extends through the surface and every acquired image lease.
    unsafe fn surface(
        device: &Device<Self>,
        window: RawWindowHandle,
        display: RawDisplayHandle,
    ) -> Result<Surface<Self>> {
        // SAFETY: both handles came from the same live host window;
        // `Application` drops the surface before that host is destroyed.
        unsafe { Self::surface_from_raw_window_handle(device, window, display) }
    }
}

type Draw<B> = dyn FnMut(&Device<B>, &mut Surface<B>, SurfaceImage<B>, [u32; 2]) -> Result<()>;

struct Application<B: WindowsBackend> {
    title: String,
    config: SurfaceConfig,
    device: Option<Device<B>>,
    surface: Option<Surface<B>>,
    draw: Box<Draw<B>>,
    input: Box<dyn FnMut(&WindowEvent)>,
    failure: Arc<Mutex<Option<Error>>>,
    suspended: bool,
    frames: u64,
    frame_limit: Option<u64>,
}

impl<B: WindowsBackend> Application<B> {
    fn fail(&self, host: &HostContext<'_>, error: Error) {
        if let Ok(mut slot) = self.failure.lock() {
            *slot = Some(error);
        }
        host.exit();
    }

    fn open(&mut self, host: &HostContext<'_>) -> Result<()> {
        let window = host.window().ok_or(Error::SurfaceLost)?;
        let handle = window
            .window_handle()
            .map_err(|error| Error::Backend(error.to_string()))?
            .as_raw();
        let display = window
            .display_handle()
            .map_err(|error| Error::Backend(error.to_string()))?
            .as_raw();
        let device = B::open()?;
        // SAFETY: the host owns `window` for this lifecycle and drops the
        // surface on `SurfaceDestroyed` before that ownership ends.
        let mut surface = unsafe { B::surface(&device, handle, display)? };
        block_on(surface.configure(&self.config))?;
        self.device = Some(device);
        self.surface = Some(surface);
        window.request_redraw();
        Ok(())
    }

    fn redraw(&mut self, host: &HostContext<'_>) -> Result<()> {
        if self.suspended || self.config.extent.contains(&0) {
            return Ok(());
        }
        let device = self.device.as_ref().ok_or(Error::DeviceLost)?;
        let surface = self.surface.as_mut().ok_or(Error::SurfaceLost)?;
        let image = block_on(surface.acquire())?;
        // The draw closure owns scene resources. Keep it live until even a
        // failed or partially accepted frame has reached idle.
        let _scope = super::GpuScope::new(device);
        (self.draw)(device, surface, image, self.config.extent)?;
        self.frames += 1;
        if self.frame_limit.is_some_and(|limit| self.frames >= limit) {
            host.exit();
            return Ok(());
        }
        if let Some(window) = host.window() {
            window.request_redraw();
        }
        Ok(())
    }

    fn resize(&mut self, width: u32, height: u32) -> Result<()> {
        if width == 0 || height == 0 {
            return Ok(());
        }
        if let Some(surface) = self.surface.as_mut() {
            let caps = surface.capabilities()?;
            self.config.extent = [
                width.clamp(caps.min_extent[0], caps.max_extent[0]),
                height.clamp(caps.min_extent[1], caps.max_extent[1]),
            ];
            block_on(surface.configure(&self.config))?;
        } else {
            self.config.extent = [width, height];
        }
        Ok(())
    }
}

impl<B: WindowsBackend> HostApplication for Application<B> {
    fn resumed(&mut self, host: &mut HostContext<'_>) {
        if host.window().is_none() {
            let result =
                WindowConfig::new(&self.title, self.config.extent[0], self.config.extent[1])
                    .map_err(|error| Error::Backend(error.to_string()))
                    .and_then(|config| {
                        host.create_window(config)
                            .map(|_| ())
                            .map_err(|error| Error::Backend(error.to_string()))
                    });
            if let Err(error) = result {
                self.fail(host, error);
            }
        }
    }

    fn window_event(&mut self, host: &mut HostContext<'_>, event: WindowEvent) {
        if matches!(
            event,
            WindowEvent::CursorMoved { .. }
                | WindowEvent::CursorLeft
                | WindowEvent::MouseInput { .. }
                | WindowEvent::MouseWheel { .. }
                | WindowEvent::KeyboardInput { .. }
                | WindowEvent::TextInput(_)
                | WindowEvent::Focused(_)
        ) {
            (self.input)(&event);
        }
        let result = match event {
            WindowEvent::SurfaceCreated => self.open(host),
            WindowEvent::SurfaceDestroyed => {
                if let Some(device) = self.device.as_ref() {
                    drop(super::GpuScope::new(device));
                }
                self.surface = None;
                self.device = None;
                Ok(())
            }
            WindowEvent::Suspended | WindowEvent::Minimized => {
                self.suspended = true;
                Ok(())
            }
            WindowEvent::Resumed => {
                self.suspended = false;
                if let Some(window) = host.window() {
                    window.request_redraw();
                }
                Ok(())
            }
            WindowEvent::Resized { width, height } | WindowEvent::Restored { width, height } => {
                self.suspended = width == 0 || height == 0;
                let result = self.resize(width, height);
                if result.is_ok() && !self.suspended {
                    if let Some(window) = host.window() {
                        window.request_redraw();
                    }
                }
                result
            }
            WindowEvent::RedrawRequested => self.redraw(host),
            WindowEvent::CloseRequested => {
                host.exit();
                Ok(())
            }
            _ => Ok(()),
        };
        if let Err(error) = result {
            self.fail(host, error);
        }
    }
}

fn run_windows<B: WindowsBackend>(
    title: &str,
    config: SurfaceConfig,
    draw: impl FnMut(&Device<B>, &mut Surface<B>, SurfaceImage<B>, [u32; 2]) -> Result<()> + 'static,
    input: impl FnMut(&WindowEvent) + 'static,
) -> Result<()> {
    let mut frame_limit = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--frames" {
            let count = args
                .next()
                .ok_or(Error::InvalidRange)?
                .parse::<u64>()
                .map_err(|_| Error::InvalidRange)?;
            if count == 0 {
                return Err(Error::InvalidRange);
            }
            frame_limit = Some(count);
        } else if arg == "--backend" {
            args.next().ok_or(Error::InvalidRange)?;
        } else {
            return Err(Error::Unsupported("example command-line argument"));
        }
    }
    let failure = Arc::new(Mutex::new(None));
    let application = Application {
        title: title.into(),
        config,
        device: None,
        surface: None,
        draw: Box::new(draw),
        input: Box::new(input),
        failure: Arc::clone(&failure),
        suspended: false,
        frames: 0,
        frame_limit,
    };
    HostRuntime::new()
        .map_err(|error| Error::Backend(error.to_string()))?
        .run(application)
        .map_err(|error| Error::Backend(error.to_string()))?;
    let result = failure.lock().map_err(|_| Error::InvalidState)?.take();
    result.map_or(Ok(()), Err)
}

#[cfg(feature = "dx12")]
impl ExampleHost<Dx12> for Dx12Host {
    fn run(
        self,
        title: &str,
        config: SurfaceConfig,
        draw: impl FnMut(&Device<Dx12>, &mut Surface<Dx12>, SurfaceImage<Dx12>, [u32; 2]) -> Result<()>
        + 'static,
    ) -> Result<()> {
        run_windows::<Dx12>(title, config, draw, |_| {})
    }
}

#[cfg(feature = "vulkan")]
impl ExampleHost<fluxel_rhi::vulkan::Vulkan> for VulkanHost {
    fn run(
        self,
        title: &str,
        config: SurfaceConfig,
        draw: impl FnMut(
            &Device<fluxel_rhi::vulkan::Vulkan>,
            &mut Surface<fluxel_rhi::vulkan::Vulkan>,
            SurfaceImage<fluxel_rhi::vulkan::Vulkan>,
            [u32; 2],
        ) -> Result<()>
        + 'static,
    ) -> Result<()> {
        run_windows::<fluxel_rhi::vulkan::Vulkan>(title, config, draw, |_| {})
    }
}

#[cfg(feature = "dx12")]
impl ExampleInputHost<Dx12> for Dx12Host {
    fn run_with_input(
        self,
        title: &str,
        config: SurfaceConfig,
        draw: impl FnMut(&Device<Dx12>, &mut Surface<Dx12>, SurfaceImage<Dx12>, [u32; 2]) -> Result<()>
        + 'static,
        input: impl FnMut(&WindowEvent) + 'static,
    ) -> Result<()> {
        run_windows::<Dx12>(title, config, draw, input)
    }
}

#[cfg(feature = "vulkan")]
impl ExampleInputHost<fluxel_rhi::vulkan::Vulkan> for VulkanHost {
    fn run_with_input(
        self,
        title: &str,
        config: SurfaceConfig,
        draw: impl FnMut(
            &Device<fluxel_rhi::vulkan::Vulkan>,
            &mut Surface<fluxel_rhi::vulkan::Vulkan>,
            SurfaceImage<fluxel_rhi::vulkan::Vulkan>,
            [u32; 2],
        ) -> Result<()>
        + 'static,
        input: impl FnMut(&WindowEvent) + 'static,
    ) -> Result<()> {
        run_windows::<fluxel_rhi::vulkan::Vulkan>(title, config, draw, input)
    }
}
