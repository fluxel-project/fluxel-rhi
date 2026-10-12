//! Win32 window and event loop for statically typed DX12 examples.

use std::sync::{Arc, Mutex};

use fluxel_host::{HostApplication, HostContext, HostRuntime, WindowConfig, WindowEvent};
use fluxel_rhi::{
    api::{Device, Error, Result, Surface, SurfaceConfig, SurfaceImage},
    dx12::{Dx12, Dx12Instance},
};
use raw_window_handle::HasWindowHandle;

use super::{ExampleHost, block_on};

pub struct Dx12Host;

type Draw =
    dyn FnMut(&Device<Dx12>, &mut Surface<Dx12>, SurfaceImage<Dx12>, [u32; 2]) -> Result<()>;

struct Application {
    title: String,
    config: SurfaceConfig,
    device: Option<Device<Dx12>>,
    surface: Option<Surface<Dx12>>,
    draw: Box<Draw>,
    failure: Arc<Mutex<Option<Error>>>,
    suspended: bool,
    frames: u64,
    frame_limit: Option<u64>,
}

impl Application {
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
        let instance = Dx12Instance::new()?;
        let device = block_on(instance.open_default())?;
        let mut surface = unsafe { Dx12::surface_from_raw_window_handle(&device, handle)? };
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
        self.config.extent = [width, height];
        if let Some(surface) = self.surface.as_mut() {
            block_on(surface.configure(&self.config))?;
        }
        Ok(())
    }
}

impl HostApplication for Application {
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
        let result = match event {
            WindowEvent::SurfaceCreated => self.open(host),
            WindowEvent::SurfaceDestroyed => {
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

impl ExampleHost<Dx12> for Dx12Host {
    fn run(
        self,
        title: &str,
        config: SurfaceConfig,
        draw: impl FnMut(&Device<Dx12>, &mut Surface<Dx12>, SurfaceImage<Dx12>, [u32; 2]) -> Result<()>
        + 'static,
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
}
