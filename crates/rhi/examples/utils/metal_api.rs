//! AppKit window and CAMetalLayer adapter for the typed Metal examples.
//!
//! `fluxel-host` owns the native window and event loop. This module borrows its
//! `NSView`, attaches a retained `CAMetalLayer`, and keeps RHI Surface lifetime
//! inside the matching host surface interval. It does not own frame scheduling
//! beyond requesting the next redraw, or make renderer retirement decisions.

use std::sync::{Arc, Mutex};

use fluxel_host::{HostApplication, HostContext, HostRuntime, WindowConfig, WindowEvent};
use fluxel_rhi::api::{Device, Error, Result, Surface, SurfaceConfig, SurfaceImage};
use fluxel_rhi::metal::{Metal, MetalInstance};
use objc2::rc::Retained;
use objc2_app_kit::NSView;
use objc2_quartz_core::{CALayer, CAMetalLayer};
use raw_window_handle::{HasWindowHandle, RawWindowHandle};

use super::{ExampleHost, block_on};

/// Host adapter selecting the single typed Metal backend at startup.
pub struct MetalHost;

type Draw = dyn FnMut(&Device<Metal>, &mut Surface<Metal>, SurfaceImage<Metal>, [u32; 2]) -> Result<()>
    + 'static;

struct Application {
    title: String,
    config: SurfaceConfig,
    device: Option<Device<Metal>>,
    surface: Option<Surface<Metal>>,
    view: Option<Retained<NSView>>,
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
        let raw = window
            .window_handle()
            .map_err(|error| Error::Backend(error.to_string()))?
            .as_raw();
        let RawWindowHandle::AppKit(handle) = raw else {
            return Err(Error::Unsupported("AppKit window handle"));
        };

        // SAFETY: raw-window-handle guarantees this pointer refers to the live
        // NSView owned by `window`; callbacks run on AppKit's main thread, and
        // the host keeps the window alive until SurfaceDestroyed. Retain adds
        // an owned reference before the borrowed window handle can expire.
        let view = unsafe { Retained::<NSView>::retain(handle.ns_view.as_ptr().cast()) }
            .ok_or(Error::SurfaceLost)?;
        let layer = CAMetalLayer::layer();
        view.setWantsLayer(true);
        let layer_as_super: &CALayer = &layer;
        view.setLayer(Some(layer_as_super));

        let device = block_on(MetalInstance::new()?.open_default())?;
        let mut surface = Metal::surface_from_layer(&device, layer)?;
        block_on(surface.configure(&self.config))?;
        self.view = Some(view);
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
        let _scope = super::GpuScope::new(device);
        (self.draw)(device, surface, image, self.config.extent)?;
        self.frames += 1;
        if self.frame_limit.is_some_and(|limit| self.frames >= limit) {
            host.exit();
        } else if let Some(window) = host.window() {
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
                if let Some(device) = self.device.as_ref() {
                    drop(super::GpuScope::new(device));
                }
                self.surface = None;
                self.device = None;
                self.view = None;
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

impl ExampleHost<Metal> for MetalHost {
    fn run(
        self,
        title: &str,
        config: SurfaceConfig,
        draw: impl FnMut(
            &Device<Metal>,
            &mut Surface<Metal>,
            SurfaceImage<Metal>,
            [u32; 2],
        ) -> Result<()>
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
            } else if arg == "--backend" {
                if args.next().as_deref() != Some("metal") {
                    return Err(Error::Unsupported("requested example backend"));
                }
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
            view: None,
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
