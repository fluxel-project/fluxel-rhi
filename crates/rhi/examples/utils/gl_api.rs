//! Win32 WGL host for immediate-mode GL examples.
//!
//! The adopted WGL context remains on the native event-loop thread. Each
//! redraw acquires one surface lease and invokes the example's immediate
//! workload directly; this module owns neither a command queue nor frame
//! scheduling. It exits after a destroyed surface because an example closure
//! may retain resources tied to that adopted context.

use std::sync::{Arc, Mutex};

use fluxel_host::{HostApplication, HostContext, HostRuntime, WindowConfig, WindowEvent};
use fluxel_rhi::{
    api::{
        Device, DeviceRequest, Error, Instance, InstanceId, Result, Surface, SurfaceConfig,
        SurfaceImage,
    },
    gl::{Gl, GlInstance},
};
use raw_window_handle::{HasWindowHandle, RawWindowHandle};

use super::block_on;

type Draw = dyn FnMut(&Device<Gl>, &mut Surface<Gl>, SurfaceImage<Gl>, [u32; 2]) -> Result<()>;
type Destroy = dyn FnMut();

#[repr(C)]
struct ClientRect {
    left: i32,
    top: i32,
    right: i32,
    bottom: i32,
}

unsafe extern "system" {
    fn GetClientRect(window: *mut core::ffi::c_void, rect: *mut ClientRect) -> i32;
}

/// Observes the actual Win32 drawable dimensions before configuring WGL.
///
/// WGL owns no resizable swapchain: after the native owner began reporting
/// `GetClientRect`, Core configuration must use the same physical client size
/// rather than the requested winit window extent.
fn client_extent(window: &(impl HasWindowHandle + ?Sized)) -> Result<[u32; 2]> {
    let raw = window
        .window_handle()
        .map_err(|error| Error::Backend(error.to_string()))?
        .as_raw();
    let RawWindowHandle::Win32(handle) = raw else {
        return Err(Error::Unsupported("Win32 WGL window handle"));
    };
    let mut rect = ClientRect {
        left: 0,
        top: 0,
        right: 0,
        bottom: 0,
    };
    // SAFETY: `handle` was borrowed from the live HostWindow for this callback;
    // the RECT is valid writable memory for the duration of the Win32 query.
    if unsafe { GetClientRect(handle.hwnd.get() as *mut _, &mut rect) } == 0 {
        return Err(Error::Backend("GetClientRect failed".into()));
    }
    let width = u32::try_from(
        rect.right
            .checked_sub(rect.left)
            .ok_or(Error::SurfaceLost)?,
    )
    .map_err(|_| Error::SurfaceLost)?;
    let height = u32::try_from(
        rect.bottom
            .checked_sub(rect.top)
            .ok_or(Error::SurfaceLost)?,
    )
    .map_err(|_| Error::SurfaceLost)?;
    if width == 0 || height == 0 {
        return Err(Error::SurfaceLost);
    }
    Ok([width, height])
}

struct Application {
    title: String,
    config: SurfaceConfig,
    device: Option<Device<Gl>>,
    surface: Option<Surface<Gl>>,
    draw: Box<Draw>,
    drop_resources: Box<Destroy>,
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
        self.config.extent = client_extent(window)?;
        // SAFETY: `Application` retains the host window through this one
        // surface lifecycle. It drops the surface and device on destruction,
        // and all callbacks, GL calls, and final resource drops stay on the
        // HostRuntime owner thread.
        let native = unsafe { GlInstance::from_wgl_window(window, self.config.extent)? };
        let instance = unsafe { Instance::<Gl>::from_native(InstanceId::allocate()?, native) };
        let adapter = instance
            .enumerate()?
            .into_iter()
            .next()
            .ok_or(Error::Unsupported("WGL adapter"))?;
        let device = block_on(instance.open(DeviceRequest {
            adapter: adapter.id,
            required_features: Vec::new(),
        }))?;
        let mut surface = Gl::surface(&device)?;
        if let Err(error) = block_on(surface.configure(&self.config)) {
            eprintln!(
                "WGL initial configure failed for {:?}: {error:?}",
                self.config.extent
            );
            return Err(error);
        }
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
        let image = match block_on(surface.acquire()) {
            Ok(image) => image,
            Err(Error::SurfaceOutdated) => {
                // Winit can deliver a redraw while a Win32 client resize is
                // already visible to WGL but its resize event is still queued.
                // No GPU work has been issued for this lease, so reconfigure
                // against the actual client rectangle and acquire once more.
                let previous = self.config.extent;
                self.config.extent = client_extent(host.window().ok_or(Error::SurfaceLost)?)?;
                eprintln!(
                    "WGL acquire observed a resized drawable: {previous:?} -> {:?}",
                    self.config.extent
                );
                block_on(surface.configure(&self.config))?;
                block_on(surface.acquire()).map_err(|error| {
                    eprintln!("WGL acquire after reconfigure failed: {error:?}");
                    error
                })?
            }
            Err(error) => return Err(error),
        };
        (self.draw)(device, surface, image, self.config.extent).map_err(|error| {
            eprintln!("WGL immediate frame failed after acquire: {error:?}");
            error
        })?;
        self.frames += 1;
        if self.frame_limit.is_some_and(|limit| self.frames >= limit) {
            host.exit();
        } else if let Some(window) = host.window() {
            window.request_redraw();
        }
        Ok(())
    }

    fn resize(&mut self, host: &HostContext<'_>, width: u32, height: u32) -> Result<()> {
        if width == 0 || height == 0 {
            return Ok(());
        }
        // The Host event is a notification, not proof of the physical WGL
        // drawable size. Both this path and the backend now query the live
        // Win32 client rectangle before accepting a configuration generation.
        self.config.extent = client_extent(host.window().ok_or(Error::SurfaceLost)?)?;
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
                // A new WGL drawable needs a new context and fresh example
                // resources. This narrow first-example host exits instead of
                // accidentally reusing an old device through its draw closure.
                (self.drop_resources)();
                self.surface = None;
                self.device = None;
                host.exit();
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
                let result = self.resize(host, width, height);
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

fn frame_limit() -> Result<Option<u64>> {
    let mut args = std::env::args().skip(1);
    let mut frames = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--frames" => {
                let count = args
                    .next()
                    .ok_or(Error::InvalidRange)?
                    .parse::<u64>()
                    .map_err(|_| Error::InvalidRange)?;
                frames = Some((count != 0).then_some(count).ok_or(Error::InvalidRange)?);
            }
            "--backend" => {
                if args.next().as_deref() != Some("gl") {
                    return Err(Error::Unsupported("WGL GL example backend"));
                }
            }
            _ => return Err(Error::Unsupported("example command-line argument")),
        }
    }
    Ok(frames)
}

/// Runs one WGL example using the adopted window drawable.
///
/// The callback must issue and finish its [`fluxel_rhi::api::ImmediateSession`]
/// before returning, handing the acquired lease to its plan's present step.
pub fn run_wgl(
    title: &str,
    config: SurfaceConfig,
    drop_resources: impl FnMut() + 'static,
    draw: impl FnMut(&Device<Gl>, &mut Surface<Gl>, SurfaceImage<Gl>, [u32; 2]) -> Result<()> + 'static,
) -> Result<()> {
    let failure = Arc::new(Mutex::new(None));
    let application = Application {
        title: title.into(),
        config,
        device: None,
        surface: None,
        draw: Box::new(draw),
        drop_resources: Box::new(drop_resources),
        failure: Arc::clone(&failure),
        suspended: false,
        frames: 0,
        frame_limit: frame_limit()?,
    };
    HostRuntime::new()
        .map_err(|error| Error::Backend(error.to_string()))?
        .run(application)
        .map_err(|error| Error::Backend(error.to_string()))?;
    failure
        .lock()
        .map_err(|_| Error::InvalidState)?
        .take()
        .map_or(Ok(()), Err)
}
