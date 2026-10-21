//! NativeActivity lifecycle and retained window/device/surface ownership.
//! All callbacks, scene mutations and GLES operations stay on this event-loop thread.
//! The host requests redraws and reports errors; scene semantics remain in the original examples.

mod draw;
mod initialize;

use super::*;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowAttributes},
};

#[cfg(feature = "vulkan")]
struct Gpu {
    device: Device<fluxel_rhi::vulkan::Vulkan>,
    surface: Surface<fluxel_rhi::vulkan::Vulkan>,
    scene: Scene<fluxel_rhi::vulkan::Vulkan>,
    #[cfg(any(
        feature = "demo03",
        feature = "demo04",
        feature = "demo07",
        feature = "demo08",
        feature = "demo10"
    ))]
    time: std::time::Instant,
    #[cfg(feature = "demo02")]
    pixels: Vec<u8>,
    #[cfg(feature = "demo02")]
    pixel_extent: [u32; 2],
    lane: LaneId,
    extent: [u32; 2],
    format: TextureFormat,
    alpha: CompositeAlphaMode,
    // Rust drops fields in declaration order. The adopted native window must
    // outlive the surface and the EGL context, including error/unwind paths.
    window: Arc<Window>,
}

#[cfg(feature = "gles")]
struct Gpu {
    device: Device<fluxel_rhi::gl::Gl>,
    surface: Surface<fluxel_rhi::gl::Gl>,
    scene: Scene<fluxel_rhi::gl::Gl>,
    #[cfg(any(
        feature = "demo03",
        feature = "demo04",
        feature = "demo07",
        feature = "demo08",
        feature = "demo10"
    ))]
    time: std::time::Instant,
    #[cfg(feature = "demo02")]
    pixels: Vec<u8>,
    #[cfg(feature = "demo02")]
    pixel_extent: [u32; 2],
    lane: LaneId,
    extent: [u32; 2],
    format: TextureFormat,
    alpha: CompositeAlphaMode,
    // Keep the borrowed ANativeWindow alive through EGL/context destruction.
    window: Arc<Window>,
}

#[derive(Default)]
struct App {
    gpu: Option<Gpu>,
    frames: u64,
    capture_path: Option<std::path::PathBuf>,
    surface_lost_deadline: Option<std::time::Instant>,
}

impl App {
    // A native surface may disappear before Android delivers suspended(). Stop
    // drawing immediately and retain all GPU objects until queued work drains.
    // Recovery waits for resumed(), when a new ANativeWindow is available.
    fn retire_gpu(&mut self, _event: &ActiveEventLoop) {
        if let Some(gpu) = self.gpu.take() {
            // The guard drains before the scene and borrowed native window drop.
            // A failed drain aborts instead of releasing potentially in-flight data.
            drop(utils::GpuScope::new(&gpu.device));
        }
        self.frames = 0;
    }

    fn surface_lost(&mut self, event: &ActiveEventLoop, operation: &str) {
        eprintln!("RHI_SURFACE_LOST={operation}; awaiting suspension or resume");
        self.retire_gpu(event);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        self.surface_lost_deadline = Some(deadline);
        event.set_control_flow(ControlFlow::WaitUntil(deadline));
    }
}

impl ApplicationHandler for App {
    fn suspended(&mut self, event: &ActiveEventLoop) {
        // Android may invalidate ANativeWindow after this callback returns.
        // Drain native work and release the adopted surface while it is valid.
        self.retire_gpu(event);
        self.surface_lost_deadline = None;
        event.set_control_flow(ControlFlow::Wait);
        eprintln!("RHI_SUSPENDED");
    }
    fn resumed(&mut self, event: &ActiveEventLoop) {
        self.surface_lost_deadline = None;
        event.set_control_flow(ControlFlow::Poll);
        if self.gpu.is_none() {
            if let Err(error) = self.init(event) {
                eprintln!("RHI_INIT_ERROR={error:?}");
                event.exit();
            }
        }
    }
    fn about_to_wait(&mut self, event: &ActiveEventLoop) {
        if self
            .surface_lost_deadline
            .is_some_and(|deadline| std::time::Instant::now() >= deadline)
        {
            eprintln!("RHI_DRAW_ERROR=surface lost without Android lifecycle transition");
            self.surface_lost_deadline = None;
            event.exit();
        }
    }
    fn window_event(
        &mut self,
        event: &ActiveEventLoop,
        window_id: winit::window::WindowId,
        value: WindowEvent,
    ) {
        // Ignore redraw and resize events queued for a retired Android window.
        if self
            .gpu
            .as_ref()
            .is_none_or(|gpu| gpu.window.id() != window_id)
        {
            return;
        }
        match value {
            WindowEvent::RedrawRequested => {
                if let Err(error) = self.draw() {
                    if matches!(error, Error::SurfaceLost) {
                        self.surface_lost(event, "draw");
                    } else {
                        eprintln!("RHI_DRAW_ERROR={error:?}");
                        event.exit();
                    }
                    return;
                }
                if let Some(gpu) = &self.gpu {
                    gpu.window.request_redraw();
                }
            }
            WindowEvent::CloseRequested => event.exit(),
            WindowEvent::Resized(size) => {
                if let Some(gpu) = &mut self.gpu {
                    if size.width > 0 && size.height > 0 {
                        let caps = match gpu.surface.capabilities() {
                            Ok(c) => c,
                            Err(Error::SurfaceLost) => {
                                self.surface_lost(event, "resize capabilities");
                                return;
                            }
                            Err(e) => {
                                eprintln!("RHI_RESIZE_ERROR={e:?}");
                                event.exit();
                                return;
                            }
                        };
                        gpu.extent = if caps.min_extent == caps.max_extent {
                            caps.min_extent
                        } else {
                            [
                                size.width.clamp(caps.min_extent[0], caps.max_extent[0]),
                                size.height.clamp(caps.min_extent[1], caps.max_extent[1]),
                            ]
                        };
                        #[cfg(feature = "vulkan")]
                        let latency = 2;
                        #[cfg(feature = "gles")]
                        let latency = 1;
                        let config = SurfaceConfig {
                            extent: gpu.extent,
                            format: gpu.format,
                            present_mode: PresentMode::Fifo,
                            alpha_mode: gpu.alpha,
                            color_space: ColorSpace::Srgb,
                            usage: SurfaceUsage::RENDER_ATTACHMENT,
                            maximum_frame_latency: latency,
                        };
                        if let Err(e) = utils::block_on(gpu.surface.configure(&config)) {
                            if matches!(e, Error::SurfaceLost) {
                                self.surface_lost(event, "resize configure");
                            } else {
                                eprintln!("RHI_RESIZE_ERROR={e:?}");
                                event.exit();
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }
}

pub(super) fn run(app: winit::platform::android::activity::AndroidApp) {
    let capture_path = app
        .external_data_path()
        .map(|path| path.join("12_screenshot.ppm"));
    let mut builder = EventLoop::builder();
    winit::platform::android::EventLoopBuilderExtAndroid::with_android_app(&mut builder, app);
    let event_loop = match builder.build() {
        Ok(v) => v,
        Err(e) => {
            eprintln!("EVENT_LOOP_ERROR={e}");
            return;
        }
    };
    event_loop.set_control_flow(ControlFlow::Poll);
    if let Err(error) = event_loop.run_app(&mut App {
        capture_path,
        ..App::default()
    }) {
        eprintln!("EVENT_LOOP_ERROR={error}");
    }
}
