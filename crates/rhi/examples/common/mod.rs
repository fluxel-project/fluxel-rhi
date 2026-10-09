//! Shared native lifecycle framework for runnable RHI examples.
//!
//! A concrete example owns only portable RHI work.  Its platform adapter owns
//! the `fluxel-host` to RHI composition: backend provider selection, opening a
//! device, and creation and retirement of the presentation target.  This keeps
//! raw window handles and backend-native objects out of individual demos.
//!
//! It is placed in `common/` rather than at `examples/` root so Cargo does not
//! discover it as a standalone example binary.  A runnable example imports it
//! with `mod common;`.
//!
//! # Native backend matrix
//!
//! `dx12`, `vulkan`, and `gl4` are Windows selections.  `gles3` and `vulkan`
//! are Android selections.  `metal` is the Apple selection for macOS and iOS.
//! There is no native `webgl2` or `webgpu` selection: Chrome's canvas, RAF,
//! visibility, and context-loss lifecycle are owned by the browser adapter in
//! `fluxel-jsbridge`.  A browser runner may use the same [`Example`] trait but
//! must not construct [`NativeExampleRunner`].
//!
//! The selection parser deliberately reports platform and build-feature
//! mismatches before a platform adapter attempts to open a device.  Runtime
//! support remains subject to the selected adapter's actual driver/device
//! capabilities, which the adapter reports when it opens the session.

use std::{error::Error, fmt, num::NonZeroU32};

use fluxel_host::{
    HostApplication, HostContext, HostWindow, WindowConfig, WindowError, WindowEvent,
};

#[cfg(all(target_os = "android", feature = "vulkan"))]
pub mod android;
#[cfg(target_vendor = "apple")]
pub mod apple;
pub mod shader;
#[cfg(target_arch = "wasm32")]
pub mod web;
#[cfg(all(
    windows,
    any(feature = "dx12", feature = "vulkan", feature = "native-gl-wgl")
))]
pub mod windows;
#[cfg(all(
    windows,
    any(feature = "dx12", feature = "vulkan", feature = "native-gl-wgl")
))]
#[allow(unused_imports)]
pub use windows::{WindowsPlatform, WindowsSession};
#[cfg(all(windows, feature = "vulkan"))]
#[allow(unused_imports)]
pub use windows::{WindowsVulkanPlatform, WindowsVulkanSession};

/// Runs a platform-independent demo through the adapter selected for this target.
pub fn run_example<D: Example + 'static>(title: &str, demo: D) -> Result<(), Box<dyn Error>> {
    #[cfg(all(
        windows,
        any(feature = "dx12", feature = "vulkan", feature = "native-gl-wgl")
    ))]
    {
        return windows::run_example(title, demo);
    }
    #[cfg(all(target_os = "android", feature = "vulkan"))]
    {
        let _ = (title, demo);
        Err("Android examples are started by the NativeActivity entry point".into())
    }
    #[cfg(not(any(
        all(
            windows,
            any(feature = "dx12", feature = "vulkan", feature = "native-gl-wgl")
        ),
        all(target_os = "android", feature = "vulkan")
    )))]
    {
        let _ = (title, demo);
        Err("no example platform adapter is registered for this target".into())
    }
}

#[cfg(all(target_os = "android", feature = "vulkan"))]
#[unsafe(no_mangle)]
pub fn android_main(app: fluxel_host::AndroidApp) {
    if let Err(error) = android::run_example("Basic indexed triangle", crate::create_example(), app)
    {
        eprintln!("01_triangle: {error}");
    }
}

/// Drives an RHI future on the current thread for simple native example callbacks.
pub fn block_on<F: std::future::Future>(future: F) -> F::Output {
    struct ThreadWaker(std::thread::Thread);
    impl std::task::Wake for ThreadWaker {
        fn wake(self: std::sync::Arc<Self>) {
            self.0.unpark();
        }
        fn wake_by_ref(self: &std::sync::Arc<Self>) {
            self.0.unpark();
        }
    }
    let waker = std::task::Waker::from(std::sync::Arc::new(ThreadWaker(std::thread::current())));
    let mut context = std::task::Context::from_waker(&waker);
    let mut future = std::pin::pin!(future);
    loop {
        match std::future::Future::poll(future.as_mut(), &mut context) {
            std::task::Poll::Ready(value) => return value,
            std::task::Poll::Pending => std::thread::park(),
        }
    }
}

/// The backend requested on an example command line.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeBackend {
    /// Direct3D 12 on Windows.
    Dx12,
    /// Vulkan on Windows and Android.
    Vulkan,
    /// A desktop OpenGL 4.x context on Windows.
    Gl4,
    /// An OpenGL ES 3.x context on Android.
    Gles3,
    /// Metal on macOS and iOS.
    Metal,
}

impl NativeBackend {
    /// Parses the stable backend spelling used by the command-line runner.
    pub fn parse(value: &str) -> Result<Self, FrameworkError> {
        match value {
            "dx12" => Ok(Self::Dx12),
            "vulkan" => Ok(Self::Vulkan),
            "gl4" => Ok(Self::Gl4),
            "gles3" => Ok(Self::Gles3),
            "metal" => Ok(Self::Metal),
            "webgl2" | "webgpu" => Err(FrameworkError::BrowserBackend(value.to_owned())),
            _ => Err(FrameworkError::UnknownBackend(value.to_owned())),
        }
    }

    /// Returns whether this backend belongs to the current native target.
    ///
    /// This only checks the target family.  Feature selection and device
    /// discovery remain the platform adapter's responsibility, so an example
    /// can emit one precise diagnostic for a missing RHI backend feature or a
    /// driver capability instead of encoding those details in CLI parsing.
    pub const fn is_native_on_current_target(self) -> bool {
        match self {
            Self::Dx12 | Self::Gl4 => cfg!(target_os = "windows"),
            Self::Vulkan => cfg!(any(target_os = "windows", target_os = "android")),
            Self::Gles3 => cfg!(target_os = "android"),
            Self::Metal => cfg!(any(target_os = "macos", target_os = "ios")),
        }
    }

    /// The stable spelling accepted by [`Self::parse`].
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Dx12 => "dx12",
            Self::Vulkan => "vulkan",
            Self::Gl4 => "gl4",
            Self::Gles3 => "gles3",
            Self::Metal => "metal",
        }
    }
}

/// Common command-line settings for every native RHI example.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RunnerOptions {
    /// The RHI backend to open.
    pub backend: NativeBackend,
    /// Initial host client extent.
    pub width: NonZeroU32,
    /// Initial host client extent.
    pub height: NonZeroU32,
    /// Optional frame count for a finite native smoke run.
    pub frames: Option<NonZeroU32>,
    /// Window title supplied to `fluxel-host`.
    pub title: String,
}

impl RunnerOptions {
    /// Creates fixed backend settings for packaged targets without a CLI.
    pub fn for_backend(title: impl Into<String>, backend: NativeBackend) -> Self {
        Self {
            backend,
            width: NonZeroU32::new(1280).expect("literal is non-zero"),
            height: NonZeroU32::new(720).expect("literal is non-zero"),
            frames: None,
            title: format!("{}: {}", backend.as_str(), title.into()),
        }
    }

    /// Parses `--backend NAME`, `--width PIXELS`, `--height PIXELS`, and
    /// optional `--frames COUNT`.
    ///
    /// The application supplies its own title so all examples retain a clear
    /// identity in screenshots and automated desktop runs.
    pub fn parse<I>(title: impl Into<String>, arguments: I) -> Result<Self, FrameworkError>
    where
        I: IntoIterator<Item = String>,
    {
        let mut backend = None;
        let mut width = NonZeroU32::new(1280).expect("literal is non-zero");
        let mut height = NonZeroU32::new(720).expect("literal is non-zero");
        let mut frames = None;
        let mut arguments = arguments.into_iter();

        while let Some(argument) = arguments.next() {
            match argument.as_str() {
                "--backend" => {
                    let value = arguments
                        .next()
                        .ok_or(FrameworkError::MissingValue("--backend"))?;
                    backend = Some(NativeBackend::parse(&value)?);
                }
                "--width" => width = parse_dimension("--width", arguments.next())?,
                "--height" => height = parse_dimension("--height", arguments.next())?,
                "--frames" => frames = Some(parse_dimension("--frames", arguments.next())?),
                "--help" | "-h" => return Err(FrameworkError::HelpRequested),
                _ => return Err(FrameworkError::UnknownArgument(argument)),
            }
        }

        let backend = backend.ok_or(FrameworkError::MissingValue("--backend"))?;
        if !backend.is_native_on_current_target() {
            return Err(FrameworkError::UnsupportedTarget(backend));
        }
        Ok(Self {
            backend,
            width,
            height,
            frames,
            title: title.into(),
        })
    }

    /// Usage text shared by every native RHI example binary.
    pub const fn usage() -> &'static str {
        "--backend <dx12|vulkan|gl4|gles3|metal> [--width PIXELS] [--height PIXELS] [--frames COUNT]"
    }

    fn window_config(&self) -> Result<WindowConfig, WindowError> {
        WindowConfig::new(self.title.clone(), self.width.get(), self.height.get())
    }
}

fn parse_dimension(
    flag: &'static str,
    value: Option<String>,
) -> Result<NonZeroU32, FrameworkError> {
    let value = value.ok_or(FrameworkError::MissingValue(flag))?;
    let value = value
        .parse::<u32>()
        .map_err(|_| FrameworkError::InvalidDimension {
            flag,
            value: value.clone(),
        })?;
    NonZeroU32::new(value).ok_or_else(|| FrameworkError::InvalidDimension {
        flag,
        value: value.to_string(),
    })
}

fn lifecycle_result(
    operation: &'static str,
    callback: Result<(), Box<dyn Error + Send + Sync>>,
    cleanup: Result<(), Box<dyn Error + Send + Sync>>,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    match (callback, cleanup) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(error), Ok(())) | (Ok(()), Err(error)) => Err(error),
        (Err(callback), Err(cleanup)) => Err(Box::new(LifecycleError {
            operation,
            callback,
            cleanup,
        })),
    }
}

#[derive(Debug)]
struct LifecycleError {
    operation: &'static str,
    callback: Box<dyn Error + Send + Sync>,
    cleanup: Box<dyn Error + Send + Sync>,
}

impl fmt::Display for LifecycleError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{} callback failed ({}) and its session cleanup also failed ({})",
            self.operation, self.callback, self.cleanup
        )
    }
}

impl Error for LifecycleError {}

/// Portable demo code implemented once for every native backend.
///
/// Individual examples should create buffers, textures, pipelines, recorder
/// work, submissions, and readback only through Fluxel RHI in these callbacks.
/// They must not receive a raw handle, WebGL object, Metal object, or Vulkan/
/// DX12 object.
pub trait Example {
    /// Texture uses required from every acquired presentation frame.
    ///
    /// Raster examples retain the default color-attachment configuration.
    /// A demo that reads an acquired frame back may add `COPY_SRC`; the native
    /// adapter retains this request when it reconfigures after a resize.
    fn presentation_usage(&self) -> fluxel_rhi::api::resource::TextureUsage {
        fluxel_rhi::api::resource::TextureUsage::COLOR_ATTACHMENT
    }

    /// Performs one-time portable RHI setup after the adapter opened a session.
    fn init(
        &mut self,
        context: &mut ExampleContext<'_>,
    ) -> Result<(), Box<dyn Error + Send + Sync>>;

    /// Updates demo state once for each redraw before [`Self::render`].
    fn update(
        &mut self,
        context: &mut ExampleContext<'_>,
    ) -> Result<(), Box<dyn Error + Send + Sync>>;

    /// Records, submits, and presents one frame through the portable RHI session.
    fn render(
        &mut self,
        context: &mut ExampleContext<'_>,
    ) -> Result<(), Box<dyn Error + Send + Sync>>;

    /// Reconfigures portable RHI resources for a non-zero drawable extent.
    fn resize(
        &mut self,
        context: &mut ExampleContext<'_>,
        width: u32,
        height: u32,
    ) -> Result<(), Box<dyn Error + Send + Sync>>;

    /// Drops device-owned state after a device loss.  A later resume creates a new session.
    fn device_lost(&mut self) -> Result<(), Box<dyn Error + Send + Sync>>;

    /// Releases portable RHI state before the native host exits.
    fn close(&mut self) -> Result<(), Box<dyn Error + Send + Sync>>;
}

/// Backend-neutral resources a demo may use during one lifecycle callback.
/// Platform adapters construct this view from their private session type.
pub struct ExampleContext<'a> {
    device: &'a fluxel_rhi::api::platform::Device,
    presentation: &'a mut fluxel_rhi::api::presentation::ConfiguredPresentation,
    extent: fluxel_rhi::api::presentation::Extent2d,
}

impl<'a> ExampleContext<'a> {
    pub fn new(
        device: &'a fluxel_rhi::api::platform::Device,
        presentation: &'a mut fluxel_rhi::api::presentation::ConfiguredPresentation,
        extent: fluxel_rhi::api::presentation::Extent2d,
    ) -> Self {
        Self {
            device,
            presentation,
            extent,
        }
    }

    pub fn device(&self) -> &'a fluxel_rhi::api::platform::Device {
        self.device
    }
    pub fn presentation_mut(
        &mut self,
    ) -> &mut fluxel_rhi::api::presentation::ConfiguredPresentation {
        self.presentation
    }
    pub fn extent(&self) -> fluxel_rhi::api::presentation::Extent2d {
        self.extent
    }
}

/// Backend and surface composition owned by the platform-specific framework layer.
///
/// Implementations select the requested RHI feature/provider and retain the
/// host window for the lifetime of the presentation target.  The associated
/// session is opaque to the framework and to a demo's CLI code.
pub trait NativePlatform {
    /// The adapter's portable RHI device/surface/session aggregate.
    type Session;

    /// Opens the requested provider/device and configures its first presentation target.
    fn open(
        &mut self,
        backend: NativeBackend,
        window: &HostWindow,
        width: u32,
        height: u32,
        presentation_usage: fluxel_rhi::api::resource::TextureUsage,
    ) -> Result<Self::Session, Box<dyn Error + Send + Sync>>;

    /// Reports device loss without exposing backend-native loss values.
    fn poll_device_loss(
        &mut self,
        session: &mut Self::Session,
    ) -> Result<bool, Box<dyn Error + Send + Sync>>;

    /// Reconfigures the presentation state for the host's current non-zero extent.
    ///
    /// This runs before [`Example::resize`] so the demo never creates
    /// extent-dependent resources against an obsolete presentation lease.
    fn resize_session(
        &mut self,
        session: &mut Self::Session,
        width: u32,
        height: u32,
    ) -> Result<(), Box<dyn Error + Send + Sync>>;

    /// Borrows the portable RHI view while keeping platform session details private.
    fn example_context<'a>(&self, session: &'a mut Self::Session) -> ExampleContext<'a>;

    /// Retires presentation state before `SurfaceDestroyed` or process exit.
    fn close_session(&mut self, session: Self::Session)
    -> Result<(), Box<dyn Error + Send + Sync>>;
}

/// A `fluxel-host` callback bridge for one CLI-selected native RHI example.
pub struct NativeExampleRunner<P: NativePlatform, D> {
    options: RunnerOptions,
    platform: P,
    demo: D,
    session: Option<P::Session>,
    closing: bool,
    rendered_frames: u32,
}

impl<P, D> NativeExampleRunner<P, D>
where
    P: NativePlatform,
    D: Example,
{
    /// Creates a runner.  Call `HostRuntime::run(runner)` from the target's
    /// normal platform entry point; Android uses `from_android_app` first.
    pub fn new(options: RunnerOptions, platform: P, demo: D) -> Self {
        Self {
            options,
            platform,
            demo,
            session: None,
            closing: false,
            rendered_frames: 0,
        }
    }

    fn fail(host: &HostContext<'_>, error: impl fmt::Display) {
        eprintln!("Fluxel RHI example failed: {error}");
        host.exit();
    }

    fn close_session(&mut self) -> Result<(), Box<dyn Error + Send + Sync>> {
        if let Some(session) = self.session.take() {
            self.platform.close_session(session)?;
        }
        Ok(())
    }

    fn after_device_loss(&mut self) -> Result<(), Box<dyn Error + Send + Sync>> {
        let Some(session) = self.session.take() else {
            return Ok(());
        };
        // Demo-owned views, pipelines, and resource handles must be released
        // while their device is still alive.  The platform adapter then retires
        // presentation and the device itself.
        let callback = self.demo.device_lost();
        let cleanup = self.platform.close_session(session);
        lifecycle_result("device-loss", callback, cleanup)
    }

    fn close_once(&mut self) -> Result<(), Box<dyn Error + Send + Sync>> {
        if !self.closing {
            self.closing = true;
            // `close` is the final demo-owned teardown and must precede device
            // and presentation retirement.  Always attempt both steps, even
            // when the callback itself reports an error.
            let callback = self.demo.close();
            let cleanup = self.close_session();
            lifecycle_result("close", callback, cleanup)?;
        }
        Ok(())
    }

    fn open_session(&mut self, host: &HostContext<'_>) -> Result<(), Box<dyn Error + Send + Sync>> {
        if self.closing || self.session.is_some() {
            return Ok(());
        }
        let window = host.window().ok_or(FrameworkError::SurfaceWithoutWindow)?;
        let presentation_usage = self.demo.presentation_usage();
        let mut session = self.platform.open(
            self.options.backend,
            window,
            self.options.width.get(),
            self.options.height.get(),
            presentation_usage,
        )?;
        let init_result = {
            let mut context = self.platform.example_context(&mut session);
            self.demo.init(&mut context)
        };
        if let Err(init) = init_result {
            // `init` may have created device-owned demo state before failing.
            // The newly opened session has not entered `self` yet, so retire it
            // locally instead of leaving a target registration alive.
            let demo_cleanup = self.demo.device_lost();
            let session_cleanup = self.platform.close_session(session);
            let cleanup = lifecycle_result("init cleanup", demo_cleanup, session_cleanup);
            return lifecycle_result("init", Err(init), cleanup);
        }
        self.session = Some(session);
        window.request_redraw();
        Ok(())
    }

    fn on_redraw(&mut self, host: &HostContext<'_>) -> Result<(), Box<dyn Error + Send + Sync>> {
        #[cfg(target_os = "android")]
        eprintln!("FLUXEL_FRAME_TICK");
        let lost = {
            let Some(session) = self.session.as_mut() else {
                return Ok(());
            };
            self.platform.poll_device_loss(session)?
        };
        if lost {
            self.after_device_loss()?;
            self.open_session(host)?;
            return Ok(());
        }
        #[cfg(target_os = "android")]
        eprintln!("FLUXEL_AFTER_POLL");
        let session = self
            .session
            .as_mut()
            .expect("loss handling preserves a live session or returns");
        {
            let mut context = self.platform.example_context(session);
            self.demo.update(&mut context)?;
            #[cfg(target_os = "android")]
            eprintln!("FLUXEL_BEFORE_RENDER");
            self.demo.render(&mut context)?;
        }
        #[cfg(target_os = "android")]
        eprintln!("FLUXEL_AFTER_RENDER");
        self.rendered_frames = self.rendered_frames.saturating_add(1);
        if self
            .options
            .frames
            .is_some_and(|limit| self.rendered_frames >= limit.get())
        {
            host.exit();
            return Ok(());
        }
        if let Some(window) = host.window() {
            window.request_redraw();
        }
        #[cfg(target_os = "android")]
        eprintln!("FLUXEL_AFTER_REQUEST_REDRAW");
        Ok(())
    }
}

impl<P, D> HostApplication for NativeExampleRunner<P, D>
where
    P: NativePlatform + 'static,
    D: Example + 'static,
{
    fn resumed(&mut self, host: &mut HostContext<'_>) {
        if self.closing {
            return;
        }
        // `fluxel-host` guarantees that its runner follows this callback with
        // SurfaceCreated for a live drawable.  Opening the RHI presentation
        // target there makes the native-drawable interval explicit.
        let result = self
            .options
            .window_config()
            .and_then(|config| host.create_window(config).map(|_| ()));
        if let Err(error) = result {
            Self::fail(host, error);
        }
    }

    fn window_event(&mut self, host: &mut HostContext<'_>, event: WindowEvent) {
        let result = match event {
            WindowEvent::Resized { width, height } | WindowEvent::Restored { width, height } => {
                if width == 0 || height == 0 {
                    Ok(())
                } else if let Some(session) = self.session.as_mut() {
                    self.platform
                        .resize_session(session, width, height)
                        .and_then(|()| {
                            let mut context = self.platform.example_context(session);
                            self.demo.resize(&mut context, width, height)
                        })
                } else {
                    Ok(())
                }
            }
            WindowEvent::SurfaceCreated => self.open_session(host),
            WindowEvent::RedrawRequested => self.on_redraw(host),
            WindowEvent::SurfaceDestroyed => self.after_device_loss(),
            WindowEvent::CloseRequested => {
                let result = self.close_once();
                host.exit();
                result
            }
            WindowEvent::Minimized | WindowEvent::Suspended | WindowEvent::Resumed => Ok(()),
            _ => Ok(()),
        };
        if let Err(error) = result {
            Self::fail(host, error);
        }
    }
}

/// CLI or lifecycle errors that can be reported before backend-specific work begins.
#[derive(Debug)]
pub enum FrameworkError {
    /// No value followed a known option.
    MissingValue(&'static str),
    /// A pixel dimension was zero or not an unsigned integer.
    InvalidDimension { flag: &'static str, value: String },
    /// The backend spelling is not part of the native runner contract.
    UnknownBackend(String),
    /// Browser backends require the `fluxel-jsbridge` browser runner.
    BrowserBackend(String),
    /// The selected native backend has no host/platform implementation here.
    UnsupportedTarget(NativeBackend),
    /// The host broke the callback ordering required for presentation setup.
    SurfaceWithoutWindow,
    /// An option is not part of the common runner contract.
    UnknownArgument(String),
    /// The caller asked to print [`RunnerOptions::usage`].
    HelpRequested,
}

impl fmt::Display for FrameworkError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingValue(flag) => write!(formatter, "{flag} requires a value"),
            Self::InvalidDimension { flag, value } => {
                write!(
                    formatter,
                    "{flag} requires a non-zero pixel count, got {value:?}"
                )
            }
            Self::UnknownBackend(value) => write!(
                formatter,
                "unknown backend {value:?}; use {}",
                RunnerOptions::usage()
            ),
            Self::BrowserBackend(value) => write!(
                formatter,
                "{value} is a browser backend; run it through fluxel-jsbridge's browser host adapter"
            ),
            Self::UnsupportedTarget(backend) => write!(
                formatter,
                "backend `{}` is unavailable on this native target",
                backend.as_str()
            ),
            Self::SurfaceWithoutWindow => {
                formatter.write_str("host reported SurfaceCreated without a live HostWindow")
            }
            Self::UnknownArgument(value) => write!(
                formatter,
                "unknown argument {value:?}; use {}",
                RunnerOptions::usage()
            ),
            Self::HelpRequested => formatter.write_str(RunnerOptions::usage()),
        }
    }
}

impl Error for FrameworkError {}
