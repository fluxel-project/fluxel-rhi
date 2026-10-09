//! Windows implementation of the native example platform seam.
//!
//! This module selects a public Fluxel RHI provider for a Win32 window.  It
//! supports the enabled DX12, Vulkan, and WGL OpenGL adapters without exposing
//! an HWND or a backend-native object to an example.
//!
//! The registration guard remains in [`WindowsSession`] until after the
//! configured presentation lease and device are released.  That ordering keeps
//! the backend-private Win32 target valid for every object that may still refer
//! to it.

#![cfg(all(
    windows,
    any(feature = "dx12", feature = "vulkan", feature = "native-gl-wgl")
))]

use std::{
    error::Error,
    fmt,
    future::Future,
    pin::pin,
    sync::{Arc, Mutex},
    task::{Context, Poll, Wake, Waker},
};

use fluxel_host::HostWindow;
use fluxel_rhi::api::{
    platform::{
        AdapterSelection, Device, DeviceRequestDescriptor, DeviceRequirements, DeviceStatus,
        PlatformProvider, PresentationTargetRegistration,
    },
    presentation::{
        ConfiguredPresentation, Extent2d, PresentationConfiguration, PresentationExtent,
        PresentationExtentControl, PresentationTarget,
    },
    resource::TextureUsage,
};
#[cfg(feature = "dx12")]
use fluxel_rhi::create_dx12_provider;
#[cfg(feature = "vulkan")]
use fluxel_rhi::create_vulkan_provider;
#[cfg(feature = "native-gl-wgl")]
use fluxel_rhi::{WglProvider, create_wgl_provider};

use crate::common::{
    Example, ExampleContext, NativeBackend, NativeExampleRunner, NativePlatform, RunnerOptions,
};

pub fn run_example<D: Example + 'static>(title: &str, demo: D) -> Result<(), Box<dyn Error>> {
    let mut options = RunnerOptions::parse(title, std::env::args().skip(1))?;
    options.title = format!("{}: {title}", options.backend.as_str());
    let failure = Arc::new(Mutex::new(None));
    fluxel_host::HostRuntime::new()?.run(
        NativeExampleRunner::new(options, WindowsPlatform, demo)
            .with_failure_slot(Arc::clone(&failure)),
    )?;
    let failure = failure
        .lock()
        .map_err(|_| std::io::Error::other("example failure state lock was poisoned"))?
        .take();
    if let Some(failure) = failure {
        return Err(std::io::Error::other(failure).into());
    }
    Ok(())
}

/// Opens a session for a live Windows host window.
#[derive(Default)]
pub struct WindowsPlatform;

/// Portable RHI state that a Windows example receives in its callbacks.
///
/// The provider and registration are retained privately to preserve the target
/// lifetime.  Demo code can use the device and configured presentation lease,
/// but cannot observe a Win32 or backend-native object.
pub struct WindowsSession {
    provider: Option<WindowsProvider>,
    registration: Option<PresentationTargetRegistration>,
    device: Option<Device>,
    presentation: Option<ConfiguredPresentation>,
    format: fluxel_rhi::api::format::TextureFormat,
    extent_control: PresentationExtentControl,
    extent: Extent2d,
    presentation_usage: TextureUsage,
}

impl WindowsSession {
    /// The device used for portable resource and pipeline creation.
    pub fn device(&self) -> &Device {
        self.device
            .as_ref()
            .expect("a live WindowsSession always has a device")
    }

    /// The configured presentation lease used to acquire the next frame.
    pub fn presentation_mut(&mut self) -> &mut ConfiguredPresentation {
        self.presentation
            .as_mut()
            .expect("a live WindowsSession always has presentation")
    }

    /// The current host drawable extent used by this session.
    ///
    /// Demos use this when allocating size-dependent depth, MSAA, or
    /// offscreen textures during `init` and after `resize`.
    pub fn extent(&self) -> Extent2d {
        self.extent
    }

    fn presentation_configuration(&self, width: u32, height: u32) -> PresentationConfiguration {
        PresentationConfiguration::new(self.format)
            .with_extent(presentation_extent(self.extent_control, width, height))
            .with_usage(self.presentation_usage)
    }

    fn close(mut self) -> Result<(), Box<dyn Error + Send + Sync>> {
        // A configured lease is released before the device and before the
        // registration guard; this lets the backend destroy its swapchain while
        // the native target it was created for is still registered.
        drop(self.presentation.take());

        // Waiting is a shutdown operation only.  It is never part of the frame
        // loop; the framework calls it only after no new work can be recorded.
        // A lost device may refuse this call, but all later teardown still runs.
        let idle_result = self
            .device
            .as_ref()
            .map(Device::wait_idle_blocking)
            .transpose();
        drop(self.device.take());

        // Dropping the guard retires the backend-private target.  It must follow
        // both surface/device destruction because either may still use it.
        drop(self.registration.take());
        drop(self.provider.take());

        idle_result
            .map(|_| ())
            .map_err(|error| Box::new(error) as Box<dyn Error + Send + Sync>)
    }
}

/// The provider owner retained by a Windows session.
///
/// DX12 and Vulkan acquire a target through the shared registration API. WGL
/// owns its context and target together, so it must retain the WGL facade until
/// the configured presentation lease and all device work have been retired.
enum WindowsProvider {
    Registered(PlatformProvider),
    #[cfg(feature = "native-gl-wgl")]
    Wgl(WglProvider),
}

impl WindowsProvider {
    fn provider(&self) -> &PlatformProvider {
        match self {
            Self::Registered(provider) => provider,
            #[cfg(feature = "native-gl-wgl")]
            Self::Wgl(provider) => provider.provider(),
        }
    }

    fn resize(&self, width: u32, height: u32) -> Result<(), Box<dyn Error + Send + Sync>> {
        match self {
            Self::Registered(_) => Ok(()),
            #[cfg(feature = "native-gl-wgl")]
            Self::Wgl(provider) => {
                provider.resize([width, height])?;
                Ok(())
            }
        }
    }
}

impl NativePlatform for WindowsPlatform {
    type Session = WindowsSession;

    fn open(
        &mut self,
        backend: NativeBackend,
        window: &HostWindow,
        width: u32,
        height: u32,
        presentation_usage: TextureUsage,
    ) -> Result<Self::Session, Box<dyn Error + Send + Sync>> {
        let (provider, registration, target) = open_provider(backend, window, width, height)?;
        #[cfg(feature = "native-gl-wgl")]
        if let WindowsProvider::Wgl(wgl) = &provider {
            use crate::common::shader::{ExampleShaderTarget, set_gl_target};
            let shader_target = match wgl.shader_target() {
                fluxel_rhi_gl::GlShaderTarget::Desktop { version } => {
                    ExampleShaderTarget::Glsl { version }
                }
                fluxel_rhi_gl::GlShaderTarget::Embedded { version } => {
                    ExampleShaderTarget::GlslEs { version }
                }
                fluxel_rhi_gl::GlShaderTarget::WebGl2 => {
                    ExampleShaderTarget::GlslEs { version: 300 }
                }
            };
            set_gl_target(shader_target);
        }

        // The target is part of the device request, rather than a late surface
        // preflight, so Vulkan selects a queue family that can actually present.
        let request =
            DeviceRequestDescriptor::new(AdapterSelection::Default, DeviceRequirements::new())
                .require_presentation_target(target.clone());
        let device = block_on(provider.provider().request_device(request))?;

        let capabilities = device.presentation_capabilities(&target)?;
        let format = *capabilities
            .formats()
            .first()
            .ok_or(AdapterError::NoPresentationFormat)?;
        let extent_control = capabilities.extent_control();
        let extent = match extent_control {
            PresentationExtentControl::HostManaged {
                current: Some(current),
            } => current,
            _ => Extent2d { width, height },
        };
        let configuration = PresentationConfiguration::new(format)
            .with_extent(presentation_extent(
                extent_control,
                extent.width,
                extent.height,
            ))
            .with_usage(presentation_usage);
        let presentation = block_on(device.configure_presentation(&target, &configuration))?;

        Ok(WindowsSession {
            provider: Some(provider),
            registration,
            device: Some(device),
            presentation: Some(presentation),
            format,
            extent_control,
            extent,
            presentation_usage,
        })
    }

    fn poll_device_loss(
        &mut self,
        session: &mut Self::Session,
    ) -> Result<bool, Box<dyn Error + Send + Sync>> {
        let device = session.device();
        if device.status() == DeviceStatus::Lost {
            return Ok(true);
        }

        if let Err(error) = device.poll() {
            // A native poll may be the call that observes loss.  Re-read the
            // authoritative device status so it reaches the demo's device-loss
            // callback rather than being reported as an unrelated runner fault.
            if device.status() == DeviceStatus::Lost {
                return Ok(true);
            }
            return Err(error.into());
        }
        Ok(device.status() == DeviceStatus::Lost)
    }

    fn resize_session(
        &mut self,
        session: &mut Self::Session,
        width: u32,
        height: u32,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        session
            .provider
            .as_ref()
            .expect("a live WindowsSession always has a provider")
            .resize(width, height)?;
        let configuration = session.presentation_configuration(width, height);
        block_on(session.presentation_mut().reconfigure(&configuration))?;
        session.extent = Extent2d { width, height };
        Ok(())
    }

    fn example_context<'a>(&self, session: &'a mut Self::Session) -> ExampleContext<'a> {
        let extent = session.extent;
        let device = session
            .device
            .as_ref()
            .expect("a live WindowsSession always has a device");
        let presentation = session
            .presentation
            .as_mut()
            .expect("a live WindowsSession always has presentation");
        ExampleContext::new(device, presentation, extent)
    }

    fn close_session(
        &mut self,
        session: Self::Session,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        session.close()
    }
}

fn presentation_extent(
    extent_control: PresentationExtentControl,
    width: u32,
    height: u32,
) -> PresentationExtent {
    match extent_control {
        PresentationExtentControl::Configurable { .. } => {
            PresentationExtent::Exact(Extent2d { width, height })
        }
        // A host-managed target receives a reconfigure on resize but keeps its
        // host-owned extent policy; passing Exact there would be invalid.
        PresentationExtentControl::HostManaged { .. } => PresentationExtent::HostManaged,
        _ => PresentationExtent::HostManaged,
    }
}

/// Adapter construction/configuration failures that are independent of a demo.
#[derive(Debug)]
enum AdapterError {
    WrongBackend(NativeBackend),
    NoPresentationFormat,
}

impl fmt::Display for AdapterError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongBackend(backend) => write!(
                formatter,
                "the requested Windows backend `{}` is not enabled in this build",
                backend.as_str()
            ),
            Self::NoPresentationFormat => {
                formatter.write_str("the presentation target reported no usable format")
            }
        }
    }
}

impl Error for AdapterError {}

/// Selects the public provider constructor for the requested Windows backend.
///
/// Keeping this dispatch in the host adapter makes the examples themselves
/// backend-neutral: their only input remains a [`Device`] and a configured
/// portable presentation lease.
fn open_provider(
    backend: NativeBackend,
    window: &HostWindow,
    width: u32,
    height: u32,
) -> Result<
    (
        WindowsProvider,
        Option<PresentationTargetRegistration>,
        PresentationTarget,
    ),
    Box<dyn Error + Send + Sync>,
> {
    #[cfg(feature = "native-gl-wgl")]
    if backend == NativeBackend::Gl4 {
        let wgl = create_wgl_provider(window, [width, height])?;
        let target = wgl.presentation_target().clone();
        return Ok((WindowsProvider::Wgl(wgl), None, target));
    }

    #[cfg(not(any(feature = "dx12", feature = "vulkan")))]
    {
        let _ = window;
        return Err(AdapterError::WrongBackend(backend).into());
    }

    #[cfg(any(feature = "dx12", feature = "vulkan"))]
    {
        let provider = match backend {
            #[cfg(feature = "dx12")]
            NativeBackend::Dx12 => create_dx12_provider()?,
            #[cfg(feature = "vulkan")]
            NativeBackend::Vulkan => create_vulkan_provider()?,
            #[cfg(not(feature = "dx12"))]
            NativeBackend::Dx12 => return Err(AdapterError::WrongBackend(backend).into()),
            #[cfg(not(feature = "vulkan"))]
            NativeBackend::Vulkan => return Err(AdapterError::WrongBackend(backend).into()),
            NativeBackend::Gl4 | NativeBackend::Gles3 | NativeBackend::Metal => {
                return Err(AdapterError::WrongBackend(backend).into());
            }
        };

        // This public call retains every Win32 detail below the RHI boundary and
        // returns only an opaque target plus its lifetime guard.
        let registration = provider.register_presentation_target(window)?;
        let target = registration.target().clone();
        Ok((
            WindowsProvider::Registered(provider),
            Some(registration),
            target,
        ))
    }
}

/// Compatibility names for callers that used the original Vulkan-only adapter.
#[cfg(feature = "vulkan")]
pub type WindowsVulkanPlatform = WindowsPlatform;
#[cfg(feature = "vulkan")]
pub type WindowsVulkanSession = WindowsSession;

/// Runs the small async public-RHI operations during native startup.
///
/// Device creation/configuration is driven before the host render loop starts.
/// The waker blocks the current startup thread only while a provider
/// request is pending; normal frame execution remains event driven.
fn block_on<F: Future>(future: F) -> F::Output {
    struct ThreadWaker(std::thread::Thread);

    impl Wake for ThreadWaker {
        fn wake(self: Arc<Self>) {
            self.0.unpark();
        }

        fn wake_by_ref(self: &Arc<Self>) {
            self.0.unpark();
        }
    }

    let waker = Waker::from(Arc::new(ThreadWaker(std::thread::current())));
    let mut context = Context::from_waker(&waker);
    let mut future = pin!(future);
    loop {
        match future.as_mut().poll(&mut context) {
            Poll::Ready(value) => return value,
            Poll::Pending => std::thread::park(),
        }
    }
}
