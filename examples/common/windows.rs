//! Windows Vulkan implementation of the native example platform seam.
//!
//! This module implements exactly one adapter: a Win32 window presented through
//! Fluxel RHI's public Vulkan provider composition API.  It does not provide a
//! DX12, GL, Android, Metal, or browser adapter.
//!
//! The registration guard remains in [`WindowsVulkanSession`] until after the
//! configured presentation lease and device are released.  That ordering keeps
//! the backend-private Win32 target valid for every object that may still refer
//! to it.

#![cfg(all(windows, feature = "vulkan"))]

use std::{
    error::Error,
    fmt,
    future::Future,
    pin::pin,
    sync::Arc,
    task::{Context, Poll, Wake, Waker},
};

use fluxel_host::HostWindow;
use fluxel_rhi::{
    api::{
        platform::{
            AdapterSelection, Device, DeviceRequestDescriptor, DeviceRequirements, DeviceStatus,
            PlatformProvider, PresentationTargetRegistration,
        },
        presentation::{
            ConfiguredPresentation, Extent2d, PresentationConfiguration, PresentationExtent,
            PresentationExtentControl,
        },
    },
    create_vulkan_provider,
};

use crate::common::{
    Example, ExampleContext, NativeBackend, NativeExampleRunner, NativePlatform, RunnerOptions,
};

pub fn run_example<D: Example + 'static>(title: &str, demo: D) -> Result<(), Box<dyn Error>> {
    let mut options = RunnerOptions::parse(title, std::env::args().skip(1))?;
    options.title = format!("{}: {title}", options.backend.as_str());
    if options.backend != NativeBackend::Vulkan {
        return Err("Windows Vulkan adapter cannot run the requested backend".into());
    }
    fluxel_host::HostRuntime::new()?.run(NativeExampleRunner::new(
        options,
        WindowsVulkanPlatform,
        demo,
    ))?;
    Ok(())
}

/// Opens a Vulkan session for a live Windows host window.
#[derive(Default)]
pub struct WindowsVulkanPlatform;

/// Portable RHI state that a Windows Vulkan example receives in its callbacks.
///
/// The provider and registration are retained privately to preserve the target
/// lifetime.  Demo code can use the device and configured presentation lease,
/// but cannot observe a Win32 or Vulkan-native object.
pub struct WindowsVulkanSession {
    provider: Option<PlatformProvider>,
    registration: Option<PresentationTargetRegistration>,
    device: Option<Device>,
    presentation: Option<ConfiguredPresentation>,
    format: fluxel_rhi::api::format::TextureFormat,
    extent_control: PresentationExtentControl,
    extent: Extent2d,
}

impl WindowsVulkanSession {
    /// The device used for portable resource and pipeline creation.
    pub fn device(&self) -> &Device {
        self.device
            .as_ref()
            .expect("a live WindowsVulkanSession always has a device")
    }

    /// The configured presentation lease used to acquire the next frame.
    pub fn presentation_mut(&mut self) -> &mut ConfiguredPresentation {
        self.presentation
            .as_mut()
            .expect("a live WindowsVulkanSession always has presentation")
    }

    /// The current host drawable extent used by this session.
    ///
    /// Demos use this when allocating size-dependent depth, MSAA, or
    /// offscreen textures during `init` and after `resize`.
    pub fn extent(&self) -> Extent2d {
        self.extent
    }

    fn presentation_configuration(&self, width: u32, height: u32) -> PresentationConfiguration {
        PresentationConfiguration::new(self.format).with_extent(presentation_extent(
            self.extent_control,
            width,
            height,
        ))
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

impl NativePlatform for WindowsVulkanPlatform {
    type Session = WindowsVulkanSession;

    fn open(
        &mut self,
        backend: NativeBackend,
        window: &HostWindow,
        width: u32,
        height: u32,
    ) -> Result<Self::Session, Box<dyn Error + Send + Sync>> {
        if backend != NativeBackend::Vulkan {
            return Err(AdapterError::WrongBackend(backend).into());
        }

        let provider = create_vulkan_provider()?;

        // This public call retains every Win32 detail below the RHI boundary and
        // returns only an opaque target plus its lifetime guard.
        let registration = provider.register_presentation_target(window)?;
        let target = registration.target().clone();

        // The target is part of the device request, rather than a late surface
        // preflight, so Vulkan selects a queue family that can actually present.
        let request =
            DeviceRequestDescriptor::new(AdapterSelection::Default, DeviceRequirements::new())
                .require_presentation_target(target.clone());
        let device = block_on(provider.request_device(request))?;

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
        let configuration = PresentationConfiguration::new(format).with_extent(
            presentation_extent(extent_control, extent.width, extent.height),
        );
        let presentation = block_on(device.configure_presentation(&target, &configuration))?;

        Ok(WindowsVulkanSession {
            provider: Some(provider),
            registration: Some(registration),
            device: Some(device),
            presentation: Some(presentation),
            format,
            extent_control,
            extent,
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
            .expect("a live WindowsVulkanSession always has a device");
        let presentation = session
            .presentation
            .as_mut()
            .expect("a live WindowsVulkanSession always has presentation");
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
                "the Windows Vulkan adapter cannot open backend `{}`",
                backend.as_str()
            ),
            Self::NoPresentationFormat => {
                formatter.write_str("the Vulkan presentation target reported no usable format")
            }
        }
    }
}

impl Error for AdapterError {}

/// Runs the small async public-RHI operations during native startup.
///
/// Vulkan device creation/configuration is driven before the host render loop
/// starts.  The waker blocks the current startup thread only while a provider
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
