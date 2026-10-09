//! macOS implementation of the native example platform seam.
//!
//! `fluxel-host` owns the AppKit window and this adapter registers its opaque
//! raw-window handle with the Metal provider. The provider privately creates
//! and retains the `CAMetalLayer`; examples receive only portable RHI objects.

#![cfg(target_os = "macos")]

use std::{
    error::Error,
    fmt,
    sync::{Arc, Mutex},
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
        resource::TextureUsage,
    },
    create_metal_provider,
};

use crate::common::{
    Example, ExampleContext, NativeBackend, NativeExampleRunner, NativePlatform, RunnerOptions,
    block_on,
};

/// Starts a macOS example using the shared event-driven host runner.
pub fn run_example<D: Example + 'static>(title: &str, demo: D) -> Result<(), Box<dyn Error>> {
    let mut options = RunnerOptions::parse(title, std::env::args().skip(1))?;
    options.title = format!("{}: {title}", options.backend.as_str());
    let failure = Arc::new(Mutex::new(None));
    fluxel_host::HostRuntime::new()?.run(
        NativeExampleRunner::new(options, ApplePlatform, demo)
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

/// Opens a portable Metal session for a live AppKit host window.
#[derive(Default)]
pub struct ApplePlatform;

/// Private host composition retained for one macOS example session.
pub struct AppleSession {
    provider: Option<PlatformProvider>,
    registration: Option<PresentationTargetRegistration>,
    device: Option<Device>,
    presentation: Option<ConfiguredPresentation>,
    format: fluxel_rhi::api::format::TextureFormat,
    extent_control: PresentationExtentControl,
    extent: Extent2d,
    presentation_usage: TextureUsage,
}

impl AppleSession {
    fn presentation_configuration(&self) -> PresentationConfiguration {
        PresentationConfiguration::new(self.format)
            .with_extent(match self.extent_control {
                PresentationExtentControl::Configurable { .. } => {
                    PresentationExtent::Exact(self.extent)
                }
                PresentationExtentControl::HostManaged { .. } => PresentationExtent::HostManaged,
                _ => PresentationExtent::HostManaged,
            })
            .with_usage(self.presentation_usage)
    }

    fn close(mut self) -> Result<(), Box<dyn Error + Send + Sync>> {
        drop(self.presentation.take());
        let idle_result = self
            .device
            .as_ref()
            .map(Device::wait_idle_blocking)
            .transpose();
        drop(self.device.take());
        drop(self.registration.take());
        drop(self.provider.take());
        idle_result
            .map(|_| ())
            .map_err(|error| Box::new(error) as Box<dyn Error + Send + Sync>)
    }
}

impl NativePlatform for ApplePlatform {
    type Session = AppleSession;

    fn open(
        &mut self,
        backend: NativeBackend,
        window: &HostWindow,
        width: u32,
        height: u32,
        presentation_usage: TextureUsage,
    ) -> Result<Self::Session, Box<dyn Error + Send + Sync>> {
        if backend != NativeBackend::Metal {
            return Err(AdapterError::WrongBackend(backend).into());
        }
        let provider = create_metal_provider()?;
        // This call privately turns the AppKit view into a retained CAMetalLayer
        // and returns an opaque target plus its lifetime guard.
        let registration = provider.register_presentation_target(window)?;
        let target = registration.target().clone();
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
        let configuration = PresentationConfiguration::new(format)
            .with_extent(PresentationExtent::HostManaged)
            .with_usage(presentation_usage);
        let presentation = block_on(device.configure_presentation(&target, &configuration))?;
        Ok(AppleSession {
            provider: Some(provider),
            registration: Some(registration),
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
        let device = session
            .device
            .as_ref()
            .expect("live AppleSession has a device");
        if device.status() == DeviceStatus::Lost {
            return Ok(true);
        }
        if let Err(error) = device.poll() {
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
        session.extent = Extent2d { width, height };
        let configuration = session.presentation_configuration();
        block_on(
            session
                .presentation
                .as_mut()
                .expect("live AppleSession has presentation")
                .reconfigure(&configuration),
        )?;
        Ok(())
    }

    fn example_context<'a>(&self, session: &'a mut Self::Session) -> ExampleContext<'a> {
        let extent = session.extent;
        ExampleContext::new(
            session
                .device
                .as_ref()
                .expect("live AppleSession has a device"),
            session
                .presentation
                .as_mut()
                .expect("live AppleSession has presentation"),
            extent,
        )
    }

    fn close_session(
        &mut self,
        session: Self::Session,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        session.close()
    }
}

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
                "the requested macOS backend `{}` is not enabled in this build",
                backend.as_str()
            ),
            Self::NoPresentationFormat => {
                formatter.write_str("the presentation target reported no usable format")
            }
        }
    }
}

impl Error for AdapterError {}
