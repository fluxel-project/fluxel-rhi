//! Browser platform adapters shared by the WebGPU and WebGL2 example runners.
//!
//! The browser owns canvas, animation-frame, visibility, resize, and context/device
//! loss events through `fluxel-jsbridge`. RHI resource and submission ownership
//! stays in the backend; browser-native objects remain private to this module.

#![cfg(target_arch = "wasm32")]

use fluxel_rhi::api::error::{RhiError, RhiErrorKind, RhiResult};
use fluxel_rhi::api::platform::{
    AdapterSelection, Device, DeviceRequestDescriptor, DeviceRequirements,
};
use fluxel_rhi::api::presentation::{
    ConfiguredPresentation, Extent2d, PresentationConfiguration, PresentationExtent,
    PresentationTarget,
};
use fluxel_rhi::api::resource::TextureUsage;
use web_sys::HtmlCanvasElement;

/// A browser-owned WebGPU device and its configured canvas presentation lease.
///
/// This is deliberately the smallest browser host surface used by the examples:
/// the canvas remains with the browser, while all rendering continues through
/// the portable RHI [`Device`].
pub struct WebGpuExampleSession {
    device: Device,
    presentation: ConfiguredPresentation,
    target: PresentationTarget,
    extent: Extent2d,
}

impl WebGpuExampleSession {
    /// Opens the portable device used by the example.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Borrows the configured presentation lease.
    pub fn presentation_mut(&mut self) -> &mut ConfiguredPresentation {
        &mut self.presentation
    }

    /// Current canvas extent in physical pixels.
    pub const fn extent(&self) -> Extent2d {
        self.extent
    }

    /// Opaque presentation identity retained by this browser session.
    pub fn target(&self) -> &PresentationTarget {
        &self.target
    }
}

/// Opens a WebGPU device and configures `canvas` as a color-attachment target.
///
/// The browser backend registers a canvas only after its `GPUDevice` promise
/// resolves, so this intentionally requests the device first and then creates
/// the portable presentation lease.
#[cfg(feature = "webgpu")]
pub async fn open_webgpu_session(
    canvas: HtmlCanvasElement,
    presentation_usage: TextureUsage,
) -> RhiResult<WebGpuExampleSession> {
    let provider = fluxel_rhi::create_webgpu_provider()?;
    let device = provider
        .request_device(DeviceRequestDescriptor::new(
            AdapterSelection::Default,
            DeviceRequirements::new(),
        ))
        .await?;
    let target = fluxel_rhi_webgpu::register_canvas(&device, canvas)?;
    let capabilities = device.presentation_capabilities(&target)?;
    let format = *capabilities.formats().first().ok_or_else(|| {
        RhiError::new(
            RhiErrorKind::Unsupported,
            "the browser WebGPU canvas exposes no presentation format",
        )
        .at("examples::web::open_webgpu_session")
    })?;
    let extent = match capabilities.extent_control() {
        fluxel_rhi::api::presentation::PresentationExtentControl::HostManaged {
            current: Some(extent),
        } => extent,
        _ => {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "the browser WebGPU canvas did not report a drawable extent",
            )
            .at("examples::web::open_webgpu_session"));
        }
    };
    let configuration = PresentationConfiguration::new(format)
        .with_extent(PresentationExtent::HostManaged)
        .with_usage(presentation_usage);
    let presentation = device
        .configure_presentation(&target, &configuration)
        .await?;
    Ok(WebGpuExampleSession {
        device,
        presentation,
        target,
        extent,
    })
}
