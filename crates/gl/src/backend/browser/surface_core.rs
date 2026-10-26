//! Static Core surface SPI for the browser-owned WebGL2 default framebuffer.
//!
//! The browser module owns the generation and lease state. This file only
//! connects those facts to Core; it does not create a swapchain, a scheduler,
//! or a second presentation path.

use std::future::{Ready, ready};

use fluxel_rhi_core::api::{self, Result, Support};

use super::{BrowserSurface, BrowserSurfaceImage};
use crate::backend::rhi::Gl;

impl Gl {
    /// Associates this WebGL2 device's canvas default framebuffer with Core.
    ///
    /// The `BrowserOwner` retained by the device is the context-lifetime
    /// authority. The resulting surface contains no DOM object in Core's
    /// public model and remains local to the browser thread.
    pub fn surface(device: &api::Device<Self>) -> Result<api::Surface<Self>> {
        // SAFETY: `device` is statically a Core GL device. Cloning its native
        // carrier keeps the same browser owner alive for every surface lease.
        let native = unsafe { device.native_ref() }.clone();
        let surface = BrowserSurface::new(native.0.owner.clone());
        // SAFETY: this new browser surface is bound to exactly `device`; no
        // lease exists before Core publishes the returned Surface wrapper.
        Ok(unsafe { api::Surface::from_native(device, api::SurfaceId::allocate()?, surface) })
    }
}

// SAFETY: BrowserSurface serializes reconfigure, acquisition, handoff, and
// Drop through one Rc<RefCell> state. BrowserSurfaceImage retains its exact
// generation and releases only that generation's exclusive lease.

impl BrowserSurfaceImage {
    /// Describes this live canvas default framebuffer for raster validation.
    /// WebGL exposes framebuffer zero rather than an allocatable texture, so
    /// this returns immutable frame identity instead of a GL object handle.
    pub(crate) fn default_target(
        &self,
    ) -> Result<crate::backend::api::GlDefaultFramebufferTarget> {
        if !self.is_current_for(self.generation) {
            return Err(api::Error::SurfaceOutdated);
        }
        let state = self.surface.state.borrow();
        let extent = state.extent;
        let color_format = match state.format {
            api::TextureFormat::Rgba8Unorm => crate::backend::api::GlFormat::Rgba8Unorm,
            api::TextureFormat::Rgba8UnormSrgb => crate::backend::api::GlFormat::Rgba8Srgb,
            _ => return Err(api::Error::SurfaceOutdated),
        };
        Ok(crate::backend::api::GlDefaultFramebufferTarget {
            frame_serial: self.serial,
            context: self.surface.owner.context_stamp(),
            width: extent[0],
            height: extent[1],
            sample_count: 1,
            color_format,
        })
    }
}

unsafe impl fluxel_rhi_core::backend::NativeImageLease for BrowserSurfaceImage {
    fn try_handoff(&mut self, generation: u64) -> Result<()> {
        BrowserSurfaceImage::try_handoff(self, generation)
    }

    fn is_current(&self, generation: u64) -> bool {
        self.is_current_for(generation)
    }
}

impl fluxel_rhi_core::backend::SurfaceBackend for Gl {
    type ConfigureFuture<'a> = Ready<Result<()>>;
    type AcquireFuture<'a> = Ready<Result<BrowserSurfaceImage>>;

    fn generation(surface: &BrowserSurface) -> u64 {
        surface.generation()
    }

    fn capabilities(surface: &BrowserSurface) -> Result<api::SurfaceCapabilities> {
        surface.capabilities()
    }

    fn supports_config(surface: &BrowserSurface, desc: &api::SurfaceConfig) -> Support<()> {
        surface.supports_config(desc)
    }

    fn configure<'a>(
        surface: &'a mut BrowserSurface,
        desc: &'a api::SurfaceConfig,
    ) -> Self::ConfigureFuture<'a> {
        ready(surface.configure(desc))
    }

    fn acquire<'a>(surface: &'a mut BrowserSurface) -> Self::AcquireFuture<'a> {
        ready(surface.acquire())
    }
}

/// Confirms browser presentation after Core performed the irrevocable lease
/// handoff. There is no native swap call after WebGL `flush`.
pub(crate) fn present_handed_off(image: &BrowserSurfaceImage) -> Result<()> {
    image.present_handed_off()
}
