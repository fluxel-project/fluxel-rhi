//! WebGL2 canvas surface generations and exclusive default-framebuffer leases.
//!
//! A browser compositor presents the canvas outside WebGL.  `present` therefore
//! publishes work with `flush` and hands off the lease, while a separate owner
//! fence supplies GPU-completion evidence.  Resize and context loss invalidate
//! all outstanding leases through their configuration generation.

use std::{cell::RefCell, rc::Rc};

use fluxel_rhi_core::api::{
    ColorSpace, CompositeAlphaMode, Error, PresentMode, Result, Support, SurfaceCapabilities,
    SurfaceConfig, SurfaceUsage, TextureFormat,
};

use super::owner::BrowserOwner;

/// Browser-private state shared by a WebGL2 canvas surface and its leases.
#[derive(Clone)]
pub struct BrowserSurface {
    pub(super) owner: BrowserOwner,
    pub(super) state: Rc<RefCell<SurfaceState>>,
}

/// Exclusive lease of the canvas default framebuffer for one configuration.
pub struct BrowserSurfaceImage {
    pub(super) surface: BrowserSurface,
    pub(super) generation: u64,
    handed_off: bool,
    pub(super) serial: u64,
}

pub(super) struct SurfaceState {
    generation: u64,
    configured: bool,
    acquired: bool,
    live_serial: Option<u64>,
    next_serial: u64,
    pub(super) extent: [u32; 2],
}

impl BrowserSurface {
    /// Creates the surface state for one browser owner. The caller associates
    /// it with the same Core device that owns `owner`.
    pub(crate) fn new(owner: BrowserOwner) -> Self {
        Self {
            owner,
            state: Rc::new(RefCell::new(SurfaceState {
                generation: 1,
                configured: false,
                acquired: false,
                live_serial: None,
                next_serial: 1,
                extent: [0, 0],
            })),
        }
    }

    /// Reconfigures the RHI-owned canvas backing store. The browser may only
    /// give this surface its default framebuffer, so a live lease forbids a
    /// generation change.
    pub(crate) fn configure(&self, config: &SurfaceConfig) -> Result<()> {
        if !matches!(self.supports_config(config), Support::Supported(())) {
            return Err(Error::Unsupported("WebGL2 canvas surface configuration"));
        }
        let mut state = self.state.borrow_mut();
        if state.acquired {
            return Err(Error::InvalidState);
        }
        let extent = self.owner.resize_drawing_buffer(config.extent)?;
        state.generation = state.generation.checked_add(1).ok_or(Error::OutOfMemory)?;
        state.configured = true;
        state.extent = extent;
        Ok(())
    }

    /// Acquires the default framebuffer if its drawing buffer has nonzero size.
    /// A zero-area canvas reports `SurfaceOutdated` so callers can wait for the
    /// host resize lifecycle instead of receiving an unusable image.
    pub(crate) fn acquire(&self) -> Result<BrowserSurfaceImage> {
        let mut state = self.state.borrow_mut();
        if !state.configured || state.acquired {
            return Err(Error::InvalidState);
        }
        let actual = self.owner.drawing_buffer_extent()?;
        if actual == [0, 0] || actual != state.extent {
            return Err(Error::SurfaceOutdated);
        }
        let serial = state.next_serial;
        state.next_serial = serial.checked_add(1).ok_or(Error::OutOfMemory)?;
        state.acquired = true;
        state.live_serial = Some(serial);
        Ok(BrowserSurfaceImage {
            surface: self.clone(),
            generation: state.generation,
            handed_off: false,
            serial,
        })
    }

    /// Returns the current configuration generation for Core's lease check.
    pub(crate) fn generation(&self) -> u64 {
        self.state.borrow().generation
    }

    /// Reports the portable default-framebuffer route that WebGL2 can prove.
    /// Browser canvas presentation exposes no configurable swapchain format or
    /// present mode, so the capability list deliberately stays narrow.
    pub(crate) fn capabilities(&self) -> Result<SurfaceCapabilities> {
        self.owner.drawing_buffer_extent()?;
        Ok(SurfaceCapabilities {
            formats: vec![TextureFormat::Rgba8UnormSrgb],
            present_modes: vec![PresentMode::Fifo],
            alpha_modes: vec![CompositeAlphaMode::Opaque],
            color_spaces: vec![ColorSpace::Srgb],
            usages: SurfaceUsage::RENDER_ATTACHMENT,
            min_extent: [1, 1],
            max_extent: [i32::MAX as u32; 2],
        })
    }

    /// Checks the browser-default-framebuffer configuration without changing
    /// the backing store. `configure` verifies the actual chosen canvas size.
    pub(crate) fn supports_config(&self, config: &SurfaceConfig) -> Support<()> {
        if config.extent.contains(&0)
            || !matches!(config.format, TextureFormat::Rgba8UnormSrgb)
            || !matches!(config.present_mode, PresentMode::Fifo)
            || !matches!(config.alpha_mode, CompositeAlphaMode::Opaque)
            || !matches!(config.color_space, ColorSpace::Srgb)
            || config.usage.0 != SurfaceUsage::RENDER_ATTACHMENT.0
            || config.maximum_frame_latency == 0
        {
            Support::Unsupported("WebGL2 canvas surface configuration")
        } else {
            Support::Supported(())
        }
    }

    /// Invalidates every lease before forwarding terminal loss to the context
    /// owner. Restoring a browser context creates a new surface/owner pair;
    /// this method never attempts to reuse the old default framebuffer.
    pub(crate) fn mark_context_lost(&self) {
        let mut state = self.state.borrow_mut();
        state.generation = state.generation.wrapping_add(1).max(1);
        state.configured = false;
        state.acquired = false;
        state.live_serial = None;
        drop(state);
        self.owner.mark_context_lost();
    }

    fn release(&self, generation: u64, serial: u64) {
        let mut state = self.state.borrow_mut();
        if state.generation == generation && state.live_serial == Some(serial) {
            state.acquired = false;
            state.live_serial = None;
        }
    }
}

impl BrowserSurfaceImage {
    /// Tests whether this lease still names the surface's current configuration.
    pub(crate) fn is_current_for(&self, generation: u64) -> bool {
        generation == self.generation && !self.handed_off && {
            let state = self.surface.state.borrow();
            state.generation == self.generation && state.live_serial == Some(self.serial)
        }
    }

    /// Hands the default framebuffer to the browser compositor. This has no
    /// GPU-completion meaning; callers needing retirement evidence use a
    /// `BrowserCompletion` produced by the encoder. A flush failure leaves the
    /// lease live so Core can return a clean present rejection.
    pub(crate) fn try_handoff(&mut self, generation: u64) -> Result<()> {
        if !self.is_current_for(generation) {
            return Err(Error::SurfaceOutdated);
        }
        self.surface.owner.flush()?;
        self.handed_off = true;
        self.surface.release(self.generation, self.serial);
        Ok(())
    }

    /// Browser canvas presentation has no second native present operation
    /// after a successful flush and lease handoff.
    pub(crate) fn present_handed_off(&self) -> Result<()> {
        if self.handed_off {
            Ok(())
        } else {
            Err(Error::InvalidState)
        }
    }
}

impl Drop for BrowserSurfaceImage {
    fn drop(&mut self) {
        if !self.handed_off {
            self.surface.release(self.generation, self.serial);
        }
    }
}
