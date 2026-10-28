//! Core surface SPI for one thread-affine GL drawable.
//!
//! A GL surface does not own a swapchain: WGL/EGL retain the host drawable and
//! this module retains only configuration generations and one exclusive lease.
//! Reconfiguration invalidates the old generation before publishing the new
//! configuration.  A lease retains the device state, which in turn retains the
//! platform context, so the host handle cannot be used after the context owner
//! has been dropped.  Native presentation remains immediate: after Core marks
//! the lease handed off, this module makes the owner context current and asks
//! the platform owner to swap its drawable.
//!
//! This module deliberately does not choose frame pacing, create a command
//! queue, or infer resource hazards.  Those belong to the renderer and Core
//! submission contracts respectively.

#[cfg(not(all(target_arch = "wasm32", feature = "webgl2")))]
use std::{
    cell::RefCell,
    future::{Ready, ready},
    rc::Rc,
};

#[cfg(not(all(target_arch = "wasm32", feature = "webgl2")))]
use fluxel_rhi_core::api::{self, Error, Result, Support};

#[cfg(not(all(target_arch = "wasm32", feature = "webgl2")))]
use super::{Gl, GlDevice};
#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
pub(crate) use crate::backend::browser::{
    BrowserSurface as GlSurface, BrowserSurfaceImage as GlSurfaceImage,
};

/// Native surface retained by Core's typed [`api::Surface`].
///
/// Clones share one generation record.  Core owns the mutable surface wrapper;
/// image leases use the shared state only to validate and retire themselves
/// after that wrapper has moved on to a later configuration.
#[derive(Clone)]
#[cfg(not(all(target_arch = "wasm32", feature = "webgl2")))]
pub struct GlSurface(Rc<RefCell<SurfaceState>>);

/// One exclusive default-framebuffer acquisition.
///
/// There is no separately allocated GL image: the native default framebuffer
/// is the acquired target.  The lease's serial prevents an old drop from
/// clearing a newer acquisition after reconfiguration.
#[cfg(not(all(target_arch = "wasm32", feature = "webgl2")))]
pub struct GlSurfaceImage {
    surface: GlSurface,
    generation: u64,
    serial: u64,
    handed_off: bool,
}

/// Read-only proof that a pass still targets its acquired configuration.
///
/// This does not own or release the exclusive lease. The original image stays
/// with Core, while a depth pass may outlive the borrow used by `begin_raster`.
#[derive(Clone)]
#[cfg(not(all(target_arch = "wasm32", feature = "webgl2")))]
pub(crate) struct GlSurfaceTargetToken {
    surface: GlSurface,
    generation: u64,
    serial: u64,
}

#[cfg(not(all(target_arch = "wasm32", feature = "webgl2")))]
impl GlSurfaceTargetToken {
    pub(crate) fn is_current(&self) -> bool {
        let state = self.surface.0.borrow();
        state.generation == self.generation
            && matches!(
                state.live,
                Some(LiveLease { generation, serial })
                    if generation == self.generation && serial == self.serial
            )
    }
}

#[cfg(not(all(target_arch = "wasm32", feature = "webgl2")))]
struct SurfaceState {
    device: GlDevice,
    context_stamp: crate::backend::api::ContextStamp,
    formats: Vec<api::TextureFormat>,
    extent: [u32; 2],
    generation: u64,
    configured: Option<api::SurfaceConfig>,
    live: Option<LiveLease>,
    next_serial: u64,
}

#[derive(Clone, Copy)]
#[cfg(not(all(target_arch = "wasm32", feature = "webgl2")))]
struct LiveLease {
    generation: u64,
    serial: u64,
}

#[cfg(not(all(target_arch = "wasm32", feature = "webgl2")))]
impl Gl {
    /// Associates this device's already-adopted native drawable with Core.
    ///
    /// The host continues to own its window or EGL native window.  It must keep
    /// that object alive until this surface, all its leases, and the device have
    /// been dropped on the GL owner thread.
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    pub fn surface(device: &api::Device<Self>) -> Result<api::Surface<Self>> {
        use crate::backend::{api::GlFamilyApi as _, native::NativeContextOwner as _};

        // SAFETY: `device` is statically a Core GL device.  The clone retained
        // by SurfaceState keeps the adopted context alive for every lease.
        let native = unsafe { device.native_ref() }.clone();
        let (extent, supports_presentation, context_stamp) = native
            .0
            .with_context("create GL surface", |context| {
                Ok((
                    context.drawable_extent(),
                    context.supports_presentation(),
                    context.context_stamp(),
                ))
            })
            .map_err(native_error)?;
        if !supports_presentation {
            return Err(Error::Unsupported("a window-backed GL drawable"));
        }
        let extent = extent.ok_or(Error::SurfaceLost)?;
        if extent.contains(&0) {
            return Err(Error::SurfaceLost);
        }
        let facts = native
            .0
            .with_native("inspect GL surface format", |provider| {
                Ok(provider.discovery().surface_facts())
            })
            .map_err(native_error)?;
        let formats = formats_from_facts(facts)?;
        let surface = GlSurface(Rc::new(RefCell::new(SurfaceState {
            device: native,
            context_stamp,
            formats,
            extent,
            generation: 0,
            configured: None,
            live: None,
            next_serial: 0,
        })));
        // SAFETY: the state was created solely for this device and contains no
        // lease until Core returns the newly associated Surface to its caller.
        Ok(unsafe { api::Surface::from_native(device, api::SurfaceId::allocate()?, surface) })
    }

    /// Featureless builds cannot synthesize a host drawable.
    #[cfg(not(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl")))]
    pub fn surface(_: &api::Device<Self>) -> Result<api::Surface<Self>> {
        Err(Error::Unsupported("a native GL presentation context"))
    }
}

#[cfg(not(all(target_arch = "wasm32", feature = "webgl2")))]
impl GlSurfaceImage {
    pub(crate) fn target_token(&self) -> GlSurfaceTargetToken {
        GlSurfaceTargetToken {
            surface: self.surface.clone(),
            generation: self.generation,
            serial: self.serial,
        }
    }

    pub(crate) fn default_target(&self) -> Result<crate::backend::api::GlDefaultFramebufferTarget> {
        use fluxel_rhi_core::backend::NativeImageLease as _;

        if !self.is_current(self.generation) {
            return Err(Error::SurfaceOutdated);
        }
        let state = self.surface.0.borrow();
        let config = state.configured.as_ref().ok_or(Error::SurfaceOutdated)?;
        Ok(crate::backend::api::GlDefaultFramebufferTarget {
            frame_serial: self.serial,
            context: state.context_stamp,
            width: state.extent[0],
            height: state.extent[1],
            sample_count: 1,
            color_format: match config.format {
                api::TextureFormat::Rgba8Unorm => crate::backend::api::GlFormat::Rgba8Unorm,
                _ => return Err(Error::SurfaceOutdated),
            },
        })
    }
}

#[cfg(not(all(target_arch = "wasm32", feature = "webgl2")))]
// SAFETY: SurfaceState serializes handoff, abandonment, and configuration in
// one RefCell. Each lease carries its exact generation/serial pair, so a stale
// Drop cannot clear a later generation's live lease.
unsafe impl fluxel_rhi_core::backend::NativeImageLease for GlSurfaceImage {
    fn try_handoff(&mut self, generation: u64) -> Result<()> {
        if !self.is_current(generation) {
            return Err(Error::SurfaceOutdated);
        }
        #[cfg(all(feature = "native-gles-egl", not(target_arch = "wasm32")))]
        capture_default_framebuffer_if_requested(self)?;
        let mut state = self.surface.0.borrow_mut();
        state.live = None;
        self.handed_off = true;
        Ok(())
    }

    fn is_current(&self, generation: u64) -> bool {
        let state = self.surface.0.borrow();
        !self.handed_off
            && self.generation == generation
            && state.generation == generation
            && matches!(
                state.live,
                Some(LiveLease {
                    generation: live_generation,
                    serial,
                }) if live_generation == generation && serial == self.serial
            )
    }
}

/// Opt-in local visual diagnostics for EGL runs where compositor capture is
/// unavailable. Reading happens before the lease handoff, while the exact
/// default framebuffer is still authenticated; normal runs perform no I/O.
#[cfg(all(feature = "native-gles-egl", not(target_arch = "wasm32")))]
fn capture_default_framebuffer_if_requested(image: &GlSurfaceImage) -> Result<()> {
    let Some(path) = std::env::var_os("FLUXEL_RHI_GL_CAPTURE_PATH") else {
        return Ok(());
    };

    let target = image.default_target()?;
    let device = image.surface.0.borrow().device.clone();
    let readback = device
        .0
        .with_native("capture GL surface for local diagnostics", |provider| {
            provider.read_default_framebuffer(target)
        })
        .map_err(native_error)?;
    let capacity = 32usize
        .checked_add(readback.bytes.len().saturating_mul(3) / 4)
        .ok_or(Error::OutOfMemory)?;
    let mut ppm = format!("P6\n{} {}\n255\n", target.width, target.height).into_bytes();
    ppm.reserve(capacity);
    for rgba in readback.bytes.chunks_exact(4) {
        ppm.extend_from_slice(&rgba[..3]);
    }
    std::fs::write(path, ppm)
        .map_err(|error| Error::Backend(format!("writing GL diagnostic capture failed: {error}")))
}

#[cfg(not(all(target_arch = "wasm32", feature = "webgl2")))]
impl Drop for GlSurfaceImage {
    fn drop(&mut self) {
        if self.handed_off {
            return;
        }
        let mut state = self.surface.0.borrow_mut();
        if matches!(
            state.live,
            Some(LiveLease {
                generation,
                serial,
            }) if generation == self.generation && serial == self.serial
        ) {
            state.live = None;
        }
    }
}

#[cfg(not(all(target_arch = "wasm32", feature = "webgl2")))]
impl fluxel_rhi_core::backend::SurfaceBackend for Gl {
    type ConfigureFuture<'a> = Ready<Result<()>>;
    type AcquireFuture<'a> = Ready<Result<GlSurfaceImage>>;

    fn generation(surface: &GlSurface) -> u64 {
        surface.0.borrow().generation
    }

    fn capabilities(surface: &GlSurface) -> Result<api::SurfaceCapabilities> {
        let state = surface.0.borrow();
        Ok(api::SurfaceCapabilities {
            formats: state.formats.clone(),
            present_modes: vec![api::PresentMode::Fifo],
            alpha_modes: vec![api::CompositeAlphaMode::Opaque],
            color_spaces: vec![api::ColorSpace::Srgb],
            usages: api::SurfaceUsage::RENDER_ATTACHMENT,
            min_extent: [1, 1],
            // GL reports the actual viewport bound while resizing the native
            // drawable.  This cached field is the current Host extent, not a
            // maximum: treating it as one would reject every later resize.
            max_extent: [i32::MAX as u32; 2],
        })
    }

    fn supports_config(surface: &GlSurface, desc: &api::SurfaceConfig) -> Support<()> {
        let Ok(caps) = Self::capabilities(surface) else {
            return Support::Unsupported("GL default framebuffer facts");
        };
        if !caps.formats.contains(&desc.format)
            || desc.present_mode != api::PresentMode::Fifo
            || desc.alpha_mode != api::CompositeAlphaMode::Opaque
            || desc.color_space != api::ColorSpace::Srgb
            || desc.usage != api::SurfaceUsage::RENDER_ATTACHMENT
            || desc.maximum_frame_latency != 1
            || desc.extent.contains(&0)
        {
            Support::Unsupported("GL default framebuffer surface configuration")
        } else {
            Support::Supported(())
        }
    }

    fn configure<'a>(
        surface: &'a mut GlSurface,
        desc: &'a api::SurfaceConfig,
    ) -> Self::ConfigureFuture<'a> {
        ready(configure(surface, desc))
    }

    fn acquire<'a>(surface: &'a mut GlSurface) -> Self::AcquireFuture<'a> {
        ready(acquire(surface))
    }
}

/// Swaps a lease which Core has already handed off at the irrevocable boundary.
///
/// A failure here is intentionally returned only after handoff: callers must
/// classify it as an immediate `Indeterminate` outcome, never return the
/// binding for retry.  The platform may have accepted the swap before reporting
/// a driver error.
#[cfg(not(all(target_arch = "wasm32", feature = "webgl2")))]
pub(crate) fn present_handed_off(image: &GlSurfaceImage) -> Result<()> {
    if !image.handed_off {
        return Err(Error::InvalidState);
    }
    let device = image.surface.0.borrow().device.clone();
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    {
        use crate::backend::native::NativeContextOwner as _;

        return device
            .0
            .with_context("present GL surface", |context| {
                context.swap_buffers()?;
                Ok(())
            })
            .map_err(native_error);
    }
    #[cfg(not(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl")))]
    {
        let _ = device;
        Err(Error::Unsupported("a native GL presentation context"))
    }
}

/// Browser presentation has already flushed during its native handoff; this
/// forwarding seam keeps ImmediateBackend independent of the platform owner.
#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
pub(crate) fn present_handed_off(image: &GlSurfaceImage) -> fluxel_rhi_core::api::Result<()> {
    image.present_handed_off()
}

#[cfg(not(all(target_arch = "wasm32", feature = "webgl2")))]
fn configure(surface: &GlSurface, desc: &api::SurfaceConfig) -> Result<()> {
    if !matches!(
        <Gl as fluxel_rhi_core::backend::SurfaceBackend>::supports_config(surface, desc),
        Support::Supported(())
    ) {
        return Err(Error::Unsupported(
            "GL default framebuffer surface configuration",
        ));
    }
    let device = surface.0.borrow().device.clone();
    let observed = resize_and_observe(&device, desc.extent)?;
    if observed != desc.extent {
        return Err(Error::SurfaceOutdated);
    }
    let mut state = surface.0.borrow_mut();
    state.generation = state.generation.checked_add(1).ok_or(Error::OutOfMemory)?;
    state.extent = observed;
    state.configured = Some(desc.clone());
    // The generation bump makes outstanding leases stale.  Clearing the live
    // slot is atomic with it, and an old lease's Drop checks both values before
    // it can alter state again.
    state.live = None;
    Ok(())
}

#[cfg(not(all(target_arch = "wasm32", feature = "webgl2")))]
fn acquire(surface: &GlSurface) -> Result<GlSurfaceImage> {
    let device = surface.0.borrow().device.clone();
    let observed = observe_extent(&device)?;
    let mut state = surface.0.borrow_mut();
    if observed != state.extent {
        state.generation = state.generation.checked_add(1).ok_or(Error::OutOfMemory)?;
        state.extent = observed;
        state.configured = None;
        state.live = None;
        return Err(Error::SurfaceOutdated);
    }
    if state.configured.is_none() {
        return Err(Error::InvalidState);
    }
    if state.live.is_some() {
        return Err(Error::InvalidState);
    }
    state.next_serial = state.next_serial.checked_add(1).ok_or(Error::OutOfMemory)?;
    let serial = state.next_serial;
    let generation = state.generation;
    state.live = Some(LiveLease { generation, serial });
    Ok(GlSurfaceImage {
        surface: surface.clone(),
        generation,
        serial,
        handed_off: false,
    })
}

#[cfg(not(all(target_arch = "wasm32", feature = "webgl2")))]
fn resize_and_observe(device: &GlDevice, requested: [u32; 2]) -> Result<[u32; 2]> {
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    {
        use crate::backend::native::NativeContextOwner as _;

        return device
            .0
            .with_context("configure GL surface", |context| {
                context.resize_drawable(requested)?;
                context
                    .drawable_extent()
                    .ok_or(crate::backend::api::GlError::Unsupported {
                        operation: "configure GL surface",
                        reason: "the Host did not report a native drawable extent",
                    })
            })
            .map_err(native_error);
    }
    #[cfg(not(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl")))]
    {
        let _ = (device, requested);
        Err(Error::Unsupported("a native GL presentation context"))
    }
}

#[cfg(not(all(target_arch = "wasm32", feature = "webgl2")))]
fn observe_extent(device: &GlDevice) -> Result<[u32; 2]> {
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    {
        use crate::backend::native::NativeContextOwner as _;

        return device
            .0
            .with_context("acquire GL surface", |context| {
                context
                    .drawable_extent()
                    .ok_or(crate::backend::api::GlError::Unsupported {
                        operation: "acquire GL surface",
                        reason: "the Host did not report a native drawable extent",
                    })
            })
            .map_err(native_error);
    }
    #[cfg(not(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl")))]
    {
        let _ = device;
        Err(Error::Unsupported("a native GL presentation context"))
    }
}

#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[cfg(not(all(target_arch = "wasm32", feature = "webgl2")))]
fn formats_from_facts(
    facts: crate::backend::api::GlSurfaceFacts,
) -> Result<Vec<api::TextureFormat>> {
    match facts {
        // The default framebuffer has no portable encoding query.  This exact
        // observation is the one configuration we can name without guessing;
        // other component widths remain unsupported rather than being coerced
        // to an RHI format with different alpha or transfer semantics.
        crate::backend::api::GlSurfaceFacts::Observed {
            color_bits: [8, 8, 8, 8],
        } => Ok(vec![api::TextureFormat::Rgba8Unorm]),
        crate::backend::api::GlSurfaceFacts::Observed { .. }
        | crate::backend::api::GlSurfaceFacts::Unavailable => Err(Error::Unsupported(
            "an observed RGBA8 native GL default framebuffer",
        )),
    }
}

#[cfg(not(all(target_arch = "wasm32", feature = "webgl2")))]
fn native_error(error: crate::backend::api::GlError) -> Error {
    match error {
        crate::backend::api::GlError::ContextLost { .. } => Error::DeviceLost,
        crate::backend::api::GlError::OutOfMemory { .. } => Error::OutOfMemory,
        other => Error::Backend(format!("{other:?}")),
    }
}
