//! CAMetalLayer configuration, drawable acquisition, and Surface leases.
//!
//! The host owns the view/layer and its application lifecycle; this module
//! retains the supplied layer, binds it to this Metal device, and issues
//! generation-bound drawable leases. Configure publishes a new generation only
//! after validating the complete request and applying native layer settings.
//! Drawable acquisition calls `nextDrawable` synchronously and reports an
//! unavailable drawable instead of fabricating a texture. Enabling
//! `allowsNextDrawableTimeout` bounds the wait, but does not make this call
//! asynchronous: the calling thread can still block until a drawable arrives
//! or the configured timeout expires. Wrapping it in `Ready` only reflects the
//! SPI shape and must not be treated as a nonblocking UI-thread operation.

#[cfg(target_vendor = "apple")]
use super::MetalDevice;
use super::{Metal, MetalSurfaceImage};
use fluxel_rhi_core::api::{
    self, ColorSpace, CompositeAlphaMode, PresentMode, Support, SurfaceBackend,
    SurfaceCapabilities, SurfaceConfig, SurfaceUsage,
};
use std::{
    cell::RefCell,
    future::{Ready, ready},
    rc::Rc,
    sync::{Arc, Mutex},
};

const MAX_SURFACE_EXTENT: u32 = 8192;
const MAX_FRAME_LATENCY: u32 = 3;

/// A private-owner CAMetalLayer association used by the generic Core Surface.
#[derive(Clone)]
pub struct MetalSurface {
    state: Rc<RefCell<MetalSurfaceState>>,
}

struct MetalSurfaceState {
    #[cfg(target_vendor = "apple")]
    layer: objc2::rc::Retained<objc2_quartz_core::CAMetalLayer>,
    generation: Arc<Mutex<u64>>,
    #[cfg(target_vendor = "apple")]
    configured: bool,
}

impl MetalSurface {
    /// Binds a host-owned layer to this device and retains it for Surface use.
    ///
    /// The host remains responsible for attaching the layer to a live view and
    /// updating the configured drawable size when the view resizes.
    #[cfg(target_vendor = "apple")]
    pub fn from_layer(
        device: &MetalDevice,
        layer: objc2::rc::Retained<objc2_quartz_core::CAMetalLayer>,
    ) -> api::Result<Self> {
        layer.setDevice(Some(&device._state._device));
        layer.setFramebufferOnly(true);
        // Prevent an unbounded nextDrawable wait when the host pauses or
        // configures a zero-sized view. A timed-out acquire returns SurfaceOutdated.
        layer.setAllowsNextDrawableTimeout(true);
        Ok(Self {
            state: Rc::new(RefCell::new(MetalSurfaceState {
                layer,
                generation: Arc::new(Mutex::new(0)),
                #[cfg(target_vendor = "apple")]
                configured: false,
            })),
        })
    }

    /// Creates an unavailable placeholder on non-Apple targets.
    #[cfg(not(target_vendor = "apple"))]
    pub fn unavailable() -> Self {
        Self {
            state: Rc::new(RefCell::new(MetalSurfaceState {
                generation: Arc::new(Mutex::new(0)),
            })),
        }
    }
}

impl SurfaceBackend for Metal {
    type ConfigureFuture<'a> = Ready<api::Result<()>>;
    type AcquireFuture<'a> = Ready<api::Result<MetalSurfaceImage>>;

    fn generation(surface: &MetalSurface) -> u64 {
        surface
            .state
            .borrow()
            .generation
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .to_owned()
    }

    fn capabilities(_: &MetalSurface) -> api::Result<SurfaceCapabilities> {
        Ok(SurfaceCapabilities {
            formats: vec![
                api::TextureFormat::Bgra8Unorm,
                api::TextureFormat::Bgra8UnormSrgb,
            ],
            present_modes: vec![PresentMode::Fifo],
            alpha_modes: vec![CompositeAlphaMode::Opaque],
            color_spaces: vec![ColorSpace::Srgb],
            usages: SurfaceUsage::RENDER_ATTACHMENT,
            min_extent: [1, 1],
            max_extent: [MAX_SURFACE_EXTENT, MAX_SURFACE_EXTENT],
        })
    }

    fn supports_config(surface: &MetalSurface, desc: &SurfaceConfig) -> Support<()> {
        if desc.extent[0] == 0
            || desc.extent[1] == 0
            || desc.extent[0] > MAX_SURFACE_EXTENT
            || desc.extent[1] > MAX_SURFACE_EXTENT
            || !matches!(
                desc.format,
                api::TextureFormat::Bgra8Unorm | api::TextureFormat::Bgra8UnormSrgb
            )
            || desc.present_mode != PresentMode::Fifo
            || desc.alpha_mode != CompositeAlphaMode::Opaque
            || desc.color_space != ColorSpace::Srgb
            || desc.usage.0 != SurfaceUsage::RENDER_ATTACHMENT.0
            || desc.maximum_frame_latency == 0
            || desc.maximum_frame_latency > MAX_FRAME_LATENCY
        {
            return Support::Unsupported(
                "Metal Surface configuration is outside the supported subset",
            );
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = surface;
            Support::Unsupported("Metal requires an Apple target")
        }
        #[cfg(target_vendor = "apple")]
        {
            let _ = surface;
            Support::Supported(())
        }
    }

    fn configure<'a>(
        surface: &'a mut MetalSurface,
        desc: &'a SurfaceConfig,
    ) -> Self::ConfigureFuture<'a> {
        if let Support::Unsupported(reason) = Self::supports_config(surface, desc) {
            return ready(Err(api::Error::Unsupported(reason)));
        }
        #[allow(unused_mut)] // Apple builds update `configured` on this state.
        let mut state = surface.state.borrow_mut();
        // Clone the shared epoch before locking it so we can mutate the layer
        // and configured flag while the epoch lock prevents a racing handoff.
        let generation_state = state.generation.clone();
        #[allow(unused_mut)]
        let mut generation = generation_state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let Some(next_generation) = generation.checked_add(1) else {
            return ready(Err(api::Error::InvalidRange));
        };
        #[cfg(target_vendor = "apple")]
        {
            use objc2_core_foundation::CGSize;
            use objc2_metal::MTLPixelFormat;
            // Invalidate images before changing the layer: submission can
            // validate an existing lease without borrowing this Surface. The
            // shared lock serializes invalidation with present handoff.
            *generation = next_generation;
            state.layer.setPixelFormat(match desc.format {
                api::TextureFormat::Bgra8Unorm => MTLPixelFormat::BGRA8Unorm,
                api::TextureFormat::Bgra8UnormSrgb => MTLPixelFormat::BGRA8Unorm_sRGB,
                _ => unreachable!("supports_config checked the format"),
            });
            state.layer.setDrawableSize(CGSize {
                width: f64::from(desc.extent[0]),
                height: f64::from(desc.extent[1]),
            });
            state
                .layer
                .setMaximumDrawableCount(desc.maximum_frame_latency as usize);
            state.layer.setDisplaySyncEnabled(true);
            state.layer.setFramebufferOnly(true);
            state.layer.setAllowsNextDrawableTimeout(true);
            state.configured = true;
            ready(Ok(()))
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = (desc, next_generation);
            drop(generation);
            drop(state);
            let _ = surface;
            ready(Err(api::Error::Unsupported(
                "Metal requires an Apple target",
            )))
        }
    }

    fn acquire<'a>(surface: &'a mut MetalSurface) -> Self::AcquireFuture<'a> {
        #[cfg(target_vendor = "apple")]
        {
            use objc2_quartz_core::CAMetalDrawable as _;

            let state = surface.state.borrow();

            if !state.configured {
                return ready(Err(api::Error::InvalidState));
            }
            let generation = *state
                .generation
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let Some(drawable) = state.layer.nextDrawable() else {
                return ready(Err(api::Error::SurfaceOutdated));
            };
            let texture = drawable.texture();
            ready(Ok(MetalSurfaceImage {
                generation,
                surface_generation: state.generation.clone(),
                handed_off: false,
                drawable,
                texture,
            }))
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = surface;
            ready(Err(api::Error::Unsupported(
                "Metal requires an Apple target",
            )))
        }
    }
}
