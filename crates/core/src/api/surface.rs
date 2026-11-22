//! Surface configuration and exclusive image-acquisition contract.
//!
//! A surface image lease is valid for one configuration generation. The host
//! owns window lifetime; the renderer chooses frame pacing and retirement.
#![allow(missing_docs, dead_code, unused_variables, unused_imports)]
use super::*;
use std::{
    future::Future,
    marker::PhantomData,
    ops::Range,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

pub struct Surface<B: Backend> {
    pub device: DeviceId,
    pub id: SurfaceId,
    native: B::NativeSurface,
    stats: Arc<RhiStats>,
    configured: Option<ConfiguredLeaseFacts>,
}

/// Facts that Core publishes only after a successful native configuration.
///
/// The generation is sampled after `SurfaceBackend::configure` resolves, so
/// an acquired image can carry the exact configuration that produced it.
struct ConfiguredLeaseFacts {
    generation: u64,
    config: SurfaceConfig,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PresentMode {
    /// Present queued images in submission order with display synchronization.
    Fifo,
    /// Present with display synchronization and allow a newer image to replace
    /// one already queued for presentation. Acceptance does not establish
    /// that the image was displayed.
    Mailbox,
    /// Present without waiting for display synchronization, allowing tearing.
    Immediate,
}
/// Preference for choosing a configuration. The returned mode is explicit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PresentModePreference {
    AutoVsync,
    AutoNoVsync,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompositeAlphaMode {
    Opaque,
    Premultiplied,
    Postmultiplied,
    Inherit,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ColorSpace {
    Srgb,
    DisplayP3,
    ExtendedLinearSrgb,
    Hdr10,
}
/// Surface image uses, combinable with `|` and `|=`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SurfaceUsage(pub u32);
impl SurfaceUsage {
    pub const RENDER_ATTACHMENT: Self = Self(1);
    pub const COPY_SRC: Self = Self(2);
    pub const COPY_DST: Self = Self(4);
    pub const TEXTURE_BINDING: Self = Self(8);
    pub const STORAGE_BINDING: Self = Self(16);
}
impl std::ops::BitOr for SurfaceUsage {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl std::ops::BitOrAssign for SurfaceUsage {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
#[derive(Clone, Debug)]
pub struct SurfaceConfig {
    pub extent: [u32; 2],
    pub format: TextureFormat,
    pub present_mode: PresentMode,
    pub alpha_mode: CompositeAlphaMode,
    pub color_space: ColorSpace,
    pub usage: SurfaceUsage,
    pub maximum_frame_latency: u32,
}
#[derive(Clone, Debug)]
pub struct SurfaceCapabilities {
    pub formats: Vec<TextureFormat>,
    pub present_modes: Vec<PresentMode>,
    pub alpha_modes: Vec<CompositeAlphaMode>,
    pub color_spaces: Vec<ColorSpace>,
    pub usages: SurfaceUsage,
    pub min_extent: [u32; 2],
    pub max_extent: [u32; 2],
}

/// Static HDR10 mastering and content-light metadata for a presentable surface.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Hdr10Metadata {
    /// CIE 1931 xy primaries in red, green, blue order, in units of 1/50,000.
    pub primaries: [[u16; 2]; 3],
    /// CIE 1931 xy white point, in units of 1/50,000.
    pub white_point: [u16; 2],
    /// Peak luminance of the mastering display, in nits.
    pub max_mastering_nits: u32,
    /// Minimum luminance of the mastering display, in units of 1/10,000 nit.
    pub min_mastering_nits_x10000: u32,
    /// Maximum content light level, in nits. Zero means unspecified.
    pub max_content_nits: u16,
    /// Maximum frame-average light level, in nits. Zero means unspecified.
    pub max_frame_average_nits: u16,
}

impl Hdr10Metadata {
    fn validate(&self) -> Result<()> {
        if self
            .primaries
            .iter()
            .chain([&self.white_point])
            .any(|&[x, y]| u32::from(x) + u32::from(y) > 50_000)
            || u64::from(self.min_mastering_nits_x10000)
                > u64::from(self.max_mastering_nits) * 10_000
            || (self.max_content_nits != 0 && self.max_frame_average_nits > self.max_content_nits)
        {
            return Err(Error::InvalidRange);
        }
        Ok(())
    }
}

/// A snapshot of the output currently associated with a surface.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfaceOutputInfo {
    /// Color space currently active at the operating-system output.
    pub color_space: Option<ColorSpace>,
    pub bits_per_color: u32,
    /// CIE 1931 xy chromaticities in red, green, blue order.
    pub primaries: [[f32; 2]; 3],
    pub white_point: [f32; 2],
    pub min_luminance_nits: f32,
    pub max_luminance_nits: f32,
    pub max_full_frame_luminance_nits: f32,
    /// Current SDR reference white on the output, when the system reports it.
    pub sdr_white_level_nits: Option<f32>,
}

impl SurfaceOutputInfo {
    /// Ratio of peak display luminance to SDR reference white.
    pub fn hdr_headroom(&self) -> Option<f32> {
        let white = self.sdr_white_level_nits?;
        if !white.is_finite()
            || white <= 0.0
            || !self.max_luminance_nits.is_finite()
            || self.max_luminance_nits < white
        {
            return None;
        }
        let ratio = self.max_luminance_nits / white;
        ratio.is_finite().then_some(ratio)
    }
}
impl<B: SurfaceBackend> Surface<B> {
    /// # Safety: caller associates exactly one native surface with this Device.
    pub unsafe fn from_native(device: &Device<B>, id: SurfaceId, native: B::NativeSurface) -> Self {
        Self {
            device: device.id,
            id,
            native,
            stats: device.stats.clone(),
            configured: None,
        }
    }
    pub fn generation(&self) -> u64 {
        B::generation(&self.native)
    }
    pub fn capabilities(&self) -> Result<SurfaceCapabilities> {
        B::capabilities(&self.native)
    }
    /// Reads the current output independently of the swapchain configuration.
    /// Returns `None` when the backend cannot identify or describe that output.
    pub fn output_info(&self) -> Result<Option<SurfaceOutputInfo>> {
        B::output_info(&self.native)
    }
    pub fn configuration_support(&self, desc: &SurfaceConfig) -> Support<()> {
        B::supports_config(&self.native, desc)
    }
    /// Checks whether the configured surface and current output accept an HDR10 metadata write.
    pub fn hdr10_metadata_support(&self) -> Support<()> {
        let Some(configured) = self.configured.as_ref() else {
            return Support::Unsupported("configured HDR10 surface");
        };
        if configured.config.color_space != ColorSpace::Hdr10
            || configured.generation != self.generation()
        {
            return Support::Unsupported("configured HDR10 surface");
        }
        B::hdr10_metadata_support(&self.native, configured.generation)
    }
    /// Writes static HDR10 metadata for the current configuration. Pass `None`
    /// to clear it, including after moving away from an HDR-active output.
    /// Reconfiguration clears the metadata. Success reports backend acceptance.
    pub fn set_hdr10_metadata(&mut self, metadata: Option<&Hdr10Metadata>) -> Result<()> {
        let configured = self.configured.as_ref().ok_or(Error::InvalidState)?;
        if configured.generation != self.generation() {
            return Err(Error::SurfaceOutdated);
        }
        if configured.config.color_space != ColorSpace::Hdr10 {
            return Err(Error::Unsupported("configured HDR10 surface"));
        }
        if let Some(metadata) = metadata {
            metadata.validate()?;
        }
        B::set_hdr10_metadata(&mut self.native, configured.generation, metadata)
    }
    /// Suggests an SDR render-attachment configuration for the given extent.
    /// Prefers an sRGB attachment view when available. The result can be
    /// inspected or modified before an explicit `configure` call.
    pub fn suggest_configuration(
        &self,
        extent: [u32; 2],
        preference: PresentModePreference,
    ) -> Result<SurfaceConfig> {
        let caps = self.capabilities()?;
        if (0..2).any(|axis| {
            extent[axis] < caps.min_extent[axis] || extent[axis] > caps.max_extent[axis]
        }) {
            return Err(Error::InvalidRange);
        }
        if !caps.color_spaces.contains(&ColorSpace::Srgb)
            || caps.usages.0 & SurfaceUsage::RENDER_ATTACHMENT.0 == 0
        {
            return Err(Error::Unsupported("default SDR surface configuration"));
        }
        let modes: &[PresentMode] = match preference {
            PresentModePreference::AutoVsync => &[PresentMode::Fifo, PresentMode::Mailbox],
            PresentModePreference::AutoNoVsync => &[
                PresentMode::Immediate,
                PresentMode::Mailbox,
                PresentMode::Fifo,
            ],
        };
        let is_srgb = |format| {
            matches!(
                format,
                TextureFormat::Rgba8UnormSrgb | TextureFormat::Bgra8UnormSrgb
            )
        };
        let formats = caps
            .formats
            .iter()
            .copied()
            .filter(|&format| is_srgb(format))
            .chain(
                caps.formats
                    .iter()
                    .copied()
                    .filter(|&format| !is_srgb(format)),
            );
        for mode in modes
            .iter()
            .copied()
            .filter(|mode| caps.present_modes.contains(mode))
        {
            for format in formats.clone() {
                for alpha_mode in [CompositeAlphaMode::Opaque, CompositeAlphaMode::Inherit]
                    .into_iter()
                    .filter(|mode| caps.alpha_modes.contains(mode))
                {
                    let mut config = SurfaceConfig {
                        extent,
                        format,
                        present_mode: mode,
                        alpha_mode,
                        color_space: ColorSpace::Srgb,
                        usage: SurfaceUsage::RENDER_ATTACHMENT,
                        maximum_frame_latency: 2,
                    };
                    if matches!(self.configuration_support(&config), Support::Supported(())) {
                        return Ok(config);
                    }
                    config.maximum_frame_latency = 1;
                    if matches!(self.configuration_support(&config), Support::Supported(())) {
                        return Ok(config);
                    }
                }
            }
        }
        Err(Error::Unsupported("default SDR surface configuration"))
    }
    /// Snapshot of the successful configuration for the current native
    /// generation. Returns `None` before configuration or after invalidation.
    pub fn configuration(&self) -> Option<&SurfaceConfig> {
        let configured = self.configured.as_ref()?;
        (configured.generation == self.generation()).then_some(&configured.config)
    }
    pub async fn configure(&mut self, desc: &SurfaceConfig) -> Result<()> {
        if let Support::Unsupported(reason) = B::supports_config(&self.native, desc) {
            return Err(Error::Unsupported(reason));
        }
        B::configure(&mut self.native, desc).await?;
        self.configured = Some(ConfiguredLeaseFacts {
            generation: self.generation(),
            config: desc.clone(),
        });
        Ok(())
    }
    pub async fn acquire(&mut self) -> Result<SurfaceImage<B>> {
        let native = B::acquire(&mut self.native).await?;
        // Read the generation after acquisition: a backend's atomic acquire
        // transaction is the authority for the lease generation it returned.
        // Caching an earlier value would let a resize race mint a stale lease.
        let generation = self.generation();
        let Some(configured) = self.configured.as_ref() else {
            return Err(Error::InvalidState);
        };
        // A backend may observe an out-of-band resize while acquiring. Never
        // mint a lease with configuration facts from a different generation.
        if configured.generation != generation {
            return Err(Error::SurfaceOutdated);
        }
        self.stats.record_surface_acquire();
        Ok(SurfaceImage {
            device: self.device,
            surface: self.id,
            generation,
            format: configured.config.format,
            extent: configured.config.extent,
            native,
            stats: self.stats.clone(),
        })
    }
}
