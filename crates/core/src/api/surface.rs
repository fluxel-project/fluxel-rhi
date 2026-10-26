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
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PresentMode {
    Fifo,
    Mailbox,
    Immediate,
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
impl<B: SurfaceBackend> Surface<B> {
    /// # Safety: caller associates exactly one native surface with this Device.
    pub unsafe fn from_native(device: &Device<B>, id: SurfaceId, native: B::NativeSurface) -> Self {
        Self {
            device: device.id,
            id,
            native,
            stats: device.stats.clone(),
        }
    }
    pub fn generation(&self) -> u64 {
        B::generation(&self.native)
    }
    pub fn capabilities(&self) -> Result<SurfaceCapabilities> {
        B::capabilities(&self.native)
    }
    pub fn configuration_support(&self, desc: &SurfaceConfig) -> Support<()> {
        B::supports_config(&self.native, desc)
    }
    pub async fn configure(&mut self, desc: &SurfaceConfig) -> Result<()> {
        if let Support::Unsupported(reason) = B::supports_config(&self.native, desc) {
            return Err(Error::Unsupported(reason));
        }
        B::configure(&mut self.native, desc).await
    }
    pub async fn acquire(&mut self) -> Result<SurfaceImage<B>> {
        let native = B::acquire(&mut self.native).await?;
        self.stats.record_surface_acquire();
        // Read the generation after acquisition: a backend's atomic acquire
        // transaction is the authority for the lease generation it returned.
        // Caching an earlier value would let a resize race mint a stale lease.
        Ok(SurfaceImage {
            device: self.device,
            surface: self.id,
            generation: self.generation(),
            native,
            stats: self.stats.clone(),
        })
    }
}
