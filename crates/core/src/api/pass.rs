//! Raster-pass attachment, viewport, and load/store vocabulary.
//!
//! The renderer determines pass order and resource transitions. Backends lower
//! this immutable pass description without deriving a second hazard model.
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

/// Framebuffer viewport: upper-left origin, X right, Y down, depth in [0, 1].
/// Clip/NDC Y points up; backends map it to this framebuffer convention.
#[derive(Clone, Copy, Debug)]
pub struct Viewport {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub min_depth: f32,
    pub max_depth: f32,
}
/// Rectangle in framebuffer pixels, measured from the upper-left origin.
#[derive(Clone, Copy, Debug)]
pub struct Rect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}
#[derive(Clone, Copy, Debug)]
pub enum LoadOp<T> {
    Load,
    Clear(T),
    Discard,
}
#[derive(Clone, Copy, Debug)]
pub enum StoreOp {
    Store,
    Discard,
}
#[derive(Clone, Copy, Debug)]
pub enum ClearColor {
    Float([f32; 4]),
    Uint([u32; 4]),
    Sint([i32; 4]),
}
/// Clear value for one color slot in an active raster pass.
#[derive(Clone, Copy, Debug)]
pub struct ColorAttachmentClear {
    /// Color attachment slot from [`RasterPassDesc::colors`].
    pub attachment: u32,
    /// Value converted according to the target attachment's numeric type.
    pub value: ClearColor,
}

/// One native backbuffer lease may be a render attachment only while acquired.
/// WebGL2 default framebuffer is intentionally not a fake TextureView.
pub enum AttachmentView<'a, B: Backend> {
    Texture(&'a TextureView<B>),
    Surface(&'a SurfaceImage<B>),
}
impl<B: Backend> AttachmentView<'_, B> {
    pub fn device(&self) -> DeviceId {
        match self {
            Self::Texture(v) => v.device,
            Self::Surface(v) => v.device,
        }
    }
    pub fn is_current(&self) -> bool {
        match self {
            Self::Texture(_) => true,
            Self::Surface(v) => v.is_current(),
        }
    }
}
pub struct ColorAttachment<'a, B: Backend> {
    pub view: AttachmentView<'a, B>,
    pub resolve_target: Option<AttachmentView<'a, B>>,
    pub load: LoadOp<ClearColor>,
    pub store: StoreOp,
}
pub struct DepthStencilAttachment<'a, B: Backend> {
    pub view: AttachmentView<'a, B>,
    pub depth: Option<(LoadOp<f32>, StoreOp)>,
    pub stencil: Option<(LoadOp<u32>, StoreOp)>,
    pub depth_read_only: bool,
    pub stencil_read_only: bool,
}
pub struct PassTimestampWrites<'a, B: Backend> {
    pub set: &'a QuerySet<B>,
    pub begin: Option<u32>,
    pub end: Option<u32>,
}
pub struct RasterPassDesc<'a, B: Backend> {
    pub label: Option<&'a str>,
    pub colors: &'a [Option<ColorAttachment<'a, B>>],
    pub depth_stencil: Option<DepthStencilAttachment<'a, B>>,
    pub render_area: Rect,
    pub multiview_mask: Option<u32>,
    pub timestamp_writes: Option<PassTimestampWrites<'a, B>>,
    /// Query set fixed at render-pass creation for backends such as WebGPU.
    /// Individual query slots are begun and ended through `Queries`.
    pub occlusion_query_set: Option<&'a QuerySet<B>>,
    /// Required when this pass executes inherited secondary raster work.  The
    /// backend validates this declaration against its real attachments.
    pub secondary_inheritance: Option<SecondaryRasterDesc>,
}
#[derive(Clone, Copy, Debug)]
pub struct TexelCopyBufferLayout {
    pub offset: u64,
    /// Byte distance between adjacent texel rows, or adjacent block rows for
    /// compressed formats.
    pub bytes_per_row: u32,
    /// Number of texel rows per image, or block rows per image for compressed
    /// formats. The unit is the same row unit used by `bytes_per_row`.
    pub rows_per_image: u32,
}
pub struct TextureRegion<'a, B: Backend> {
    pub texture: &'a Texture<B>,
    pub mip: u32,
    /// Texel origin; for 2D images (0, 0) is upper-left and Y increases down.
    pub origin: [u32; 3],
    pub extent: [u32; 3],
    pub aspect: TextureAspects,
}
