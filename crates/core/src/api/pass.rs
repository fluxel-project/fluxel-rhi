//! Statically typed RHI contract component.
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

#[derive(Clone, Copy, Debug)]
pub struct Viewport {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub min_depth: f32,
    pub max_depth: f32,
}
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
    /// Required when this pass executes inherited secondary raster work.  The
    /// backend validates this declaration against its real attachments.
    pub secondary_inheritance: Option<SecondaryRasterDesc>,
}
#[derive(Clone, Copy, Debug)]
pub struct TexelCopyBufferLayout {
    pub offset: u64,
    pub bytes_per_row: u32,
    pub rows_per_image: u32,
}
pub struct TextureRegion<'a, B: Backend> {
    pub texture: &'a Texture<B>,
    pub mip: u32,
    pub origin: [u32; 3],
    pub extent: [u32; 3],
    pub aspect: TextureAspects,
}
