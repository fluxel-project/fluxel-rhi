//! Raster pipeline descriptors and static backend pipeline creation SPI.
//!
//! Pipeline objects are device-local. Shader selection and pipeline caching
//! policy belong to callers or backends, not this portable contract.
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
pub enum PrimitiveTopology {
    PointList,
    LineList,
    LineStrip,
    TriangleList,
    TriangleStrip,
}
#[derive(Clone, Copy, Debug)]
pub enum IndexFormat {
    Uint16,
    Uint32,
}
#[derive(Clone, Copy, Debug)]
pub enum VertexStepMode {
    Vertex,
    Instance,
}
#[derive(Clone, Copy, Debug)]
pub enum VertexFormat {
    Float32(u8),
    Float16(u8),
    Uint32(u8),
    Sint32(u8),
    Uint16(u8),
    Sint16(u8),
    Uint8(u8),
    Sint8(u8),
    Unorm8(u8),
    Snorm8(u8),
    Unorm16(u8),
    Snorm16(u8),
    Float64(u8),
    Uint64(u8),
}
#[derive(Clone, Debug)]
pub struct VertexAttribute {
    pub location: u32,
    pub offset: u64,
    pub format: VertexFormat,
}
#[derive(Clone, Debug)]
pub struct VertexBufferLayout {
    pub stride: u64,
    pub step_mode: VertexStepMode,
    pub attributes: Vec<VertexAttribute>,
}
/// Front-facing winding after viewport mapping, in framebuffer coordinates.
#[derive(Clone, Copy, Debug)]
pub enum FrontFace {
    Ccw,
    Cw,
}
#[derive(Clone, Copy, Debug)]
pub enum CullMode {
    None,
    Front,
    Back,
}
#[derive(Clone, Copy, Debug)]
pub enum PolygonMode {
    Fill,
    Line,
    Point,
}
#[derive(Clone, Copy, Debug)]
pub struct PrimitiveState {
    pub topology: PrimitiveTopology,
    pub strip_index_format: Option<IndexFormat>,
    pub front_face: FrontFace,
    pub cull_mode: CullMode,
    pub polygon_mode: PolygonMode,
    pub unclipped_depth: bool,
    pub conservative: bool,
}
#[derive(Clone, Copy, Debug)]
pub enum BlendFactor {
    Zero,
    One,
    SrcColor,
    OneMinusSrcColor,
    DstColor,
    OneMinusDstColor,
    SrcAlpha,
    OneMinusSrcAlpha,
    DstAlpha,
    OneMinusDstAlpha,
    Constant,
    OneMinusConstant,
    SrcAlphaSaturated,
    Src1,
    OneMinusSrc1,
}
#[derive(Clone, Copy, Debug)]
pub enum BlendOperation {
    Add,
    Subtract,
    ReverseSubtract,
    Min,
    Max,
}
#[derive(Clone, Copy, Debug)]
pub struct BlendComponent {
    pub src_factor: BlendFactor,
    pub dst_factor: BlendFactor,
    pub operation: BlendOperation,
}
#[derive(Clone, Copy, Debug)]
pub struct BlendState {
    pub color: BlendComponent,
    pub alpha: BlendComponent,
}
#[derive(Clone, Debug)]
pub struct ColorTargetState {
    pub format: TextureFormat,
    pub blend: Option<BlendState>,
    pub write_mask: u8,
}
#[derive(Clone, Copy, Debug)]
pub enum StencilOperation {
    Keep,
    Zero,
    Replace,
    Invert,
    IncrementClamp,
    DecrementClamp,
    IncrementWrap,
    DecrementWrap,
}
#[derive(Clone, Copy, Debug)]
pub struct StencilFaceState {
    pub compare: CompareFunction,
    pub fail: StencilOperation,
    pub depth_fail: StencilOperation,
    pub pass: StencilOperation,
}
#[derive(Clone, Copy, Debug)]
pub struct DepthStencilState {
    pub format: TextureFormat,
    pub depth_write_enabled: bool,
    pub depth_compare: CompareFunction,
    pub stencil_front: StencilFaceState,
    pub stencil_back: StencilFaceState,
    pub stencil_read_mask: u32,
    pub stencil_write_mask: u32,
    pub depth_bias_constant: i32,
    pub depth_bias_slope_scale: f32,
    pub depth_bias_clamp: f32,
}
#[derive(Clone, Copy, Debug)]
pub struct MultisampleState {
    pub count: u32,
    pub mask: u64,
    pub alpha_to_coverage_enabled: bool,
}
#[derive(Clone)]
pub struct PipelineStage<B: Backend> {
    pub module: Shader<B>,
    pub entry: String,
    pub constants: Vec<(String, SpecializationValue)>,
}
#[derive(Clone, Copy, Debug)]
pub enum SpecializationValue {
    Bool(bool),
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F16Bits(u16),
    F32(f32),
    F64(f64),
}
#[derive(Clone)]
pub struct RasterPipelineDesc<B: Backend> {
    pub label: Option<String>,
    pub layout: PipelineLayout<B>,
    pub vertex: PipelineStage<B>,
    pub fragment: Option<PipelineStage<B>>,
    pub vertex_buffers: Vec<VertexBufferLayout>,
    pub color_targets: Vec<Option<ColorTargetState>>,
    pub primitive: PrimitiveState,
    pub depth_stencil: Option<DepthStencilState>,
    pub multisample: MultisampleState,
    pub multiview: Option<u32>,
    pub cache: Option<PipelineCache<B>>,
}
#[derive(Clone)]
pub struct ComputePipelineDesc<B: Backend> {
    pub label: Option<String>,
    pub layout: PipelineLayout<B>,
    pub compute: PipelineStage<B>,
    pub required_subgroup_size: Option<u32>,
    pub cache: Option<PipelineCache<B>>,
}

pub trait PipelineBackend: Backend {
    type RasterPipelineFuture<'a>: Future<Output = Result<Self::NativeRasterPipeline>> + 'a
    where
        Self: 'a;
    fn create_raster_pipeline<'a>(
        device: &'a Self::NativeDevice,
        desc: &'a RasterPipelineDesc<Self>,
    ) -> Self::RasterPipelineFuture<'a>;
}
impl<B: PipelineBackend> Device<B> {
    pub async fn raster_pipeline(&self, desc: RasterPipelineDesc<B>) -> Result<RasterPipeline<B>> {
        // Pipeline pieces may be structurally compatible while belonging to
        // different devices. Validate provenance before a backend starts a
        // potentially expensive asynchronous native pipeline compilation.
        if desc.layout.device != self.id
            || desc.vertex.module.device != self.id
            || desc
                .fragment
                .as_ref()
                .is_some_and(|s| s.module.device != self.id)
            || desc.cache.as_ref().is_some_and(|c| c.device != self.id)
        {
            return Err(Error::WrongDevice);
        }
        if desc.color_targets.len() > self.caps.limits.max_color_attachments as usize
            || desc.vertex_buffers.len() > self.caps.limits.max_vertex_buffers as usize
            || desc
                .vertex_buffers
                .iter()
                .map(|buffer| buffer.attributes.len())
                .sum::<usize>()
                > self.caps.limits.max_vertex_attributes as usize
        {
            return Err(Error::InvalidRange);
        }
        if let Some(views) = desc.multiview {
            // A nonzero count is required because a zero-mask pipeline has no
            // native attachment interpretation on several APIs.
            if !self.caps.multiview {
                return Err(Error::Unsupported("multiview"));
            }
            if views == 0 || views > self.caps.limits.max_multiview_views {
                return Err(Error::InvalidRange);
            }
        }
        // SAFETY: shader and layout provenance were checked against this device.
        Ok(unsafe {
            RasterPipeline::from_native(
                self.id,
                B::create_raster_pipeline(&self.native, &desc).await?,
            )
        })
    }
}

pub trait CacheBackend: Backend {
    fn create_cache(
        device: &Self::NativeDevice,
        data: Option<&[u8]>,
    ) -> Result<Self::NativePipelineCache>;
    fn serialize_cache(cache: &Self::NativePipelineCache) -> Result<Vec<u8>>;
}
impl<B: CacheBackend> Device<B> {
    pub fn create_pipeline_cache(&self, data: Option<&[u8]>) -> Result<PipelineCache<B>> {
        // SAFETY: the SPI created this cache for this device.
        Ok(unsafe { PipelineCache::from_native(self.id, B::create_cache(&self.native, data)?) })
    }
    pub fn serialize_pipeline_cache(&self, cache: &PipelineCache<B>) -> Result<Vec<u8>> {
        if cache.device != self.id {
            return Err(Error::WrongDevice);
        }
        B::serialize_cache(&cache.native)
    }
}
