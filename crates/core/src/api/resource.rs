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

// Resource creation Usage is immutable; Uses below is per-operation state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BufferUsage(pub u64);
impl BufferUsage {
    pub const COPY_SRC: Self = Self(1 << 0);
    pub const COPY_DST: Self = Self(1 << 1);
    pub const VERTEX: Self = Self(1 << 2);
    pub const INDEX: Self = Self(1 << 3);
    pub const UNIFORM: Self = Self(1 << 4);
    pub const STORAGE: Self = Self(1 << 5);
    pub const INDIRECT: Self = Self(1 << 6);
    pub const MAP_READ: Self = Self(1 << 7);
    pub const MAP_WRITE: Self = Self(1 << 8);
    pub const QUERY_RESOLVE: Self = Self(1 << 9);
    pub const ACCEL_BUILD_INPUT: Self = Self(1 << 10);
    pub const ACCEL_SCRATCH: Self = Self(1 << 11);
    pub const SHADER_TABLE: Self = Self(1 << 12);
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TextureUsage(pub u64);
impl TextureUsage {
    pub const COPY_SRC: Self = Self(1 << 0);
    pub const COPY_DST: Self = Self(1 << 1);
    pub const TEXTURE_BINDING: Self = Self(1 << 2);
    pub const STORAGE_BINDING: Self = Self(1 << 3);
    pub const RENDER_ATTACHMENT: Self = Self(1 << 4);
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum TextureFormat {
    R8Unorm,
    R8Snorm,
    R8Uint,
    R8Sint,
    Rg8Unorm,
    Rg8Snorm,
    Rg8Uint,
    Rg8Sint,
    Rgba8Unorm,
    Rgba8UnormSrgb,
    Rgba8Snorm,
    Rgba8Uint,
    Rgba8Sint,
    Bgra8Unorm,
    Bgra8UnormSrgb,
    R16Unorm,
    R16Snorm,
    R16Uint,
    R16Sint,
    R16Float,
    Rg16Unorm,
    Rg16Snorm,
    Rg16Uint,
    Rg16Sint,
    Rg16Float,
    Rgba16Unorm,
    Rgba16Snorm,
    Rgba16Uint,
    Rgba16Sint,
    Rgba16Float,
    R32Uint,
    R32Sint,
    R32Float,
    Rg32Uint,
    Rg32Sint,
    Rg32Float,
    Rgba32Uint,
    Rgba32Sint,
    Rgba32Float,
    Rgb10a2Unorm,
    Rgb10a2Uint,
    Rg11b10Ufloat,
    Rgb9e5Ufloat,
    Depth16Unorm,
    Depth24Plus,
    Depth24PlusStencil8,
    Depth32Float,
    Depth32FloatStencil8,
    Stencil8,
    Bc1RgbaUnorm,
    Bc1RgbaUnormSrgb,
    Bc2RgbaUnorm,
    Bc2RgbaUnormSrgb,
    Bc3RgbaUnorm,
    Bc3RgbaUnormSrgb,
    Bc4RUnorm,
    Bc4RSnorm,
    Bc5RgUnorm,
    Bc5RgSnorm,
    Bc6hRgbUfloat,
    Bc6hRgbFloat,
    Bc7RgbaUnorm,
    Bc7RgbaUnormSrgb,
    Etc2Rgb8Unorm,
    Etc2Rgb8UnormSrgb,
    Etc2Rgba8Unorm,
    Etc2Rgba8UnormSrgb,
    Etc2Rgb8A1Unorm,
    Etc2Rgb8A1UnormSrgb,
    EacR11Unorm,
    EacR11Snorm,
    EacRg11Unorm,
    EacRg11Snorm,
    Astc {
        block: [u8; 2],
        srgb: bool,
        hdr: bool,
    },
    R64Uint,
    Rg64Uint,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TextureDimension {
    D1,
    D2,
    D3,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TextureViewDimension {
    D1,
    D2,
    D2Array,
    Cube,
    CubeArray,
    D3,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TextureAspects(pub u8);
impl TextureAspects {
    pub const COLOR: Self = Self(1);
    pub const DEPTH: Self = Self(2);
    pub const STENCIL: Self = Self(4);
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MemoryClass {
    Device,
    Upload,
    Readback,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FilterMode {
    Nearest,
    Linear,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AddressMode {
    ClampToEdge,
    Repeat,
    MirrorRepeat,
    ClampToBorder,
    ClampToZero,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompareFunction {
    Never,
    Less,
    Equal,
    LessEqual,
    Greater,
    NotEqual,
    GreaterEqual,
    Always,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BorderColor {
    TransparentBlack,
    OpaqueBlack,
    OpaqueWhite,
    ZeroInteger,
}

#[derive(Clone, Debug)]
pub struct BufferDesc {
    pub label: Option<String>,
    pub size: u64,
    pub usage: BufferUsage,
    pub memory: MemoryClass,
}
#[derive(Clone, Debug)]
pub struct TextureDesc {
    pub label: Option<String>,
    pub dimension: TextureDimension,
    /// Width, height, and depth. D1/D2 textures use depth 1; array layers are
    /// described separately so D2 arrays are not mistaken for D3 textures.
    pub extent: [u32; 3],
    pub array_layers: u32,
    pub format: TextureFormat,
    pub mip_levels: u32,
    pub samples: u32,
    pub usage: TextureUsage,
    pub view_formats: Vec<TextureFormat>,
}
#[derive(Clone, Debug)]
pub struct TextureViewDesc {
    pub format: TextureFormat,
    pub dimension: TextureViewDimension,
    pub aspects: TextureAspects,
    pub mips: Range<u32>,
    pub layers: Range<u32>,
}
#[derive(Clone, Debug)]
pub struct SamplerDesc {
    pub label: Option<String>,
    pub address: [AddressMode; 3],
    pub mag_filter: FilterMode,
    pub min_filter: FilterMode,
    pub mip_filter: FilterMode,
    pub lod: Range<f32>,
    pub anisotropy: u16,
    pub compare: Option<CompareFunction>,
    pub border: Option<BorderColor>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShaderStage {
    Vertex,
    Fragment,
    Compute,
    Task,
    Mesh,
    RayGeneration,
    AnyHit,
    ClosestHit,
    Miss,
    Intersection,
    Callable,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StorageAccess {
    Read,
    Write,
    ReadWrite,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TextureSampleType {
    Float,
    UnfilterableFloat,
    Sint,
    Uint,
    Depth,
}
#[derive(Clone, Debug)]
pub enum BindingType {
    UniformBuffer {
        dynamic_offset: bool,
        min_size: u64,
    },
    StorageBuffer {
        dynamic_offset: bool,
        min_size: u64,
        access: StorageAccess,
    },
    SampledTexture {
        sample: TextureSampleType,
        dimension: TextureViewDimension,
        multisampled: bool,
    },
    StorageTexture {
        format: TextureFormat,
        dimension: TextureViewDimension,
        access: StorageAccess,
    },
    Sampler {
        filtering: bool,
        comparison: bool,
    },
    AccelerationStructure,
    ExternalTexture,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BindingCount {
    One,
    Fixed(u32),
    RuntimeSized,
}
#[derive(Clone, Debug)]
pub struct BindGroupLayoutEntry {
    pub binding: u32,
    pub visibility: Vec<ShaderStage>,
    pub ty: BindingType,
    pub count: BindingCount,
    pub partially_bound: bool,
    pub nonuniform_indexing: bool,
}
#[derive(Clone, Debug)]
pub struct BindGroupLayoutDesc {
    pub label: Option<String>,
    pub entries: Vec<BindGroupLayoutEntry>,
}
#[derive(Clone)]
pub enum BindingResource<'a, B: Backend> {
    Buffer {
        buffer: &'a Buffer<B>,
        range: Range<u64>,
    },
    TextureView(&'a TextureView<B>),
    Sampler(&'a Sampler<B>),
    Acceleration(&'a AccelerationStructure<B>),
    Array(Vec<Option<BindingResource<'a, B>>>),
}
pub struct BindGroupEntry<'a, B: Backend> {
    pub binding: u32,
    pub resource: BindingResource<'a, B>,
}
pub struct BindGroupDesc<'a, B: Backend> {
    pub label: Option<&'a str>,
    pub layout: &'a BindGroupLayout<B>,
    pub entries: &'a [BindGroupEntry<'a, B>],
}
#[derive(Clone, Debug)]
pub struct PushConstantRange {
    pub stages: Vec<ShaderStage>,
    pub range: Range<u32>,
}
pub struct PipelineLayoutDesc<'a, B: Backend> {
    pub label: Option<&'a str>,
    pub groups: &'a [&'a BindGroupLayout<B>],
    pub push_constants: &'a [PushConstantRange],
}

pub(crate) fn binding_resource_device<B: Backend>(
    v: &BindingResource<'_, B>,
    device: DeviceId,
) -> bool {
    match v {
        BindingResource::Buffer { buffer, .. } => buffer.device == device,
        BindingResource::TextureView(v) => v.device == device,
        BindingResource::Sampler(v) => v.device == device,
        BindingResource::Acceleration(v) => v.device == device,
        BindingResource::Array(a) => a
            .iter()
            .flatten()
            .all(|v| binding_resource_device(v, device)),
    }
}

pub(crate) fn binding_matches<B: Backend>(
    ty: &BindingType,
    resource: &BindingResource<'_, B>,
) -> bool {
    match resource {
        BindingResource::Array(values) => values
            .iter()
            .flatten()
            .all(|value| binding_matches(ty, value)),
        BindingResource::Buffer { .. } => matches!(
            ty,
            BindingType::UniformBuffer { .. } | BindingType::StorageBuffer { .. }
        ),
        BindingResource::TextureView(_) => matches!(
            ty,
            BindingType::SampledTexture { .. } | BindingType::StorageTexture { .. }
        ),
        BindingResource::Sampler(_) => matches!(ty, BindingType::Sampler { .. }),
        BindingResource::Acceleration(_) => matches!(ty, BindingType::AccelerationStructure),
    }
}

pub enum Support<T> {
    Supported(T),
    Unsupported(&'static str),
}
pub struct BufferSupport {
    pub alignment: u64,
    pub max_size: u64,
}
pub struct TextureSupport {
    pub max_extent: [u32; 3],
    pub max_array_layers: u32,
    pub max_mips: u32,
    pub sample_counts: Vec<u32>,
    pub compatible_view_formats: Vec<TextureFormat>,
    pub filterable: bool,
    pub blendable: bool,
    pub storage: bool,
}
