//! Data types exported by the constrained embedded glTF loader.
//!
//! These CPU-side types retain asset identity and decoded geometry only; GPU
//! resource allocation belongs to each example workload.

use super::*;

pub const VKGLTF_VERTEX_STRIDE: usize = 96;

/// Per-primitive range in [`EmbeddedModel`]'s shared buffers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DrawPrimitive {
    pub first_index: u32,
    pub index_count: u32,
    pub first_vertex: u32,
    pub vertex_count: u32,
    /// The glTF material selected by this primitive, if it names one.
    pub material_index: Option<usize>,
}

/// An encoded image as it appears in the glTF document.
///
/// Pixels deliberately remain encoded: callers choose the decoder and GPU
/// format from `mime_type`, rather than the loader silently converting them.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EmbeddedImage {
    pub mime_type: String,
    pub bytes: Vec<u8>,
}

/// glTF sampler fields. Missing values retain the glTF defaults.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EmbeddedSampler {
    pub mag_filter: Option<u32>,
    pub min_filter: Option<u32>,
    pub wrap_s: u32,
    pub wrap_t: u32,
}

/// A glTF texture maps an image source to an optional glTF sampler.
///
/// `vkglTF::Model::loadMaterials` follows `material → texture → source` for
/// every image binding. Its `Texture::fromglTfImage` creates its own Vulkan
/// sampler, so the C++ path does not consume `sampler_index`; it remains here
/// so ports can preserve the document's sampler semantics when needed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EmbeddedTexture {
    pub image_index: usize,
    pub sampler_index: Option<usize>,
}

/// The material references needed by the selected examples.
#[derive(Clone, Debug, PartialEq)]
pub struct EmbeddedMaterial {
    pub base_color_texture: Option<usize>,
    pub normal_texture: Option<usize>,
    pub metallic_roughness_texture: Option<usize>,
    pub occlusion_texture: Option<usize>,
    pub emissive_texture: Option<usize>,
    pub base_color_factor: [f32; 4],
}

/// GPU-ready output of an embedded glTF model load.
#[derive(Clone, Debug, Default)]
pub struct EmbeddedModel {
    /// C++ `vkglTF::Vertex` records, 96 bytes each.
    pub vertices: Vec<u8>,
    /// Globally rebased `u32` little-endian indices.
    pub indices: Vec<u8>,
    pub primitives: Vec<DrawPrimitive>,
    /// Source images, preserving their original encoded bytes and MIME types.
    pub images: Vec<EmbeddedImage>,
    pub samplers: Vec<EmbeddedSampler>,
    pub textures: Vec<EmbeddedTexture>,
    pub materials: Vec<EmbeddedMaterial>,
}

impl EmbeddedModel {
    pub fn vertex_count(&self) -> u32 {
        (self.vertices.len() / VKGLTF_VERTEX_STRIDE) as u32
    }
    pub fn index_count(&self) -> u32 {
        (self.indices.len() / 4) as u32
    }
}

/// The flags used by the selected SaschaWillems C++ ports.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LoadOptions {
    pub pretransform_vertices: bool,
    pub premultiply_vertex_colors: bool,
    pub flip_y: bool,
}

impl LoadOptions {
    /// No C++ loader preprocessing.
    pub const NONE: Self = Self {
        pretransform_vertices: false,
        premultiply_vertex_colors: false,
        flip_y: false,
    };
    /// `PreTransformVertices | FlipY`, used by examples that preserve material colour separately.
    pub const PRETRANSFORM_FLIP_Y: Self = Self {
        pretransform_vertices: true,
        premultiply_vertex_colors: false,
        flip_y: true,
    };
    /// `PreTransformVertices | PreMultiplyVertexColors | FlipY`.
    pub const CPP_PORT: Self = Self {
        pretransform_vertices: true,
        premultiply_vertex_colors: true,
        flip_y: true,
    };

    pub const fn with_pretransform_vertices(mut self, enabled: bool) -> Self {
        self.pretransform_vertices = enabled;
        self
    }
    pub const fn with_premultiply_vertex_colors(mut self, enabled: bool) -> Self {
        self.premultiply_vertex_colors = enabled;
        self
    }
    pub const fn with_flip_y(mut self, enabled: bool) -> Self {
        self.flip_y = enabled;
        self
    }
}

/// A refusal from the constrained embedded-data glTF reader.
#[derive(Debug, Clone)]
pub struct GltfError(String);
impl GltfError {
    pub(super) fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}
impl fmt::Display for GltfError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl Error for GltfError {}
