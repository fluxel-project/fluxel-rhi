//! Small, deliberately constrained glTF 2.0 loader for the example assets.
//!
//! It accepts only JSON `.gltf` files whose buffers are embedded base64 data
//! URIs.  Output vertices have the exact `vkglTF::Vertex` ABI used by
//! SaschaWillems/Vulkan: 96 bytes, containing position, normal, UV, colour,
//! joints, weights, and tangent in that order.

use std::{error::Error, fmt};

use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde_json::Value;

/// `vkglTF::Vertex` byte stride.
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
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}
impl fmt::Display for GltfError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl Error for GltfError {}

/// Loads every primitive reachable from the active scene.
pub fn load_embedded_model(bytes: &[u8], options: LoadOptions) -> Result<EmbeddedModel, GltfError> {
    let document: Value = serde_json::from_slice(bytes)
        .map_err(|error| GltfError::new(format!("invalid glTF JSON: {error}")))?;
    let buffers = decode_buffers(&document)?;
    let images = decode_images(&document, &buffers)?;
    let samplers = decode_samplers(&document)?;
    let textures = decode_textures(&document, images.len(), samplers.len())?;
    let materials = decode_materials(&document, textures.len())?;
    let nodes = array(&document, "nodes")?;
    let roots: Vec<usize> = match document.get("scene").and_then(Value::as_u64) {
        Some(scene) => array_index(array(&document, "scenes")?, scene as usize, "scene")?
            .get("nodes")
            .and_then(Value::as_array)
            .ok_or_else(|| GltfError::new("active glTF scene has no nodes"))?
            .iter()
            .map(index)
            .collect::<Result<_, _>>()?,
        None => (0..nodes.len())
            .filter(|candidate| {
                !nodes.iter().any(|node| {
                    node.get("children")
                        .and_then(Value::as_array)
                        .is_some_and(|children| {
                            children
                                .iter()
                                .any(|child| child.as_u64() == Some(*candidate as u64))
                        })
                })
            })
            .collect(),
    };
    let mut output = EmbeddedModel {
        images,
        samplers,
        textures,
        materials,
        ..EmbeddedModel::default()
    };
    for root in roots {
        walk_node(
            &document,
            &buffers,
            root,
            Mat4::IDENTITY,
            options,
            &mut output,
        )?;
    }
    Ok(output)
}

/// Loads the first primitive of the first node reachable from the active scene.
///
/// This is the efficient entry point for `rock01.gltf` and `lavaplanet.gltf`.
pub fn load_embedded_first_mesh(bytes: &[u8]) -> Result<EmbeddedModel, GltfError> {
    let mut model = load_embedded_model(bytes, LoadOptions::CPP_PORT)?;
    let first = *model
        .primitives
        .first()
        .ok_or_else(|| GltfError::new("glTF scene contains no primitives"))?;
    let vertex_start = first.first_vertex as usize * VKGLTF_VERTEX_STRIDE;
    let vertex_end = vertex_start + first.vertex_count as usize * VKGLTF_VERTEX_STRIDE;
    let index_start = first.first_index as usize * 4;
    let index_end = index_start + first.index_count as usize * 4;
    let mut indices = Vec::with_capacity(index_end - index_start);
    for index in model.indices[index_start..index_end].chunks_exact(4) {
        let rebased = u32::from_le_bytes(index.try_into().expect("u32 index")) - first.first_vertex;
        indices.extend_from_slice(&rebased.to_le_bytes());
    }
    model.vertices = model.vertices[vertex_start..vertex_end].to_vec();
    model.indices = indices;
    model.primitives = vec![DrawPrimitive {
        first_index: 0,
        first_vertex: 0,
        index_count: first.index_count,
        vertex_count: first.vertex_count,
        material_index: first.material_index,
    }];
    Ok(model)
}

fn walk_node(
    document: &Value,
    buffers: &[Vec<u8>],
    node_index: usize,
    parent: Mat4,
    options: LoadOptions,
    output: &mut EmbeddedModel,
) -> Result<(), GltfError> {
    let node = array_index(array(document, "nodes")?, node_index, "node")?;
    let world = parent.mul(node_matrix(node)?);
    if let Some(mesh_index) = node.get("mesh").and_then(Value::as_u64) {
        let mesh = array_index(array(document, "meshes")?, mesh_index as usize, "mesh")?;
        for primitive in mesh
            .get("primitives")
            .and_then(Value::as_array)
            .ok_or_else(|| GltfError::new("mesh has no primitives"))?
        {
            append_primitive(document, buffers, primitive, world, options, output)?;
        }
    }
    for child in node
        .get("children")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        walk_node(document, buffers, index(child)?, world, options, output)?;
    }
    Ok(())
}

fn append_primitive(
    document: &Value,
    buffers: &[Vec<u8>],
    primitive: &Value,
    world: Mat4,
    options: LoadOptions,
    output: &mut EmbeddedModel,
) -> Result<(), GltfError> {
    if primitive
        .get("mode")
        .and_then(Value::as_u64)
        .is_some_and(|mode| mode != 4)
    {
        return Err(GltfError::new(
            "only TRIANGLES glTF primitives are supported",
        ));
    }
    let attributes = primitive
        .get("attributes")
        .and_then(Value::as_object)
        .ok_or_else(|| GltfError::new("primitive has no attributes"))?;
    let positions = read_f32(
        document,
        buffers,
        index(
            attributes
                .get("POSITION")
                .ok_or_else(|| GltfError::new("primitive has no POSITION"))?,
        )?,
        false,
    )?;
    if positions.components != 3 {
        return Err(GltfError::new("POSITION must be VEC3"));
    }
    let count = positions.count;
    let normals = optional_f32(document, buffers, attributes.get("NORMAL"), count, 3, false)?;
    let uvs = optional_f32(
        document,
        buffers,
        attributes.get("TEXCOORD_0"),
        count,
        2,
        false,
    )?;
    let colors = optional_f32(document, buffers, attributes.get("COLOR_0"), count, 4, true)?;
    let tangents = optional_f32(
        document,
        buffers,
        attributes.get("TANGENT"),
        count,
        4,
        false,
    )?;
    let joints = optional_f32(
        document,
        buffers,
        attributes.get("JOINTS_0"),
        count,
        4,
        false,
    )?;
    let weights = optional_f32(
        document,
        buffers,
        attributes.get("WEIGHTS_0"),
        count,
        4,
        true,
    )?;
    let material_index = primitive
        .get("material")
        .and_then(Value::as_u64)
        .map(|value| value as usize);
    if let Some(material) = material_index {
        if material >= output.materials.len() {
            return Err(GltfError::new("primitive selects a missing material"));
        }
    }
    let material = material_index
        .map(|value| output.materials[value].base_color_factor)
        .unwrap_or([1.0; 4]);
    let first_vertex = output.vertex_count();
    for vertex in 0..count {
        let mut position = [
            positions.at(vertex, 0),
            positions.at(vertex, 1),
            positions.at(vertex, 2),
        ];
        let mut normal = normalize3([
            normals.at(vertex, 0),
            normals.at(vertex, 1),
            normals.at(vertex, 2),
        ]);
        if options.pretransform_vertices {
            position = world.point(position);
            normal = normalize3(world.direction(normal));
        }
        if options.flip_y {
            position[1] *= -1.0;
            normal[1] *= -1.0;
        }
        let mut color = [
            colors.at(vertex, 0),
            colors.at(vertex, 1),
            colors.at(vertex, 2),
            colors.at(vertex, 3),
        ];
        if options.premultiply_vertex_colors {
            for component in 0..4 {
                color[component] *= material[component];
            }
        }
        let record = [
            position[0],
            position[1],
            position[2],
            normal[0],
            normal[1],
            normal[2],
            uvs.at(vertex, 0),
            uvs.at(vertex, 1),
            color[0],
            color[1],
            color[2],
            color[3],
            joints.at(vertex, 0),
            joints.at(vertex, 1),
            joints.at(vertex, 2),
            joints.at(vertex, 3),
            weights.at(vertex, 0),
            weights.at(vertex, 1),
            weights.at(vertex, 2),
            weights.at(vertex, 3),
            tangents.at(vertex, 0),
            tangents.at(vertex, 1),
            tangents.at(vertex, 2),
            tangents.at(vertex, 3),
        ];
        for value in record {
            output.vertices.extend_from_slice(&value.to_le_bytes());
        }
    }
    let first_index = output.index_count();
    let indices = match primitive.get("indices") {
        Some(accessor) => read_indices(document, buffers, index(accessor)?)?,
        None => (0..count as u32).collect(),
    };
    for value in &indices {
        output
            .indices
            .extend_from_slice(&(value + first_vertex).to_le_bytes());
    }
    output.primitives.push(DrawPrimitive {
        first_index,
        index_count: indices.len() as u32,
        first_vertex,
        vertex_count: count as u32,
        material_index,
    });
    Ok(())
}

struct Attribute {
    values: Vec<f32>,
    count: usize,
    components: usize,
}
impl Attribute {
    fn at(&self, vertex: usize, component: usize) -> f32 {
        self.values
            .get(vertex * self.components + component)
            .copied()
            .unwrap_or(match component {
                3 => 1.0,
                _ => 0.0,
            })
    }
}

fn optional_f32(
    document: &Value,
    buffers: &[Vec<u8>],
    accessor: Option<&Value>,
    count: usize,
    components: usize,
    normalized: bool,
) -> Result<Attribute, GltfError> {
    match accessor {
        Some(accessor) => {
            let attribute = read_f32(document, buffers, index(accessor)?, normalized)?;
            if attribute.count != count
                || !(attribute.components == components
                    || components == 4 && attribute.components == 3)
            {
                return Err(GltfError::new(
                    "primitive attributes have mismatched counts or dimensions",
                ));
            }
            Ok(attribute)
        }
        None => Ok(Attribute {
            values: Vec::new(),
            count,
            components: 0,
        }),
    }
}

fn read_f32(
    document: &Value,
    buffers: &[Vec<u8>],
    accessor: usize,
    force_normalized: bool,
) -> Result<Attribute, GltfError> {
    let raw = accessor_bytes(document, buffers, accessor)?;
    let mut values = Vec::with_capacity(raw.count * raw.components);
    for element in 0..raw.count {
        for component in 0..raw.components {
            values.push(component_f32(
                raw.element(element, component)?,
                raw.component_type,
                raw.normalized || force_normalized,
            )?);
        }
    }
    Ok(Attribute {
        values,
        count: raw.count,
        components: raw.components,
    })
}

fn read_indices(
    document: &Value,
    buffers: &[Vec<u8>],
    accessor: usize,
) -> Result<Vec<u32>, GltfError> {
    let raw = accessor_bytes(document, buffers, accessor)?;
    if raw.components != 1 {
        return Err(GltfError::new("index accessor must be SCALAR"));
    }
    (0..raw.count)
        .map(|element| {
            let bytes = raw.element(element, 0)?;
            match raw.component_type {
                5121 => Ok(bytes[0] as u32),
                5123 => Ok(u16::from_le_bytes(
                    bytes
                        .try_into()
                        .map_err(|_| GltfError::new("bad u16 index"))?,
                ) as u32),
                5125 => Ok(u32::from_le_bytes(
                    bytes
                        .try_into()
                        .map_err(|_| GltfError::new("bad u32 index"))?,
                )),
                _ => Err(GltfError::new(
                    "indices must use UNSIGNED_BYTE, UNSIGNED_SHORT, or UNSIGNED_INT",
                )),
            }
        })
        .collect()
}

struct RawAccessor<'a> {
    data: &'a [u8],
    start: usize,
    stride: usize,
    count: usize,
    components: usize,
    component_type: u64,
    normalized: bool,
    component_size: usize,
}
impl<'a> RawAccessor<'a> {
    fn element(&self, element: usize, component: usize) -> Result<&'a [u8], GltfError> {
        let start = self
            .start
            .checked_add(
                element
                    .checked_mul(self.stride)
                    .ok_or_else(|| GltfError::new("accessor offset overflow"))?,
            )
            .and_then(|offset| offset.checked_add(component * self.component_size))
            .ok_or_else(|| GltfError::new("accessor offset overflow"))?;
        self.data
            .get(start..start + self.component_size)
            .ok_or_else(|| GltfError::new("accessor data is truncated"))
    }
}

fn accessor_bytes<'a>(
    document: &'a Value,
    buffers: &'a [Vec<u8>],
    accessor_index: usize,
) -> Result<RawAccessor<'a>, GltfError> {
    let accessor = array_index(array(document, "accessors")?, accessor_index, "accessor")?;
    if accessor.get("sparse").is_some() {
        return Err(GltfError::new(
            "sparse accessors are outside the selected-example loader",
        ));
    }
    let view = array_index(
        array(document, "bufferViews")?,
        index(
            accessor
                .get("bufferView")
                .ok_or_else(|| GltfError::new("accessor has no bufferView"))?,
        )?,
        "bufferView",
    )?;
    let component_type = number(accessor, "componentType")?;
    let component_size = component_size(component_type)?;
    let components = match string(accessor, "type")? {
        "SCALAR" => 1,
        "VEC2" => 2,
        "VEC3" => 3,
        "VEC4" => 4,
        _ => {
            return Err(GltfError::new(
                "only scalar and vector accessors are supported",
            ));
        }
    };
    let stride = view
        .get("byteStride")
        .and_then(Value::as_u64)
        .unwrap_or((components * component_size) as u64) as usize;
    let buffer = buffers
        .get(number(view, "buffer")? as usize)
        .ok_or_else(|| GltfError::new("bufferView selects a missing buffer"))?;
    let start = view.get("byteOffset").and_then(Value::as_u64).unwrap_or(0) as usize
        + accessor
            .get("byteOffset")
            .and_then(Value::as_u64)
            .unwrap_or(0) as usize;
    Ok(RawAccessor {
        data: buffer,
        start,
        stride,
        count: number(accessor, "count")? as usize,
        components,
        component_type,
        normalized: accessor
            .get("normalized")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        component_size,
    })
}

fn component_f32(bytes: &[u8], ty: u64, normalized: bool) -> Result<f32, GltfError> {
    let value = match ty {
        5126 => {
            return Ok(f32::from_le_bytes(
                bytes
                    .try_into()
                    .map_err(|_| GltfError::new("bad float accessor"))?,
            ));
        }
        5120 => i8::from_le_bytes(
            bytes
                .try_into()
                .map_err(|_| GltfError::new("bad i8 accessor"))?,
        ) as f32,
        5121 => bytes[0] as f32,
        5122 => i16::from_le_bytes(
            bytes
                .try_into()
                .map_err(|_| GltfError::new("bad i16 accessor"))?,
        ) as f32,
        5123 => u16::from_le_bytes(
            bytes
                .try_into()
                .map_err(|_| GltfError::new("bad u16 accessor"))?,
        ) as f32,
        5125 => u32::from_le_bytes(
            bytes
                .try_into()
                .map_err(|_| GltfError::new("bad u32 accessor"))?,
        ) as f32,
        _ => return Err(GltfError::new("unsupported accessor component type")),
    };
    Ok(if normalized {
        match ty {
            5120 => (value / 127.0).max(-1.0),
            5121 => value / 255.0,
            5122 => (value / 32767.0).max(-1.0),
            5123 => value / 65535.0,
            5125 => value / 4_294_967_295.0,
            _ => value,
        }
    } else {
        value
    })
}
fn component_size(ty: u64) -> Result<usize, GltfError> {
    match ty {
        5120 | 5121 => Ok(1),
        5122 | 5123 => Ok(2),
        5125 | 5126 => Ok(4),
        _ => Err(GltfError::new("unsupported accessor component type")),
    }
}

fn decode_buffers(document: &Value) -> Result<Vec<Vec<u8>>, GltfError> {
    array(document, "buffers")?
        .iter()
        .map(|buffer| {
            let uri = string(buffer, "uri")?;
            let (_, payload) = uri
                .split_once(",")
                .ok_or_else(|| GltfError::new("embedded buffer URI has no payload"))?;
            if !uri.starts_with("data:") || !uri.contains(";base64,") {
                return Err(GltfError::new("only base64 data URI buffers are supported"));
            }
            STANDARD
                .decode(payload)
                .map_err(|error| GltfError::new(format!("invalid base64 buffer: {error}")))
        })
        .collect()
}

fn decode_images(document: &Value, buffers: &[Vec<u8>]) -> Result<Vec<EmbeddedImage>, GltfError> {
    let Some(images) = document.get("images").and_then(Value::as_array) else {
        return Ok(Vec::new());
    };
    images
        .iter()
        .map(|image| {
            if let Some(uri) = image.get("uri").and_then(Value::as_str) {
                let (mime_type, bytes) = decode_data_uri(uri)?;
                return Ok(EmbeddedImage { mime_type, bytes });
            }
            let view_index = index(
                image
                    .get("bufferView")
                    .ok_or_else(|| GltfError::new("image has neither uri nor bufferView"))?,
            )?;
            let view = array_index(
                array(document, "bufferViews")?,
                view_index,
                "image bufferView",
            )?;
            let buffer = buffers
                .get(number(view, "buffer")? as usize)
                .ok_or_else(|| GltfError::new("image bufferView selects a missing buffer"))?;
            let offset = view.get("byteOffset").and_then(Value::as_u64).unwrap_or(0) as usize;
            let length = number(view, "byteLength")? as usize;
            let bytes = buffer
                .get(offset..offset.saturating_add(length))
                .ok_or_else(|| GltfError::new("image bufferView bytes are truncated"))?
                .to_vec();
            let mime_type = string(image, "mimeType")?.to_owned();
            Ok(EmbeddedImage { mime_type, bytes })
        })
        .collect()
}

fn decode_data_uri(uri: &str) -> Result<(String, Vec<u8>), GltfError> {
    let (header, payload) = uri
        .split_once(",")
        .ok_or_else(|| GltfError::new("data URI has no payload"))?;
    let mime_type = header
        .strip_prefix("data:")
        .and_then(|value| value.strip_suffix(";base64"))
        .filter(|mime| !mime.is_empty())
        .ok_or_else(|| GltfError::new("only base64 image data URIs are supported"))?;
    let bytes = STANDARD
        .decode(payload)
        .map_err(|error| GltfError::new(format!("invalid base64 image: {error}")))?;
    Ok((mime_type.to_owned(), bytes))
}

fn decode_samplers(document: &Value) -> Result<Vec<EmbeddedSampler>, GltfError> {
    let Some(samplers) = document.get("samplers").and_then(Value::as_array) else {
        return Ok(Vec::new());
    };
    samplers
        .iter()
        .map(|sampler| {
            Ok(EmbeddedSampler {
                mag_filter: sampler
                    .get("magFilter")
                    .and_then(Value::as_u64)
                    .map(|value| value as u32),
                min_filter: sampler
                    .get("minFilter")
                    .and_then(Value::as_u64)
                    .map(|value| value as u32),
                // glTF 2.0 defaults to REPEAT.
                wrap_s: sampler
                    .get("wrapS")
                    .and_then(Value::as_u64)
                    .unwrap_or(10_497) as u32,
                wrap_t: sampler
                    .get("wrapT")
                    .and_then(Value::as_u64)
                    .unwrap_or(10_497) as u32,
            })
        })
        .collect()
}

fn decode_textures(
    document: &Value,
    image_count: usize,
    sampler_count: usize,
) -> Result<Vec<EmbeddedTexture>, GltfError> {
    let Some(textures) = document.get("textures").and_then(Value::as_array) else {
        return Ok(Vec::new());
    };
    textures
        .iter()
        .map(|texture| {
            let image_index = index(
                texture
                    .get("source")
                    .ok_or_else(|| GltfError::new("texture has no image source"))?,
            )?;
            if image_index >= image_count {
                return Err(GltfError::new("texture selects a missing image"));
            }
            let sampler_index = texture.get("sampler").map(index).transpose()?;
            if sampler_index.is_some_and(|sampler| sampler >= sampler_count) {
                return Err(GltfError::new("texture selects a missing sampler"));
            }
            Ok(EmbeddedTexture {
                image_index,
                sampler_index,
            })
        })
        .collect()
}

fn decode_materials(
    document: &Value,
    texture_count: usize,
) -> Result<Vec<EmbeddedMaterial>, GltfError> {
    let Some(materials) = document.get("materials").and_then(Value::as_array) else {
        return Ok(Vec::new());
    };
    materials
        .iter()
        .map(|material| {
            let pbr = material.get("pbrMetallicRoughness");
            let base_color_texture = pbr
                .and_then(|value| value.get("baseColorTexture"))
                .map(texture_index)
                .transpose()?;
            let metallic_roughness_texture = pbr
                .and_then(|value| value.get("metallicRoughnessTexture"))
                .map(texture_index)
                .transpose()?;
            let normal_texture = material
                .get("normalTexture")
                .map(texture_index)
                .transpose()?;
            let occlusion_texture = material
                .get("occlusionTexture")
                .map(texture_index)
                .transpose()?;
            let emissive_texture = material
                .get("emissiveTexture")
                .map(texture_index)
                .transpose()?;
            for texture in [
                base_color_texture,
                metallic_roughness_texture,
                normal_texture,
                occlusion_texture,
                emissive_texture,
            ]
            .into_iter()
            .flatten()
            {
                if texture >= texture_count {
                    return Err(GltfError::new("material selects a missing texture"));
                }
            }
            Ok(EmbeddedMaterial {
                base_color_texture,
                normal_texture,
                metallic_roughness_texture,
                occlusion_texture,
                emissive_texture,
                base_color_factor: material_color_value(pbr)?,
            })
        })
        .collect()
}

fn texture_index(value: &Value) -> Result<usize, GltfError> {
    index(
        value
            .get("index")
            .ok_or_else(|| GltfError::new("material texture reference has no index"))?,
    )
}

fn material_color_value(pbr: Option<&Value>) -> Result<[f32; 4], GltfError> {
    let Some(values) = pbr
        .and_then(|value| value.get("baseColorFactor"))
        .and_then(Value::as_array)
    else {
        return Ok([1.0; 4]);
    };
    if values.len() != 4 {
        return Err(GltfError::new("baseColorFactor must have four elements"));
    }
    let mut color = [0.0; 4];
    for (index, value) in values.iter().enumerate() {
        color[index] = value
            .as_f64()
            .ok_or_else(|| GltfError::new("baseColorFactor must be numeric"))?
            as f32;
    }
    Ok(color)
}
#[derive(Clone, Copy)]
struct Mat4([f32; 16]);
impl Mat4 {
    const IDENTITY: Self = Self([
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ]);
    fn mul(self, other: Self) -> Self {
        let mut out = [0.0; 16];
        for column in 0..4 {
            for row in 0..4 {
                out[column * 4 + row] = (0..4)
                    .map(|k| self.0[k * 4 + row] * other.0[column * 4 + k])
                    .sum();
            }
        }
        Self(out)
    }
    fn point(self, v: [f32; 3]) -> [f32; 3] {
        let x = self.0[0] * v[0] + self.0[4] * v[1] + self.0[8] * v[2] + self.0[12];
        let y = self.0[1] * v[0] + self.0[5] * v[1] + self.0[9] * v[2] + self.0[13];
        let z = self.0[2] * v[0] + self.0[6] * v[1] + self.0[10] * v[2] + self.0[14];
        let w = self.0[3] * v[0] + self.0[7] * v[1] + self.0[11] * v[2] + self.0[15];
        if w != 0.0 {
            [x / w, y / w, z / w]
        } else {
            [x, y, z]
        }
    }
    fn direction(self, v: [f32; 3]) -> [f32; 3] {
        [
            self.0[0] * v[0] + self.0[4] * v[1] + self.0[8] * v[2],
            self.0[1] * v[0] + self.0[5] * v[1] + self.0[9] * v[2],
            self.0[2] * v[0] + self.0[6] * v[1] + self.0[10] * v[2],
        ]
    }
}
fn node_matrix(node: &Value) -> Result<Mat4, GltfError> {
    if let Some(values) = node.get("matrix").and_then(Value::as_array) {
        if values.len() != 16 {
            return Err(GltfError::new("node matrix must contain sixteen values"));
        }
        let mut matrix = [0.0; 16];
        for (index, value) in values.iter().enumerate() {
            matrix[index] = value
                .as_f64()
                .ok_or_else(|| GltfError::new("node matrix must be numeric"))?
                as f32;
        }
        return Ok(Mat4(matrix));
    }
    let translation = vector(node.get("translation"), [0.0, 0.0, 0.0])?;
    let scale = vector(node.get("scale"), [1.0, 1.0, 1.0])?;
    let rotation = vector4(node.get("rotation"), [0.0, 0.0, 0.0, 1.0])?;
    let [x, y, z, w] = rotation;
    let rotation = Mat4([
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y + z * w),
        2.0 * (x * z - y * w),
        0.0,
        2.0 * (x * y - z * w),
        1.0 - 2.0 * (x * x + z * z),
        2.0 * (y * z + x * w),
        0.0,
        2.0 * (x * z + y * w),
        2.0 * (y * z - x * w),
        1.0 - 2.0 * (x * x + y * y),
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]);
    Ok(Mat4([
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        translation[0],
        translation[1],
        translation[2],
        1.0,
    ])
    .mul(rotation)
    .mul(Mat4([
        scale[0], 0.0, 0.0, 0.0, 0.0, scale[1], 0.0, 0.0, 0.0, 0.0, scale[2], 0.0, 0.0, 0.0, 0.0,
        1.0,
    ])))
}
fn normalize3(value: [f32; 3]) -> [f32; 3] {
    let length = (value[0] * value[0] + value[1] * value[1] + value[2] * value[2]).sqrt();
    if length > 0.0 {
        [value[0] / length, value[1] / length, value[2] / length]
    } else {
        value
    }
}
fn vector(value: Option<&Value>, default: [f32; 3]) -> Result<[f32; 3], GltfError> {
    match value {
        None => Ok(default),
        Some(value) => {
            let values = value
                .as_array()
                .ok_or_else(|| GltfError::new("node vector must be an array"))?;
            if values.len() != 3 {
                return Err(GltfError::new("node vector must have three values"));
            }
            Ok([
                values[0]
                    .as_f64()
                    .ok_or_else(|| GltfError::new("node vector must be numeric"))?
                    as f32,
                values[1]
                    .as_f64()
                    .ok_or_else(|| GltfError::new("node vector must be numeric"))?
                    as f32,
                values[2]
                    .as_f64()
                    .ok_or_else(|| GltfError::new("node vector must be numeric"))?
                    as f32,
            ])
        }
    }
}
fn vector4(value: Option<&Value>, default: [f32; 4]) -> Result<[f32; 4], GltfError> {
    match value {
        None => Ok(default),
        Some(value) => {
            let values = value
                .as_array()
                .ok_or_else(|| GltfError::new("node quaternion must be an array"))?;
            if values.len() != 4 {
                return Err(GltfError::new("node quaternion must have four values"));
            }
            Ok([
                values[0]
                    .as_f64()
                    .ok_or_else(|| GltfError::new("node quaternion must be numeric"))?
                    as f32,
                values[1]
                    .as_f64()
                    .ok_or_else(|| GltfError::new("node quaternion must be numeric"))?
                    as f32,
                values[2]
                    .as_f64()
                    .ok_or_else(|| GltfError::new("node quaternion must be numeric"))?
                    as f32,
                values[3]
                    .as_f64()
                    .ok_or_else(|| GltfError::new("node quaternion must be numeric"))?
                    as f32,
            ])
        }
    }
}
fn array<'a>(value: &'a Value, name: &str) -> Result<&'a Vec<Value>, GltfError> {
    value
        .get(name)
        .and_then(Value::as_array)
        .ok_or_else(|| GltfError::new(format!("glTF has no {name} array")))
}
fn array_index<'a>(values: &'a [Value], index: usize, name: &str) -> Result<&'a Value, GltfError> {
    values
        .get(index)
        .ok_or_else(|| GltfError::new(format!("{name} index {index} is out of range")))
}
fn index(value: &Value) -> Result<usize, GltfError> {
    value
        .as_u64()
        .map(|value| value as usize)
        .ok_or_else(|| GltfError::new("glTF index is not unsigned"))
}
fn number(value: &Value, name: &str) -> Result<u64, GltfError> {
    value
        .get(name)
        .and_then(Value::as_u64)
        .ok_or_else(|| GltfError::new(format!("missing numeric {name}")))
}
fn string<'a>(value: &'a Value, name: &str) -> Result<&'a str, GltfError> {
    value
        .get(name)
        .and_then(Value::as_str)
        .ok_or_else(|| GltfError::new(format!("missing string {name}")))
}
