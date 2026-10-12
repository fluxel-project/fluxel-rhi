use super::*;

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

pub(super) fn walk_node(
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

pub(super) fn append_primitive(
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
        // glTF defaults a missing COLOR_0 attribute to opaque white. Keep this
        // rule local to colour: other optional vertex attributes retain their
        // ABI-specific zero/default-component handling.
        let mut color = if attributes.contains_key("COLOR_0") {
            [
                colors.at(vertex, 0),
                colors.at(vertex, 1),
                colors.at(vertex, 2),
                colors.at(vertex, 3),
            ]
        } else {
            [1.0; 4]
        };
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

#[cfg(test)]
mod tests {
    use super::*;

    const ROCK_WITHOUT_COLOR_0: &[u8] = include_bytes!("../../assets/models/rock01.gltf");

    #[test]
    fn missing_color_0_writes_opaque_white_vertex_colours() {
        let model = load_embedded_model(ROCK_WITHOUT_COLOR_0, LoadOptions::NONE)
            .expect("rock asset without COLOR_0 loads");

        assert!(!model.vertices.is_empty());
        for vertex in model.vertices.chunks_exact(VKGLTF_VERTEX_STRIDE) {
            let color = std::array::from_fn(|component| {
                let offset = 32 + component * 4;
                f32::from_le_bytes(vertex[offset..offset + 4].try_into().expect("colour bytes"))
            });
            assert_eq!(color, [1.0; 4]);
        }
    }
}
