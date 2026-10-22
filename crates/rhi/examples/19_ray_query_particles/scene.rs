//! Reads the fixed NVIDIA glTF scene and keeps mesh data in source coordinates.
//!
//! The source scene contains five indexed triangle meshes. Instance transforms
//! stay separate so the same transforms can be supplied to the TLAS and shader.
use serde_json::Value;

const GLTF: &str = include_str!("../assets/19_ray_query_particles/ParticleScene.gltf");
const BIN: &[u8] = include_bytes!("../assets/19_ray_query_particles/ParticleScene.bin");

pub struct Mesh {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub indices: Vec<u32>,
    pub transform: [[f32; 4]; 3],
}

/// Expand the reference meshes into world-space triangles for a single BLAS.
/// Each triangle stores three world-space normals for barycentric interpolation.
pub fn world_triangles(meshes: &[Mesh]) -> (Vec<f32>, Vec<f32>) {
    let mut vertices = Vec::new();
    let mut normals = Vec::new();
    for mesh in meshes {
        let position = |index: u32| {
            let p = mesh.positions[index as usize];
            mesh.transform
                .map(|row| row[0] * p[0] + row[1] * p[1] + row[2] * p[2] + row[3])
        };
        for indices in mesh.indices.chunks_exact(3) {
            let [a, b, c] = [
                position(indices[0]),
                position(indices[1]),
                position(indices[2]),
            ];
            vertices.extend_from_slice(&a);
            vertices.extend_from_slice(&b);
            vertices.extend_from_slice(&c);
            // The source nodes contain orthogonal rotation/scale transforms;
            // inverse-transpose is each column divided by its squared length.
            let scale_squared = [0, 1, 2].map(|column| {
                mesh.transform
                    .iter()
                    .map(|row| row[column] * row[column])
                    .sum::<f32>()
            });
            for &index in indices {
                let n = mesh.normals[index as usize];
                for row in mesh.transform {
                    normals.push((0..3).map(|i| row[i] * n[i] / scale_squared[i]).sum());
                }
                normals.push(0.);
            }
        }
    }
    (vertices, normals)
}

fn accessor<'a, 'b>(
    doc: &'b Value,
    bin: &'a [u8],
    index: usize,
) -> Result<(&'a [u8], usize, u64, &'b str), String> {
    let a = doc["accessors"].get(index).ok_or("missing accessor")?;
    let view = doc["bufferViews"]
        .get(a["bufferView"].as_u64().ok_or("missing view")? as usize)
        .ok_or("missing buffer view")?;
    if a.get("sparse").is_some() || view["buffer"].as_u64() != Some(0) {
        return Err("unsupported glTF accessor".into());
    }
    let offset = view["byteOffset"].as_u64().unwrap_or(0) + a["byteOffset"].as_u64().unwrap_or(0);
    let length = view["byteLength"]
        .as_u64()
        .ok_or("missing buffer view length")?;
    let view_start = view["byteOffset"].as_u64().unwrap_or(0);
    let end = view_start.checked_add(length).ok_or("view overflow")?;
    if offset > end || end > bin.len() as u64 {
        return Err("glTF view out of bounds".into());
    }
    Ok((
        &bin[offset as usize..end as usize],
        a["count"].as_u64().ok_or("missing accessor count")? as usize,
        a["componentType"]
            .as_u64()
            .ok_or("missing component type")?,
        a["type"].as_str().ok_or("missing accessor type")?,
    ))
}

fn vec3(doc: &Value, index: usize) -> Result<Vec<[f32; 3]>, String> {
    let (bytes, count, ty, shape) = accessor(doc, BIN, index)?;
    if ty != 5126 || shape != "VEC3" || bytes.len() < count * 12 {
        return Err("invalid position/normal accessor".into());
    }
    Ok(bytes[..count * 12]
        .chunks_exact(12)
        .map(|v| {
            [
                f32::from_le_bytes(v[0..4].try_into().unwrap()),
                f32::from_le_bytes(v[4..8].try_into().unwrap()),
                f32::from_le_bytes(v[8..12].try_into().unwrap()),
            ]
        })
        .collect())
}
fn indices(doc: &Value, index: usize) -> Result<Vec<u32>, String> {
    let (bytes, count, ty, shape) = accessor(doc, BIN, index)?;
    if shape != "SCALAR" {
        return Err("invalid index accessor".into());
    }
    let width = match ty {
        5123 => 2,
        5125 => 4,
        _ => return Err("unsupported index format".into()),
    };
    if bytes.len() < count * width {
        return Err("index accessor out of bounds".into());
    }
    Ok(bytes[..count * width]
        .chunks_exact(width)
        .map(|v| {
            if width == 2 {
                u16::from_le_bytes(v.try_into().unwrap()) as u32
            } else {
                u32::from_le_bytes(v.try_into().unwrap())
            }
        })
        .collect())
}
fn array<const N: usize>(value: &Value, fallback: [f32; N]) -> Result<[f32; N], String> {
    if value.is_null() {
        return Ok(fallback);
    }
    let values = value.as_array().ok_or("invalid transform")?;
    if values.len() != N {
        return Err("invalid transform size".into());
    }
    let mut out = fallback;
    for (dst, src) in out.iter_mut().zip(values) {
        *dst = src.as_f64().ok_or("invalid transform value")? as f32;
    }
    Ok(out)
}
fn transform(node: &Value) -> Result<[[f32; 4]; 3], String> {
    if node.get("matrix").is_some() {
        return Err("matrix transform not supported by this scene".into());
    }
    let [x, y, z, w] = array(&node["rotation"], [0., 0., 0., 1.])?;
    let [sx, sy, sz] = array(&node["scale"], [1., 1., 1.])?;
    let [tx, ty, tz] = array(&node["translation"], [0., 0., 0.])?;
    Ok([
        [
            (1. - 2. * (y * y + z * z)) * sx,
            (2. * (x * y - z * w)) * sy,
            (2. * (x * z + y * w)) * sz,
            tx,
        ],
        [
            (2. * (x * y + z * w)) * sx,
            (1. - 2. * (x * x + z * z)) * sy,
            (2. * (y * z - x * w)) * sz,
            ty,
        ],
        [
            (2. * (x * z - y * w)) * sx,
            (2. * (y * z + x * w)) * sy,
            (1. - 2. * (x * x + y * y)) * sz,
            tz,
        ],
    ])
}

pub fn load() -> Result<Vec<Mesh>, String> {
    let doc: Value = serde_json::from_str(GLTF).map_err(|e| e.to_string())?;
    let mut result = Vec::new();
    for node in doc["nodes"].as_array().ok_or("missing nodes")? {
        let Some(index) = node["mesh"].as_u64() else {
            continue;
        };
        let mesh = doc["meshes"].get(index as usize).ok_or("missing mesh")?;
        let primitives = mesh["primitives"].as_array().ok_or("missing primitives")?;
        if primitives.len() != 1 {
            return Err("unsupported primitive count".into());
        }
        let prim = &primitives[0];
        if prim["mode"].as_u64().unwrap_or(4) != 4 {
            return Err("expected triangles".into());
        }
        let attr = &prim["attributes"];
        let positions = vec3(
            &doc,
            attr["POSITION"].as_u64().ok_or("missing positions")? as usize,
        )?;
        let normals = vec3(
            &doc,
            attr["NORMAL"].as_u64().ok_or("missing normals")? as usize,
        )?;
        let indices = indices(
            &doc,
            prim["indices"].as_u64().ok_or("missing indices")? as usize,
        )?;
        if positions.len() != normals.len()
            || indices.len() % 3 != 0
            || indices.iter().any(|&i| i as usize >= positions.len())
        {
            return Err("invalid mesh indices or normals".into());
        }
        result.push(Mesh {
            positions,
            normals,
            indices,
            transform: transform(node)?,
        });
    }
    if result.len() != 5 {
        return Err("unexpected source scene".into());
    }
    Ok(result)
}
