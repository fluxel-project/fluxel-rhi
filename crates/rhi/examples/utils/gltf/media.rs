//! Embedded data-URI buffer and image decoding for example glTF assets.
//!
//! It accepts the deliberately narrow media subset documented by the parent
//! loader and leaves texture format selection to the consuming example.

use super::*;

pub(super) fn decode_buffers(document: &Value) -> Result<Vec<Vec<u8>>, GltfError> {
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

pub(super) fn decode_images(
    document: &Value,
    buffers: &[Vec<u8>],
) -> Result<Vec<EmbeddedImage>, GltfError> {
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

pub(super) fn decode_data_uri(uri: &str) -> Result<(String, Vec<u8>), GltfError> {
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

pub(super) fn decode_samplers(document: &Value) -> Result<Vec<EmbeddedSampler>, GltfError> {
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

pub(super) fn decode_textures(
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

pub(super) fn decode_materials(
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

pub(super) fn texture_index(value: &Value) -> Result<usize, GltfError> {
    index(
        value
            .get("index")
            .ok_or_else(|| GltfError::new("material texture reference has no index"))?,
    )
}

pub(super) fn material_color_value(pbr: Option<&Value>) -> Result<[f32; 4], GltfError> {
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
