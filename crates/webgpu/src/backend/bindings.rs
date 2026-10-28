//! WebGPU bind-group layout lowering and feature gates.
//!
//! The conversion is intentionally fail-closed: capabilities which the opened
//! browser device did not establish remain unavailable instead of being lowered
//! to a nearby but semantically different WebGPU layout.

use fluxel_rhi_core::api::{self, Error, Result};
use wasm_bindgen::JsValue;

use super::resource::{descriptor, field, texture_format};

pub(super) fn binding_layout_entry(
    entry: &api::BindGroupLayoutEntry,
    features: &std::collections::BTreeSet<String>,
) -> Result<JsValue> {
    if !matches!(entry.count, api::BindingCount::One)
        || entry.partially_bound
        || entry.nonuniform_indexing
    {
        return Err(Error::Unsupported("WebGPU binding arrays"));
    }
    let native = descriptor();
    field(&native, "binding", JsValue::from_f64(entry.binding as f64))?;
    let mut visibility = 0;
    for stage in &entry.visibility {
        visibility |= match stage {
            api::ShaderStage::Vertex => 1,
            api::ShaderStage::Fragment => 2,
            api::ShaderStage::Compute => 4,
            _ => return Err(Error::Unsupported("WebGPU shader stage visibility")),
        };
    }
    field(&native, "visibility", JsValue::from_f64(visibility as f64))?;
    match &entry.ty {
        api::BindingType::UniformBuffer {
            dynamic_offset,
            min_size,
        }
        | api::BindingType::StorageBuffer {
            dynamic_offset,
            min_size,
            ..
        } => {
            let buffer = descriptor();
            let ty = match &entry.ty {
                api::BindingType::UniformBuffer { .. } => "uniform",
                api::BindingType::StorageBuffer {
                    access: api::StorageAccess::Read,
                    ..
                } => "read-only-storage",
                _ => "storage",
            };
            field(&buffer, "type", JsValue::from_str(ty))?;
            field(
                &buffer,
                "hasDynamicOffset",
                JsValue::from_bool(*dynamic_offset),
            )?;
            field(
                &buffer,
                "minBindingSize",
                JsValue::from_f64(*min_size as f64),
            )?;
            field(&native, "buffer", buffer)?;
        }
        api::BindingType::SampledTexture {
            sample,
            dimension,
            multisampled,
        } => {
            if *multisampled && matches!(sample, api::TextureSampleType::Float) {
                return Err(Error::InvalidState);
            }
            let texture = descriptor();
            field(
                &texture,
                "sampleType",
                JsValue::from_str(match sample {
                    api::TextureSampleType::Float => "float",
                    api::TextureSampleType::UnfilterableFloat => "unfilterable-float",
                    api::TextureSampleType::Sint => "sint",
                    api::TextureSampleType::Uint => "uint",
                    api::TextureSampleType::Depth => "depth",
                }),
            )?;
            field(
                &texture,
                "viewDimension",
                JsValue::from_str(view_dimension(*dimension)),
            )?;
            field(&texture, "multisampled", JsValue::from_bool(*multisampled))?;
            field(&native, "texture", texture)?;
        }
        api::BindingType::Sampler {
            filtering,
            comparison,
        } => {
            let sampler = descriptor();
            field(
                &sampler,
                "type",
                JsValue::from_str(if *comparison {
                    "comparison"
                } else if *filtering {
                    "filtering"
                } else {
                    "non-filtering"
                }),
            )?;
            field(&native, "sampler", sampler)?;
        }
        api::BindingType::StorageTexture {
            access,
            format,
            dimension,
        } => {
            if matches!(access, api::StorageAccess::Atomic) {
                return Err(Error::Unsupported("WebGPU texture atomic binding"));
            }
            if !matches!(access, api::StorageAccess::Write)
                && !features.contains("readonly_and_readwrite_storage_textures")
            {
                return Err(Error::Unsupported(
                    "WebGPU read storage texture language extension",
                ));
            }
            let storage = descriptor();
            field(
                &storage,
                "access",
                JsValue::from_str(match access {
                    api::StorageAccess::Read => "read-only",
                    api::StorageAccess::Write => "write-only",
                    api::StorageAccess::ReadWrite => "read-write",
                    api::StorageAccess::Atomic => unreachable!("atomic texture access rejected"),
                }),
            )?;
            field(
                &storage,
                "format",
                JsValue::from_str(texture_format(*format)?),
            )?;
            field(
                &storage,
                "viewDimension",
                JsValue::from_str(view_dimension(*dimension)),
            )?;
            field(&native, "storageTexture", storage)?;
        }
        _ => return Err(Error::Unsupported("WebGPU binding layout entry")),
    }
    Ok(native)
}

fn view_dimension(dimension: api::TextureViewDimension) -> &'static str {
    match dimension {
        api::TextureViewDimension::D1 => "1d",
        api::TextureViewDimension::D2 => "2d",
        api::TextureViewDimension::D2Array => "2d-array",
        api::TextureViewDimension::Cube => "cube",
        api::TextureViewDimension::CubeArray => "cube-array",
        api::TextureViewDimension::D3 => "3d",
    }
}
