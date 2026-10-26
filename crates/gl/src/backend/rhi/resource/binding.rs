//! Core bind group declarations lowered to the private GL program ABI.
//!
//! Layouts describe logical group/binding keys. Program linking later resolves
//! those keys to native UBO points and combined sampler units. This module
//! never treats a Core binding number as a GL slot or records draw commands.

use fluxel_rhi_core::api::{self, BindingCount, BindingType, Error, Result};

use super::super::types::{GlBindingLayoutEntry, GlBindingSlot};
use crate::backend::api::{GlBindingLocation, GlLogicalBinding, GlShaderResourceKind};

pub(super) fn lower_bind_group_layout(
    desc: &api::BindGroupLayoutDesc,
) -> Result<Vec<GlBindingLayoutEntry>> {
    let mut entries = Vec::with_capacity(desc.entries.len());
    for entry in &desc.entries {
        if matches!(
            entry.count,
            BindingCount::Fixed(0) | BindingCount::RuntimeSized { max_count: 0 }
        ) {
            return Err(Error::InvalidRange);
        }
        if entries
            .iter()
            .any(|value: &GlBindingLayoutEntry| value.binding == entry.binding)
        {
            return Err(Error::InvalidState);
        }
        let slot = match entry.ty {
            BindingType::UniformBuffer {
                dynamic_offset,
                min_size,
            } => GlBindingSlot::UniformBuffer {
                dynamic_offset,
                min_size,
            },
            BindingType::StorageBuffer {
                dynamic_offset,
                min_size,
                access,
            } => GlBindingSlot::StorageBuffer {
                dynamic_offset,
                min_size,
                access,
            },
            BindingType::SampledTexture {
                sample,
                dimension,
                multisampled,
            } if matches!(
                dimension,
                api::TextureViewDimension::D2
                    | api::TextureViewDimension::D1
                    | api::TextureViewDimension::D2Array
                    | api::TextureViewDimension::D3
                    | api::TextureViewDimension::Cube
                    | api::TextureViewDimension::CubeArray
            ) =>
            {
                GlBindingSlot::SampledTexture {
                    dimension,
                    sample,
                    multisampled,
                }
            }
            BindingType::StorageTexture {
                format,
                dimension,
                access,
            } if matches!(
                dimension,
                api::TextureViewDimension::D1
                    | api::TextureViewDimension::D2
                    | api::TextureViewDimension::D2Array
                    | api::TextureViewDimension::D3
            ) =>
            {
                GlBindingSlot::StorageTexture {
                    dimension,
                    format,
                    access,
                }
            }
            BindingType::Sampler {
                filtering,
                comparison,
            } => GlBindingSlot::Sampler {
                filtering,
                comparison,
            },
            _ => return Err(Error::Unsupported("GL binding declaration")),
        };
        entries.push(GlBindingLayoutEntry {
            binding: entry.binding,
            slot,
            count: entry.count,
            partially_bound: entry.partially_bound,
            nonuniform_indexing: entry.nonuniform_indexing,
        });
    }
    entries.sort_by_key(|entry| entry.binding);
    Ok(entries)
}

/// Reject an impossible runtime capacity before shader expansion can allocate
/// per-element IR. More precise per-stage and combined-pair checks occur when
/// linking the selected pipeline; this is only the absolute native ceiling.
pub(super) fn validate_runtime_capacities(
    entries: &[GlBindingLayoutEntry],
    limits: crate::backend::api::GlLimits,
) -> Result<()> {
    for entry in entries {
        let BindingCount::RuntimeSized { max_count } = entry.count else {
            continue;
        };
        let limit = match entry.slot {
            GlBindingSlot::UniformBuffer { .. } => limits
                .max_uniform_buffer_bindings
                .min(limits.max_combined_uniform_blocks),
            GlBindingSlot::StorageBuffer { .. } => limits
                .max_storage_buffer_bindings
                .min(limits.max_combined_storage_blocks),
            GlBindingSlot::StorageTexture { .. } => {
                limits.max_image_units.min(limits.max_combined_image_units)
            }
            GlBindingSlot::SampledTexture { .. } | GlBindingSlot::Sampler { .. } => {
                limits.max_combined_texture_image_units
            }
        };
        if max_count > limit {
            return Err(Error::Unsupported(
                "GL runtime binding array capacity exceeds native resource limit",
            ));
        }
    }
    Ok(())
}

/// Each image/sampler pair uses a linked GLSL sampler unit. `textureLoad`
/// also lowers to a GLSL sampler uniform, but needs only the texture unit.
/// Naga reflection supplies the
/// emitted names before linking. Direct GLSL may use `binding_N` for group
/// zero; higher groups use `group_G_binding_N` to avoid logical-name aliasing.
pub(crate) fn lower_program_bindings(
    group: u32,
    entries: &[GlBindingLayoutEntry],
) -> Result<Vec<GlLogicalBinding>> {
    let textures: Vec<_> = entries
        .iter()
        .filter(|entry| matches!(entry.slot, GlBindingSlot::SampledTexture { .. }))
        .collect();
    let samplers: Vec<_> = entries
        .iter()
        .filter(|entry| matches!(entry.slot, GlBindingSlot::Sampler { .. }))
        .collect();
    let location = |binding| GlBindingLocation { group, binding };
    let name = |binding| match group {
        0 => format!("binding_{binding}"),
        group => format!("group_{group}_binding_{binding}"),
    };
    Ok(entries
        .iter()
        .map(|entry| {
            let (kind, pair, names) = match entry.slot {
                GlBindingSlot::UniformBuffer { .. } => (
                    GlShaderResourceKind::UniformBuffer,
                    None,
                    vec![name(entry.binding)],
                ),
                GlBindingSlot::StorageBuffer { access, .. } => (
                    GlShaderResourceKind::StorageBuffer(match access {
                        api::StorageAccess::Read => {
                            crate::backend::api::GlStorageBufferUsage::ReadOnly
                        }
                        api::StorageAccess::Write | api::StorageAccess::ReadWrite => {
                            crate::backend::api::GlStorageBufferUsage::ReadWrite
                        }
                    }),
                    None,
                    vec![name(entry.binding)],
                ),
                GlBindingSlot::SampledTexture { .. } => (
                    GlShaderResourceKind::Texture,
                    (textures.len() == 1 && samplers.len() == 1)
                        .then(|| location(samplers[0].binding)),
                    vec![name(entry.binding)],
                ),
                GlBindingSlot::StorageTexture { access, .. } => (
                    GlShaderResourceKind::StorageImage(match access {
                        api::StorageAccess::Read => {
                            crate::backend::api::GlStorageImageAccess::ReadOnly
                        }
                        api::StorageAccess::Write => {
                            crate::backend::api::GlStorageImageAccess::WriteOnly
                        }
                        api::StorageAccess::ReadWrite => {
                            crate::backend::api::GlStorageImageAccess::ReadWrite
                        }
                    }),
                    None,
                    vec![name(entry.binding)],
                ),
                GlBindingSlot::Sampler { .. } => (
                    GlShaderResourceKind::Sampler,
                    (textures.len() == 1 && samplers.len() == 1)
                        .then(|| location(textures[0].binding)),
                    textures
                        .first()
                        .filter(|_| textures.len() == 1 && samplers.len() == 1)
                        .map(|texture| vec![name(texture.binding)])
                        .unwrap_or_default(),
                ),
            };
            GlLogicalBinding {
                element: None,
                names,
                location: location(entry.binding),
                kind,
                array_count: match entry.count {
                    BindingCount::One => 1,
                    BindingCount::Fixed(count) => count,
                    BindingCount::RuntimeSized { max_count } => max_count,
                },
                fixed_array: !matches!(entry.count, BindingCount::One),
                image_type: match entry.slot {
                    GlBindingSlot::StorageTexture {
                        dimension, format, ..
                    } => {
                        use api::TextureFormat as F;
                        let dimension = match dimension {
                            api::TextureViewDimension::D2 => {
                                crate::backend::api::GlSampledDimension::D2
                            }
                            api::TextureViewDimension::D2Array => {
                                crate::backend::api::GlSampledDimension::D2Array
                            }
                            api::TextureViewDimension::D3 => {
                                crate::backend::api::GlSampledDimension::D3
                            }
                            _ => unreachable!("validated storage image dimension"),
                        };
                        let kind = match format {
                            F::R8Uint
                            | F::Rg8Uint
                            | F::Rgba8Uint
                            | F::R16Uint
                            | F::Rg16Uint
                            | F::Rgba16Uint
                            | F::R32Uint
                            | F::Rg32Uint
                            | F::Rgba32Uint
                            | F::Rgb10a2Uint
                            | F::R64Uint
                            | F::Rg64Uint => crate::backend::api::GlSampledKind::Uint,
                            F::R8Sint
                            | F::Rg8Sint
                            | F::Rgba8Sint
                            | F::R16Sint
                            | F::Rg16Sint
                            | F::Rgba16Sint
                            | F::R32Sint
                            | F::Rg32Sint
                            | F::Rgba32Sint => crate::backend::api::GlSampledKind::Sint,
                            _ => crate::backend::api::GlSampledKind::Float,
                        };
                        Some((dimension, kind))
                    }
                    _ => None,
                },
                pair,
                image_unit: None,
                sampled_type: match entry.slot {
                    GlBindingSlot::SampledTexture {
                        dimension,
                        sample,
                        multisampled,
                    } => Some(crate::backend::api::GlSampledBindingType::Texture {
                        dimension: match dimension {
                            api::TextureViewDimension::D1 => {
                                crate::backend::api::GlSampledDimension::D1
                            }
                            api::TextureViewDimension::D2 => {
                                crate::backend::api::GlSampledDimension::D2
                            }
                            api::TextureViewDimension::D2Array => {
                                crate::backend::api::GlSampledDimension::D2Array
                            }
                            api::TextureViewDimension::D3 => {
                                crate::backend::api::GlSampledDimension::D3
                            }
                            api::TextureViewDimension::Cube => {
                                crate::backend::api::GlSampledDimension::Cube
                            }
                            api::TextureViewDimension::CubeArray => {
                                crate::backend::api::GlSampledDimension::CubeArray
                            }
                            _ => unreachable!("validated sampled texture dimension"),
                        },
                        kind: match sample {
                            api::TextureSampleType::Float
                            | api::TextureSampleType::UnfilterableFloat => {
                                crate::backend::api::GlSampledKind::Float
                            }
                            api::TextureSampleType::Sint => {
                                crate::backend::api::GlSampledKind::Sint
                            }
                            api::TextureSampleType::Uint => {
                                crate::backend::api::GlSampledKind::Uint
                            }
                            api::TextureSampleType::Depth => {
                                crate::backend::api::GlSampledKind::Depth
                            }
                        },
                        multisampled,
                    }),
                    GlBindingSlot::Sampler { comparison, .. } => {
                        Some(crate::backend::api::GlSampledBindingType::Sampler { comparison })
                    }
                    _ => None,
                },
            }
        })
        .collect())
}

#[cfg(test)]
#[path = "../../../tests/backend/rhi/resource/binding__tests.rs"]
mod tests;
