//! Core bind group declarations lowered to the private GL program ABI.
//!
//! Layouts describe logical group/binding keys. Program linking later resolves
//! those keys to native UBO points and combined sampler units. This module
//! never treats a Core binding number as a GL slot or records draw commands.

use fluxel_rhi_core::api::{self, BindingCount, BindingType, Error, Result};

use super::super::types::{GlBindingLayoutEntry, GlBindingSlot};
use crate::backend::gl::api::{GlBindingLocation, GlLogicalBinding, GlShaderResourceKind};

pub(super) fn lower_bind_group_layout(
    desc: &api::BindGroupLayoutDesc,
) -> Result<Vec<GlBindingLayoutEntry>> {
    let mut entries = Vec::with_capacity(desc.entries.len());
    for entry in &desc.entries {
        if entry.count != BindingCount::One || entry.partially_bound || entry.nonuniform_indexing {
            return Err(Error::Unsupported("GL bind group arrays or indexing"));
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
            BindingType::SampledTexture {
                sample: api::TextureSampleType::Float,
                dimension,
                multisampled: false,
            } if matches!(
                dimension,
                api::TextureViewDimension::D2 | api::TextureViewDimension::D2Array
            ) =>
            {
                GlBindingSlot::SampledTexture { dimension }
            }
            BindingType::StorageTexture {
                format,
                dimension,
                access,
            } if matches!(dimension, api::TextureViewDimension::D2) => {
                GlBindingSlot::StorageTexture {
                    dimension,
                    format,
                    access,
                }
            }
            BindingType::Sampler {
                comparison: false, ..
            } => GlBindingSlot::Sampler,
            _ => return Err(Error::Unsupported("GL binding declaration")),
        };
        entries.push(GlBindingLayoutEntry {
            binding: entry.binding,
            slot,
        });
    }
    entries.sort_by_key(|entry| entry.binding);
    Ok(entries)
}

/// This GL binding route supports any number of UBOs and at most one sampled
/// texture. Filtering samples require its WGSL sampler binding; `textureLoad`
/// also lowers to a GLSL sampler uniform, but needs only the texture unit and
/// therefore permits a texture-only group. Naga reflection supplies the
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
        .filter(|entry| matches!(entry.slot, GlBindingSlot::Sampler))
        .collect();
    if textures.len() > 1 || samplers.len() > textures.len() {
        return Err(Error::Unsupported("GL texture/sampler pair layout"));
    }
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
                GlBindingSlot::SampledTexture { .. } => (
                    GlShaderResourceKind::CombinedTextureSampler,
                    samplers.first().map(|other| location(other.binding)),
                    vec![name(entry.binding)],
                ),
                GlBindingSlot::StorageTexture { access, .. } => (
                    GlShaderResourceKind::StorageImage(match access {
                        api::StorageAccess::Read => {
                            crate::backend::gl::api::GlStorageImageAccess::ReadOnly
                        }
                        api::StorageAccess::Write => {
                            crate::backend::gl::api::GlStorageImageAccess::WriteOnly
                        }
                        api::StorageAccess::ReadWrite => {
                            crate::backend::gl::api::GlStorageImageAccess::ReadWrite
                        }
                    }),
                    None,
                    vec![name(entry.binding)],
                ),
                GlBindingSlot::Sampler => (
                    GlShaderResourceKind::CombinedTextureSampler,
                    textures.first().map(|other| location(other.binding)),
                    textures
                        .first()
                        .map(|texture| vec![name(texture.binding)])
                        .unwrap_or_default(),
                ),
            };
            GlLogicalBinding {
                names,
                location: location(entry.binding),
                kind,
                array_count: 1,
                pair,
                image_unit: None,
            }
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn group_index_keeps_same_numbered_uniforms_distinct() {
        let entries = [GlBindingLayoutEntry {
            binding: 0,
            slot: GlBindingSlot::UniformBuffer {
                dynamic_offset: false,
                min_size: 16,
            },
        }];
        let group_zero = lower_program_bindings(0, &entries).expect("group zero layout");
        let group_one = lower_program_bindings(1, &entries).expect("group one layout");

        assert_ne!(group_zero[0].location, group_one[0].location);
        assert_eq!(group_zero[0].names, ["binding_0"]);
        assert_eq!(group_one[0].names, ["group_1_binding_0"]);
    }

    #[test]
    fn texture_load_layout_needs_no_sampler_binding() {
        let entries = [GlBindingLayoutEntry {
            binding: 0,
            slot: GlBindingSlot::SampledTexture {
                dimension: api::TextureViewDimension::D2,
            },
        }];
        let lowered = lower_program_bindings(0, &entries).expect("texture-only layout");
        assert_eq!(lowered.len(), 1);
        assert_eq!(lowered[0].pair, None);
        assert!(matches!(
            lowered[0].kind,
            GlShaderResourceKind::CombinedTextureSampler
        ));
    }

    #[test]
    fn texture_load_layout_does_not_require_a_sampler_resource() {
        let entries = [GlBindingLayoutEntry {
            binding: 0,
            slot: GlBindingSlot::SampledTexture {
                dimension: api::TextureViewDimension::D2,
            },
        }];
        let lowered = lower_program_bindings(0, &entries).expect("textureLoad layout");

        assert_eq!(lowered.len(), 1);
        assert_eq!(
            lowered[0].kind,
            GlShaderResourceKind::CombinedTextureSampler
        );
        assert_eq!(lowered[0].pair, None);
        assert_eq!(lowered[0].names, ["binding_0"]);
    }
}
