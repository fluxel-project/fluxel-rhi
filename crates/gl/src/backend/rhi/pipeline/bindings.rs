//! Reconcile shader reflection with the logical pipeline binding ABI.

use super::*;

/// Replaces placeholder ABI names with the names Naga actually emitted for
/// this pair of entry points. Native program reflection needs those names;
/// Naga's explicit storage-image unit is carried beside them for bind groups.
pub(crate) fn with_naga_binding_names(
    mut layout: crate::backend::api::GlPipelineLayout,
    shaders: &[&super::super::types::GlShader],
) -> Result<crate::backend::api::GlPipelineLayout> {
    let mut expanded = Vec::new();
    for binding in std::mem::take(&mut layout.bindings) {
        let mut elements: Vec<_> = shaders
            .iter()
            .flat_map(|shader| &shader.naga_bindings)
            .filter(|value| {
                value.group == binding.location.group && value.binding == binding.location.binding
            })
            .filter_map(|value| value.element)
            .collect();
        elements.sort_unstable();
        elements.dedup();
        if elements.is_empty() {
            expanded.push(binding);
        } else {
            for element in elements {
                if element >= binding.array_count {
                    return Err(Error::InvalidRange);
                }
                let mut item = binding.clone();
                item.element = Some(element);
                item.array_count = 1;
                expanded.push(item);
            }
        }
    }
    layout.bindings = expanded;
    for binding in &mut layout.bindings {
        let mut names = Vec::new();
        let mut image_units = Vec::new();
        for shader in shaders {
            image_units.extend(
                shader
                    .naga_bindings
                    .iter()
                    .filter(|value| {
                        value.group == binding.location.group
                            && value.binding == binding.location.binding
                            && value.element == binding.element
                    })
                    .filter_map(|value| value.image_unit),
            );
            names.extend(
                shader
                    .naga_bindings
                    .iter()
                    .filter(|value| {
                        value.group == binding.location.group
                            && value.binding == binding.location.binding
                            && value.element == binding.element
                    })
                    .map(|value| value.name.clone()),
            );
        }
        if !names.is_empty() {
            names.sort();
            names.dedup();
            binding.names = names;
        }
        if !image_units.is_empty() {
            image_units.sort_unstable();
            image_units.dedup();
            binding.image_unit = image_units.first().copied();
        }
    }
    Ok(layout)
}

pub(crate) fn validate_naga_resource_types(
    layout: &super::super::types::GlPipelineLayout,
    shaders: &[&super::super::types::GlShader],
) -> Result<()> {
    use super::super::types::{GlBindingSlot as L, GlNagaResourceType as R};
    use fluxel_rhi_core::api::TextureSampleType as S;
    for resource in shaders.iter().flat_map(|shader| &shader.naga_bindings) {
        let Some(resource_type) = resource.resource_type else {
            continue;
        };
        let entry = layout
            .groups
            .get(resource.group as usize)
            .and_then(|group| {
                group
                    .entries
                    .iter()
                    .find(|entry| entry.binding == resource.binding)
            })
            .ok_or(Error::InvalidState)?;
        let valid = match (resource_type, entry.slot) {
            (R::Sampler { comparison: a }, L::Sampler { comparison: b, .. }) => a == b,
            (
                R::StorageTexture {
                    dimension: a,
                    format: af,
                    access: aa,
                },
                L::StorageTexture {
                    dimension: b,
                    format: bf,
                    access: ba,
                },
            ) => a == b && af == bf && aa == ba,
            (
                R::SampledTexture {
                    dimension: a,
                    sample: sa,
                    multisampled: ma,
                },
                L::SampledTexture {
                    dimension: b,
                    sample: sb,
                    multisampled: mb,
                },
            ) => a == b && ma == mb && (sa == sb || (sa == S::Float && sb == S::UnfilterableFloat)),
            _ => false,
        };
        if !valid {
            return Err(Error::InvalidState);
        }
    }
    Ok(())
}
