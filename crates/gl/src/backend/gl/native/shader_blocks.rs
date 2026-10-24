//! Numeric and textual aliases for linked uniform/storage block resources.

use crate::backend::gl::api::{GlError, GlLogicalBinding, GlPipelineLayout, GlShaderResourceKind};

const OP: &str = "assign GL block bindings";

fn validation(message: impl Into<String>) -> GlError {
    GlError::Validation {
        operation: OP,
        message: message.into(),
    }
}

fn namespace(kind: GlShaderResourceKind) -> usize {
    match kind {
        GlShaderResourceKind::UniformBuffer => 0,
        GlShaderResourceKind::StorageBuffer(_) => 1,
        GlShaderResourceKind::Texture | GlShaderResourceKind::CombinedTextureSampler => 2,
        GlShaderResourceKind::Sampler => 3,
        GlShaderResourceKind::StorageImage(_) => 4,
    }
}
fn span(binding: &GlLogicalBinding) -> u32 {
    if binding.fixed_array && binding.element.is_none() {
        binding.array_count
    } else {
        1
    }
}
/// Native binding namespaces are independent; array declarations reserve a
/// contiguous span without expanding a potentially large host-side vector.
pub(super) fn ordinals(
    layout: &GlPipelineLayout,
) -> Result<Vec<(u32, &GlLogicalBinding)>, GlError> {
    let mut bindings: Vec<_> = layout.bindings.iter().collect();
    bindings.sort_by_key(|binding| (binding.location, binding.element));
    let mut next = [0_u32; 5];
    let mut result = Vec::with_capacity(bindings.len());
    for binding in bindings {
        let cursor = &mut next[namespace(binding.kind)];
        result.push((*cursor, binding));
        *cursor = cursor
            .checked_add(span(binding))
            .ok_or_else(|| validation("binding ordinal overflow"))?;
    }
    Ok(result)
}
/// Finds a declaration and element offset in one native binding namespace.
pub(super) fn at(
    layout: &GlPipelineLayout,
    kind: GlShaderResourceKind,
    ordinal: u32,
) -> Result<Option<(&GlLogicalBinding, u32)>, GlError> {
    for (base, binding) in ordinals(layout)? {
        if namespace(binding.kind) == namespace(kind)
            && ordinal >= base
            && ordinal - base < span(binding)
        {
            return Ok(Some((binding, ordinal - base)));
        }
    }
    Ok(None)
}

/// Consumes active native block indices for one logical declaration.
pub(super) fn block_indices(
    binding: &GlLogicalBinding,
    numeric_base: Option<u32>,
    blocks: &mut Vec<(u32, String)>,
) -> Result<Vec<(u32, Vec<u32>)>, GlError> {
    let count = if binding.fixed_array && binding.element.is_none() {
        binding.array_count
    } else {
        1
    };
    let mut result = Vec::new();
    for element in 0..count {
        let logical = binding.element.unwrap_or(element);
        let aliases: Vec<String> = match numeric_base {
            Some(base) => vec![format!(
                "@{}",
                base.checked_add(element)
                    .ok_or_else(|| validation("block binding overflow"))?
            )],
            None => binding
                .names
                .iter()
                .map(|name| {
                    if binding.fixed_array && binding.element.is_none() {
                        format!("{name}[{element}]")
                    } else {
                        name.clone()
                    }
                })
                .collect(),
        };
        let mut indices = Vec::new();
        blocks.retain(|(index, active)| {
            if aliases.iter().any(|alias| alias == active) {
                indices.push(*index);
                false
            } else {
                true
            }
        });
        if !indices.is_empty() {
            result.push((logical, indices));
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::gl::api::{GlBindingLocation, GlShaderResourceKind};
    fn binding(binding: u32, count: u32) -> GlLogicalBinding {
        GlLogicalBinding {
            names: vec![format!("b{binding}")],
            location: GlBindingLocation { group: 0, binding },
            element: None,
            kind: GlShaderResourceKind::UniformBuffer,
            array_count: count,
            fixed_array: count > 1,
            pair: None,
            image_unit: None,
            sampled_type: None,
            image_type: None,
        }
    }
    #[test]
    fn numeric_array_holes_and_duplicate_native_indices_are_consumed() {
        let binding = binding(0, 3);
        let mut blocks = vec![(4, "@10".into()), (5, "@10".into()), (9, "@12".into())];
        assert_eq!(
            block_indices(&binding, Some(10), &mut blocks).unwrap(),
            vec![(0, vec![4, 5]), (2, vec![9])]
        );
        assert!(blocks.is_empty());
    }
    #[test]
    fn array_span_does_not_capture_next_binding() {
        let layout = GlPipelineLayout {
            bindings: vec![binding(0, 3), binding(1, 1)],
        };
        assert_eq!(
            at(&layout, GlShaderResourceKind::UniformBuffer, 2)
                .unwrap()
                .unwrap()
                .1,
            2
        );
        let next = at(&layout, GlShaderResourceKind::UniformBuffer, 3)
            .unwrap()
            .unwrap();
        assert_eq!(next.0.location.binding, 1);
        assert_eq!(next.1, 0);
    }
    #[test]
    fn resource_namespaces_do_not_consume_each_others_native_slots() {
        let mut image = binding(1, 1);
        image.kind = GlShaderResourceKind::StorageImage(
            crate::backend::gl::api::GlStorageImageAccess::WriteOnly,
        );
        let layout = GlPipelineLayout {
            bindings: vec![binding(0, 8), image],
        };
        assert_eq!(
            at(
                &layout,
                GlShaderResourceKind::StorageImage(
                    crate::backend::gl::api::GlStorageImageAccess::WriteOnly
                ),
                0
            )
            .unwrap()
            .unwrap()
            .0
            .location
            .binding,
            1
        );
    }
}
