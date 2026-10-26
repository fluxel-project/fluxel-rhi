//! Numeric and textual aliases for linked uniform/storage block resources.

use crate::backend::api::{GlError, GlLogicalBinding, GlPipelineLayout, GlShaderResourceKind};

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
#[path = "../../tests/backend/native/shader_blocks__tests.rs"]
mod tests;
