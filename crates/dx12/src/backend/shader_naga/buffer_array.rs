//! Bounds the patched HLSL buffer-array route before lowering or specialization.
//!
//! Resource-array selection is distinct from buffer member addressing. Only
//! fixed compute arrays are admitted by this route; the existing specialization
//! of runtime arrays must not silently extend that promise.

use fluxel_rhi_core::api::{self, Error, Result};
use std::collections::HashSet;

pub(super) fn validate(module: &naga::Module, entries: &[api::ShaderEntryPoint]) -> Result<()> {
    let mut has_buffer_array = false;
    for (_, global) in module.global_variables.iter() {
        if !matches!(
            global.space,
            naga::AddressSpace::Storage { .. } | naga::AddressSpace::Uniform
        ) {
            continue;
        }
        if let naga::TypeInner::BindingArray { size, .. } = module.types[global.ty].inner {
            has_buffer_array = true;
            if !matches!(size, naga::ArraySize::Constant(_)) {
                return Err(Error::Unsupported(
                    "DX12 WGSL runtime buffer binding arrays",
                ));
            }
        }
    }
    if !has_buffer_array {
        return Ok(());
    }
    if entries
        .iter()
        .any(|entry| entry.stage != api::ShaderStage::Compute)
    {
        return Err(Error::Unsupported(
            "DX12 WGSL buffer binding arrays require compute",
        ));
    }
    // Naga's buffer-length helper is keyed by global, not descriptor element.
    // Do not generate an unindexed GetDimensions call for an array resource.
    if module
        .functions
        .iter()
        .any(|(_, function)| uses_array_length(function))
        || module
            .entry_points
            .iter()
            .any(|entry| uses_array_length(&entry.function))
    {
        return Err(Error::Unsupported(
            "DX12 WGSL buffer binding array length queries",
        ));
    }
    Ok(())
}

fn uses_array_length(function: &naga::Function) -> bool {
    function
        .expressions
        .iter()
        .any(|(_, expression)| matches!(expression, naga::Expression::ArrayLength(_)))
}

/// A fixed WGSL declaration must not borrow a larger runtime/partial layout's
/// register span. Exact layout provenance is required even for static selectors.
pub(super) fn validate_bindings(
    module: &naga::Module,
    info: &naga::valid::ModuleInfo,
    entries: &[(u32, api::BindGroupLayoutEntry)],
) -> Result<()> {
    let mut arrays = module
        .global_variables
        .iter()
        .filter_map(
            |(handle, global)| match (global.space, &module.types[global.ty].inner) {
                (
                    naga::AddressSpace::Storage { .. } | naga::AddressSpace::Uniform,
                    naga::TypeInner::BindingArray {
                        size: naga::ArraySize::Constant(size),
                        ..
                    },
                ) => Some((handle, global, *size)),
                _ => None,
            },
        )
        .peekable();
    if arrays.peek().is_none() {
        return Ok(());
    }
    let nonuniform = nonuniform_buffers(module, info)?;
    for (handle, global, size) in arrays {
        let binding = global.binding.as_ref().ok_or(Error::InvalidState)?;
        let entry = entries
            .iter()
            .find(|(group, entry)| *group == binding.group && entry.binding == binding.binding)
            .map(|(_, entry)| entry)
            .ok_or(Error::Unsupported("DX12 WGSL buffer array layout metadata"))?;
        if entry.count != api::BindingCount::Fixed(size.get())
            || (!entry.nonuniform_indexing && nonuniform.contains(&handle))
            || entry.partially_bound
            || entry.visibility.is_empty()
            || entry
                .visibility
                .iter()
                .any(|stage| *stage != api::ShaderStage::Compute)
            || !matches!(
                entry.ty,
                api::BindingType::StorageBuffer {
                    dynamic_offset: false,
                    ..
                } | api::BindingType::UniformBuffer {
                    dynamic_offset: false,
                    ..
                }
            )
        {
            return Err(Error::Unsupported(
                "DX12 WGSL buffer array requires matching fixed fully-bound compute layout",
            ));
        }
    }
    Ok(())
}

fn nonuniform_buffers(
    module: &naga::Module,
    info: &naga::valid::ModuleInfo,
) -> Result<HashSet<naga::Handle<naga::GlobalVariable>>> {
    let mut buffers = HashSet::new();
    let functions = module
        .functions
        .iter()
        .map(|(handle, function)| (function, &info[handle]));
    let entry_points = module
        .entry_points
        .iter()
        .enumerate()
        .map(|(index, entry)| (&entry.function, info.get_entry_point(index)));
    for (function, facts) in functions.chain(entry_points) {
        for (_, expression) in function.expressions.iter() {
            let naga::Expression::Access { base, index } = *expression else {
                continue;
            };
            if facts[index].uniformity.non_uniform_result.is_none() {
                continue;
            }
            let naga::TypeInner::Pointer {
                base: ty, space, ..
            } = *facts[base].ty.inner_with(&module.types)
            else {
                continue;
            };
            if !matches!(
                space,
                naga::AddressSpace::Uniform | naga::AddressSpace::Storage { .. }
            ) || !matches!(module.types[ty].inner, naga::TypeInner::BindingArray { .. })
            {
                continue;
            }
            let naga::Expression::GlobalVariable(global) = function.expressions[base] else {
                return Err(Error::Unsupported(
                    "DX12 indirect buffer binding array provenance",
                ));
            };
            buffers.insert(global);
        }
    }
    Ok(buffers)
}
