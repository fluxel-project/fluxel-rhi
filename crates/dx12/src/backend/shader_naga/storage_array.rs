//! Bounds the patched HLSL storage-array route before lowering or specialization.
//!
//! Resource-array selection is distinct from raw-buffer byte addressing. Only
//! fixed compute arrays are admitted by this route; the existing specialization
//! of runtime arrays must not silently extend that promise.

use fluxel_rhi_core::api::{self, Error, Result};

pub(super) fn validate(module: &naga::Module, entries: &[api::ShaderEntryPoint]) -> Result<()> {
    let mut has_storage_array = false;
    for (_, global) in module.global_variables.iter() {
        if !matches!(global.space, naga::AddressSpace::Storage { .. }) {
            continue;
        }
        if let naga::TypeInner::BindingArray { size, .. } = module.types[global.ty].inner {
            has_storage_array = true;
            if !matches!(size, naga::ArraySize::Constant(_)) {
                return Err(Error::Unsupported(
                    "DX12 WGSL runtime storage-buffer arrays",
                ));
            }
        }
    }
    if !has_storage_array {
        return Ok(());
    }
    if entries
        .iter()
        .any(|entry| entry.stage != api::ShaderStage::Compute)
    {
        return Err(Error::Unsupported(
            "DX12 WGSL storage-buffer arrays require compute",
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
            "DX12 WGSL storage-buffer array length queries",
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
    entries: &[(u32, api::BindGroupLayoutEntry)],
) -> Result<()> {
    for (_, global) in module.global_variables.iter() {
        if !matches!(global.space, naga::AddressSpace::Storage { .. }) {
            continue;
        }
        let naga::TypeInner::BindingArray {
            size: naga::ArraySize::Constant(size),
            ..
        } = module.types[global.ty].inner
        else {
            continue;
        };
        let binding = global.binding.as_ref().ok_or(Error::InvalidState)?;
        let entry = entries
            .iter()
            .find(|(group, entry)| *group == binding.group && entry.binding == binding.binding)
            .map(|(_, entry)| entry)
            .ok_or(Error::Unsupported(
                "DX12 WGSL storage array layout metadata",
            ))?;
        if entry.count != api::BindingCount::Fixed(size.get())
            || !entry.nonuniform_indexing
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
                }
            )
        {
            return Err(Error::Unsupported(
                "DX12 WGSL storage array requires matching fixed fully-bound compute layout",
            ));
        }
    }
    Ok(())
}
