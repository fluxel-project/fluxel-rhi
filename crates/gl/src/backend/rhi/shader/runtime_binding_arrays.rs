//! Bounded runtime binding arrays specialized to the pipeline layout capacity.
//!
//! Each global gets its own interned type: Naga may share the original dynamic
//! type between bindings with different capacities. Core validates each bound
//! group independently and forbids shaders from accessing absent elements.

use super::super::types::GlPipelineLayout;
use fluxel_rhi_core::api::{BindingCount, Error, Result};

pub(super) fn specialize(
    module: &mut naga::Module,
    layout: Option<&GlPipelineLayout>,
) -> Result<()> {
    specialize_with(module, |binding| {
        let entry = layout
            .and_then(|layout| layout.groups.get(binding.group as usize))
            .and_then(|group| {
                group
                    .entries
                    .iter()
                    .find(|entry| entry.binding == binding.binding)
            })
            .ok_or(Error::InvalidState)?;
        match entry.count {
            BindingCount::RuntimeSized { max_count } => Ok(max_count),
            _ => Err(Error::InvalidState),
        }
    })
}

fn specialize_with(
    module: &mut naga::Module,
    mut capacity: impl FnMut(&naga::ResourceBinding) -> Result<u32>,
) -> Result<()> {
    let arrays: Vec<_> = module
        .global_variables
        .iter()
        .filter_map(|(handle, global)| match module.types[global.ty].inner {
            naga::TypeInner::BindingArray {
                base,
                size: naga::ArraySize::Dynamic,
            } => Some((handle, base, global.binding.clone())),
            _ => None,
        })
        .collect();
    for (handle, base, binding) in arrays {
        let binding = binding.ok_or(Error::InvalidState)?;
        let max_count = capacity(&binding)?;
        let count = std::num::NonZeroU32::new(max_count).ok_or(Error::InvalidRange)?;
        let ty = naga::Type {
            name: None,
            inner: naga::TypeInner::BindingArray {
                base,
                size: naga::ArraySize::Constant(count),
            },
        };
        let new_ty = module.types.insert(ty, naga::Span::UNDEFINED);
        module.global_variables[handle].ty = new_ty;
    }
    Ok(())
}

#[cfg(test)]
#[path = "../../../tests/backend/rhi/shader/runtime_binding_arrays__tests.rs"]
mod tests;
