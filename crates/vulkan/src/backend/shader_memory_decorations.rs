//! Admission for the legacy SPIR-V coherent/volatile memory decorations.
//!
//! Vulkan's GLSL450 memory model permits these variable decorations for the
//! selected StorageBuffer route. The Vulkan memory model uses per-access
//! semantics instead and rejects these decorations, so route selection is
//! checked before passing the module to the driver.

use fluxel_rhi_core::api::{self, Error, Result};

const OP_MEMORY_MODEL: u32 = 14;
const OP_ENTRY_POINT: u32 = 15;
const OP_VARIABLE: u32 = 59;
const OP_TYPE_POINTER: u32 = 32;
const OP_TYPE_STRUCT: u32 = 30;
const OP_DECORATE: u32 = 71;
const OP_MEMBER_DECORATE: u32 = 72;
const STORAGE_BUFFER: u32 = 12;
const MEMORY_MODEL_GLSL450: u32 = 1;
const DECORATION_BLOCK: u32 = 2;
const DECORATION_VOLATILE: u32 = 21;
const DECORATION_COHERENT: u32 = 23;

pub(super) fn validate(desc: &api::ShaderDesc, words: &[u32]) -> Result<()> {
    let declared_coherent = desc
        .required_features
        .iter()
        .any(|feature| feature == api::SHADER_FEATURE_MEMORY_COHERENT_SPIRV);
    let declared_volatile = desc
        .required_features
        .iter()
        .any(|feature| feature == api::SHADER_FEATURE_MEMORY_VOLATILE_SPIRV);

    let mut memory_model = None;
    let mut entry_point_models = Vec::new();
    let mut pointer_types = std::collections::HashMap::new();
    let mut struct_member_counts = std::collections::HashMap::new();
    let mut block_structs = std::collections::HashSet::new();
    let mut variables = Vec::new();
    let mut object_decorations = Vec::new();
    let mut member_decorations = Vec::new();

    let mut offset = 5usize;
    while offset < words.len() {
        let header = words[offset];
        let count = (header >> 16) as usize;
        let opcode = header & 0xffff;
        let end = offset
            .checked_add(count)
            .filter(|end| count != 0 && *end <= words.len())
            .ok_or_else(|| Error::Shader("invalid SPIR-V instruction length".into()))?;
        let instruction = &words[offset..end];
        match opcode {
            OP_MEMORY_MODEL if count == 3 => {
                if memory_model.replace(instruction[2]).is_some() {
                    return Err(Error::Shader("duplicate SPIR-V memory model".into()));
                }
            }
            OP_ENTRY_POINT if count >= 4 => entry_point_models.push(instruction[1]),
            OP_TYPE_POINTER if count == 4 => {
                pointer_types.insert(instruction[1], (instruction[2], instruction[3]));
            }
            OP_TYPE_STRUCT if count >= 2 => {
                struct_member_counts.insert(instruction[1], count - 2);
            }
            OP_VARIABLE if count >= 4 => {
                variables.push((instruction[1], instruction[2], instruction[3]));
            }
            OP_DECORATE if count == 3 => {
                if instruction[2] == DECORATION_BLOCK {
                    block_structs.insert(instruction[1]);
                } else if is_memory_decoration(instruction[2]) {
                    object_decorations.push((instruction[1], instruction[2]));
                }
            }
            OP_MEMBER_DECORATE if count >= 4 && is_memory_decoration(instruction[3]) => {
                member_decorations.push((instruction[1], instruction[2], instruction[3]));
            }
            _ => {}
        }
        offset = end;
    }

    let has_memory_decorations = !object_decorations.is_empty() || !member_decorations.is_empty();
    let selected_target = if has_memory_decorations {
        let mut storage_buffer_targets = Vec::new();
        for (pointer_type, variable, storage_class) in variables {
            if storage_class != STORAGE_BUFFER {
                continue;
            }
            let (pointer_storage, pointee) =
                pointer_types
                    .get(&pointer_type)
                    .copied()
                    .ok_or(Error::Unsupported(
                        "Vulkan coherent/volatile StorageBuffer variable requires a pointer type",
                    ))?;
            if pointer_storage != STORAGE_BUFFER {
                return Err(Error::Unsupported(
                    "Vulkan coherent/volatile StorageBuffer variable requires a StorageBuffer pointer",
                ));
            }
            if !struct_member_counts.contains_key(&pointee) {
                return Err(Error::Unsupported(
                    "Vulkan coherent/volatile StorageBuffer pointer must target a structure",
                ));
            }
            storage_buffer_targets.push((variable, pointee));
        }
        match storage_buffer_targets.as_slice() {
            [(variable, structure)] if block_structs.contains(structure) => (*variable, *structure),
            [_] => {
                return Err(Error::Unsupported(
                    "Vulkan coherent/volatile StorageBuffer target requires Block decoration",
                ));
            }
            [] => {
                return Err(Error::Unsupported(
                    "Vulkan coherent/volatile decorations require one Block StorageBuffer target",
                ));
            }
            _ => {
                return Err(Error::Unsupported(
                    "Vulkan coherent/volatile route permits exactly one StorageBuffer target",
                ));
            }
        }
    } else {
        (u32::MAX, u32::MAX)
    };

    let mut found_coherent = false;
    let mut found_volatile = false;
    for (target, decoration) in object_decorations {
        if target != selected_target.0 {
            return Err(Error::Unsupported(
                "Vulkan coherent/volatile object decoration must target the StorageBuffer variable",
            ));
        }
        mark_decoration(decoration, &mut found_coherent, &mut found_volatile);
    }
    for (target, _member, decoration) in member_decorations {
        let member_count = (target == selected_target.1)
            .then(|| struct_member_counts.get(&target).copied())
            .flatten();
        if member_count.is_none_or(|count| _member as usize >= count) {
            return Err(Error::Unsupported(
                "Vulkan coherent/volatile member outside StorageBuffer or struct bounds",
            ));
        }
        mark_decoration(decoration, &mut found_coherent, &mut found_volatile);
    }

    if found_coherent != declared_coherent || found_volatile != declared_volatile {
        return Err(Error::Shader(
            "SPIR-V coherent/volatile decorations do not match required_features".into(),
        ));
    }
    if declared_coherent || declared_volatile {
        if memory_model != Some(MEMORY_MODEL_GLSL450) {
            return Err(Error::Unsupported(
                "Vulkan coherent/volatile decorations require GLSL450 memory model",
            ));
        }
        if entry_point_models.is_empty() || entry_point_models.iter().any(|model| *model != 5) {
            return Err(Error::Unsupported(
                "Vulkan coherent/volatile SPIR-V route requires compute entry points",
            ));
        }
    }
    Ok(())
}

fn is_memory_decoration(decoration: u32) -> bool {
    matches!(decoration, DECORATION_VOLATILE | DECORATION_COHERENT)
}

fn mark_decoration(decoration: u32, coherent: &mut bool, volatile: &mut bool) {
    match decoration {
        DECORATION_COHERENT => *coherent = true,
        DECORATION_VOLATILE => *volatile = true,
        _ => unreachable!("caller filters memory decorations"),
    }
}

#[cfg(test)]
#[path = "../tests/backend/shader_memory_decorations.rs"]
mod tests;
