//! Private structural preflight and specialization ABI for direct desktop-GL SPIR-V.
//!
//! Names resolve through `OpName`; a decimal pipeline-constant name denotes an
//! `OpDecorate SpecId` directly, matching the Vulkan direct-artifact contract.
//! This is intentionally not a complete SPIR-V validator: direct artifacts are
//! trusted native GL artifacts, and driver specialization/linking remains the
//! authority for capabilities, interfaces, and all remaining SPIR-V rules.

use std::collections::{HashMap, HashSet};

use fluxel_rhi_core::api::{Error, Result, ShaderStage, SpecializationValue};

use crate::backend::api::GlShaderSource;

pub(crate) fn validate_entry(words: &[u32], entry: &str, stage: ShaderStage) -> Result<()> {
    let wanted = match stage {
        ShaderStage::Vertex => 0,
        ShaderStage::Fragment => 4,
        ShaderStage::Compute => 5,
        _ => return Err(Error::Unsupported("GL SPIR-V shader stage")),
    };
    let mut matched = false;
    for instruction in instructions(words)? {
        if opcode(instruction) == 71
            && instruction.len() >= 4
            && instruction[2] == 34
            && instruction[3] != 0
        {
            return Err(Error::Shader(
                "OpenGL SPIR-V requires DescriptorSet zero".into(),
            ));
        }
        if opcode(instruction) == 15
            && instruction.len() >= 4
            && instruction[1] == wanted
            && string(&instruction[3..]).as_deref() == Some(entry)
        {
            matched = true;
        }
    }
    if matched {
        return Ok(());
    }
    Err(Error::Shader(format!(
        "SPIR-V entry point {entry} does not match the requested GL stage"
    )))
}

/// A real native fragment stage for a depth-only SPIR-V raster pipeline.
/// OpenGL cannot link a SPIR-V vertex executable with a GLSL fallback stage.
pub(crate) fn empty_fragment() -> crate::backend::api::GlSpirvSource {
    crate::backend::api::GlSpirvSource {
        // Shader capability; Logical/GLSL450; Fragment main/OriginLowerLeft;
        // void function with one label and return. No resources or outputs.
        words: vec![
            0x0723_0203,
            0x0001_0000,
            0,
            5,
            0,
            0x0002_0011,
            1,
            0x0003_000e,
            0,
            1,
            0x0005_000f,
            4,
            4,
            0x6e69_616d,
            0,
            0x0003_0010,
            4,
            8,
            0x0002_0013,
            1,
            0x0003_0021,
            2,
            1,
            0x0005_0036,
            1,
            4,
            0,
            2,
            0x0002_00f8,
            3,
            0x0001_00fd,
            0x0001_0038,
        ],
        specialization: Vec::new(),
    }
}

pub(super) fn specialize(
    source: &mut GlShaderSource,
    constants: &[(String, SpecializationValue)],
) -> Result<()> {
    let Some(spirv) = &mut source.spirv else {
        return Ok(());
    };
    if constants.is_empty() {
        return Ok(());
    }
    let ids = specialization_ids(&spirv.words)?;
    let names = names(&spirv.words)?;
    let types = specialization_types(&spirv.words)?;
    let mut selected = HashSet::new();
    let mut values = Vec::with_capacity(constants.len());
    for (name, value) in constants {
        let id = names
            .get(name)
            .and_then(|target| ids.get(target))
            .copied()
            .or_else(|| {
                name.parse::<u32>()
                    .ok()
                    .filter(|id| ids.values().any(|known| known == id))
            })
            .ok_or_else(|| {
                Error::Shader(format!("unknown SPIR-V specialization constant {name}"))
            })?;
        if !selected.insert(id) {
            return Err(Error::InvalidRange);
        }
        let target = ids
            .iter()
            .find_map(|(target, known)| (*known == id).then_some(*target))
            .ok_or(Error::InvalidRange)?;
        if types.get(&target) != Some(&value_type(*value)) {
            return Err(Error::Shader(format!(
                "SPIR-V specialization constant {name} has a different type"
            )));
        }
        values.push((id, specialization_word(*value)?));
    }
    values.sort_unstable_by_key(|(id, _)| *id);
    spirv.specialization = values;
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ScalarType {
    Bool,
    Int { width: u32, signed: bool },
    Float { width: u32 },
}

fn value_type(value: SpecializationValue) -> ScalarType {
    match value {
        SpecializationValue::Bool(_) => ScalarType::Bool,
        SpecializationValue::U8(_) => ScalarType::Int {
            width: 8,
            signed: false,
        },
        SpecializationValue::I8(_) => ScalarType::Int {
            width: 8,
            signed: true,
        },
        SpecializationValue::U16(_) => ScalarType::Int {
            width: 16,
            signed: false,
        },
        SpecializationValue::I16(_) => ScalarType::Int {
            width: 16,
            signed: true,
        },
        SpecializationValue::U32(_) => ScalarType::Int {
            width: 32,
            signed: false,
        },
        SpecializationValue::I32(_) => ScalarType::Int {
            width: 32,
            signed: true,
        },
        SpecializationValue::U64(_) => ScalarType::Int {
            width: 64,
            signed: false,
        },
        SpecializationValue::I64(_) => ScalarType::Int {
            width: 64,
            signed: true,
        },
        SpecializationValue::F16Bits(_) => ScalarType::Float { width: 16 },
        SpecializationValue::F32(_) => ScalarType::Float { width: 32 },
        SpecializationValue::F64(_) => ScalarType::Float { width: 64 },
    }
}

fn specialization_types(words: &[u32]) -> Result<HashMap<u32, ScalarType>> {
    let mut declared = HashMap::new();
    let mut result_types = HashMap::new();
    for inst in instructions(words)? {
        match opcode(inst) {
            20 if inst.len() >= 2 => {
                declared.insert(inst[1], ScalarType::Bool);
            }
            21 if inst.len() >= 4 => {
                declared.insert(
                    inst[1],
                    ScalarType::Int {
                        width: inst[2],
                        signed: inst[3] != 0,
                    },
                );
            }
            22 if inst.len() >= 3 => {
                declared.insert(inst[1], ScalarType::Float { width: inst[2] });
            }
            48..=50 if inst.len() >= 3 => {
                result_types.insert(inst[2], inst[1]);
            }
            _ => {}
        }
    }
    Ok(result_types
        .into_iter()
        .filter_map(|(target, ty)| declared.get(&ty).copied().map(|kind| (target, kind)))
        .collect())
}

fn specialization_word(value: SpecializationValue) -> Result<u32> {
    Ok(match value {
        SpecializationValue::Bool(value) => u32::from(value),
        SpecializationValue::U8(value) => u32::from(value),
        SpecializationValue::I8(value) => value as i32 as u32,
        SpecializationValue::U16(value) => u32::from(value),
        SpecializationValue::I16(value) => value as i32 as u32,
        SpecializationValue::U32(value) => value,
        SpecializationValue::I32(value) => value as u32,
        SpecializationValue::F16Bits(value) => u32::from(value),
        SpecializationValue::F32(value) => value.to_bits(),
        SpecializationValue::U64(_) | SpecializationValue::I64(_) | SpecializationValue::F64(_) => {
            return Err(Error::Unsupported(
                "64-bit GL SPIR-V specialization constant",
            ));
        }
    })
}

fn specialization_ids(words: &[u32]) -> Result<HashMap<u32, u32>> {
    let mut decorations = HashMap::new();
    let mut groups = HashSet::new();
    let mut uses = Vec::new();
    for inst in instructions(words)? {
        match opcode(inst) {
            71 if inst.len() >= 4 && inst[2] == 1 => {
                decorations.insert(inst[1], inst[3]);
            }
            73 if inst.len() >= 2 => {
                groups.insert(inst[1]);
            }
            74 if inst.len() >= 3 => uses.push((inst[1], inst[2..].to_vec())),
            _ => {}
        }
    }
    let mut ids = decorations.clone();
    for (group, targets) in uses {
        if let Some(id) = decorations.get(&group) {
            for target in targets {
                ids.insert(target, *id);
            }
        }
    }
    ids.retain(|target, _| !groups.contains(target));
    Ok(ids)
}
fn names(words: &[u32]) -> Result<HashMap<String, u32>> {
    let mut result = HashMap::new();
    for inst in instructions(words)? {
        if opcode(inst) == 5 && inst.len() >= 3 {
            if let Some(name) = string(&inst[2..]) {
                result.insert(name, inst[1]);
            }
        }
    }
    Ok(result)
}
fn instructions(words: &[u32]) -> Result<Vec<&[u32]>> {
    if words.len() < 5 || words[0] != 0x0723_0203 {
        return Err(Error::Shader("invalid SPIR-V header".into()));
    };
    let mut output = Vec::new();
    let mut offset = 5;
    while offset < words.len() {
        let count = (words[offset] >> 16) as usize;
        if count == 0 || offset + count > words.len() {
            return Err(Error::Shader("invalid SPIR-V instruction".into()));
        };
        output.push(&words[offset..offset + count]);
        offset += count;
    }
    Ok(output)
}
fn opcode(inst: &[u32]) -> u32 {
    inst[0] & 0xffff
}
fn string(words: &[u32]) -> Option<String> {
    let bytes: Vec<_> = words
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .take_while(|byte| *byte != 0)
        .collect();
    String::from_utf8(bytes).ok()
}

#[cfg(test)]
#[path = "../../../tests/backend/rhi/shader/spirv__tests.rs"]
mod tests;
