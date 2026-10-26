//! Convert named SPIR-V specialization constants to Vulkan map entries.
use super::super::VulkanShader;
use ash::vk;
use fluxel_rhi_core::api::{self, Error, Result};
use std::collections::{HashMap, HashSet};

fn specialization_ids(words: &[u32]) -> Result<HashMap<u32, u32>> {
    let mut decorations = HashMap::new();
    let mut groups = HashSet::new();
    let mut uses = Vec::new();
    let mut offset = 5;
    while offset < words.len() {
        let count = (words[offset] >> 16) as usize;
        if count == 0 || offset + count > words.len() {
            return Err(Error::Shader("invalid SPIR-V instruction".into()));
        }
        let inst = &words[offset..offset + count];
        match inst[0] & 0xffff {
            71 if count >= 4 && inst[2] == 1 => {
                decorations.insert(inst[1], inst[3]);
            }
            73 if count >= 2 => {
                groups.insert(inst[1]);
            }
            74 if count >= 3 => {
                uses.push((inst[1], inst[2..].to_vec()));
            }
            _ => {}
        }
        offset += count;
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

pub(crate) struct Specialization {
    entries: Vec<vk::SpecializationMapEntry>,
    data: Vec<u8>,
}
impl Specialization {
    pub(crate) fn new(
        shader: &VulkanShader,
        constants: &[(String, api::SpecializationValue)],
    ) -> Result<Self> {
        let mut names = HashMap::new();
        let ids = specialization_ids(shader.words())?;
        let mut types = HashMap::new();
        let mut result_types = HashMap::new();
        let words = shader.words();
        let mut offset = 5;
        while offset < words.len() {
            let count = (words[offset] >> 16) as usize;
            if count == 0 || offset + count > words.len() {
                return Err(Error::Shader("invalid SPIR-V instruction".into()));
            }
            let inst = &words[offset..offset + count];
            match words[offset] & 0xffff {
                5 if count >= 3 => {
                    let bytes: Vec<_> = inst[2..]
                        .iter()
                        .flat_map(|w| w.to_le_bytes())
                        .take_while(|b| *b != 0)
                        .collect();
                    if let Ok(name) = String::from_utf8(bytes) {
                        names.insert(name, inst[1]);
                    }
                }
                20 if count >= 2 => {
                    types.insert(inst[1], (0u32, 32u32));
                }
                21 if count >= 4 => {
                    types.insert(inst[1], (if inst[3] == 0 { 1 } else { 2 }, inst[2]));
                }
                22 if count >= 3 => {
                    types.insert(inst[1], (3, inst[2]));
                }
                48..=50 if count >= 3 => {
                    result_types.insert(inst[2], inst[1]);
                }
                _ => {}
            }
            offset += count;
        }
        let mut entries = Vec::new();
        let mut data = Vec::new();
        for (name, value) in constants {
            let id = names
                .get(name)
                .and_then(|target| ids.get(target))
                .copied()
                .or_else(|| {
                    name.parse::<u32>()
                        .ok()
                        .filter(|id| ids.values().any(|v| v == id))
                })
                .ok_or_else(|| Error::Shader(format!("unknown specialization constant {name}")))?;
            let target = ids
                .iter()
                .find_map(|(target, spec_id)| (*spec_id == id).then_some(target))
                .ok_or(Error::InvalidRange)?;
            let ty = result_types
                .get(target)
                .and_then(|id| types.get(id))
                .copied()
                .ok_or(Error::InvalidRange)?;
            let expected = match value {
                api::SpecializationValue::Bool(_) => (0, 32),
                api::SpecializationValue::U8(_) => (1, 8),
                api::SpecializationValue::I8(_) => (2, 8),
                api::SpecializationValue::U16(_) => (1, 16),
                api::SpecializationValue::I16(_) => (2, 16),
                api::SpecializationValue::U32(_) => (1, 32),
                api::SpecializationValue::I32(_) => (2, 32),
                api::SpecializationValue::U64(_) => (1, 64),
                api::SpecializationValue::I64(_) => (2, 64),
                api::SpecializationValue::F16Bits(_) => (3, 16),
                api::SpecializationValue::F32(_) => (3, 32),
                api::SpecializationValue::F64(_) => (3, 64),
            };
            if ty != expected {
                return Err(Error::Shader(format!(
                    "specialization constant {name} has a different type"
                )));
            }
            if entries
                .iter()
                .any(|entry: &vk::SpecializationMapEntry| entry.constant_id == id)
            {
                return Err(Error::InvalidRange);
            }
            let bytes: Vec<u8> = match value {
                api::SpecializationValue::Bool(v) => u32::from(*v).to_ne_bytes().to_vec(),
                api::SpecializationValue::U8(v) => v.to_ne_bytes().to_vec(),
                api::SpecializationValue::I8(v) => v.to_ne_bytes().to_vec(),
                api::SpecializationValue::U16(v) => v.to_ne_bytes().to_vec(),
                api::SpecializationValue::I16(v) => v.to_ne_bytes().to_vec(),
                api::SpecializationValue::U32(v) => v.to_ne_bytes().to_vec(),
                api::SpecializationValue::I32(v) => v.to_ne_bytes().to_vec(),
                api::SpecializationValue::U64(v) => v.to_ne_bytes().to_vec(),
                api::SpecializationValue::I64(v) => v.to_ne_bytes().to_vec(),
                api::SpecializationValue::F16Bits(v) => v.to_ne_bytes().to_vec(),
                api::SpecializationValue::F32(v) => v.to_ne_bytes().to_vec(),
                api::SpecializationValue::F64(v) => v.to_ne_bytes().to_vec(),
            };
            entries.push(vk::SpecializationMapEntry {
                constant_id: id,
                offset: u32::try_from(data.len()).map_err(|_| Error::InvalidRange)?,
                size: bytes.len(),
            });
            data.extend(bytes);
        }
        Ok(Self { entries, data })
    }
    pub(crate) fn info(&self) -> vk::SpecializationInfo<'_> {
        vk::SpecializationInfo::default()
            .map_entries(&self.entries)
            .data(&self.data)
    }

    /// Native stages without overrides deliberately omit the optional
    /// `VkSpecializationInfo`. Aside from matching the absence of public
    /// overrides, this avoids passing an empty structure through ray-pipeline
    /// creation, which was observed to crash a native driver.
    pub(crate) fn native_info(&self) -> Option<vk::SpecializationInfo<'_>> {
        (!self.entries.is_empty()).then(|| self.info())
    }
}

#[cfg(test)]
#[path = "../../tests/backend/pipeline/specialization.rs"]
mod tests;
