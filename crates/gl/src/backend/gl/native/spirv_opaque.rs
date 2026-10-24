//! Reflection of opaque direct-SPIR-V uniforms which GL may report unnamed.
//!
//! This deliberately reads only `UniformConstant` variables.  Blocks and
//! ordinary uniforms have different GL reflection rules and must not acquire a
//! synthetic sampler or image identity merely because their type graph happens
//! to contain an integer or pointer.

use std::collections::BTreeMap;

use crate::backend::gl::api::{GlError, GlShaderSource};

const OPERATION: &str = "reflect GL SPIR-V opaque uniform";
const MAGIC: u32 = 0x0723_0203;
const UNIFORM_CONSTANT: u32 = 0;

/// A numeric identity for an opaque uniform preserved by the SPIR-V ABI.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct OpaqueUniform {
    pub binding: u32,
    pub location: u32,
    pub array_count: u32,
    pub uniform_type: u32,
    pub image: bool,
}

#[derive(Clone, Copy)]
enum Type {
    Int {
        signed: bool,
    },
    Float,
    Image {
        sampled_type: u32,
        dim: u32,
        depth: u32,
        arrayed: bool,
        ms: bool,
        sampled: u32,
    },
    SampledImage {
        image: u32,
    },
    Pointer {
        storage: u32,
        pointee: u32,
    },
    Array {
        element: u32,
        length: u32,
    },
}

#[derive(Default, Clone, Copy)]
struct Decoration {
    binding: Option<u32>,
    set: Option<u32>,
    location: Option<u32>,
    spec_id: Option<u32>,
}

fn invalid(message: impl Into<String>) -> GlError {
    GlError::Validation {
        operation: OPERATION,
        message: message.into(),
    }
}

/// Recovers sampler and storage-image metadata from direct SPIR-V stages.
pub(super) fn uniforms<'a>(
    sources: impl IntoIterator<Item = &'a GlShaderSource>,
) -> Result<Vec<OpaqueUniform>, GlError> {
    let mut merged = BTreeMap::<(bool, u32, u32), OpaqueUniform>::new();
    let mut by_binding = BTreeMap::<(bool, u32), OpaqueUniform>::new();
    let mut by_location = BTreeMap::<u32, OpaqueUniform>::new();
    let mut location_elements = BTreeMap::<u32, OpaqueUniform>::new();
    for source in sources {
        let Some(spirv) = &source.spirv else { continue };
        let reflected = one(&spirv.words, &spirv.specialization)?;
        for uniform in reflected {
            let key = (uniform.image, uniform.binding, uniform.location);
            if let Some(previous) = by_binding.get(&(uniform.image, uniform.binding)) {
                if previous.location != uniform.location
                    || previous.uniform_type != uniform.uniform_type
                    || previous.array_count != uniform.array_count
                    || previous.image != uniform.image
                {
                    return Err(invalid(format!(
                        "binding {} disagrees between shader stages",
                        uniform.binding
                    )));
                }
            }
            if let Some(previous) = by_location.get(&uniform.location) {
                if previous.binding != uniform.binding
                    || previous.uniform_type != uniform.uniform_type
                    || previous.array_count != uniform.array_count
                    || previous.image != uniform.image
                {
                    return Err(invalid(format!(
                        "location {} aliases different opaque uniforms",
                        uniform.location
                    )));
                }
            }
            let last = uniform
                .location
                .checked_add(uniform.array_count - 1)
                .ok_or_else(|| {
                    invalid(format!(
                        "opaque array at location {} overflows GL locations",
                        uniform.location
                    ))
                })?;
            for element_location in uniform.location..=last {
                if let Some(previous) = location_elements.get(&element_location) {
                    if previous != &uniform {
                        return Err(invalid(format!(
                            "location {element_location} aliases opaque uniform array"
                        )));
                    }
                }
                location_elements.insert(element_location, uniform);
            }
            if let Some(previous) = merged.get(&key) {
                if previous != &uniform {
                    return Err(invalid(format!(
                        "binding {} and location {} disagree between shader stages",
                        uniform.binding, uniform.location
                    )));
                }
            } else {
                merged.insert(key, uniform);
            }
            by_binding.insert((uniform.image, uniform.binding), uniform);
            by_location.insert(uniform.location, uniform);
        }
    }
    Ok(merged.into_values().collect())
}

fn one(words: &[u32], specialization: &[(u32, u32)]) -> Result<Vec<OpaqueUniform>, GlError> {
    if words.len() < 5 || words[0] != MAGIC {
        return Err(invalid("SPIR-V header is invalid"));
    }
    let mut types = BTreeMap::new();
    let mut decorations = BTreeMap::<u32, Decoration>::new();
    let mut constants = BTreeMap::new();
    let mut variables = Vec::new();
    let mut at = 5;
    while at < words.len() {
        let first = words[at];
        let count = (first >> 16) as usize;
        let opcode = first & 0xffff;
        if count == 0
            || at
                .checked_add(count)
                .filter(|end| *end <= words.len())
                .is_none()
        {
            return Err(invalid("SPIR-V instruction has an invalid word count"));
        }
        let p = &words[at + 1..at + count];
        match opcode {
            21 if p.len() >= 3 => {
                types.insert(p[0], Type::Int { signed: p[2] != 0 });
            }
            22 if p.len() >= 2 => {
                types.insert(p[0], Type::Float);
            }
            25 if p.len() >= 7 => {
                types.insert(
                    p[0],
                    Type::Image {
                        sampled_type: p[1],
                        dim: p[2],
                        depth: p[3],
                        arrayed: p[4] != 0,
                        ms: p[5] != 0,
                        sampled: p[6],
                    },
                );
            }
            27 if p.len() >= 2 => {
                types.insert(p[0], Type::SampledImage { image: p[1] });
            }
            28 if p.len() >= 3 => {
                types.insert(
                    p[0],
                    Type::Array {
                        element: p[1],
                        length: p[2],
                    },
                );
            }
            32 if p.len() >= 3 => {
                types.insert(
                    p[0],
                    Type::Pointer {
                        storage: p[1],
                        pointee: p[2],
                    },
                );
            }
            43 | 50 if p.len() >= 3 => {
                constants.insert(p[1], p[2]);
            }
            59 if p.len() >= 3 => {
                variables.push((p[1], p[0], p[2]));
            }
            71 if p.len() >= 3 => {
                let decoration = decorations.entry(p[0]).or_default();
                match p[1] {
                    1 if p.len() >= 3 => decoration.spec_id = Some(p[2]),
                    30 if p.len() >= 3 => decoration.location = Some(p[2]),
                    33 if p.len() >= 3 => decoration.binding = Some(p[2]),
                    34 if p.len() >= 3 => decoration.set = Some(p[2]),
                    _ => {}
                }
            }
            _ => {}
        }
        at += count;
    }
    for (&id, decoration) in &decorations {
        if let Some(spec_id) = decoration.spec_id {
            if let Some((_, value)) = specialization
                .iter()
                .find(|(candidate, _)| *candidate == spec_id)
            {
                constants.insert(id, *value);
            }
        }
    }
    let mut result = Vec::new();
    for (variable, pointer, storage) in variables {
        if storage != UNIFORM_CONSTANT {
            continue;
        }
        let (mut ty, mut array_count) = match types.get(&pointer).copied() {
            Some(Type::Pointer {
                storage: UNIFORM_CONSTANT,
                pointee,
            }) => (pointee, 1_u32),
            Some(Type::Pointer { .. }) => continue,
            _ => {
                return Err(invalid(format!(
                    "opaque variable {variable} has no UniformConstant pointer type"
                )));
            }
        };
        while let Some(Type::Array { element, length }) = types.get(&ty).copied() {
            let length = constants.get(&length).copied().ok_or_else(|| {
                invalid(format!(
                    "opaque array {variable} has no supported constant length"
                ))
            })?;
            if length == 0 {
                return Err(invalid(format!("opaque array {variable} has zero length")));
            }
            array_count = array_count
                .checked_mul(length)
                .ok_or_else(|| invalid(format!("opaque array {variable} length overflows u32")))?;
            ty = element;
        }
        let (image_type, image) = match types.get(&ty).copied() {
            Some(Type::SampledImage { image }) => (image, false),
            Some(Type::Image { sampled: 2, .. }) => (ty, true),
            // A scalar or a non-opaque UniformConstant is not a GL opaque uniform.
            Some(_) => continue,
            None => {
                return Err(invalid(format!(
                    "opaque variable {variable} has an unknown type"
                )));
            }
        };
        let Type::Image {
            sampled_type,
            dim,
            depth,
            arrayed,
            ms,
            sampled,
            ..
        } = types
            .get(&image_type)
            .copied()
            .ok_or_else(|| invalid("opaque image type is missing"))?
        else {
            return Err(invalid("opaque uniform does not resolve to OpTypeImage"));
        };
        let kind = match types.get(&sampled_type) {
            Some(Type::Float) => 0,
            Some(Type::Int { signed: true }) => 1,
            Some(Type::Int { signed: false }) => 2,
            _ => {
                return Err(invalid(format!(
                    "opaque variable {variable} has an unsupported sampled scalar type"
                )));
            }
        };
        let uniform_type = if image {
            if depth != 0 {
                return Err(invalid(
                    "SPIR-V storage image with depth is unsupported by GL",
                ));
            }
            image_token(dim, arrayed, ms, kind)?
        } else {
            if sampled != 1 {
                return Err(invalid(format!(
                    "sampled image {variable} has invalid Sampled operand {sampled}"
                )));
            }
            sampler_token(dim, depth, arrayed, ms, kind)?
        };
        let decoration = decorations.get(&variable).copied().unwrap_or_default();
        if decoration.set.unwrap_or(0) != 0 {
            return Err(invalid(format!(
                "opaque variable {variable} uses descriptor set other than zero"
            )));
        }
        result.push(OpaqueUniform {
            binding: decoration.binding.ok_or_else(|| {
                invalid(format!(
                    "opaque variable {variable} has no Binding decoration"
                ))
            })?,
            location: decoration.location.ok_or_else(|| {
                invalid(format!(
                    "opaque variable {variable} has no Location decoration"
                ))
            })?,
            array_count,
            uniform_type,
            image,
        });
    }
    Ok(result)
}

fn sampler_token(dim: u32, depth: u32, arrayed: bool, ms: bool, kind: u32) -> Result<u32, GlError> {
    if depth > 1 {
        return Err(invalid("SPIR-V sampler has an invalid Depth operand"));
    }
    if depth != 0 && kind != 0 {
        return Err(invalid("integer depth sampler is unsupported"));
    }
    let token = match (dim, arrayed, ms, depth != 0, kind) {
        (0, false, false, false, 0) => 0x8b5d,
        (1, false, false, false, 0) => 0x8b5e,
        (2, false, false, false, 0) => 0x8b5f,
        (3, false, false, false, 0) => 0x8b60,
        (5, false, false, false, 0) => 0x8dc2,
        (1, true, false, false, 0) => 0x8dc1,
        (3, true, false, false, 0) => 0x900c,
        (0, false, false, true, 0) => 0x8b61,
        (1, false, false, true, 0) => 0x8b62,
        (1, true, false, true, 0) => 0x8dc4,
        (3, false, false, true, 0) => 0x8dc5,
        (3, true, false, true, 0) => 0x900d,
        (1, false, true, false, 0) => 0x9108,
        (1, true, true, false, 0) => 0x910b,
        (0, false, false, false, 1) => 0x8dc9,
        (1, false, false, false, 1) => 0x8dca,
        (2, false, false, false, 1) => 0x8dcb,
        (3, false, false, false, 1) => 0x8dcc,
        (5, false, false, false, 1) => glow::INT_SAMPLER_BUFFER,
        (1, true, false, false, 1) => 0x8dcf,
        (3, true, false, false, 1) => 0x900e,
        (1, false, true, false, 1) => 0x9109,
        (1, true, true, false, 1) => 0x910c,
        (0, false, false, false, 2) => 0x8dd1,
        (1, false, false, false, 2) => 0x8dd2,
        (2, false, false, false, 2) => 0x8dd3,
        (3, false, false, false, 2) => 0x8dd4,
        (5, false, false, false, 2) => glow::UNSIGNED_INT_SAMPLER_BUFFER,
        (1, true, false, false, 2) => 0x8dd7,
        (3, true, false, false, 2) => 0x900f,
        (1, false, true, false, 2) => 0x910a,
        (1, true, true, false, 2) => 0x910d,
        _ => {
            return Err(invalid(
                "SPIR-V sampler dimension or flags are unsupported by GL",
            ));
        }
    };
    Ok(token)
}

fn image_token(dim: u32, arrayed: bool, ms: bool, kind: u32) -> Result<u32, GlError> {
    let base = match (dim, arrayed, ms) {
        (0, false, false) => 0x904c,
        (1, false, false) => 0x904d,
        (2, false, false) => 0x904e,
        (3, false, false) => 0x9050,
        (5, false, false) => 0x9051,
        (0, true, false) => 0x9052,
        (1, true, false) => 0x9053,
        (3, true, false) => 0x9054,
        (1, false, true) => 0x9055,
        (1, true, true) => 0x9056,
        _ => {
            return Err(invalid(
                "SPIR-V storage-image dimension or flags are unsupported by GL",
            ));
        }
    };
    Ok(base + kind * 11)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::gl::api::{GlShaderDialect, GlShaderKey, GlShaderStage, GlSpirvSource};

    fn instruction(op: u32, operands: &[u32]) -> Vec<u32> {
        let mut out = vec![((operands.len() as u32 + 1) << 16) | op];
        out.extend_from_slice(operands);
        out
    }
    fn source(mut body: Vec<u32>, specs: Vec<(u32, u32)>) -> GlShaderSource {
        let mut words = vec![MAGIC, 0x10000, 0, 32, 0];
        words.append(&mut body);
        GlShaderSource {
            stage: GlShaderStage::Fragment,
            dialect: GlShaderDialect::Desktop { version: 460 },
            entry_point: "main".into(),
            source_hash: GlShaderKey([0; 32]),
            text: String::new(),
            spirv: Some(GlSpirvSource {
                words,
                specialization: specs,
            }),
            debug_name: None,
        }
    }
    fn sampler(variable: u32, binding: u32, location: u32) -> Vec<u32> {
        let mut v = Vec::new();
        v.extend(instruction(22, &[1, 32]));
        v.extend(instruction(25, &[2, 1, 1, 0, 0, 0, 1, 0]));
        v.extend(instruction(27, &[3, 2]));
        v.extend(instruction(32, &[4, 0, 3]));
        v.extend(instruction(59, &[4, variable, 0]));
        v.extend(instruction(71, &[variable, 33, binding]));
        v.extend(instruction(71, &[variable, 30, location]));
        v
    }
    #[test]
    fn opaque_uniforms_sort_binding_not_declaration_order() {
        let mut body = sampler(10, 4, 8);
        body.extend(sampler(11, 1, 2));
        let got = uniforms([&source(body, vec![])]).unwrap();
        assert_eq!(
            got.iter().map(|u| u.binding).collect::<Vec<_>>(),
            vec![1, 4]
        );
    }
    #[test]
    fn ubo_before_sampler_is_not_opaque() {
        let mut body = instruction(32, &[7, 2, 1]);
        body.extend(instruction(59, &[7, 9, 2]));
        body.extend(sampler(10, 1, 0));
        assert_eq!(uniforms([&source(body, vec![])]).unwrap().len(), 1);
    }
    #[test]
    fn storage_image_is_distinct_from_sampler() {
        let mut body = Vec::new();
        body.extend(instruction(22, &[1, 32]));
        body.extend(instruction(25, &[2, 1, 1, 0, 0, 0, 2, 0]));
        body.extend(instruction(32, &[3, 0, 2]));
        body.extend(instruction(59, &[3, 4, 0]));
        body.extend(instruction(71, &[4, 33, 3]));
        body.extend(instruction(71, &[4, 30, 7]));
        let got = uniforms([&source(body, vec![])]).unwrap();
        assert_eq!(got[0].uniform_type, 0x904d);
        assert!(got[0].image);
    }
    #[test]
    fn specialization_controls_array_length() {
        // Replace the variable pointer with an array pointer in a fresh module.
        let mut a = Vec::new();
        a.extend(instruction(22, &[1, 32]));
        a.extend(instruction(25, &[2, 1, 1, 0, 0, 0, 1, 0]));
        a.extend(instruction(27, &[3, 2]));
        a.extend(instruction(21, &[4, 32, 0]));
        a.extend(instruction(50, &[4, 5, 2]));
        a.extend(instruction(71, &[5, 1, 9]));
        a.extend(instruction(28, &[6, 3, 5]));
        a.extend(instruction(32, &[7, 0, 6]));
        a.extend(instruction(59, &[7, 8, 0]));
        a.extend(instruction(71, &[8, 33, 1]));
        a.extend(instruction(71, &[8, 30, 0]));
        assert_eq!(
            uniforms([&source(a, vec![(9, 6)])]).unwrap()[0].array_count,
            6
        );
    }
}
