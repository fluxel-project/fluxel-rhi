//! Explicit direct-SPIR-V vertex interface locations.

use crate::backend::gl::api::{GlError, GlShaderSource, GlShaderStage, GlVertexInputReflection};
use std::collections::BTreeMap;
const MAGIC: u32 = 0x0723_0203;
fn invalid(message: impl Into<String>) -> GlError {
    GlError::Validation {
        operation: "reflect GL SPIR-V vertex input",
        message: message.into(),
    }
}

/// Recovers explicitly located scalar and vector vertex inputs from SPIR-V.
///
/// Matrices deliberately remain outside this helper: their location footprint
/// depends on the scalar width and profile, while the current Core attribute
/// contract only needs the scalar/vector vocabulary.  Inputs without a direct
/// location are skipped because they are normally built-in structs whose member
/// decorations are not a user vertex-attribute ABI.
pub(super) fn inputs<'a>(
    sources: impl IntoIterator<Item = &'a GlShaderSource>,
) -> Result<Vec<GlVertexInputReflection>, GlError> {
    let mut result = BTreeMap::<u32, GlVertexInputReflection>::new();
    for source in sources {
        if source.stage != GlShaderStage::Vertex {
            continue;
        }
        let Some(spirv) = &source.spirv else { continue };
        let words = &spirv.words;
        if words.len() < 5 || words[0] != MAGIC {
            return Err(invalid("SPIR-V header is invalid"));
        }
        let mut scalars = BTreeMap::<u32, ()>::new();
        let mut vectors = BTreeMap::<u32, (u32, u32)>::new();
        let mut pointers = BTreeMap::<u32, (u32, u32)>::new();
        let mut decorations = BTreeMap::<u32, (Option<u32>, bool)>::new();
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
                    scalars.insert(p[0], ());
                }
                22 if p.len() >= 2 => {
                    scalars.insert(p[0], ());
                }
                23 if p.len() >= 3 => {
                    vectors.insert(p[0], (p[1], p[2]));
                }
                32 if p.len() >= 3 => {
                    pointers.insert(p[0], (p[1], p[2]));
                }
                59 if p.len() >= 3 => variables.push((p[1], p[0], p[2])),
                71 if p.len() >= 3 => {
                    let decoration = decorations.entry(p[0]).or_insert((None, false));
                    match p[1] {
                        11 => decoration.1 = true,
                        30 => decoration.0 = Some(p[2]),
                        _ => {}
                    }
                }
                _ => {}
            }
            at += count;
        }
        for (variable, pointer, storage) in variables {
            if storage != 1 {
                continue;
            } // Input
            let (location, builtin) = decorations.get(&variable).copied().unwrap_or((None, false));
            if builtin || location.is_none() {
                continue;
            }
            let (pointer_storage, pointee) = pointers
                .get(&pointer)
                .copied()
                .ok_or_else(|| invalid(format!("vertex input {variable} has no pointer type")))?;
            if pointer_storage != 1 {
                return Err(invalid(format!(
                    "vertex input {variable} does not use an Input pointer type"
                )));
            }
            let columns = if scalars.contains_key(&pointee) {
                1
            } else if let Some((component, count)) = vectors.get(&pointee) {
                if !scalars.contains_key(component) || !(2..=4).contains(count) {
                    return Err(invalid(format!(
                        "vertex input {variable} has an unsupported vector type"
                    )));
                }
                *count as u8
            } else {
                return Err(invalid(format!(
                    "vertex input {variable} is not a supported scalar or vector"
                )));
            };
            let location = location.unwrap();
            let input = GlVertexInputReflection {
                name: format!("@{location}"),
                location,
                columns,
            };
            if let Some(previous) = result.insert(location, input.clone()) {
                if previous.columns != input.columns {
                    return Err(invalid(format!(
                        "vertex location {location} disagrees between shader stages"
                    )));
                }
            }
        }
    }
    Ok(result.into_values().collect())
}
