//! Adapts WebGPU upper-left texture rows to GL's lower-left native texture rows.
//!
//! Naga emits ordinary GLSL texture builtins, whose normalized and integer
//! coordinates follow GL storage orientation. This narrow source pass only
//! rewrites reflected 2D/2D-array image operations; it does not change shader
//! inputs, framebuffer coordinates, cube directions, or resource state.

use std::collections::HashSet;

const SAMPLE_CALLS: &[&str] = &[
    "texture",
    "textureLod",
    "textureGrad",
    "textureOffset",
    "textureLodOffset",
    "textureGradOffset",
    "textureGather",
    "textureGatherOffset",
    "textureGatherOffsets",
];

/// Rewrites coordinates only when Naga reflection proves the first argument
/// is a 2D image declared by this shader. Generated GLSL contains no user
/// macros at this point, so balanced call parsing preserves nested expressions.
pub(super) fn lower(
    source: &str,
    sampled_images: &HashSet<&str>,
    storage_images: &HashSet<&str>,
) -> String {
    let bytes = source.as_bytes();
    let mut edits = Vec::new();
    let mut cursor = 0;
    let mut needs_uv = false;
    let mut needs_texel = false;

    while cursor < bytes.len() {
        if !is_ident_start(bytes[cursor]) {
            cursor += 1;
            continue;
        }
        let name_start = cursor;
        cursor += 1;
        while cursor < bytes.len() && is_ident_continue(bytes[cursor]) {
            cursor += 1;
        }
        let name = &source[name_start..cursor];
        let open = skip_space(bytes, cursor);
        if open == bytes.len() || bytes[open] != b'(' {
            continue;
        }
        let Some((close, args)) = call_arguments(bytes, open) else {
            continue;
        };
        let Some(first) = args.first() else {
            cursor = close + 1;
            continue;
        };
        let resource = source[first.clone()].trim();
        let transformed = if SAMPLE_CALLS.contains(&name) && sampled_images.contains(resource) {
            args.get(1).map(|coordinate| {
                needs_uv = true;
                format!(
                    "fluxel_internal_flip_uv({})",
                    source[coordinate.clone()].trim()
                )
            })
        } else if name == "texelFetch" && sampled_images.contains(resource) {
            args.get(1).zip(args.get(2)).map(|(coordinate, level)| {
                needs_texel = true;
                format!(
                    "fluxel_internal_flip_texel({}, textureSize({}, {}))",
                    source[coordinate.clone()].trim(),
                    resource,
                    source[level.clone()].trim()
                )
            })
        } else if (name == "imageLoad" || name == "imageStore") && storage_images.contains(resource)
        {
            args.get(1).map(|coordinate| {
                needs_texel = true;
                format!(
                    "fluxel_internal_flip_texel({}, imageSize({}))",
                    source[coordinate.clone()].trim(),
                    resource
                )
            })
        } else {
            None
        };
        if let (Some(coordinate), Some(replacement)) = (args.get(1), transformed) {
            edits.push((coordinate.start, coordinate.end, replacement));
        }
        cursor = close + 1;
    }

    if edits.is_empty() {
        return source.to_owned();
    }
    let mut result = source.to_owned();
    for (start, end, replacement) in edits.into_iter().rev() {
        result.replace_range(start..end, &replacement);
    }
    let helpers = format!(
        "\nvec2 fluxel_internal_flip_uv(vec2 uv) {{ return vec2(uv.x, 1.0 - uv.y); }}\n\
         vec3 fluxel_internal_flip_uv(vec3 uv) {{ return vec3(uv.x, 1.0 - uv.y, uv.z); }}\n{}",
        if needs_texel {
            "ivec2 fluxel_internal_flip_texel(ivec2 p, ivec2 size) { return ivec2(p.x, size.y - 1 - p.y); }\n\
             ivec3 fluxel_internal_flip_texel(ivec3 p, ivec3 size) { return ivec3(p.x, size.y - 1 - p.y, p.z); }\n"
        } else {
            ""
        }
    );
    if needs_uv || needs_texel {
        let insertion = declaration_insertion_point(&result);
        result.insert_str(insertion, &helpers);
    }
    result
}

/// Keep helper definitions ahead of Naga functions that may call them.
/// GLSL drivers do not consistently allow a helper definition to appear only
/// after an earlier function body has used it.
fn declaration_insertion_point(source: &str) -> usize {
    let mut offset = 0;
    for line in source.split_inclusive('\n') {
        if line.trim_start().starts_with('#') {
            offset += line.len();
        } else {
            break;
        }
    }
    offset
}

fn call_arguments(bytes: &[u8], open: usize) -> Option<(usize, Vec<std::ops::Range<usize>>)> {
    let mut args = Vec::new();
    let mut start = open.checked_add(1)?;
    let mut depth = 1_u32;
    for index in start..bytes.len() {
        match bytes[index] {
            b'(' | b'[' => depth = depth.checked_add(1)?,
            b')' => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    if start < index || !args.is_empty() {
                        args.push(trim_range(bytes, start..index));
                    }
                    return Some((index, args));
                }
            }
            b']' => depth = depth.checked_sub(1)?,
            b',' if depth == 1 => {
                args.push(trim_range(bytes, start..index));
                start = index + 1;
            }
            _ => {}
        }
    }
    None
}

fn trim_range(bytes: &[u8], mut range: std::ops::Range<usize>) -> std::ops::Range<usize> {
    while range.start < range.end && bytes[range.start].is_ascii_whitespace() {
        range.start += 1;
    }
    while range.start < range.end && bytes[range.end - 1].is_ascii_whitespace() {
        range.end -= 1;
    }
    range
}

fn skip_space(bytes: &[u8], mut cursor: usize) -> usize {
    while cursor < bytes.len() && bytes[cursor].is_ascii_whitespace() {
        cursor += 1;
    }
    cursor
}

const fn is_ident_start(byte: u8) -> bool {
    byte == b'_' || byte.is_ascii_alphabetic()
}

const fn is_ident_continue(byte: u8) -> bool {
    is_ident_start(byte) || byte.is_ascii_digit()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rewrites_only_reflected_2d_texture_operations() {
        let sampled = HashSet::from(["source_image"]);
        let storage = HashSet::from(["result_image"]);
        let source = "#version 430 core\nvoid main() {\n  vec4 a = textureLod(source_image, uv, 0.0);\n  vec4 b = texelFetch(source_image, pos, 0);\n  imageStore(result_image, pos, a + b);\n  vec4 c = textureLod(cube_image, uv, 0.0);\n}\n";

        let actual = lower(source, &sampled, &storage);

        assert!(actual.contains("textureLod(source_image, fluxel_internal_flip_uv(uv), 0.0)"));
        assert!(actual.contains(
            "texelFetch(source_image, fluxel_internal_flip_texel(pos, textureSize(source_image, 0)), 0)"
        ));
        assert!(actual.contains(
            "imageStore(result_image, fluxel_internal_flip_texel(pos, imageSize(result_image)), a + b)"
        ));
        assert!(actual.contains("textureLod(cube_image, uv, 0.0)"));
    }

    #[test]
    fn keeps_array_layer_while_flipping_texture_rows() {
        let sampled = HashSet::from(["array_image"]);
        let storage = HashSet::from(["array_output"]);
        let source = "#version 430 core\n#extension GL_ARB_shader_image_load_store : require\nvoid main() {\n  vec4 c = texture(array_image, uv_layer);\n  imageStore(array_output, texel, c);\n}\n";

        let actual = lower(source, &sampled, &storage);

        assert!(actual.contains("texture(array_image, fluxel_internal_flip_uv(uv_layer))"));
        assert!(actual.contains(
            "imageStore(array_output, fluxel_internal_flip_texel(texel, imageSize(array_output)), c)"
        ));
        assert!(
            actual.find("vec2 fluxel_internal_flip_uv").unwrap()
                < actual.find("void main").unwrap()
        );
    }
}
