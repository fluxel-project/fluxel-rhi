//! GLSL source-level coordinate and precision adaptation.

use super::*;

/// Inserts the one GL-family clip-depth adaptation owned by this backend.
pub(super) fn apply_public_clip_depth_abi(
    source: &mut crate::backend::api::GlShaderSource,
) -> Result<()> {
    if source.stage != crate::backend::api::GlShaderStage::Vertex {
        return Ok(());
    }
    // Rename the user's entry with the GLSL preprocessor, then wrap it. A
    // wrapper also handles early returns inside main, unlike an insertion at
    // the final brace (which may belong to a different function).
    if !source.text.contains("gl_Position") {
        return Err(Error::Unsupported("GL vertex shader main/position ABI"));
    }
    let mut insertion = 0;
    let mut version_end = None;
    for line in source.text.split_inclusive('\n') {
        insertion += line.len();
        if line.trim_start().starts_with("#version") {
            version_end = Some(insertion);
            break;
        }
    }
    let insertion = if let Some(end) = version_end {
        if !source.text[..end].ends_with('\n') {
            source.text.insert(end, '\n');
            end + 1
        } else {
            end
        }
    } else {
        // ShaderFormat carries the version even when the source omits it.
        let version = match source.dialect {
            crate::backend::api::GlShaderDialect::Desktop { version } => {
                format!("#version {version} core\n")
            }
            crate::backend::api::GlShaderDialect::Embedded { version } => {
                format!("#version {version} es\n")
            }
        };
        source.text.insert_str(0, &version);
        version.len()
    };
    let mut entry_name = "fluxel_internal_vertex_main".to_owned();
    while source.text.contains(&entry_name) {
        entry_name.push('_');
    }
    source
        .text
        .insert_str(insertion, &format!("#define main {entry_name}\n"));
    source.text.push_str(&format!("\n#undef main\nvoid main() {{ {entry_name}(); gl_Position.y = -gl_Position.y; gl_Position.z = 2.0 * gl_Position.z - gl_Position.w; }}\n"));
    Ok(())
}

/// Gives GLSL ES defined defaults for unqualified scalar types.
///
/// Desktop GLSL supplies a default float precision; GLSL ES fragment shaders
/// do not. Naga leaves source-level precision choices open, so native GLES and
/// browser WebGL apply the same backend policy before compiling the source.
/// Explicit author precision statements remain authoritative.
pub(super) fn apply_embedded_precision_defaults(source: &mut crate::backend::api::GlShaderSource) {
    if !matches!(
        source.dialect,
        crate::backend::api::GlShaderDialect::Embedded { .. }
    ) {
        return;
    }
    let mut declarations = Vec::new();
    let mut body = String::with_capacity(source.text.len());
    for line in source.text.split_inclusive('\n') {
        if line.trim_start().starts_with("precision ") {
            declarations.push(line.to_owned());
        } else {
            body.push_str(line);
        }
    }
    let has_default = |ty: &str| {
        declarations.iter().any(|line| {
            line.trim_end()
                .strip_suffix(';')
                .is_some_and(|declaration| declaration.split_whitespace().last() == Some(ty))
        })
    };
    let has_float_default = has_default("float");
    let has_int_default = has_default("int");
    if !has_float_default {
        declarations.push("precision highp float;\n".into());
    }
    if !has_int_default {
        declarations.push("precision highp int;\n".into());
    }

    // Naga and the texture-origin lowering can place helper functions before
    // Naga's default precision declarations. GLSL ES requires defaults before
    // every use, so move all precision declarations after directives/comments.
    let mut insertion = 0;
    for line in body.split_inclusive('\n') {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with("//") {
            insertion += line.len();
        } else {
            break;
        }
    }
    let declarations = declarations.concat();
    body.insert_str(insertion, &declarations);
    source.text = body;
}
