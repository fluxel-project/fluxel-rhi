//! Adapt generated multiview declarations to the OVR shader ABI.

use super::*;

/// OVR supports arbitrary view-dependent vertex and fragment outputs only with
/// multiview2. Naga emits EXT_multiview outside WebGL, so normalize all
/// profiles to the extension actually used by our framebuffer attachment.
pub(in crate::backend::rhi::shader) fn lower_multiview(
    mut text: String,
    stage: api::ShaderStage,
    count: Option<std::num::NonZeroU32>,
) -> Result<String> {
    let Some(count) = count else {
        if text.contains("gl_ViewIndex") || text.contains("gl_ViewID_OVR") {
            return Err(Error::Shader(
                "view_index requires a multiview pipeline".into(),
            ));
        }
        return Ok(text);
    };
    const BASE: &str = "fluxel_internal_view_index_base";
    if text.contains(BASE) {
        return Err(Error::Shader("reserved multiview uniform name".into()));
    }
    text = text.replace(
        "#extension GL_EXT_multiview : require",
        "#extension GL_OVR_multiview2 : require",
    );
    if !text.contains("#extension GL_OVR_multiview2 : require") {
        let offset = text
            .find('\n')
            .map(|p| p + 1)
            .ok_or_else(|| Error::Shader("GL shader missing version line".into()))?;
        text.insert_str(offset, "#extension GL_OVR_multiview2 : require\n");
    }
    let uses_index = text.contains("gl_ViewIndex") || text.contains("gl_ViewID_OVR");
    text = text.replace("gl_ViewID_OVR", &format!("(gl_ViewID_OVR + uint({BASE}))"));
    text = text.replace(
        "gl_ViewIndex",
        &format!("int(gl_ViewID_OVR + uint({BASE}))"),
    );
    let insertion: usize = text
        .split_inclusive('\n')
        .take_while(|line| line.trim_start().starts_with('#'))
        .map(str::len)
        .sum();
    let mut declarations = String::new();
    if uses_index {
        declarations.push_str(&format!("uniform int {BASE};\n"));
    }
    if stage == api::ShaderStage::Vertex {
        // GLSL's num_views is a literal, not a runtime uniform. Keep an
        // existing declaration only when it matches the pipeline count.
        let mut declarations_found = Vec::new();
        let mut remaining = text.as_str();
        while let Some(start) = remaining.find("layout") {
            remaining = &remaining[start + "layout".len()..];
            let trimmed = remaining.trim_start();
            let Some(body) = trimmed.strip_prefix('(') else {
                continue;
            };
            let Some(end) = body.find(')') else {
                break;
            };
            let qualifiers = &body[..end];
            if let Some(index) = qualifiers.find("num_views") {
                let tail = qualifiers[index + "num_views".len()..].trim_start();
                let value = tail.strip_prefix('=').map(str::trim_start).unwrap_or("");
                declarations_found.push(
                    value
                        .chars()
                        .take_while(char::is_ascii_digit)
                        .collect::<String>(),
                );
            }
            remaining = &body[end + 1..];
        }
        if declarations_found.len() > 1
            || declarations_found
                .first()
                .is_some_and(|value| value.parse::<u32>() != Ok(count.get()))
        {
            return Err(Error::Shader(
                "GLSL num_views does not match pipeline multiview count".into(),
            ));
        }
        if declarations_found.is_empty() {
            declarations.push_str(&format!("layout(num_views = {}) in;\n", count.get()));
        }
    }
    text.insert_str(insertion, &declarations);
    Ok(text)
}
