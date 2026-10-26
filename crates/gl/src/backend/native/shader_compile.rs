//! Native source and direct-SPIR-V shader compilation.

use super::provider::NativeGlProvider;
use crate::backend::api::{GlError, GlFamilyApi as _, GlShaderSource, GlShaderStage};

pub(super) fn compile_shader_source(
    provider: &mut NativeGlProvider,
    op: &'static str,
    source: &GlShaderSource,
) -> Result<glow::NativeShader, GlError> {
    use glow::HasContext as _;
    let kind = match source.stage {
        GlShaderStage::Vertex => glow::VERTEX_SHADER,
        GlShaderStage::Fragment => glow::FRAGMENT_SHADER,
        GlShaderStage::Compute => glow::COMPUTE_SHADER,
    };
    if let Some(spirv) = &source.spirv {
        let functions = provider.spirv.ok_or(GlError::Unsupported {
            operation: op,
            reason: "GL_ARB_gl_spirv entry points",
        })?;
        let entry = std::ffi::CString::new(source.entry_point.as_str()).map_err(|_| {
            GlError::Validation {
                operation: op,
                message: "SPIR-V entry point contains NUL".into(),
            }
        })?;
        let byte_len = i32::try_from(
            spirv
                .words
                .len()
                .saturating_mul(core::mem::size_of::<u32>()),
        )
        .map_err(|_| GlError::Validation {
            operation: op,
            message: "SPIR-V artifact too large".into(),
        })?;
        // SAFETY: the current context owns this temporary shader and both input arrays live through specialization.
        let raw =
            unsafe { provider.gl.create_shader(kind) }.map_err(|message| GlError::Driver {
                operation: op,
                message,
            })?;
        let raw_name = raw.0.get();
        // SAFETY: the shader name is live on this context; the byte slice length is checked.
        unsafe {
            (functions.shader_binary)(
                1,
                &raw_name,
                super::spirv::GL_SHADER_BINARY_FORMAT_SPIR_V,
                spirv.words.as_ptr().cast(),
                byte_len,
            );
        }
        let ids: Vec<u32> = spirv.specialization.iter().map(|(id, _)| *id).collect();
        let values: Vec<u32> = spirv
            .specialization
            .iter()
            .map(|(_, value)| *value)
            .collect();
        // SAFETY: entry is NUL-terminated; parallel ID/value slices have identical checked lengths.
        unsafe {
            (functions.specialize_shader)(
                raw_name,
                entry.as_ptr(),
                ids.len() as u32,
                ids.as_ptr(),
                values.as_ptr(),
            );
        }
        // SAFETY: specialization completed on this live shader owned by the current context.
        let result = unsafe {
            if provider.gl.get_shader_compile_status(raw) {
                Ok(())
            } else {
                Err(GlError::Shader {
                    stage: stage_name(source.stage),
                    log: provider.gl.get_shader_info_log(raw),
                })
            }
        }
        .and_then(|()| provider.driver_error(op));
        if let Err(error) = result {
            // SAFETY: this unpublished shader is still owned exclusively by this transaction.
            unsafe { provider.gl.delete_shader(raw) };
            return Err(error);
        }
        return Ok(raw);
    }
    // SAFETY: a failed compile deletes this unpublished current-context object.
    let raw = unsafe { provider.gl.create_shader(kind) }.map_err(|message| GlError::Driver {
        operation: op,
        message,
    })?;
    let mut text = source.text.clone();
    let directive = "#extension GL_EXT_texture_cube_map_array : require\n";
    if text.contains(directive) {
        if let crate::backend::api::GlFamilyProfile::Embedded { major, minor } = provider.profile()
        {
            let names = provider.discovery.extensions();
            let has_ext = names
                .raw_reported_names()
                .any(|name| name == "GL_EXT_texture_cube_map_array");
            let has_oes = names
                .raw_reported_names()
                .any(|name| name == "GL_OES_texture_cube_map_array");
            if !has_ext && (major > 3 || (major == 3 && minor >= 2)) {
                text = text
                    .replace(directive, "")
                    .replacen("#version 300 es", "#version 320 es", 1)
                    .replacen("#version 310 es", "#version 320 es", 1);
            } else if !has_ext && has_oes {
                text = text.replace(
                    directive,
                    "#extension GL_OES_texture_cube_map_array : require\n",
                );
            } else if !has_ext {
                // SAFETY: this unpublished shader is still owned exclusively by this transaction.
                unsafe { provider.gl.delete_shader(raw) };
                return Err(GlError::Unsupported {
                    operation: op,
                    reason: "cube array shader support unavailable",
                });
            }
        }
    }
    // SAFETY: text and this unpublished shader stay live on the current context throughout compilation.
    let result = unsafe {
        provider.gl.shader_source(raw, &text);
        provider.gl.compile_shader(raw);
        if provider.gl.get_shader_compile_status(raw) {
            Ok(())
        } else {
            Err(GlError::Shader {
                stage: stage_name(source.stage),
                log: provider.gl.get_shader_info_log(raw),
            })
        }
    }
    .and_then(|()| provider.driver_error(op));
    if let Err(error) = result {
        // SAFETY: compile failed before publication, so no caller retains this shader.
        unsafe { provider.gl.delete_shader(raw) };
        return Err(error);
    }
    Ok(raw)
}

const fn stage_name(stage: GlShaderStage) -> &'static str {
    match stage {
        GlShaderStage::Vertex => "vertex",
        GlShaderStage::Fragment => "fragment",
        GlShaderStage::Compute => "compute",
    }
}
