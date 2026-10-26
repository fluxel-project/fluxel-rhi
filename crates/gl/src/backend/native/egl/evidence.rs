//! GLES source and shader compilation for EGL visual evidence.
//!
//! This is deliberately fixture-only support. Context ownership, currentness,
//! and presentation remain in the parent EGL owner and runtime modules.

use crate::backend::api::GlError;

/// GLES spelling of the shared 4x4 evidence triangle.
pub(crate) const EVIDENCE_VERTEX: &str = "#version 300 es\nprecision highp float; void main() { const vec2 p[3]=vec2[3](vec2(-0.6,-0.5),vec2(0.6,-0.5),vec2(0.0,0.0)); gl_Position=vec4(p[gl_VertexID],0.0,1.0); }";

/// GLES fragment stage for the shared 4x4 evidence triangle.
pub(crate) const EVIDENCE_FRAGMENT: &str = "#version 300 es\nprecision highp float; out vec4 o; void main() { o=vec4(48.0/255.0,176.0/255.0,112.0/255.0,1.0); }";

pub(crate) fn compile_evidence_shader(
    gl: &glow::Context,
    kind: u32,
    source: &str,
) -> Result<glow::NativeShader, GlError> {
    use glow::HasContext as _;
    // SAFETY: caller owns current EGL context for this helper.
    unsafe {
        let shader = gl.create_shader(kind).map_err(|message| GlError::Driver {
            operation: "EGL evidence shader",
            message,
        })?;
        gl.shader_source(shader, source);
        gl.compile_shader(shader);
        if gl.get_shader_compile_status(shader) {
            Ok(shader)
        } else {
            let message = gl.get_shader_info_log(shader);
            gl.delete_shader(shader);
            Err(GlError::Driver {
                operation: "EGL evidence shader",
                message,
            })
        }
    }
}
