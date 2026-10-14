//! WebGPU-to-WebGL vertex clip-space adaptation.
//!
//! Fluxel exposes WebGPU clip depth (`0..w`) while WebGL clips before
//! `depthRange` in OpenGL clip depth (`-w..w`).  The conversion therefore has
//! to happen in the vertex shader after user code writes `gl_Position`; a
//! viewport depth range can only affect values that survived clipping.

/// Wraps an ESSL vertex entry point with the public WebGPU depth convention.
///
/// The current ESSL lowering emits one conventional `void main()` or
/// `void main(void)` entry point.  Refuse a source outside that ABI rather
/// than compiling it with silently incorrect clip-space semantics.
pub(super) fn adapt_vertex_clip_depth(source: &str) -> Result<String, &'static str> {
    const USER_ENTRY: &str = "fluxel_webgpu_user_vertex_main";
    let (needle, replacement) = if source.matches("void main()").count() == 1 {
        ("void main()", format!("void {USER_ENTRY}()"))
    } else if source.matches("void main(void)").count() == 1 {
        ("void main(void)", format!("void {USER_ENTRY}(void)"))
    } else {
        return Err("WebGL2 vertex source must contain exactly one void main() entry point");
    };
    let user_source = source.replacen(needle, &replacement, 1);
    Ok(format!(
        "{user_source}\n\
         void main() {{\n\
             {USER_ENTRY}();\n\
             gl_Position.z = gl_Position.z * 2.0 - gl_Position.w;\n\
         }}\n"
    ))
}
