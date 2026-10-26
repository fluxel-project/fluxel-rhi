//! Direct GLES diagnostic for indirect gl_VertexID handling.
#[cfg(all(test, feature = "native-gles-egl"))]
#[test]
#[ignore = "requires GLES 3.1 hardware context"]
fn gles_raw_indirect_vertex_id() {
    use super::egl::{EglGlesContext, EglGlesVersion, EglPbufferSize};
    use crate::backend::api::{ContextEpoch, ContextStamp, DeviceIdentity};
    use glow::HasContext as _;
    let stamp = ContextStamp::new(DeviceIdentity::new(999).unwrap(), ContextEpoch::INITIAL);
    let mut context =
        EglGlesContext::new_pbuffer(stamp, EglPbufferSize::PROBE, EglGlesVersion::V3_1).unwrap();
    context.make_current().unwrap();
    let gl = context.load_glow().unwrap();
    // SAFETY: this thread owns and has made the pbuffer context current; all temporary GL names are local.
    unsafe {
        eprintln!(
            "RAW_GL_VERSION={}; RAW_GL_RENDERER={}",
            gl.get_parameter_string(glow::VERSION),
            gl.get_parameter_string(glow::RENDERER)
        );
        let program = gl.create_program().unwrap();
        for (kind, source) in [
            (
                glow::VERTEX_SHADER,
                "#version 310 es\nvoid main(){ vec2 p[3]=vec2[3](vec2(-1.,-1.),vec2(0.,-1.),vec2(-1.,1.)); gl_Position=vec4(p[gl_VertexID%3]+vec2(gl_VertexID>=3?1.:0.,0.),0.,1.); }",
            ),
            (
                glow::FRAGMENT_SHADER,
                "#version 310 es\nprecision highp float; out vec4 c; void main(){ c=vec4(1.,0.,0.,1.); }",
            ),
        ] {
            let shader = gl.create_shader(kind).unwrap();
            gl.shader_source(shader, source);
            gl.compile_shader(shader);
            assert!(
                gl.get_shader_compile_status(shader),
                "{}",
                gl.get_shader_info_log(shader)
            );
            gl.attach_shader(program, shader);
            gl.delete_shader(shader);
        }
        gl.link_program(program);
        assert!(
            gl.get_program_link_status(program),
            "{}",
            gl.get_program_info_log(program)
        );
        let vao = gl.create_vertex_array().unwrap();
        gl.bind_vertex_array(Some(vao));
        gl.use_program(Some(program));
        gl.viewport(0, 0, 16, 16);
        let buffer = gl.create_buffer().unwrap();
        gl.bind_buffer(glow::DRAW_INDIRECT_BUFFER, Some(buffer));
        let mut bytes = Vec::new();
        for value in [3u32, 1, 3, 0] {
            bytes.extend_from_slice(&value.to_ne_bytes());
        }
        gl.buffer_data_u8_slice(glow::DRAW_INDIRECT_BUFFER, &bytes, glow::STATIC_DRAW);
        gl.clear_color(0., 0., 0., 1.);
        gl.clear(glow::COLOR_BUFFER_BIT);
        gl.draw_arrays_indirect_offset(glow::TRIANGLES, 0);
        let mut pixels = [0u8; 16 * 16 * 4];
        gl.read_pixels(
            0,
            0,
            16,
            16,
            glow::RGBA,
            glow::UNSIGNED_BYTE,
            glow::PixelPackData::Slice(Some(&mut pixels)),
        );
        let left = pixels[(2 * 16 + 2) * 4];
        let right = pixels[(2 * 16 + 10) * 4];
        eprintln!(
            "RAW_LEFT={left} RAW_RIGHT={right} GL_ERROR={:#x}",
            gl.get_error()
        );
        gl.delete_buffer(buffer);
        gl.delete_vertex_array(vao);
        gl.delete_program(program);
        assert_eq!(
            (left, right),
            (0, 255),
            "indirect firstVertex must affect gl_VertexID"
        );
    }
}
