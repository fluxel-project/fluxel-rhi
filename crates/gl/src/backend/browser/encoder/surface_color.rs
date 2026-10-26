//! Explicit transfer-function conversion for WebGL canvas surface staging.
//!
//! Canvas default buffers are linear UNORM GL attachments. sRGB surfaces use
//! an sRGB texture for raster semantics and convert at the publication boundary.
use fluxel_rhi_core::api::{Error, Rect, Result};
use js_sys::Array;
use wasm_bindgen::JsValue;
use web_sys::{WebGl2RenderingContext as Gl, WebGlFramebuffer, WebGlTexture};

fn shader(raw: &Gl, kind: u32, source: &str) -> Result<web_sys::WebGlShader> {
    let shader = raw.create_shader(kind).ok_or(Error::OutOfMemory)?;
    raw.shader_source(&shader, source);
    raw.compile_shader(&shader);
    if !raw
        .get_shader_parameter(&shader, Gl::COMPILE_STATUS)
        .as_bool()
        .unwrap_or(false)
    {
        let message = raw.get_shader_info_log(&shader).unwrap_or_default();
        raw.delete_shader(Some(&shader));
        return Err(Error::Backend(format!(
            "WebGL2 surface conversion shader: {message}"
        )));
    }
    Ok(shader)
}
/// Draws from a texture to a chosen framebuffer attachment. `encode` converts
/// linear shader samples to stored sRGB bytes; the inverse path decodes stored
/// canvas bytes to linear, then the destination sRGB attachment encodes them.
pub(super) fn convert(
    raw: &Gl,
    source: &WebGlTexture,
    destination: Option<&WebGlFramebuffer>,
    slot: u32,
    extent: [u32; 2],
    area: Rect,
    encode: bool,
) -> Result<()> {
    let vertex = shader(
        raw,
        Gl::VERTEX_SHADER,
        "#version 300 es\nvoid main(){vec2 p=vec2((gl_VertexID<<1)&2,gl_VertexID&2); gl_Position=vec4(p*2.0-1.0,0.0,1.0);}",
    )?;
    let formula = if encode {
        "mix(12.92*c, 1.055*pow(max(c,vec3(0.0)),vec3(1.0/2.4))-0.055, greaterThan(c,vec3(0.0031308)))"
    } else {
        "mix(c/12.92, pow((c+0.055)/1.055,vec3(2.4)), greaterThan(c,vec3(0.04045)))"
    };
    let fragment_source = format!(
        "#version 300 es\nprecision highp float; uniform sampler2D image; layout(location={slot}) out vec4 outColor; void main(){{ivec2 p=ivec2(gl_FragCoord.xy); p.y={} - 1 - p.y; vec4 v=texelFetch(image,p,0); vec3 c=v.rgb; outColor=vec4({formula},v.a);}}",
        extent[1]
    );
    let fragment = match shader(raw, Gl::FRAGMENT_SHADER, &fragment_source) {
        Ok(v) => v,
        Err(e) => {
            raw.delete_shader(Some(&vertex));
            return Err(e);
        }
    };
    let program = match raw.create_program() {
        Some(v) => v,
        None => {
            raw.delete_shader(Some(&vertex));
            raw.delete_shader(Some(&fragment));
            return Err(Error::OutOfMemory);
        }
    };
    raw.attach_shader(&program, &vertex);
    raw.attach_shader(&program, &fragment);
    raw.link_program(&program);
    raw.delete_shader(Some(&vertex));
    raw.delete_shader(Some(&fragment));
    if !raw
        .get_program_parameter(&program, Gl::LINK_STATUS)
        .as_bool()
        .unwrap_or(false)
    {
        let message = raw.get_program_info_log(&program).unwrap_or_default();
        raw.delete_program(Some(&program));
        return Err(Error::Backend(format!(
            "WebGL2 surface conversion program: {message}"
        )));
    }
    let vao = match raw.create_vertex_array() {
        Some(v) => v,
        None => {
            raw.delete_program(Some(&program));
            return Err(Error::OutOfMemory);
        }
    };
    raw.bind_framebuffer(Gl::DRAW_FRAMEBUFFER, destination);
    if destination.is_some() {
        let buffers = Array::from_iter((0..=slot).map(|i| {
            JsValue::from_f64(f64::from(if i == slot {
                Gl::COLOR_ATTACHMENT0 + i
            } else {
                Gl::NONE
            }))
        }));
        raw.draw_buffers(&buffers);
    }
    raw.viewport(0, 0, extent[0] as i32, extent[1] as i32);
    raw.enable(Gl::SCISSOR_TEST);
    raw.scissor(
        area.x as i32,
        (if destination.is_none() {
            extent[1] - area.y - area.height
        } else {
            area.y
        }) as i32,
        area.width as i32,
        area.height as i32,
    );
    raw.disable(Gl::SAMPLE_ALPHA_TO_COVERAGE);
    raw.disable(Gl::BLEND);
    raw.disable(Gl::DEPTH_TEST);
    raw.disable(Gl::STENCIL_TEST);
    raw.disable(Gl::CULL_FACE);
    raw.color_mask(true, true, true, true);
    raw.use_program(Some(&program));
    raw.bind_vertex_array(Some(&vao));
    raw.active_texture(Gl::TEXTURE0);
    raw.bind_texture(Gl::TEXTURE_2D, Some(source));
    raw.bind_sampler(0, None);
    raw.tex_parameteri(Gl::TEXTURE_2D, Gl::TEXTURE_MIN_FILTER, Gl::NEAREST as i32);
    raw.tex_parameteri(Gl::TEXTURE_2D, Gl::TEXTURE_MAG_FILTER, Gl::NEAREST as i32);
    let uniform = raw.get_uniform_location(&program, "image");
    raw.uniform1i(uniform.as_ref(), 0);
    raw.draw_arrays(Gl::TRIANGLES, 0, 3);
    let error = raw.get_error();
    raw.bind_vertex_array(None);
    raw.use_program(None);
    raw.bind_texture(Gl::TEXTURE_2D, None);
    raw.disable(Gl::SCISSOR_TEST);
    raw.delete_vertex_array(Some(&vao));
    raw.delete_program(Some(&program));
    if error != Gl::NO_ERROR {
        return Err(Error::Backend(format!(
            "surface conversion draw error {error}"
        )));
    }
    Ok(())
}
/// Copies the default buffer bytes into an UNORM scratch texture before decode.
pub(super) fn preload(
    raw: &Gl,
    target: &WebGlFramebuffer,
    slot: u32,
    extent: [u32; 2],
) -> Result<()> {
    let texture = raw.create_texture().ok_or(Error::OutOfMemory)?;
    raw.bind_framebuffer(Gl::READ_FRAMEBUFFER, None);
    raw.bind_texture(Gl::TEXTURE_2D, Some(&texture));
    raw.tex_storage_2d(
        Gl::TEXTURE_2D,
        1,
        Gl::RGB8,
        extent[0] as i32,
        extent[1] as i32,
    );
    raw.copy_tex_sub_image_2d(
        Gl::TEXTURE_2D,
        0,
        0,
        0,
        0,
        0,
        extent[0] as i32,
        extent[1] as i32,
    );
    let err = raw.get_error();
    if err != Gl::NO_ERROR {
        raw.delete_texture(Some(&texture));
        return Err(Error::Backend(format!("surface preload copy error {err}")));
    }
    let result = convert(
        raw,
        &texture,
        Some(target),
        slot,
        extent,
        Rect {
            x: 0,
            y: 0,
            width: extent[0],
            height: extent[1],
        },
        false,
    );
    raw.delete_texture(Some(&texture));
    result
}
