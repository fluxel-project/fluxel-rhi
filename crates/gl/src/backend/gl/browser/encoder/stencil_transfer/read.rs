//! Exact standalone stencil-plane readback assembled from WebGL2 operations.
//!
//! The source stencil rectangle is first copied into a private S8 renderbuffer.
//! Eight ordinary stencil tests write its individual bits into two RGBA8
//! textures. A transform-feedback vertex shader packs those exact 0/255
//! channels into bytes before GPU buffer copies place the results in caller
//! rows. No blend precision or CPU readback participates in the conversion.

use fluxel_rhi_core::api::{Error, Result};
use js_sys::Array;
use wasm_bindgen::{JsCast, JsValue};
use web_sys::{
    WebGl2RenderingContext as Gl, WebGlBuffer, WebGlFramebuffer, WebGlProgram, WebGlRenderbuffer,
    WebGlSampler, WebGlShader, WebGlTexture, WebGlTransformFeedback, WebGlVertexArrayObject,
};

use super::super::super::owner::{BrowserBuffer, BrowserEncoder, BrowserTexture};
use super::write::{State, destination_attachment};
use crate::backend::gl::api::{GlFormat, GlTextureDimension, GlTextureUsage};

fn check_stage(gl: &Gl, stage: &str) -> Result<()> {
    let error = gl.get_error();
    if error == Gl::NO_ERROR {
        Ok(())
    } else {
        Err(Error::Backend(format!(
            "WebGL stencil read {stage}: GL {error}"
        )))
    }
}

fn shader(gl: &Gl, kind: u32, source: &str, list: &mut Vec<WebGlShader>) -> Result<WebGlShader> {
    let shader = gl.create_shader(kind).ok_or(Error::OutOfMemory)?;
    gl.shader_source(&shader, source);
    gl.compile_shader(&shader);
    if gl
        .get_shader_parameter(&shader, Gl::COMPILE_STATUS)
        .as_bool()
        != Some(true)
    {
        let text = gl.get_shader_info_log(&shader).unwrap_or_default();
        gl.delete_shader(Some(&shader));
        return Err(Error::Backend(format!("WebGL stencil read shader: {text}")));
    }
    list.push(shader.clone());
    Ok(shader)
}

fn object<T: JsCast>(gl: &Gl, parameter: u32) -> Result<Option<T>> {
    let value = gl
        .get_parameter(parameter)
        .map_err(|_| Error::InvalidState)?;
    if value.is_null() {
        Ok(None)
    } else {
        value.dyn_into().map(Some).map_err(|_| Error::InvalidState)
    }
}

struct Scratch<'a> {
    gl: &'a Gl,
    read_fbo: Option<WebGlFramebuffer>,
    draw_fbo: Option<WebGlFramebuffer>,
    plane: Option<WebGlRenderbuffer>,
    colors: [Option<WebGlTexture>; 2],
    sampler: Option<WebGlSampler>,
    raster_program: Option<WebGlProgram>,
    tf_program: Option<WebGlProgram>,
    vao: Option<WebGlVertexArrayObject>,
    tf: Option<WebGlTransformFeedback>,
    buffer: Option<WebGlBuffer>,
    shaders: Vec<WebGlShader>,
}

impl Drop for Scratch<'_> {
    fn drop(&mut self) {
        let gl = self.gl;
        gl.delete_buffer(self.buffer.as_ref());
        gl.delete_transform_feedback(self.tf.as_ref());
        gl.delete_vertex_array(self.vao.as_ref());
        gl.delete_program(self.tf_program.as_ref());
        gl.delete_program(self.raster_program.as_ref());
        gl.delete_sampler(self.sampler.as_ref());
        for color in &self.colors {
            gl.delete_texture(color.as_ref());
        }
        gl.delete_renderbuffer(self.plane.as_ref());
        gl.delete_framebuffer(self.draw_fbo.as_ref());
        gl.delete_framebuffer(self.read_fbo.as_ref());
        for shader in &self.shaders {
            gl.delete_shader(Some(shader));
        }
    }
}

fn attach_source(gl: &Gl, source: &BrowserTexture, mip: u32, layer: u32) -> Result<()> {
    let attachment = destination_attachment(source.desc.format);
    match (&source.raw, &source.renderbuffer) {
        (Some(texture), None) if source.desc.dimension == GlTextureDimension::D2 => gl
            .framebuffer_texture_2d(
                Gl::READ_FRAMEBUFFER,
                attachment,
                Gl::TEXTURE_2D,
                Some(texture),
                mip as i32,
            ),
        (Some(texture), None) if source.desc.dimension == GlTextureDimension::D2Array => gl
            .framebuffer_texture_layer(
                Gl::READ_FRAMEBUFFER,
                attachment,
                Some(texture),
                mip as i32,
                layer as i32,
            ),
        (None, Some(renderbuffer)) if mip == 0 && layer == 0 => gl.framebuffer_renderbuffer(
            Gl::READ_FRAMEBUFFER,
            attachment,
            Gl::RENDERBUFFER,
            Some(renderbuffer),
        ),
        _ => return Err(Error::Unsupported("WebGL stencil source representation")),
    }
    Ok(())
}

fn link_raster(gl: &Gl, scratch: &mut Scratch<'_>) -> Result<WebGlProgram> {
    let vertex = shader(
        gl,
        Gl::VERTEX_SHADER,
        "#version 300 es\nvoid main(){vec2 p[3]=vec2[3](vec2(-1.,-1.),vec2(3.,-1.),vec2(-1.,3.));gl_Position=vec4(p[gl_VertexID],0.,1.);}",
        &mut scratch.shaders,
    )?;
    let fragment = shader(
        gl,
        Gl::FRAGMENT_SHADER,
        "#version 300 es\nprecision highp float;layout(location=0) out vec4 low_bits;layout(location=1) out vec4 high_bits;void main(){low_bits=vec4(1.);high_bits=vec4(1.);}",
        &mut scratch.shaders,
    )?;
    let program = gl.create_program().ok_or(Error::OutOfMemory)?;
    gl.attach_shader(&program, &vertex);
    gl.attach_shader(&program, &fragment);
    gl.link_program(&program);
    if gl
        .get_program_parameter(&program, Gl::LINK_STATUS)
        .as_bool()
        != Some(true)
    {
        let text = gl.get_program_info_log(&program).unwrap_or_default();
        gl.delete_program(Some(&program));
        return Err(Error::Backend(format!(
            "WebGL stencil read raster link: {text}"
        )));
    }
    scratch.raster_program = Some(program.clone());
    Ok(program)
}

fn link_tf(
    gl: &Gl,
    scratch: &mut Scratch<'_>,
    width: u32,
    words_per_row: u32,
) -> Result<WebGlProgram> {
    let vertex_source = format!(
        r#"#version 300 es
precision highp float;
precision highp int;
uniform sampler2D lo;
uniform sampler2D hi;
flat out uint stencil_word;
void main() {{
  uint i=uint(gl_VertexID);
  uint x=(i%{words_per_row}u)*4u, y=i/{words_per_row}u;
  uint v=0u;
  for(uint n=0u;n<4u;n++) {{
    if(x+n<{width}u) {{
      vec4 a=texelFetch(lo,ivec2(int(x+n),int(y)),0);
      vec4 b=texelFetch(hi,ivec2(int(x+n),int(y)),0);
      uint q=(a.r>.5?1u:0u)|(a.g>.5?2u:0u)|(a.b>.5?4u:0u)|(a.a>.5?8u:0u)
            |(b.r>.5?16u:0u)|(b.g>.5?32u:0u)|(b.b>.5?64u:0u)|(b.a>.5?128u:0u);
      v|=q<<(8u*n);
    }}
  }}
  stencil_word=v;
  gl_Position=vec4(0.);
}}"#
    );
    let vertex = shader(gl, Gl::VERTEX_SHADER, &vertex_source, &mut scratch.shaders)?;
    let fragment = shader(
        gl,
        Gl::FRAGMENT_SHADER,
        "#version 300 es\nprecision highp float;void main(){}",
        &mut scratch.shaders,
    )?;
    let program = gl.create_program().ok_or(Error::OutOfMemory)?;
    gl.attach_shader(&program, &vertex);
    gl.attach_shader(&program, &fragment);
    gl.transform_feedback_varyings(
        &program,
        &Array::of1(&JsValue::from_str("stencil_word")),
        Gl::INTERLEAVED_ATTRIBS,
    );
    gl.link_program(&program);
    if gl
        .get_program_parameter(&program, Gl::LINK_STATUS)
        .as_bool()
        != Some(true)
    {
        let text = gl.get_program_info_log(&program).unwrap_or_default();
        gl.delete_program(Some(&program));
        return Err(Error::Backend(format!(
            "WebGL stencil read feedback link: {text}"
        )));
    }
    scratch.tf_program = Some(program.clone());
    Ok(program)
}

impl BrowserEncoder {
    /// Reads an exact one-byte stencil plane into a caller-provided GPU buffer.
    pub(crate) fn copy_stencil_to_buffer(
        &mut self,
        source: &BrowserTexture,
        mip: u32,
        origin: [u32; 3],
        extent: [u32; 3],
        destination: &BrowserBuffer,
        offset: u64,
        bytes_per_row: u32,
        rows_per_image: u32,
    ) -> Result<()> {
        self.same_owner(&source.owner)?;
        self.same_owner(&destination.owner)?;
        if self.has_owned_framebuffer_pass() {
            return Err(Error::InvalidState);
        }
        let bound = source.desc.mip_extent(mip).ok_or(Error::InvalidRange)?;
        let packed_array = matches!(
            source.desc.format,
            GlFormat::Depth24PlusStencil8 | GlFormat::Depth32FloatStencil8
        ) && source.desc.dimension == GlTextureDimension::D2Array;
        if !matches!(
            source.desc.format,
            GlFormat::Stencil8 | GlFormat::Depth24PlusStencil8 | GlFormat::Depth32FloatStencil8
        ) || !(source.desc.dimension == GlTextureDimension::D2 || packed_array)
            || source.desc.sample_count != 1
            || extent.contains(&0)
            || origin[0]
                .checked_add(extent[0])
                .is_none_or(|end| end > bound.width)
            || origin[1]
                .checked_add(extent[1])
                .is_none_or(|end| end > bound.height)
            || origin[2]
                .checked_add(extent[2])
                .is_none_or(|end| end > bound.depth_or_layers)
            || bytes_per_row < extent[0]
            || rows_per_image < extent[1]
            || !source.desc.usage.contains(GlTextureUsage::COPY_SOURCE)
        {
            return Err(Error::Unsupported("WebGL stencil readback shape"));
        }
        let layer_stride = u64::from(bytes_per_row)
            .checked_mul(u64::from(rows_per_image))
            .ok_or(Error::InvalidRange)?;
        let end = offset
            .checked_add(layer_stride * u64::from(extent[2] - 1))
            .and_then(|v| v.checked_add(u64::from(bytes_per_row) * u64::from(extent[1] - 1)))
            .and_then(|v| v.checked_add(u64::from(extent[0])))
            .ok_or(Error::InvalidRange)?;
        if end > destination.size || end > (1u64 << 53) - 1 {
            return Err(Error::InvalidRange);
        }
        if extent[2] > 1 {
            for layer in 0..extent[2] {
                self.copy_stencil_to_buffer(
                    source,
                    mip,
                    [origin[0], origin[1], origin[2] + layer],
                    [extent[0], extent[1], 1],
                    destination,
                    offset + u64::from(layer) * layer_stride,
                    bytes_per_row,
                    rows_per_image,
                )?;
            }
            return Ok(());
        }
        let words_per_row = extent[0].div_ceil(4);
        let words = words_per_row
            .checked_mul(extent[1])
            .ok_or(Error::InvalidRange)?;
        let feedback_bytes = u64::from(words).checked_mul(4).ok_or(Error::InvalidRange)?;
        let feedback_i32 = i32::try_from(feedback_bytes).map_err(|_| Error::InvalidRange)?;
        let words_i32 = i32::try_from(words).map_err(|_| Error::InvalidRange)?;
        let _width_i32 = i32::try_from(extent[0]).map_err(|_| Error::InvalidRange)?;
        let _height_i32 = i32::try_from(extent[1]).map_err(|_| Error::InvalidRange)?;
        {
            let state = self.owner.state.borrow();
            if state.context.active_query.is_some() {
                return Err(Error::InvalidState);
            }
        }
        let mut result = Ok(());
        let outer = self.with_context("stencil_plane_readback", |gl| {
            result = (|| -> Result<()> {
                // Capture first: this rejects active transform feedback
                // before private GL objects or shader programs are created.
                let state = State::capture(gl)?;
                let mut scratch = Scratch {
                    gl,
                    read_fbo: None,
                    draw_fbo: None,
                    plane: None,
                    colors: [None, None],
                    sampler: None,
                    raster_program: None,
                    tf_program: None,
                    vao: None,
                    tf: None,
                    buffer: None,
                    shaders: Vec::new(),
                };
                scratch.read_fbo = Some(gl.create_framebuffer().ok_or(Error::OutOfMemory)?);
                scratch.draw_fbo = Some(gl.create_framebuffer().ok_or(Error::OutOfMemory)?);
                scratch.plane = Some(gl.create_renderbuffer().ok_or(Error::OutOfMemory)?);
                scratch.colors[0] = Some(gl.create_texture().ok_or(Error::OutOfMemory)?);
                scratch.colors[1] = Some(gl.create_texture().ok_or(Error::OutOfMemory)?);
                scratch.sampler = Some(gl.create_sampler().ok_or(Error::OutOfMemory)?);
                scratch.vao = Some(gl.create_vertex_array().ok_or(Error::OutOfMemory)?);
                scratch.tf = Some(gl.create_transform_feedback().ok_or(Error::OutOfMemory)?);
                scratch.buffer = Some(gl.create_buffer().ok_or(Error::OutOfMemory)?);
                let raster = link_raster(gl, &mut scratch)?;
                let feedback = link_tf(gl, &mut scratch, extent[0], words_per_row)?;
                let read = scratch.read_fbo.as_ref().unwrap();
                let draw = scratch.draw_fbo.as_ref().unwrap();
                let plane = scratch.plane.as_ref().unwrap();
                gl.bind_framebuffer(Gl::READ_FRAMEBUFFER, Some(read));
                attach_source(gl, source, mip, origin[2])?;
                gl.read_buffer(Gl::NONE);
                if gl.check_framebuffer_status(Gl::READ_FRAMEBUFFER) != Gl::FRAMEBUFFER_COMPLETE {
                    return Err(Error::Unsupported("WebGL stencil source framebuffer"));
                }
                let plane_format = match source.desc.format {
                    GlFormat::Stencil8 => Gl::STENCIL_INDEX8,
                    GlFormat::Depth24PlusStencil8 => Gl::DEPTH24_STENCIL8,
                    GlFormat::Depth32FloatStencil8 => Gl::DEPTH32F_STENCIL8,
                    _ => unreachable!("validated stencil format"),
                };
                gl.bind_framebuffer(Gl::FRAMEBUFFER, Some(draw));
                gl.bind_renderbuffer(Gl::RENDERBUFFER, Some(plane));
                gl.renderbuffer_storage(
                    Gl::RENDERBUFFER,
                    plane_format,
                    extent[0] as i32,
                    extent[1] as i32,
                );
                // A generated renderbuffer becomes an initialized native
                // object on first bind; attach it only after allocating it.
                gl.framebuffer_renderbuffer(
                    Gl::DRAW_FRAMEBUFFER,
                    destination_attachment(source.desc.format),
                    Gl::RENDERBUFFER,
                    Some(plane),
                );
                gl.draw_buffers(&Array::of1(&Gl::NONE.into()));
                gl.read_buffer(Gl::NONE);
                check_stage(gl, "stencil staging allocation")?;
                let status = gl.check_framebuffer_status(Gl::DRAW_FRAMEBUFFER);
                if status != Gl::FRAMEBUFFER_COMPLETE {
                    return Err(Error::Backend(format!(
                        "WebGL stencil staging framebuffer: {status:#x}"
                    )));
                }
                gl.bind_framebuffer(Gl::READ_FRAMEBUFFER, Some(read));
                gl.blit_framebuffer(
                    origin[0] as i32,
                    origin[1] as i32,
                    (origin[0] + extent[0]) as i32,
                    (origin[1] + extent[1]) as i32,
                    0,
                    0,
                    extent[0] as i32,
                    extent[1] as i32,
                    Gl::STENCIL_BUFFER_BIT,
                    Gl::NEAREST,
                );
                check_stage(gl, "stencil blit")?;
                gl.bind_framebuffer(Gl::FRAMEBUFFER, Some(draw));
                for (index, color) in scratch.colors.iter().enumerate() {
                    let color = color.as_ref().unwrap();
                    gl.active_texture(Gl::TEXTURE0 + index as u32);
                    gl.bind_texture(Gl::TEXTURE_2D, Some(color));
                    gl.tex_storage_2d(
                        Gl::TEXTURE_2D,
                        1,
                        Gl::RGBA8,
                        extent[0] as i32,
                        extent[1] as i32,
                    );
                    gl.framebuffer_texture_2d(
                        Gl::FRAMEBUFFER,
                        Gl::COLOR_ATTACHMENT0 + index as u32,
                        Gl::TEXTURE_2D,
                        Some(color),
                        0,
                    );
                }
                gl.draw_buffers(&Array::of2(
                    &Gl::COLOR_ATTACHMENT0.into(),
                    &Gl::COLOR_ATTACHMENT1.into(),
                ));
                if gl.check_framebuffer_status(Gl::FRAMEBUFFER) != Gl::FRAMEBUFFER_COMPLETE {
                    return Err(Error::Unsupported("WebGL stencil bit framebuffer"));
                }
                gl.viewport(0, 0, extent[0] as i32, extent[1] as i32);
                gl.enable(Gl::SCISSOR_TEST);
                gl.scissor(0, 0, extent[0] as i32, extent[1] as i32);
                gl.disable(Gl::DEPTH_TEST);
                gl.disable(Gl::BLEND);
                gl.disable(Gl::RASTERIZER_DISCARD);
                gl.disable(Gl::DITHER);
                gl.disable(Gl::CULL_FACE);
                gl.disable(Gl::SAMPLE_COVERAGE);
                gl.enable(Gl::STENCIL_TEST);
                gl.stencil_mask(0);
                gl.color_mask(true, true, true, true);
                gl.clear_bufferfv_with_f32_array(Gl::COLOR, 0, &[0., 0., 0., 0.]);
                gl.clear_bufferfv_with_f32_array(Gl::COLOR, 1, &[0., 0., 0., 0.]);
                check_stage(gl, "bit-plane clear")?;
                gl.bind_vertex_array(scratch.vao.as_ref());
                gl.use_program(Some(&raster));
                for bit in 0..8 {
                    let mask = 1u32 << bit;
                    gl.stencil_func(Gl::EQUAL, mask as i32, mask);
                    gl.stencil_op(Gl::KEEP, Gl::KEEP, Gl::KEEP);
                    let channel = bit % 4;
                    let output = bit / 4;
                    gl.draw_buffers(&Array::of2(
                        &(if output == 0 {
                            Gl::COLOR_ATTACHMENT0
                        } else {
                            Gl::NONE
                        })
                        .into(),
                        &(if output == 1 {
                            Gl::COLOR_ATTACHMENT1
                        } else {
                            Gl::NONE
                        })
                        .into(),
                    ));
                    gl.color_mask(channel == 0, channel == 1, channel == 2, channel == 3);
                    gl.draw_arrays(Gl::TRIANGLES, 0, 3);
                    check_stage(gl, "bit-plane draw")?;
                }
                gl.draw_buffers(&Array::of2(
                    &Gl::COLOR_ATTACHMENT0.into(),
                    &Gl::COLOR_ATTACHMENT1.into(),
                ));
                gl.color_mask(true, true, true, true);
                gl.active_texture(Gl::TEXTURE0);
                gl.bind_texture(Gl::TEXTURE_2D, scratch.colors[0].as_ref());
                gl.bind_sampler(0, scratch.sampler.as_ref());
                gl.active_texture(Gl::TEXTURE1);
                gl.bind_texture(Gl::TEXTURE_2D, scratch.colors[1].as_ref());
                gl.bind_sampler(1, scratch.sampler.as_ref());
                let sampler = scratch.sampler.as_ref().unwrap();
                gl.sampler_parameteri(sampler, Gl::TEXTURE_MIN_FILTER, Gl::NEAREST as i32);
                gl.sampler_parameteri(sampler, Gl::TEXTURE_MAG_FILTER, Gl::NEAREST as i32);
                let buffer = scratch.buffer.as_ref().unwrap();
                gl.bind_buffer(Gl::TRANSFORM_FEEDBACK_BUFFER, Some(buffer));
                gl.buffer_data_with_i32(
                    Gl::TRANSFORM_FEEDBACK_BUFFER,
                    feedback_i32,
                    Gl::STREAM_COPY,
                );
                // Sampling an attachment of a currently bound draw FBO is a
                // WebGL feedback loop even while rasterization is discarded.
                gl.bind_framebuffer(Gl::DRAW_FRAMEBUFFER, None);
                gl.bind_framebuffer(Gl::READ_FRAMEBUFFER, None);
                gl.bind_transform_feedback(Gl::TRANSFORM_FEEDBACK, scratch.tf.as_ref());
                gl.bind_buffer_base(Gl::TRANSFORM_FEEDBACK_BUFFER, 0, Some(buffer));
                gl.use_program(Some(&feedback));
                gl.uniform1i(gl.get_uniform_location(&feedback, "lo").as_ref(), 0);
                gl.uniform1i(gl.get_uniform_location(&feedback, "hi").as_ref(), 1);
                gl.enable(Gl::RASTERIZER_DISCARD);
                gl.begin_transform_feedback(Gl::POINTS);
                gl.draw_arrays(Gl::POINTS, 0, words_i32);
                gl.end_transform_feedback();
                check_stage(gl, "transform feedback")?;
                gl.bind_transform_feedback(Gl::TRANSFORM_FEEDBACK, None);
                gl.bind_buffer(Gl::COPY_READ_BUFFER, Some(buffer));
                gl.bind_buffer(Gl::COPY_WRITE_BUFFER, Some(&destination.raw));
                for row in 0..extent[1] {
                    gl.copy_buffer_sub_data_with_f64_and_f64_and_f64(
                        Gl::COPY_READ_BUFFER,
                        Gl::COPY_WRITE_BUFFER,
                        (u64::from(row) * u64::from(words_per_row) * 4) as f64,
                        (offset + u64::from(row) * u64::from(bytes_per_row)) as f64,
                        extent[0] as f64,
                    );
                }
                check_stage(gl, "output copy")?;
                drop(state);
                Ok(())
            })();
        });
        // Keep the first lowering diagnostic if cleanup observes another GL
        // error, while a lost context always remains the terminal result.
        if matches!(outer, Err(Error::DeviceLost)) {
            return outer;
        }
        result.and(outer)
    }
}
