//! Exact normalized-integer readback through immediate transform feedback.
//!
//! The vertex shader reconstructs only injective UNORM encodings into their
//! original bits. GPU buffer copies move the packed words to caller-provided
//! padded rows. Private GL objects and borrowed state are scoped to this call;
//! this is neither command replay nor resource-use planning.
use fluxel_rhi_core::api::{Error, Result};
use js_sys::Array;
use wasm_bindgen::{JsCast, JsValue};
use web_sys::{
    WebGl2RenderingContext as Gl, WebGlBuffer, WebGlProgram, WebGlSampler, WebGlShader,
    WebGlTexture, WebGlTransformFeedback, WebGlVertexArrayObject,
};

use super::super::super::api::GlFormat;
use super::super::owner::{BrowserBuffer, BrowserEncoder, BrowserTexture};

#[derive(Clone, Copy)]
enum Encoding {
    Unorm16 { channels: u32 },
    Rgb10a2,
}

impl Encoding {
    fn for_format(format: GlFormat) -> Option<Self> {
        match format {
            GlFormat::Depth16Unorm | GlFormat::R16Unorm => Some(Self::Unorm16 { channels: 1 }),
            GlFormat::Rg16Unorm => Some(Self::Unorm16 { channels: 2 }),
            GlFormat::Rgba16Unorm => Some(Self::Unorm16 { channels: 4 }),
            GlFormat::Rgb10a2Unorm => Some(Self::Rgb10a2),
            _ => None,
        }
    }

    const fn bytes_per_pixel(self) -> u64 {
        match self {
            Self::Unorm16 { channels } => channels as u64 * 2,
            Self::Rgb10a2 => 4,
        }
    }
}

/// Returns whether the browser's vertex `highp` has enough mantissa bits for
/// the exact integer reconstruction used by the normalized TF fallback.
pub(crate) fn supports_format(raw: &Gl, format: GlFormat) -> bool {
    Encoding::for_format(format).is_some()
        && raw
            .get_shader_precision_format(Gl::VERTEX_SHADER, Gl::HIGH_FLOAT)
            .is_some_and(|value| value.precision() >= 23)
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

struct State<'a> {
    gl: &'a Gl,
    program: Option<WebGlProgram>,
    vao: Option<WebGlVertexArrayObject>,
    feedback: Option<WebGlTransformFeedback>,
    feedback_buffer: Option<WebGlBuffer>,
    copy_read: Option<WebGlBuffer>,
    copy_write: Option<WebGlBuffer>,
    texture: Option<WebGlTexture>,
    texture_target: u32,
    sampler: Option<WebGlSampler>,
    active: u32,
    discard: bool,
    captured: bool,
}

impl<'a> State<'a> {
    fn capture(gl: &'a Gl, texture_target: u32, texture_binding: u32) -> Result<Self> {
        if gl
            .get_parameter(Gl::TRANSFORM_FEEDBACK_ACTIVE)
            .map_err(|_| Error::InvalidState)?
            .as_bool()
            != Some(false)
        {
            return Err(Error::InvalidState);
        }
        let mut state = Self {
            gl,
            program: object(gl, Gl::CURRENT_PROGRAM)?,
            vao: object(gl, Gl::VERTEX_ARRAY_BINDING)?,
            feedback: object(gl, Gl::TRANSFORM_FEEDBACK_BINDING)?,
            feedback_buffer: object(gl, Gl::TRANSFORM_FEEDBACK_BUFFER_BINDING)?,
            copy_read: object(gl, Gl::COPY_READ_BUFFER_BINDING)?,
            copy_write: object(gl, Gl::COPY_WRITE_BUFFER_BINDING)?,
            texture: None,
            texture_target,
            sampler: None,
            active: gl
                .get_parameter(Gl::ACTIVE_TEXTURE)
                .map_err(|_| Error::InvalidState)?
                .as_f64()
                .ok_or(Error::InvalidState)? as u32,
            discard: gl.is_enabled(Gl::RASTERIZER_DISCARD),
            captured: false,
        };
        // Capture unit zero without changing it. Until both queries succeed,
        // only ACTIVE_TEXTURE has changed and must be restored on failure.
        gl.active_texture(Gl::TEXTURE0);
        let texture = object(gl, texture_binding);
        let sampler = object(gl, Gl::SAMPLER_BINDING);
        gl.active_texture(state.active);
        state.texture = texture?;
        state.sampler = sampler?;
        state.captured = true;
        Ok(state)
    }
}

impl Drop for State<'_> {
    fn drop(&mut self) {
        let gl = self.gl;
        if !self.captured {
            // Unit-zero capture failed before any binding was changed.
            gl.active_texture(self.active);
            return;
        }
        gl.bind_vertex_array(self.vao.as_ref());
        // Indexed feedback bindings belong to the feedback object. Rebinding
        // the untouched original object preserves its buffer ranges exactly.
        gl.bind_transform_feedback(Gl::TRANSFORM_FEEDBACK, self.feedback.as_ref());
        gl.bind_buffer(Gl::TRANSFORM_FEEDBACK_BUFFER, self.feedback_buffer.as_ref());
        gl.bind_buffer(Gl::COPY_READ_BUFFER, self.copy_read.as_ref());
        gl.bind_buffer(Gl::COPY_WRITE_BUFFER, self.copy_write.as_ref());
        gl.active_texture(Gl::TEXTURE0);
        gl.bind_texture(self.texture_target, self.texture.as_ref());
        gl.bind_sampler(0, self.sampler.as_ref());
        gl.active_texture(self.active);
        gl.use_program(self.program.as_ref());
        if self.discard {
            gl.enable(Gl::RASTERIZER_DISCARD);
        } else {
            gl.disable(Gl::RASTERIZER_DISCARD);
        }
    }
}

#[derive(Default)]
struct Objects {
    shaders: Vec<WebGlShader>,
    program: Option<WebGlProgram>,
    buffer: Option<WebGlBuffer>,
    feedback: Option<WebGlTransformFeedback>,
    vao: Option<WebGlVertexArrayObject>,
    sampler: Option<WebGlSampler>,
}

struct Scratch<'a> {
    gl: &'a Gl,
    objects: Objects,
}

impl Scratch<'_> {
    fn shader(&mut self, kind: u32, source: &str) -> Result<WebGlShader> {
        let shader = self.gl.create_shader(kind).ok_or(Error::OutOfMemory)?;
        self.objects.shaders.push(shader.clone());
        self.gl.shader_source(&shader, source);
        self.gl.compile_shader(&shader);
        if self
            .gl
            .get_shader_parameter(&shader, Gl::COMPILE_STATUS)
            .as_bool()
            != Some(true)
        {
            return Err(Error::Backend(format!(
                "WebGL D16 shader: {}",
                self.gl.get_shader_info_log(&shader).unwrap_or_default()
            )));
        }
        Ok(shader)
    }
}

impl Drop for Scratch<'_> {
    fn drop(&mut self) {
        let gl = self.gl;
        gl.delete_sampler(self.objects.sampler.as_ref());
        gl.delete_vertex_array(self.objects.vao.as_ref());
        gl.delete_transform_feedback(self.objects.feedback.as_ref());
        gl.delete_buffer(self.objects.buffer.as_ref());
        gl.delete_program(self.objects.program.as_ref());
        for shader in &self.objects.shaders {
            gl.delete_shader(Some(shader));
        }
    }
}

fn vertex_source(
    encoding: Encoding,
    texture_kind: &str,
    words_per_row: u32,
    words_per_layer: u32,
    mip: u32,
    origin: [u32; 3],
    extent: [u32; 3],
) -> String {
    let (sampler, coordinate, fetch) = match texture_kind {
        "2D" => (
            "sampler2D",
            format!("ivec2 q=ivec2(int(x)+{},int(y)+{});", origin[0], origin[1]),
            format!("texelFetch(image,q,{mip})"),
        ),
        "2DArray" => (
            "sampler2DArray",
            format!(
                "ivec3 q=ivec3(int(x)+{},int(y)+{},int(layer)+{});",
                origin[0], origin[1], origin[2]
            ),
            format!("texelFetch(image,q,{mip})"),
        ),
        "3D" => (
            "sampler3D",
            format!(
                "ivec3 q=ivec3(int(x)+{},int(y)+{},int(layer)+{});",
                origin[0], origin[1], origin[2]
            ),
            format!("texelFetch(image,q,{mip})"),
        ),
        _ => unreachable!("validated normalized texture kind"),
    };
    let layer = if texture_kind == "2D" {
        "uint layer=0u, in_layer=v;".to_owned()
    } else {
        format!("uint layer=v/{words_per_layer}u, in_layer=v%{words_per_layer}u;")
    };
    match encoding {
        Encoding::Unorm16 { channels } => format!(
            r#"#version 300 es
precision highp float;
precision highp int;
uniform highp {sampler} image;
flat out uint normalized_word;
void main() {{
  uint v=uint(gl_VertexID); {layer}
  uint component=(in_layer%{words_per_row}u)*2u, y=in_layer/{words_per_row}u;
  uint x=component/{channels}u, k=component%{channels}u;
  {coordinate}
  vec4 a_sample={fetch};
  uint a=uint(round(clamp(a_sample[int(k)],0.,1.)*65535.));
  uint b=0u;
  if(component+1u<{}u) {{
    uint b_component=component+1u, bx=b_component/{channels}u, bk=b_component%{channels}u;
    {coordinate_b}
    vec4 b_sample={fetch_b};
    b=uint(round(clamp(b_sample[int(bk)],0.,1.)*65535.));
  }}
  normalized_word=a|(b<<16u);
  gl_Position=vec4(0.);
}}"#,
            extent[0] * channels,
            coordinate_b = coordinate.replace("int(x)", "int(bx)"),
            fetch_b = fetch,
        ),
        Encoding::Rgb10a2 => format!(
            r#"#version 300 es
precision highp float;
precision highp int;
uniform highp {sampler} image;
flat out uint normalized_word;
void main() {{
  uint v=uint(gl_VertexID); {layer}
  uint x=in_layer%{words_per_row}u, y=in_layer/{words_per_row}u;
  {coordinate}
  vec4 sample_value=clamp({fetch},0.,1.);
  uvec4 bits=uvec4(round(sample_value*vec4(1023.,1023.,1023.,3.)));
  normalized_word=bits.x|(bits.y<<10u)|(bits.z<<20u)|(bits.w<<30u);
  gl_Position=vec4(0.);
}}"#,
        ),
    }
}

impl BrowserEncoder {
    pub(crate) fn copy_normalized_texture_to_buffer(
        &mut self,
        source: &BrowserTexture,
        mip: u32,
        origin: [u32; 3],
        extent: [u32; 3],
        dst: &BrowserBuffer,
        offset: u64,
        bytes_per_row: u32,
        rows_per_image: u32,
    ) -> Result<()> {
        use crate::backend::gl::api::{GlTextureDimension, GlTextureUsage};
        self.same_owner(&source.owner)?;
        self.same_owner(&dst.owner)?;
        let encoding = Encoding::for_format(source.desc.format).ok_or(Error::Unsupported(
            "WebGL normalized transform-feedback format",
        ))?;
        let (texture_target, texture_binding, texture_kind) = match source.desc.dimension {
            GlTextureDimension::D2 => (Gl::TEXTURE_2D, Gl::TEXTURE_BINDING_2D, "2D"),
            GlTextureDimension::D2Array => (
                Gl::TEXTURE_2D_ARRAY,
                Gl::TEXTURE_BINDING_2D_ARRAY,
                "2DArray",
            ),
            GlTextureDimension::D3 if source.desc.format != GlFormat::Depth16Unorm => {
                (Gl::TEXTURE_3D, Gl::TEXTURE_BINDING_3D, "3D")
            }
            _ => {
                return Err(Error::Unsupported(
                    "WebGL normalized feedback readback dimension",
                ));
            }
        };
        if !source.desc.usage.contains(GlTextureUsage::COPY_SOURCE) || source.desc.sample_count != 1
        {
            return Err(Error::InvalidState);
        }
        let bound = source.desc.mip_extent(mip).ok_or(Error::InvalidRange)?;
        let row_bytes = u64::from(extent[0]) * encoding.bytes_per_pixel();
        if extent.contains(&0)
            || origin[0]
                .checked_add(extent[0])
                .is_none_or(|v| v > bound.width)
            || origin[1]
                .checked_add(extent[1])
                .is_none_or(|v| v > bound.height)
            || origin[2]
                .checked_add(extent[2])
                .is_none_or(|v| v > bound.depth_or_layers)
            || (texture_kind == "2D" && (origin[2] != 0 || extent[2] != 1))
            || u64::from(bytes_per_row) < row_bytes
            || rows_per_image < extent[1]
        {
            return Err(Error::InvalidRange);
        }
        let layer_stride = u64::from(bytes_per_row)
            .checked_mul(u64::from(rows_per_image))
            .ok_or(Error::InvalidRange)?;
        let end = offset
            .checked_add(
                layer_stride
                    .checked_mul(u64::from(extent[2] - 1))
                    .ok_or(Error::InvalidRange)?,
            )
            .and_then(|v| v.checked_add(u64::from(bytes_per_row) * u64::from(extent[1] - 1)))
            .and_then(|v| v.checked_add(row_bytes))
            .ok_or(Error::InvalidRange)?;
        if end > dst.size || end > (1u64 << 53) - 1 {
            return Err(Error::InvalidRange);
        }
        let words_per_row = match encoding {
            Encoding::Unorm16 { channels } => extent[0]
                .checked_mul(channels)
                .ok_or(Error::InvalidRange)?
                .div_ceil(2),
            Encoding::Rgb10a2 => extent[0],
        };
        let words_per_layer = words_per_row
            .checked_mul(extent[1])
            .ok_or(Error::InvalidRange)?;
        let words = words_per_layer
            .checked_mul(extent[2])
            .ok_or(Error::InvalidRange)?;
        let scratch_bytes = i32::try_from(u64::from(words) * 4).map_err(|_| Error::InvalidRange)?;
        let vertices = i32::try_from(words).map_err(|_| Error::InvalidRange)?;
        let texture = source.texture()?;
        let mut result = Ok(());
        self.with_context("normalized_transform_feedback", |gl| {
            result = (|| {
                let mut scratch = Scratch {
                    gl,
                    objects: Objects::default(),
                };
                if !supports_format(gl, source.desc.format) {
                    return Err(Error::Unsupported(
                        "WebGL vertex highp cannot reconstruct normalized texture bits",
                    ));
                }
                let vertex_source = vertex_source(
                    encoding,
                    texture_kind,
                    words_per_row,
                    words_per_layer,
                    mip,
                    origin,
                    extent,
                );
                let vs = scratch.shader(Gl::VERTEX_SHADER, &vertex_source)?;
                let fs = scratch.shader(
                    Gl::FRAGMENT_SHADER,
                    "#version 300 es\nprecision highp float; void main(){}",
                )?;
                scratch.objects.program = Some(gl.create_program().ok_or(Error::OutOfMemory)?);
                let program = scratch.objects.program.as_ref().unwrap();
                gl.attach_shader(program, &vs);
                gl.attach_shader(program, &fs);
                gl.transform_feedback_varyings(
                    program,
                    &Array::of1(&JsValue::from_str("normalized_word")),
                    Gl::INTERLEAVED_ATTRIBS,
                );
                gl.link_program(program);
                if gl.get_program_parameter(program, Gl::LINK_STATUS).as_bool() != Some(true) {
                    return Err(Error::Backend(format!(
                        "WebGL normalized readback link: {}",
                        gl.get_program_info_log(program).unwrap_or_default()
                    )));
                }
                scratch.objects.buffer = Some(gl.create_buffer().ok_or(Error::OutOfMemory)?);
                scratch.objects.feedback =
                    Some(gl.create_transform_feedback().ok_or(Error::OutOfMemory)?);
                scratch.objects.vao = Some(gl.create_vertex_array().ok_or(Error::OutOfMemory)?);
                scratch.objects.sampler = Some(gl.create_sampler().ok_or(Error::OutOfMemory)?);
                // Declared last: restore bindings before deleting private objects.
                let _state = State::capture(gl, texture_target, texture_binding)?;
                let buffer = scratch.objects.buffer.as_ref().unwrap();
                let sampler = scratch.objects.sampler.as_ref().unwrap();
                gl.sampler_parameteri(
                    sampler,
                    Gl::TEXTURE_MIN_FILTER,
                    Gl::NEAREST_MIPMAP_NEAREST as i32,
                );
                gl.sampler_parameteri(sampler, Gl::TEXTURE_MAG_FILTER, Gl::NEAREST as i32);
                gl.bind_vertex_array(scratch.objects.vao.as_ref());
                gl.bind_transform_feedback(
                    Gl::TRANSFORM_FEEDBACK,
                    scratch.objects.feedback.as_ref(),
                );
                gl.bind_buffer(Gl::TRANSFORM_FEEDBACK_BUFFER, Some(buffer));
                gl.buffer_data_with_i32(
                    Gl::TRANSFORM_FEEDBACK_BUFFER,
                    scratch_bytes,
                    Gl::STREAM_COPY,
                );
                gl.bind_buffer_base(Gl::TRANSFORM_FEEDBACK_BUFFER, 0, Some(buffer));
                gl.active_texture(Gl::TEXTURE0);
                gl.bind_texture(texture_target, Some(texture));
                gl.bind_sampler(0, Some(sampler));
                gl.use_program(Some(program));
                gl.uniform1i(gl.get_uniform_location(program, "image").as_ref(), 0);
                gl.enable(Gl::RASTERIZER_DISCARD);
                gl.begin_transform_feedback(Gl::POINTS);
                gl.draw_arrays(Gl::POINTS, 0, vertices);
                gl.end_transform_feedback();
                // Detach the private TF object before using its buffer as a
                // copy source; the original object's indexed ranges stay intact.
                gl.bind_transform_feedback(Gl::TRANSFORM_FEEDBACK, None);
                gl.bind_buffer(Gl::COPY_READ_BUFFER, Some(buffer));
                gl.bind_buffer(Gl::COPY_WRITE_BUFFER, Some(&dst.raw));
                for layer in 0..extent[2] {
                    for y in 0..extent[1] {
                        gl.copy_buffer_sub_data_with_f64_and_f64_and_f64(
                            Gl::COPY_READ_BUFFER,
                            Gl::COPY_WRITE_BUFFER,
                            (u64::from(layer) * u64::from(words_per_layer)
                                + u64::from(y) * u64::from(words_per_row))
                                as f64
                                * 4.,
                            (offset
                                + u64::from(layer) * layer_stride
                                + u64::from(y) * u64::from(bytes_per_row))
                                as f64,
                            row_bytes as f64,
                        );
                    }
                }
                Ok(())
            })();
        })?;
        result
    }
}
