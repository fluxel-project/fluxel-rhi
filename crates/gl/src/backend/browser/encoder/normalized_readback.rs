//! Exact normalized-integer readback through immediate transform feedback.
//!
//! The vertex shader reconstructs only injective UNORM encodings into their
//! original bits. GPU buffer copies move the packed words to caller-provided
//! padded rows. Private GL objects and borrowed state are scoped to this call;
//! this is neither command replay nor resource-use planning.
use fluxel_rhi_core::api::{Error, Result};
use js_sys::Array;
use wasm_bindgen::JsValue;
use web_sys::{
    WebGl2RenderingContext as Gl, WebGlBuffer, WebGlProgram, WebGlSampler, WebGlShader,
    WebGlTexture, WebGlTransformFeedback, WebGlVertexArrayObject,
};

use super::super::super::api::GlFormat;
use super::super::owner::{
    BrowserBuffer, BrowserEncoder, BrowserTexture, NormalizedReadbackCache,
    NormalizedReadbackProgram, NormalizedReadbackTextureKind,
};

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
pub(crate) fn supports_format(format: GlFormat, highp_supported: bool) -> bool {
    Encoding::for_format(format).is_some() && highp_supported
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
    fn capture(gl: &'a Gl, texture_target: u32, _texture_binding: u32) -> Result<Self> {
        // The direct backend is the sole context owner. Restore a neutral
        // transfer state instead of asking WebGL to synchronously snapshot
        // the previous draw bindings on every normalized readback.
        Ok(Self {
            gl,
            program: None,
            vao: None,
            feedback: None,
            feedback_buffer: None,
            copy_read: None,
            copy_write: None,
            texture: None,
            texture_target,
            sampler: None,
            active: Gl::TEXTURE0,
            discard: false,
            captured: true,
        })
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
    buffer: Option<WebGlBuffer>,
    feedback: Option<WebGlTransformFeedback>,
    vao: Option<WebGlVertexArrayObject>,
    sampler: Option<WebGlSampler>,
}

struct Scratch<'a> {
    gl: &'a Gl,
    objects: Objects,
}

impl Drop for Scratch<'_> {
    fn drop(&mut self) {
        let gl = self.gl;
        gl.delete_sampler(self.objects.sampler.as_ref());
        gl.delete_vertex_array(self.objects.vao.as_ref());
        gl.delete_transform_feedback(self.objects.feedback.as_ref());
        gl.delete_buffer(self.objects.buffer.as_ref());
    }
}

fn program<'a>(
    gl: &Gl,
    cache: &'a mut NormalizedReadbackCache,
    key: (GlFormat, NormalizedReadbackTextureKind),
) -> Result<&'a NormalizedReadbackProgram> {
    if !cache.programs.contains_key(&key) {
        let source = vertex_source(key.0, key.1);
        let make = |kind, text: &str| -> Result<WebGlShader> {
            let shader = gl.create_shader(kind).ok_or(Error::OutOfMemory)?;
            gl.shader_source(&shader, text);
            gl.compile_shader(&shader);
            if gl
                .get_shader_parameter(&shader, Gl::COMPILE_STATUS)
                .as_bool()
                != Some(true)
            {
                let message = gl.get_shader_info_log(&shader).unwrap_or_default();
                gl.delete_shader(Some(&shader));
                return Err(Error::Backend(format!(
                    "WebGL normalized readback shader: {message}"
                )));
            }
            Ok(shader)
        };
        let vertex = make(Gl::VERTEX_SHADER, &source)?;
        let fragment = match make(
            Gl::FRAGMENT_SHADER,
            "#version 300 es\nprecision highp float;void main(){}",
        ) {
            Ok(value) => value,
            Err(error) => {
                gl.delete_shader(Some(&vertex));
                return Err(error);
            }
        };
        let Some(raw) = gl.create_program() else {
            gl.delete_shader(Some(&vertex));
            gl.delete_shader(Some(&fragment));
            return Err(Error::OutOfMemory);
        };
        gl.attach_shader(&raw, &vertex);
        gl.attach_shader(&raw, &fragment);
        gl.transform_feedback_varyings(
            &raw,
            &Array::of1(&JsValue::from_str("normalized_word")),
            Gl::INTERLEAVED_ATTRIBS,
        );
        gl.link_program(&raw);
        gl.delete_shader(Some(&vertex));
        gl.delete_shader(Some(&fragment));
        if gl.get_program_parameter(&raw, Gl::LINK_STATUS).as_bool() != Some(true) {
            let message = gl.get_program_info_log(&raw).unwrap_or_default();
            gl.delete_program(Some(&raw));
            return Err(Error::Backend(format!(
                "WebGL normalized readback link: {message}"
            )));
        };
        let uniform = |name: &str| {
            gl.get_uniform_location(&raw, name).ok_or_else(|| {
                Error::Backend(format!("WebGL normalized readback {name} uniform missing"))
            })
        };
        let image = match uniform("image") {
            Ok(value) => value,
            Err(error) => {
                gl.delete_program(Some(&raw));
                return Err(error);
            }
        };
        let origin = match uniform("origin") {
            Ok(value) => value,
            Err(error) => {
                gl.delete_program(Some(&raw));
                return Err(error);
            }
        };
        let mip = match uniform("mip") {
            Ok(value) => value,
            Err(error) => {
                gl.delete_program(Some(&raw));
                return Err(error);
            }
        };
        let words_per_row = match uniform("words_per_row") {
            Ok(value) => value,
            Err(error) => {
                gl.delete_program(Some(&raw));
                return Err(error);
            }
        };
        let words_per_layer = if key.1 == NormalizedReadbackTextureKind::D2 {
            None
        } else {
            match uniform("words_per_layer") {
                Ok(value) => Some(value),
                Err(error) => {
                    gl.delete_program(Some(&raw));
                    return Err(error);
                }
            }
        };
        let elements_per_row =
            if matches!(Encoding::for_format(key.0), Some(Encoding::Unorm16 { .. })) {
                match uniform("elements_per_row") {
                    Ok(value) => Some(value),
                    Err(error) => {
                        gl.delete_program(Some(&raw));
                        return Err(error);
                    }
                }
            } else {
                None
            };
        cache.programs.insert(
            key,
            NormalizedReadbackProgram {
                raw,
                image,
                origin,
                mip,
                words_per_row,
                words_per_layer,
                elements_per_row,
            },
        );
    }
    Ok(cache
        .programs
        .get(&key)
        .expect("cached normalized readback helper"))
}

fn vertex_source(format: GlFormat, texture_kind: NormalizedReadbackTextureKind) -> String {
    let encoding = Encoding::for_format(format).expect("normalized readback format key");
    let (sampler, layer, coordinate) = match texture_kind {
        NormalizedReadbackTextureKind::D2 => (
            "sampler2D",
            "uint in_layer=v;",
            "ivec2 q=ivec2(int(x+origin.x),int(y+origin.y));",
        ),
        NormalizedReadbackTextureKind::D2Array => (
            "sampler2DArray",
            "uint layer=v/words_per_layer, in_layer=v%words_per_layer;",
            "ivec3 q=ivec3(int(x+origin.x),int(y+origin.y),int(layer+origin.z));",
        ),
        NormalizedReadbackTextureKind::D3 => (
            "sampler3D",
            "uint layer=v/words_per_layer, in_layer=v%words_per_layer;",
            "ivec3 q=ivec3(int(x+origin.x),int(y+origin.y),int(layer+origin.z));",
        ),
    };
    let layer_uniform = if texture_kind == NormalizedReadbackTextureKind::D2 {
        ""
    } else {
        "uniform highp uint words_per_layer;"
    };
    match encoding {
        Encoding::Unorm16 { channels } => format!(
            r#"#version 300 es
precision highp float;
precision highp int;
uniform highp {sampler} image;
uniform highp uvec3 origin;
uniform highp uint mip, words_per_row, elements_per_row;
{layer_uniform}
flat out uint normalized_word;
void main() {{
  uint v=uint(gl_VertexID); {layer}
  uint component=(in_layer%words_per_row)*2u, y=in_layer/words_per_row;
  uint x=component/{channels}u, k=component%{channels}u;
  {coordinate}
  vec4 a_sample=texelFetch(image,q,int(mip));
  uint a=uint(round(clamp(a_sample[int(k)],0.,1.)*65535.));
  uint b=0u;
  if(component+1u<elements_per_row) {{
    uint b_component=component+1u, bx=b_component/{channels}u, bk=b_component%{channels}u;
    q.x=int(bx+origin.x);
    vec4 b_sample=texelFetch(image,q,int(mip));
    b=uint(round(clamp(b_sample[int(bk)],0.,1.)*65535.));
  }}
  normalized_word=a|(b<<16u);
  gl_Position=vec4(0.);
}}"#,
        ),
        Encoding::Rgb10a2 => format!(
            r#"#version 300 es
precision highp float;
precision highp int;
uniform highp {sampler} image;
uniform highp uvec3 origin;
uniform highp uint mip, words_per_row;
{layer_uniform}
flat out uint normalized_word;
void main() {{
  uint v=uint(gl_VertexID); {layer}
  uint x=in_layer%words_per_row, y=in_layer/words_per_row;
  {coordinate}
  vec4 sample_value=clamp(texelFetch(image,q,int(mip)),0.,1.);
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
        use crate::backend::api::{GlTextureDimension, GlTextureUsage};
        self.same_owner(&source.owner)?;
        self.same_owner(&dst.owner)?;
        let encoding = Encoding::for_format(source.desc.format).ok_or(Error::Unsupported(
            "WebGL normalized transform-feedback format",
        ))?;
        let (texture_target, texture_binding, texture_kind) = match source.desc.dimension {
            GlTextureDimension::D2 => (
                Gl::TEXTURE_2D,
                Gl::TEXTURE_BINDING_2D,
                NormalizedReadbackTextureKind::D2,
            ),
            GlTextureDimension::D2Array => (
                Gl::TEXTURE_2D_ARRAY,
                Gl::TEXTURE_BINDING_2D_ARRAY,
                NormalizedReadbackTextureKind::D2Array,
            ),
            GlTextureDimension::D3 if source.desc.format != GlFormat::Depth16Unorm => (
                Gl::TEXTURE_3D,
                Gl::TEXTURE_BINDING_3D,
                NormalizedReadbackTextureKind::D3,
            ),
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
            || (texture_kind == NormalizedReadbackTextureKind::D2
                && (origin[2] != 0 || extent[2] != 1))
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
        self.with_normalized_readback_context("normalized_transform_feedback", |gl, cache| {
            result = (|| {
                let mut scratch = Scratch {
                    gl,
                    objects: Objects::default(),
                };
                if !cache.highp_supported {
                    return Err(Error::Unsupported(
                        "WebGL vertex highp cannot reconstruct normalized texture bits",
                    ));
                }
                let program = program(gl, cache, (source.desc.format, texture_kind))?;
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
                gl.use_program(Some(&program.raw));
                gl.uniform1i(Some(&program.image), 0);
                gl.uniform3ui(Some(&program.origin), origin[0], origin[1], origin[2]);
                gl.uniform1ui(Some(&program.mip), mip);
                gl.uniform1ui(Some(&program.words_per_row), words_per_row);
                if let Some(words_per_layer_location) = &program.words_per_layer {
                    gl.uniform1ui(Some(words_per_layer_location), words_per_layer);
                }
                if let Some(elements_per_row) = &program.elements_per_row {
                    gl.uniform1ui(
                        Some(elements_per_row),
                        extent[0]
                            .checked_mul(match encoding {
                                Encoding::Unorm16 { channels } => channels,
                                Encoding::Rgb10a2 => unreachable!("RGB10A2 has no element uniform"),
                            })
                            .ok_or(Error::InvalidRange)?,
                    );
                }
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
