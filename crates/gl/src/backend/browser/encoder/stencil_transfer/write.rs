//! Immediate byte-plane writes for WebGL2 stencil attachments.
//!
//! A private R8UI texture receives the caller's plane. Eight discard shaders
//! then replace the corresponding stencil bit only where that input bit is
//! one; a scissored clear establishes the zero bits first. This works for an
//! S8 renderbuffer and for the stencil plane of a packed depth/stencil texture,
//! while leaving the packed depth plane untouched.

use fluxel_rhi_core::api::{Error, Result};
use js_sys::{Array, Int32Array, Uint8Array};
use wasm_bindgen::JsCast;
use web_sys::{
    WebGl2RenderingContext as Gl, WebGlBuffer, WebGlFramebuffer, WebGlProgram, WebGlRenderbuffer,
    WebGlSampler, WebGlShader, WebGlTexture, WebGlTransformFeedback, WebGlVertexArrayObject,
};

use super::super::super::owner::{BrowserBuffer, BrowserEncoder, BrowserTexture};
use crate::backend::api::{GlFormat, GlTextureDimension, GlTextureUsage};

const RED_INTEGER: u32 = 0x8D94;
const STENCIL_BACK_FAIL: u32 = 0x8801;
const STENCIL_BACK_PASS_DEPTH_FAIL: u32 = 0x8802;
const STENCIL_BACK_PASS_DEPTH_PASS: u32 = 0x8803;
const STENCIL_BACK_VALUE_MASK: u32 = 0x8CA4;
const STENCIL_BACK_WRITEMASK: u32 = 0x8CA5;

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

fn integer(gl: &Gl, parameter: u32) -> Result<i32> {
    gl.get_parameter(parameter)
        .map_err(|_| Error::InvalidState)?
        .as_f64()
        .map(|value| value as i32)
        .ok_or(Error::InvalidState)
}

fn ints(gl: &Gl, parameter: u32) -> Result<[i32; 4]> {
    let values = Int32Array::new(
        &gl.get_parameter(parameter)
            .map_err(|_| Error::InvalidState)?,
    )
    .to_vec();
    values.try_into().map_err(|_| Error::InvalidState)
}

fn bools(gl: &Gl, parameter: u32) -> Result<[bool; 4]> {
    let value = gl
        .get_parameter(parameter)
        .map_err(|_| Error::InvalidState)?;
    let values = Array::from(&value);
    if values.length() != 4 {
        return Err(Error::InvalidState);
    }
    Ok([
        values.get(0).as_bool().ok_or(Error::InvalidState)?,
        values.get(1).as_bool().ok_or(Error::InvalidState)?,
        values.get(2).as_bool().ok_or(Error::InvalidState)?,
        values.get(3).as_bool().ok_or(Error::InvalidState)?,
    ])
}

#[derive(Clone, Copy)]
struct Face {
    func: i32,
    reference: i32,
    value_mask: u32,
    write_mask: u32,
    fail: i32,
    depth_fail: i32,
    pass: i32,
}

impl Face {
    fn read(gl: &Gl, back: bool) -> Result<Self> {
        let (func, reference, value_mask, write_mask, fail, depth_fail, pass) = if back {
            (
                Gl::STENCIL_BACK_FUNC,
                Gl::STENCIL_BACK_REF,
                STENCIL_BACK_VALUE_MASK,
                STENCIL_BACK_WRITEMASK,
                STENCIL_BACK_FAIL,
                STENCIL_BACK_PASS_DEPTH_FAIL,
                STENCIL_BACK_PASS_DEPTH_PASS,
            )
        } else {
            (
                Gl::STENCIL_FUNC,
                Gl::STENCIL_REF,
                Gl::STENCIL_VALUE_MASK,
                Gl::STENCIL_WRITEMASK,
                Gl::STENCIL_FAIL,
                Gl::STENCIL_PASS_DEPTH_FAIL,
                Gl::STENCIL_PASS_DEPTH_PASS,
            )
        };
        Ok(Self {
            func: integer(gl, func)?,
            reference: integer(gl, reference)?,
            value_mask: integer(gl, value_mask)? as u32,
            write_mask: integer(gl, write_mask)? as u32,
            fail: integer(gl, fail)?,
            depth_fail: integer(gl, depth_fail)?,
            pass: integer(gl, pass)?,
        })
    }
    fn restore(self, gl: &Gl, face: u32) {
        gl.stencil_func_separate(face, self.func as u32, self.reference, self.value_mask);
        gl.stencil_mask_separate(face, self.write_mask);
        gl.stencil_op_separate(
            face,
            self.fail as u32,
            self.depth_fail as u32,
            self.pass as u32,
        );
    }
}

/// Native state changed by the private FBO and eight draws.
///
/// The encoder has no active raster pass when this is created. Still, every
/// binding and fixed state this lowering changes is restored before private
/// handles are destroyed, so a later immediate verb starts from its own facts.
pub(super) struct State<'a> {
    gl: &'a Gl,
    draw: Option<WebGlFramebuffer>,
    read: Option<WebGlFramebuffer>,
    program: Option<WebGlProgram>,
    vao: Option<WebGlVertexArrayObject>,
    renderbuffer: Option<WebGlRenderbuffer>,
    unpack: Option<WebGlBuffer>,
    copy_read: Option<WebGlBuffer>,
    copy_write: Option<WebGlBuffer>,
    feedback_buffer: Option<WebGlBuffer>,
    unpack_alignment: i32,
    unpack_row_length: i32,
    unpack_image_height: i32,
    unpack_skip_pixels: i32,
    unpack_skip_rows: i32,
    unpack_skip_images: i32,
    unpack_flip_y: bool,
    feedback: Option<WebGlTransformFeedback>,
    feedback_active: bool,
    active_texture: u32,
    unit0_texture: Option<WebGlTexture>,
    unit0_sampler: Option<WebGlSampler>,
    unit1_texture: Option<WebGlTexture>,
    unit1_sampler: Option<WebGlSampler>,
    viewport: [i32; 4],
    scissor_box: [i32; 4],
    scissor: bool,
    depth: bool,
    depth_write: bool,
    stencil: bool,
    blend: bool,
    discard: bool,
    dither: bool,
    cull: bool,
    sample_coverage: bool,
    color_mask: [bool; 4],
    front: Face,
    back: Face,
    captured: bool,
}

impl<'a> State<'a> {
    pub(super) fn capture(gl: &'a Gl) -> Result<Self> {
        let feedback_active = gl
            .get_parameter(Gl::TRANSFORM_FEEDBACK_ACTIVE)
            .map_err(|_| Error::InvalidState)?
            .as_bool()
            == Some(true);
        if feedback_active {
            return Err(Error::InvalidState);
        }
        let active_texture = integer(gl, Gl::ACTIVE_TEXTURE)? as u32;
        let mut state = Self {
            gl,
            draw: object(gl, Gl::DRAW_FRAMEBUFFER_BINDING)?,
            read: object(gl, Gl::READ_FRAMEBUFFER_BINDING)?,
            program: object(gl, Gl::CURRENT_PROGRAM)?,
            vao: object(gl, Gl::VERTEX_ARRAY_BINDING)?,
            renderbuffer: object(gl, Gl::RENDERBUFFER_BINDING)?,
            unpack: object(gl, Gl::PIXEL_UNPACK_BUFFER_BINDING)?,
            copy_read: object(gl, Gl::COPY_READ_BUFFER_BINDING)?,
            copy_write: object(gl, Gl::COPY_WRITE_BUFFER_BINDING)?,
            feedback_buffer: object(gl, Gl::TRANSFORM_FEEDBACK_BUFFER_BINDING)?,
            unpack_alignment: integer(gl, Gl::UNPACK_ALIGNMENT)?,
            unpack_row_length: integer(gl, Gl::UNPACK_ROW_LENGTH)?,
            unpack_image_height: integer(gl, Gl::UNPACK_IMAGE_HEIGHT)?,
            unpack_skip_pixels: integer(gl, Gl::UNPACK_SKIP_PIXELS)?,
            unpack_skip_rows: integer(gl, Gl::UNPACK_SKIP_ROWS)?,
            unpack_skip_images: integer(gl, Gl::UNPACK_SKIP_IMAGES)?,
            unpack_flip_y: gl
                .get_parameter(Gl::UNPACK_FLIP_Y_WEBGL)
                .map_err(|_| Error::InvalidState)?
                .as_bool()
                .ok_or(Error::InvalidState)?,
            feedback: object(gl, Gl::TRANSFORM_FEEDBACK_BINDING)?,
            feedback_active,
            active_texture,
            unit0_texture: None,
            unit0_sampler: None,
            unit1_texture: None,
            unit1_sampler: None,
            viewport: ints(gl, Gl::VIEWPORT)?,
            scissor_box: ints(gl, Gl::SCISSOR_BOX)?,
            scissor: gl.is_enabled(Gl::SCISSOR_TEST),
            depth: gl.is_enabled(Gl::DEPTH_TEST),
            depth_write: gl
                .get_parameter(Gl::DEPTH_WRITEMASK)
                .map_err(|_| Error::InvalidState)?
                .as_bool()
                .ok_or(Error::InvalidState)?,
            stencil: gl.is_enabled(Gl::STENCIL_TEST),
            blend: gl.is_enabled(Gl::BLEND),
            discard: gl.is_enabled(Gl::RASTERIZER_DISCARD),
            dither: gl.is_enabled(Gl::DITHER),
            cull: gl.is_enabled(Gl::CULL_FACE),
            sample_coverage: gl.is_enabled(Gl::SAMPLE_COVERAGE),
            color_mask: bools(gl, Gl::COLOR_WRITEMASK)?,
            front: Face::read(gl, false)?,
            back: Face::read(gl, true)?,
            captured: false,
        };
        gl.active_texture(Gl::TEXTURE0);
        let texture = object(gl, Gl::TEXTURE_BINDING_2D);
        let sampler = object(gl, Gl::SAMPLER_BINDING);
        gl.active_texture(Gl::TEXTURE1);
        let texture1 = object(gl, Gl::TEXTURE_BINDING_2D);
        let sampler1 = object(gl, Gl::SAMPLER_BINDING);
        gl.active_texture(active_texture);
        state.unit0_texture = texture?;
        state.unit0_sampler = sampler?;
        state.unit1_texture = texture1?;
        state.unit1_sampler = sampler1?;
        state.captured = true;
        Ok(state)
    }
}

impl Drop for State<'_> {
    fn drop(&mut self) {
        let gl = self.gl;
        if !self.captured {
            gl.active_texture(self.active_texture);
            return;
        }
        gl.bind_framebuffer(Gl::DRAW_FRAMEBUFFER, self.draw.as_ref());
        gl.bind_framebuffer(Gl::READ_FRAMEBUFFER, self.read.as_ref());
        gl.use_program(self.program.as_ref());
        gl.bind_vertex_array(self.vao.as_ref());
        gl.bind_renderbuffer(Gl::RENDERBUFFER, self.renderbuffer.as_ref());
        gl.bind_buffer(Gl::PIXEL_UNPACK_BUFFER, self.unpack.as_ref());
        gl.bind_buffer(Gl::COPY_READ_BUFFER, self.copy_read.as_ref());
        gl.bind_buffer(Gl::COPY_WRITE_BUFFER, self.copy_write.as_ref());
        gl.bind_buffer(Gl::TRANSFORM_FEEDBACK_BUFFER, self.feedback_buffer.as_ref());
        gl.pixel_storei(Gl::UNPACK_ALIGNMENT, self.unpack_alignment);
        gl.pixel_storei(Gl::UNPACK_ROW_LENGTH, self.unpack_row_length);
        gl.pixel_storei(Gl::UNPACK_IMAGE_HEIGHT, self.unpack_image_height);
        gl.pixel_storei(Gl::UNPACK_SKIP_PIXELS, self.unpack_skip_pixels);
        gl.pixel_storei(Gl::UNPACK_SKIP_ROWS, self.unpack_skip_rows);
        gl.pixel_storei(Gl::UNPACK_SKIP_IMAGES, self.unpack_skip_images);
        gl.pixel_storei(Gl::UNPACK_FLIP_Y_WEBGL, i32::from(self.unpack_flip_y));
        gl.bind_transform_feedback(Gl::TRANSFORM_FEEDBACK, self.feedback.as_ref());
        gl.active_texture(Gl::TEXTURE0);
        gl.bind_texture(Gl::TEXTURE_2D, self.unit0_texture.as_ref());
        gl.bind_sampler(0, self.unit0_sampler.as_ref());
        gl.active_texture(Gl::TEXTURE1);
        gl.bind_texture(Gl::TEXTURE_2D, self.unit1_texture.as_ref());
        gl.bind_sampler(1, self.unit1_sampler.as_ref());
        gl.active_texture(self.active_texture);
        gl.viewport(
            self.viewport[0],
            self.viewport[1],
            self.viewport[2],
            self.viewport[3],
        );
        gl.scissor(
            self.scissor_box[0],
            self.scissor_box[1],
            self.scissor_box[2],
            self.scissor_box[3],
        );
        for (enabled, cap) in [
            (self.scissor, Gl::SCISSOR_TEST),
            (self.depth, Gl::DEPTH_TEST),
            (self.stencil, Gl::STENCIL_TEST),
            (self.blend, Gl::BLEND),
            (self.discard, Gl::RASTERIZER_DISCARD),
            (self.dither, Gl::DITHER),
            (self.cull, Gl::CULL_FACE),
            (self.sample_coverage, Gl::SAMPLE_COVERAGE),
        ] {
            if enabled {
                gl.enable(cap);
            } else {
                gl.disable(cap);
            }
        }
        gl.depth_mask(self.depth_write);
        gl.color_mask(
            self.color_mask[0],
            self.color_mask[1],
            self.color_mask[2],
            self.color_mask[3],
        );
        self.front.restore(gl, Gl::FRONT);
        self.back.restore(gl, Gl::BACK);
    }
}

struct Scratch<'a> {
    gl: &'a Gl,
    source: Option<WebGlTexture>,
    staging: Option<WebGlBuffer>,
    framebuffer: Option<WebGlFramebuffer>,
    program: Option<WebGlProgram>,
    vao: Option<WebGlVertexArrayObject>,
    sampler: Option<WebGlSampler>,
    shaders: Vec<WebGlShader>,
}

impl Scratch<'_> {
    fn shader(&mut self, stage: u32, source: &str) -> Result<WebGlShader> {
        let gl = self.gl;
        let shader = gl.create_shader(stage).ok_or(Error::OutOfMemory)?;
        gl.shader_source(&shader, source);
        gl.compile_shader(&shader);
        if gl
            .get_shader_parameter(&shader, Gl::COMPILE_STATUS)
            .as_bool()
            != Some(true)
        {
            gl.delete_shader(Some(&shader));
            return Err(Error::Backend(format!(
                "WebGL stencil transfer shader: {}",
                gl.get_shader_info_log(&shader).unwrap_or_default()
            )));
        }
        self.shaders.push(shader.clone());
        Ok(shader)
    }
    fn program(&mut self) -> Result<&WebGlProgram> {
        let gl = self.gl;
        let vertex = self.shader(Gl::VERTEX_SHADER, "#version 300 es\nvoid main(){vec2 p[3]=vec2[3](vec2(-1.,-1.),vec2(3.,-1.),vec2(-1.,3.));gl_Position=vec4(p[gl_VertexID],0.,1.);}")?;
        let fragment = self.shader(Gl::FRAGMENT_SHADER, "#version 300 es\nprecision highp usampler2D;uniform usampler2D bytes_in;uniform ivec2 origin;uniform int bit_index;void main(){ivec2 p=ivec2(gl_FragCoord.xy)-origin;uint v=texelFetch(bytes_in,p,0).r;if((v&(1u<<uint(bit_index)))==0u)discard;}")?;
        let program = gl.create_program().ok_or(Error::OutOfMemory)?;
        gl.attach_shader(&program, &vertex);
        gl.attach_shader(&program, &fragment);
        gl.link_program(&program);
        if gl
            .get_program_parameter(&program, Gl::LINK_STATUS)
            .as_bool()
            != Some(true)
        {
            gl.delete_program(Some(&program));
            return Err(Error::Backend(format!(
                "WebGL stencil transfer link: {}",
                gl.get_program_info_log(&program).unwrap_or_default()
            )));
        }
        self.program = Some(program);
        Ok(self.program.as_ref().expect("stored program"))
    }
}

impl Drop for Scratch<'_> {
    fn drop(&mut self) {
        let gl = self.gl;
        gl.delete_sampler(self.sampler.as_ref());
        gl.delete_vertex_array(self.vao.as_ref());
        gl.delete_program(self.program.as_ref());
        gl.delete_framebuffer(self.framebuffer.as_ref());
        gl.delete_buffer(self.staging.as_ref());
        gl.delete_texture(self.source.as_ref());
        for shader in &self.shaders {
            gl.delete_shader(Some(shader));
        }
    }
}

enum Input<'a> {
    Client(&'a [u8]),
    Buffer {
        source: &'a BrowserBuffer,
        offset: u64,
    },
}

pub(super) fn destination_attachment(format: GlFormat) -> u32 {
    match format {
        GlFormat::Depth24PlusStencil8 | GlFormat::Depth32FloatStencil8 => {
            Gl::DEPTH_STENCIL_ATTACHMENT
        }
        _ => Gl::STENCIL_ATTACHMENT,
    }
}

pub(super) fn attach_destination(
    gl: &Gl,
    destination: &BrowserTexture,
    mip: u32,
    layer: u32,
) -> Result<()> {
    let attachment = destination_attachment(destination.desc.format);
    match (&destination.raw, &destination.renderbuffer) {
        (Some(texture), None) if destination.desc.dimension == GlTextureDimension::D2 => gl
            .framebuffer_texture_2d(
                Gl::FRAMEBUFFER,
                attachment,
                Gl::TEXTURE_2D,
                Some(texture),
                mip as i32,
            ),
        (Some(texture), None) if destination.desc.dimension == GlTextureDimension::D2Array => gl
            .framebuffer_texture_layer(
                Gl::FRAMEBUFFER,
                attachment,
                Some(texture),
                mip as i32,
                layer as i32,
            ),
        (None, Some(renderbuffer)) if mip == 0 && layer == 0 => gl.framebuffer_renderbuffer(
            Gl::FRAMEBUFFER,
            attachment,
            Gl::RENDERBUFFER,
            Some(renderbuffer),
        ),
        _ => {
            return Err(Error::Unsupported(
                "WebGL stencil destination representation",
            ));
        }
    }
    Ok(())
}

impl BrowserEncoder {
    /// Writes a caller-provided standalone stencil byte plane.
    pub(crate) fn write_stencil_client(
        &mut self,
        destination: &BrowserTexture,
        mip: u32,
        origin: [u32; 3],
        extent: [u32; 3],
        bytes_per_row: u32,
        rows_per_image: u32,
        bytes: &[u8],
    ) -> Result<()> {
        self.write_stencil_plane(
            destination,
            mip,
            origin,
            extent,
            bytes_per_row,
            rows_per_image,
            Input::Client(bytes),
        )
    }

    /// Writes a standalone stencil byte plane directly from a GPU buffer.
    pub(crate) fn write_stencil_buffer(
        &mut self,
        destination: &BrowserTexture,
        mip: u32,
        origin: [u32; 3],
        extent: [u32; 3],
        bytes_per_row: u32,
        rows_per_image: u32,
        source: &BrowserBuffer,
        offset: u64,
    ) -> Result<()> {
        self.write_stencil_plane(
            destination,
            mip,
            origin,
            extent,
            bytes_per_row,
            rows_per_image,
            Input::Buffer { source, offset },
        )
    }

    fn write_stencil_plane(
        &mut self,
        destination: &BrowserTexture,
        mip: u32,
        origin: [u32; 3],
        extent: [u32; 3],
        bytes_per_row: u32,
        rows_per_image: u32,
        input: Input<'_>,
    ) -> Result<()> {
        self.same_owner(&destination.owner)?;
        if let Input::Buffer { source, .. } = &input {
            self.same_owner(&source.owner)?;
        }
        if self.has_owned_framebuffer_pass() {
            return Err(Error::InvalidState);
        }
        let bound = destination
            .desc
            .mip_extent(mip)
            .ok_or(Error::InvalidRange)?;
        let packed_array = matches!(
            destination.desc.format,
            GlFormat::Depth24PlusStencil8 | GlFormat::Depth32FloatStencil8
        ) && destination.desc.dimension == GlTextureDimension::D2Array;
        if !matches!(
            destination.desc.format,
            GlFormat::Stencil8 | GlFormat::Depth24PlusStencil8 | GlFormat::Depth32FloatStencil8
        ) || !(destination.desc.dimension == GlTextureDimension::D2 || packed_array)
            || destination.desc.sample_count != 1
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
            || !destination
                .desc
                .usage
                .contains(GlTextureUsage::COPY_DESTINATION)
        {
            return Err(Error::Unsupported("WebGL stencil transfer shape"));
        }
        let layer_stride = u64::from(bytes_per_row)
            .checked_mul(u64::from(rows_per_image))
            .ok_or(Error::InvalidRange)?;
        let required = layer_stride
            .checked_mul(u64::from(extent[2] - 1))
            .and_then(|v| v.checked_add(u64::from(bytes_per_row) * u64::from(extent[1] - 1)))
            .and_then(|v| v.checked_add(u64::from(extent[0])))
            .ok_or(Error::InvalidRange)?;
        let compact_size = u64::from(extent[0])
            .checked_mul(u64::from(extent[1]))
            .ok_or(Error::InvalidRange)?;
        let compact_i32 = i32::try_from(compact_size).map_err(|_| Error::InvalidRange)?;
        let _width_i32 = i32::try_from(extent[0]).map_err(|_| Error::InvalidRange)?;
        let _height_i32 = i32::try_from(extent[1]).map_err(|_| Error::InvalidRange)?;
        match &input {
            Input::Client(bytes)
                if usize::try_from(required)
                    .ok()
                    .is_none_or(|n| n > bytes.len()) =>
            {
                return Err(Error::InvalidRange);
            }
            Input::Buffer { source, offset }
                if offset
                    .checked_add(required)
                    .is_none_or(|end| end > source.size || end > (1u64 << 53) - 1) =>
            {
                return Err(Error::InvalidRange);
            }
            _ => {}
        }
        // WebGL2 attaches one array layer per framebuffer.  Lower a valid
        // contiguous Core region as independent immediate native operations;
        // no command replay or host staging is introduced.
        if extent[2] > 1 {
            for layer in 0..extent[2] {
                let layer_input = match input {
                    Input::Client(bytes) => {
                        let start = usize::try_from(u64::from(layer) * layer_stride)
                            .map_err(|_| Error::InvalidRange)?;
                        Input::Client(&bytes[start..])
                    }
                    Input::Buffer { source, offset } => Input::Buffer {
                        source,
                        offset: offset + u64::from(layer) * layer_stride,
                    },
                };
                self.write_stencil_plane(
                    destination,
                    mip,
                    [origin[0], origin[1], origin[2] + layer],
                    [extent[0], extent[1], 1],
                    bytes_per_row,
                    rows_per_image,
                    layer_input,
                )?;
            }
            return Ok(());
        }
        {
            let state = self.owner.state.borrow();
            if state.context.active_query.is_some() {
                return Err(Error::InvalidState);
            }
        }
        let mut result = Ok(());
        self.with_context("stencil_plane_write", |gl| {
            result = (|| -> Result<()> {
                let mut scratch = Scratch { gl, source: None, staging: None, framebuffer: None, program: None, vao: None, sampler: None, shaders: Vec::new() };
                let source = gl.create_texture().ok_or(Error::OutOfMemory)?; scratch.source = Some(source.clone());
                let framebuffer = gl.create_framebuffer().ok_or(Error::OutOfMemory)?; scratch.framebuffer = Some(framebuffer.clone());
                let vao = gl.create_vertex_array().ok_or(Error::OutOfMemory)?; scratch.vao = Some(vao.clone());
                let sampler = gl.create_sampler().ok_or(Error::OutOfMemory)?; scratch.sampler = Some(sampler.clone());
                let program = scratch.program()?.clone();
                let state = State::capture(gl)?;
                gl.active_texture(Gl::TEXTURE0); gl.bind_texture(Gl::TEXTURE_2D, Some(&source));
                gl.tex_storage_2d(Gl::TEXTURE_2D, 1, Gl::R8UI, extent[0] as i32, extent[1] as i32);
                gl.sampler_parameteri(&sampler, Gl::TEXTURE_MIN_FILTER, Gl::NEAREST as i32); gl.sampler_parameteri(&sampler, Gl::TEXTURE_MAG_FILTER, Gl::NEAREST as i32);
                // Stencil planes are byte-addressed.  Normalize all unpack
                // state so odd widths and caller state cannot reinterpret
                // rows during either client or PBO upload.
                gl.pixel_storei(Gl::UNPACK_ALIGNMENT, 1);
                gl.pixel_storei(Gl::UNPACK_ROW_LENGTH, 0);
                gl.pixel_storei(Gl::UNPACK_IMAGE_HEIGHT, 0);
                gl.pixel_storei(Gl::UNPACK_SKIP_PIXELS, 0);
                gl.pixel_storei(Gl::UNPACK_SKIP_ROWS, 0);
                gl.pixel_storei(Gl::UNPACK_SKIP_IMAGES, 0);
                gl.pixel_storei(Gl::UNPACK_FLIP_Y_WEBGL, 0);
                match input {
                    Input::Client(bytes) => {
                        let mut compact = Vec::new();
                        compact.try_reserve_exact(usize::try_from(compact_size).map_err(|_| Error::OutOfMemory)?).map_err(|_| Error::OutOfMemory)?;
                        for row in 0..extent[1] as usize { let start = row * bytes_per_row as usize; compact.extend_from_slice(&bytes[start..start + extent[0] as usize]); }
                        let values = Uint8Array::from(compact.as_slice());
                        gl.bind_buffer(Gl::PIXEL_UNPACK_BUFFER, None);
                        gl.tex_sub_image_2d_with_i32_and_i32_and_u32_and_type_and_opt_array_buffer_view(Gl::TEXTURE_2D, 0, 0, 0, extent[0] as i32, extent[1] as i32, RED_INTEGER, Gl::UNSIGNED_BYTE, Some(&values)).map_err(|_| Error::Backend("WebGL stencil client upload threw".into()))?;
                    }
                    Input::Buffer { source: buffer, offset } => {
                        let staging = gl.create_buffer().ok_or(Error::OutOfMemory)?; scratch.staging = Some(staging.clone());
                        gl.bind_buffer(Gl::COPY_WRITE_BUFFER, Some(&staging)); gl.buffer_data_with_i32(Gl::COPY_WRITE_BUFFER, compact_i32, Gl::STREAM_DRAW);
                        gl.bind_buffer(Gl::COPY_READ_BUFFER, Some(&buffer.raw));
                        for row in 0..extent[1] { gl.copy_buffer_sub_data_with_f64_and_f64_and_f64(Gl::COPY_READ_BUFFER, Gl::COPY_WRITE_BUFFER, (offset + u64::from(row) * u64::from(bytes_per_row)) as f64, (u64::from(row) * u64::from(extent[0])) as f64, extent[0] as f64); }
                        gl.bind_buffer(Gl::PIXEL_UNPACK_BUFFER, Some(&staging));
                        gl.tex_sub_image_2d_with_i32_and_i32_and_u32_and_type_and_f64(Gl::TEXTURE_2D, 0, 0, 0, extent[0] as i32, extent[1] as i32, RED_INTEGER, Gl::UNSIGNED_BYTE, 0.).map_err(|_| Error::Backend("WebGL stencil PBO upload threw".into()))?;
                    }
                }
                gl.bind_framebuffer(Gl::FRAMEBUFFER, Some(&framebuffer)); attach_destination(gl, destination, mip, origin[2])?;
                gl.draw_buffers(&Array::of1(&Gl::NONE.into())); gl.read_buffer(Gl::NONE);
                if gl.check_framebuffer_status(Gl::FRAMEBUFFER) != Gl::FRAMEBUFFER_COMPLETE { return Err(Error::Unsupported("WebGL stencil framebuffer")); }
                gl.viewport(origin[0] as i32, origin[1] as i32, extent[0] as i32, extent[1] as i32);
                gl.enable(Gl::SCISSOR_TEST); gl.scissor(origin[0] as i32, origin[1] as i32, extent[0] as i32, extent[1] as i32);
                gl.disable(Gl::DEPTH_TEST); gl.disable(Gl::BLEND); gl.disable(Gl::RASTERIZER_DISCARD);
                gl.disable(Gl::DITHER); gl.disable(Gl::CULL_FACE); gl.disable(Gl::SAMPLE_COVERAGE);
                gl.depth_mask(false);
                gl.enable(Gl::STENCIL_TEST); gl.stencil_mask(u32::MAX); gl.clear_bufferiv_with_i32_array(Gl::STENCIL, 0, &[0]);
                gl.bind_vertex_array(Some(&vao)); gl.bind_sampler(0, Some(&sampler)); gl.use_program(Some(&program));
                gl.uniform1i(gl.get_uniform_location(&program, "bytes_in").as_ref(), 0);
                gl.uniform2i(gl.get_uniform_location(&program, "origin").as_ref(), origin[0] as i32, origin[1] as i32);
                for bit in 0..8 {
                    let mask = 1u32 << bit;
                    gl.uniform1i(gl.get_uniform_location(&program, "bit_index").as_ref(), bit);
                    gl.stencil_func(Gl::ALWAYS, mask as i32, mask); gl.stencil_mask(mask); gl.stencil_op(Gl::KEEP, Gl::KEEP, Gl::REPLACE);
                    gl.draw_arrays(Gl::TRIANGLES, 0, 3);
                }
                drop(state); Ok(())
            })();
        })?;
        result
    }
}
