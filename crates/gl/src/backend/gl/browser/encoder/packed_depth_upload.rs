//! D32FloatStencil8 depth-plane upload through private WebGL2 objects.
//!
//! WebGL2 has no client-array representation for
//! `FLOAT_32_UNSIGNED_INT_24_8_REV`, but does define the pixel-unpack-buffer
//! offset overload. This module therefore expands a compact float depth plane
//! into a zeroed private packed PBO, uploads it into a matching temporary
//! depth/stencil texture, and blits only `DEPTH_BUFFER_BIT` into the target.
//! The target stencil plane is never attached to a write operation.

use fluxel_rhi_core::api::{Error, Result};
use js_sys::{Array, Int32Array};
use wasm_bindgen::JsCast;
use web_sys::{WebGl2RenderingContext as Gl, WebGlBuffer, WebGlFramebuffer, WebGlTexture};

use super::super::owner::{BrowserBuffer, BrowserEncoder, BrowserTexture};
use crate::backend::gl::api::{
    GlBufferDesc, GlBufferUsage, GlFormat, GlTextureDimension, GlTextureUsage,
};

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
    let value = gl
        .get_parameter(parameter)
        .map_err(|_| Error::InvalidState)?;
    value
        .as_f64()
        .map(|value| value as i32)
        .or_else(|| value.as_bool().map(i32::from))
        .ok_or(Error::InvalidState)
}

/// The bindings and pixel-store state private to one packed depth transfer.
struct State<'a> {
    gl: &'a Gl,
    texture: Option<WebGlTexture>,
    unpack: Option<WebGlBuffer>,
    copy_read: Option<WebGlBuffer>,
    copy_write: Option<WebGlBuffer>,
    read_framebuffer: Option<WebGlFramebuffer>,
    draw_framebuffer: Option<WebGlFramebuffer>,
    alignment: i32,
    row_length: i32,
    image_height: i32,
    skip_pixels: i32,
    skip_rows: i32,
    skip_images: i32,
    flip_y: i32,
    scissor: bool,
    scissor_box: [i32; 4],
}

impl<'a> State<'a> {
    fn capture(gl: &'a Gl) -> Result<Self> {
        let scissor = gl.is_enabled(Gl::SCISSOR_TEST);
        let box_value = gl
            .get_parameter(Gl::SCISSOR_BOX)
            .map_err(|_| Error::InvalidState)?;
        let box_values = Int32Array::new(&box_value).to_vec();
        let scissor_box: [i32; 4] = box_values.try_into().map_err(|_| Error::InvalidState)?;
        Ok(Self {
            gl,
            texture: object(gl, Gl::TEXTURE_BINDING_2D)?,
            unpack: object(gl, Gl::PIXEL_UNPACK_BUFFER_BINDING)?,
            copy_read: object(gl, Gl::COPY_READ_BUFFER_BINDING)?,
            copy_write: object(gl, Gl::COPY_WRITE_BUFFER_BINDING)?,
            read_framebuffer: object(gl, Gl::READ_FRAMEBUFFER_BINDING)?,
            draw_framebuffer: object(gl, Gl::DRAW_FRAMEBUFFER_BINDING)?,
            alignment: integer(gl, Gl::UNPACK_ALIGNMENT)?,
            row_length: integer(gl, Gl::UNPACK_ROW_LENGTH)?,
            image_height: integer(gl, Gl::UNPACK_IMAGE_HEIGHT)?,
            skip_pixels: integer(gl, Gl::UNPACK_SKIP_PIXELS)?,
            skip_rows: integer(gl, Gl::UNPACK_SKIP_ROWS)?,
            skip_images: integer(gl, Gl::UNPACK_SKIP_IMAGES)?,
            flip_y: integer(gl, Gl::UNPACK_FLIP_Y_WEBGL)?,
            scissor,
            scissor_box,
        })
    }
}

impl Drop for State<'_> {
    fn drop(&mut self) {
        let gl = self.gl;
        gl.bind_texture(Gl::TEXTURE_2D, self.texture.as_ref());
        gl.bind_buffer(Gl::PIXEL_UNPACK_BUFFER, self.unpack.as_ref());
        gl.bind_buffer(Gl::COPY_READ_BUFFER, self.copy_read.as_ref());
        gl.bind_buffer(Gl::COPY_WRITE_BUFFER, self.copy_write.as_ref());
        gl.bind_framebuffer(Gl::READ_FRAMEBUFFER, self.read_framebuffer.as_ref());
        gl.bind_framebuffer(Gl::DRAW_FRAMEBUFFER, self.draw_framebuffer.as_ref());
        gl.pixel_storei(Gl::UNPACK_ALIGNMENT, self.alignment);
        gl.pixel_storei(Gl::UNPACK_ROW_LENGTH, self.row_length);
        gl.pixel_storei(Gl::UNPACK_IMAGE_HEIGHT, self.image_height);
        gl.pixel_storei(Gl::UNPACK_SKIP_PIXELS, self.skip_pixels);
        gl.pixel_storei(Gl::UNPACK_SKIP_ROWS, self.skip_rows);
        gl.pixel_storei(Gl::UNPACK_SKIP_IMAGES, self.skip_images);
        gl.pixel_storei(Gl::UNPACK_FLIP_Y_WEBGL, self.flip_y);
        gl.scissor(
            self.scissor_box[0],
            self.scissor_box[1],
            self.scissor_box[2],
            self.scissor_box[3],
        );
        if self.scissor {
            gl.enable(Gl::SCISSOR_TEST);
        } else {
            gl.disable(Gl::SCISSOR_TEST);
        }
    }
}

struct Scratch<'a> {
    gl: &'a Gl,
    packed: WebGlBuffer,
    texture: WebGlTexture,
    read: WebGlFramebuffer,
    draw: WebGlFramebuffer,
}

impl<'a> Scratch<'a> {
    fn new(gl: &'a Gl) -> Result<Self> {
        Ok(Self {
            gl,
            packed: gl.create_buffer().ok_or(Error::OutOfMemory)?,
            texture: gl.create_texture().ok_or(Error::OutOfMemory)?,
            read: gl.create_framebuffer().ok_or(Error::OutOfMemory)?,
            draw: gl.create_framebuffer().ok_or(Error::OutOfMemory)?,
        })
    }
}

impl Drop for Scratch<'_> {
    fn drop(&mut self) {
        self.gl.delete_framebuffer(Some(&self.read));
        self.gl.delete_framebuffer(Some(&self.draw));
        self.gl.delete_texture(Some(&self.texture));
        self.gl.delete_buffer(Some(&self.packed));
    }
}

impl BrowserEncoder {
    /// Uploads caller-owned compact D32 float bytes without exposing a packed
    /// depth/stencil representation through Core. The private carrier is
    /// written once, then follows the same GPU-only expansion path as a
    /// buffer source; no GPU bytes are read by the host.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn write_packed_depth32f_client(
        &mut self,
        destination: &BrowserTexture,
        mip: u32,
        origin: [u32; 3],
        extent: [u32; 3],
        bytes_per_row: u32,
        rows_per_image: u32,
        bytes: &[u8],
    ) -> Result<()> {
        if extent.contains(&0) {
            return Err(Error::InvalidRange);
        }
        let row_bytes = u64::from(extent[0]) * 4;
        let layer_stride = u64::from(bytes_per_row)
            .checked_mul(u64::from(rows_per_image))
            .ok_or(Error::InvalidRange)?;
        let required = layer_stride
            .checked_mul(u64::from(extent[2] - 1))
            .and_then(|value| {
                value.checked_add(u64::from(bytes_per_row) * u64::from(extent[1] - 1))
            })
            .and_then(|value| value.checked_add(row_bytes))
            .ok_or(Error::InvalidRange)?;
        if required > bytes.len() as u64 {
            return Err(Error::InvalidRange);
        }
        let carrier = self.owner.create_buffer(GlBufferDesc {
            size: bytes.len() as u64,
            usage: GlBufferUsage::COPY_SOURCE | GlBufferUsage::COPY_DESTINATION,
        })?;
        self.write_browser_buffer(&carrier, 0, bytes)?;
        if !self.copy_packed_depth32f_to_texture(
            &carrier,
            0,
            bytes_per_row,
            rows_per_image,
            destination,
            mip,
            origin,
            extent,
        )? {
            return Err(Error::Unsupported("WebGL packed D32FS depth-upload format"));
        }
        Ok(())
    }

    /// Lowers compact D32 float depth planes into a D32FloatStencil8 target.
    ///
    /// This accepts single-sample D2 and D2-array rectangles. The caller keeps
    /// its original buffer data on the GPU: each float is copied into byte 0
    /// of a zeroed eight-byte packed texel, so the temporary stencil word is
    /// zero without ever reading source bytes on the host.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn copy_packed_depth32f_to_texture(
        &mut self,
        source: &BrowserBuffer,
        offset: u64,
        bytes_per_row: u32,
        rows_per_image: u32,
        destination: &BrowserTexture,
        mip: u32,
        origin: [u32; 3],
        extent: [u32; 3],
    ) -> Result<bool> {
        if destination.desc.format != GlFormat::Depth32FloatStencil8 {
            return Ok(false);
        }
        self.same_owner(&source.owner)?;
        self.same_owner(&destination.owner)?;
        if self.has_owned_framebuffer_pass() {
            return Err(Error::InvalidState);
        }
        let bound = destination
            .desc
            .mip_extent(mip)
            .ok_or(Error::InvalidRange)?;
        let row_bytes = u64::from(extent[0]) * 4;
        let valid_dimension = match destination.desc.dimension {
            GlTextureDimension::D2 => origin[2] == 0 && extent[2] == 1,
            GlTextureDimension::D2Array => true,
            _ => false,
        };
        if !valid_dimension
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
            || u64::from(bytes_per_row) < row_bytes
            || rows_per_image < extent[1]
            || !destination
                .desc
                .usage
                .contains(GlTextureUsage::COPY_DESTINATION)
        {
            return Err(Error::Unsupported("WebGL packed D32FS depth-upload shape"));
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
            .and_then(|value| {
                value.checked_add(u64::from(bytes_per_row) * u64::from(extent[1] - 1))
            })
            .and_then(|value| value.checked_add(row_bytes))
            .ok_or(Error::InvalidRange)?;
        if end > source.size || end > (1u64 << 53) {
            return Err(Error::InvalidRange);
        }
        for layer in 0..extent[2] {
            self.copy_packed_depth32f_layer(
                source,
                offset + u64::from(layer) * layer_stride,
                bytes_per_row,
                destination,
                mip,
                [origin[0], origin[1], origin[2] + layer],
                [extent[0], extent[1], 1],
            )?;
        }
        Ok(true)
    }

    #[allow(clippy::too_many_arguments)]
    fn copy_packed_depth32f_layer(
        &mut self,
        source: &BrowserBuffer,
        offset: u64,
        bytes_per_row: u32,
        destination: &BrowserTexture,
        mip: u32,
        origin: [u32; 3],
        extent: [u32; 3],
    ) -> Result<()> {
        let row_bytes = u64::from(extent[0]) * 4;
        let end = offset
            .checked_add(u64::from(extent[1] - 1) * u64::from(bytes_per_row))
            .and_then(|value| value.checked_add(row_bytes))
            .ok_or(Error::InvalidRange)?;
        if end > source.size || end > (1u64 << 53) {
            return Err(Error::InvalidRange);
        }
        let packed_size = u64::from(extent[0])
            .checked_mul(u64::from(extent[1]))
            .and_then(|texels| texels.checked_mul(8))
            .ok_or(Error::InvalidRange)?;
        i32::try_from(packed_size).map_err(|_| Error::InvalidRange)?;
        let width = i32::try_from(extent[0]).map_err(|_| Error::InvalidRange)?;
        let height = i32::try_from(extent[1]).map_err(|_| Error::InvalidRange)?;
        let target = destination.texture()?;

        let mut result = Ok(());
        let outer = self.with_context("packed_d32fs_depth_upload", |gl| {
            result = (|| -> Result<()> {
                let state = State::capture(gl)?;
                let scratch = Scratch::new(gl)?;
                // `bufferData` with a typed zero array gives the temporary
                // stencil words a defined value before the GPU copies depth.
                let zeros = vec![0; usize::try_from(packed_size).map_err(|_| Error::InvalidRange)?];
                gl.bind_buffer(Gl::COPY_WRITE_BUFFER, Some(&scratch.packed));
                gl.buffer_data_with_u8_array(Gl::COPY_WRITE_BUFFER, &zeros, Gl::STREAM_DRAW);
                gl.bind_buffer(Gl::COPY_READ_BUFFER, Some(&source.raw));
                for row in 0..extent[1] {
                    for column in 0..extent[0] {
                        let source_offset = offset
                            + u64::from(row) * u64::from(bytes_per_row)
                            + u64::from(column) * 4;
                        let packed_offset =
                            (u64::from(row) * u64::from(extent[0]) + u64::from(column)) * 8;
                        gl.copy_buffer_sub_data_with_f64_and_f64_and_f64(
                            Gl::COPY_READ_BUFFER,
                            Gl::COPY_WRITE_BUFFER,
                            source_offset as f64,
                            packed_offset as f64,
                            4.0,
                        );
                    }
                }
                gl.bind_texture(Gl::TEXTURE_2D, Some(&scratch.texture));
                gl.tex_storage_2d(Gl::TEXTURE_2D, 1, Gl::DEPTH32F_STENCIL8, width, height);
                gl.bind_buffer(Gl::PIXEL_UNPACK_BUFFER, Some(&scratch.packed));
                gl.pixel_storei(Gl::UNPACK_ALIGNMENT, 8);
                gl.pixel_storei(Gl::UNPACK_ROW_LENGTH, 0);
                gl.pixel_storei(Gl::UNPACK_IMAGE_HEIGHT, 0);
                gl.pixel_storei(Gl::UNPACK_SKIP_PIXELS, 0);
                gl.pixel_storei(Gl::UNPACK_SKIP_ROWS, 0);
                gl.pixel_storei(Gl::UNPACK_SKIP_IMAGES, 0);
                gl.pixel_storei(Gl::UNPACK_FLIP_Y_WEBGL, 0);
                gl.tex_sub_image_2d_with_i32_and_i32_and_u32_and_type_and_f64(
                    Gl::TEXTURE_2D,
                    0,
                    0,
                    0,
                    width,
                    height,
                    Gl::DEPTH_STENCIL,
                    Gl::FLOAT_32_UNSIGNED_INT_24_8_REV,
                    0.0,
                )
                .map_err(|_| Error::Backend("WebGL packed D32FS PBO upload threw".into()))?;
                // Configure both depth-only FBOs through FRAMEBUFFER before
                // splitting them into READ/DRAW bindings. WebGL otherwise
                // retains COLOR_ATTACHMENT0 as the read/draw selection and
                // rejects a perfectly valid depth-only attachment set.
                gl.bind_framebuffer(Gl::FRAMEBUFFER, Some(&scratch.read));
                gl.framebuffer_texture_2d(
                    Gl::FRAMEBUFFER,
                    Gl::DEPTH_STENCIL_ATTACHMENT,
                    Gl::TEXTURE_2D,
                    Some(&scratch.texture),
                    0,
                );
                gl.draw_buffers(&Array::of1(&Gl::NONE.into()));
                gl.read_buffer(Gl::NONE);
                if gl.check_framebuffer_status(Gl::FRAMEBUFFER) != Gl::FRAMEBUFFER_COMPLETE {
                    return Err(Error::Backend(
                        "WebGL packed D32FS source framebuffer incomplete".into(),
                    ));
                }
                gl.bind_framebuffer(Gl::FRAMEBUFFER, Some(&scratch.draw));
                if destination.desc.dimension == GlTextureDimension::D2 {
                    gl.framebuffer_texture_2d(
                        Gl::FRAMEBUFFER,
                        Gl::DEPTH_STENCIL_ATTACHMENT,
                        Gl::TEXTURE_2D,
                        Some(target),
                        mip as i32,
                    );
                } else {
                    gl.framebuffer_texture_layer(
                        Gl::FRAMEBUFFER,
                        Gl::DEPTH_STENCIL_ATTACHMENT,
                        Some(target),
                        mip as i32,
                        origin[2] as i32,
                    );
                }
                gl.draw_buffers(&Array::of1(&Gl::NONE.into()));
                gl.read_buffer(Gl::NONE);
                if gl.check_framebuffer_status(Gl::FRAMEBUFFER) != Gl::FRAMEBUFFER_COMPLETE {
                    return Err(Error::Backend(
                        "WebGL packed D32FS destination framebuffer incomplete".into(),
                    ));
                }
                gl.disable(Gl::SCISSOR_TEST);
                gl.bind_framebuffer(Gl::READ_FRAMEBUFFER, Some(&scratch.read));
                gl.bind_framebuffer(Gl::DRAW_FRAMEBUFFER, Some(&scratch.draw));
                gl.blit_framebuffer(
                    0,
                    0,
                    width,
                    height,
                    origin[0] as i32,
                    origin[1] as i32,
                    origin[0] as i32 + width,
                    origin[1] as i32 + height,
                    Gl::DEPTH_BUFFER_BIT,
                    Gl::NEAREST,
                );
                let error = gl.get_error();
                drop(scratch);
                drop(state);
                if error != Gl::NO_ERROR {
                    return Err(Error::Backend(format!(
                        "WebGL packed D32FS depth blit error {error:#x}"
                    )));
                }
                Ok(())
            })();
        });
        outer?;
        result?;
        Ok(())
    }
}
