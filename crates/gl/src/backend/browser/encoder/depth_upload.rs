//! Whole-rectangle depth uploads through a private WebGL pixel-unpack buffer.
//!
//! Some WebGL implementations accept a depth PBO upload one row at a time but
//! repeatedly consume the first row.  This narrow path repacks the requested
//! D16/D32 rows entirely on the GPU, then performs one `texSubImage2D` for the
//! complete rectangle.  It is not a CPU readback and does not retain work.

use fluxel_rhi_core::api::{Error, Result};
use web_sys::{WebGl2RenderingContext as Gl, WebGlBuffer, WebGlTexture};

use super::super::super::api::GlFormat;
use super::super::owner::{BrowserBuffer, BrowserEncoder, BrowserTexture};

/// Borrowed WebGL state touched by the private PBO transfer.
///
/// This is intentionally local to one immediate call.  Restoring the exact
/// bindings and pixel-store values prevents an upload failure from leaking
/// private scratch state into later encoder verbs.
struct State<'a> {
    gl: &'a Gl,
    texture: Option<WebGlTexture>,
    texture_target: u32,
    unpack: Option<WebGlBuffer>,
    copy_read: Option<WebGlBuffer>,
    copy_write: Option<WebGlBuffer>,
    alignment: i32,
    row_length: i32,
    image_height: i32,
    skip_pixels: i32,
    skip_rows: i32,
    skip_images: i32,
    flip_y: i32,
}

impl<'a> State<'a> {
    fn capture(gl: &'a Gl, texture_target: u32, _texture_binding: u32) -> Result<Self> {
        // This backend exclusively owns the context. Reset this private
        // transfer domain to WebGL defaults instead of synchronously reading
        // state on every depth upload.
        Ok(Self {
            gl,
            texture: None,
            texture_target,
            unpack: None,
            copy_read: None,
            copy_write: None,
            alignment: 4,
            row_length: 0,
            image_height: 0,
            skip_pixels: 0,
            skip_rows: 0,
            skip_images: 0,
            flip_y: 0,
        })
    }
}

impl Drop for State<'_> {
    fn drop(&mut self) {
        let gl = self.gl;
        gl.bind_texture(self.texture_target, self.texture.as_ref());
        gl.bind_buffer(Gl::PIXEL_UNPACK_BUFFER, self.unpack.as_ref());
        gl.bind_buffer(Gl::COPY_READ_BUFFER, self.copy_read.as_ref());
        gl.bind_buffer(Gl::COPY_WRITE_BUFFER, self.copy_write.as_ref());
        gl.pixel_storei(Gl::UNPACK_ALIGNMENT, self.alignment);
        gl.pixel_storei(Gl::UNPACK_ROW_LENGTH, self.row_length);
        gl.pixel_storei(Gl::UNPACK_IMAGE_HEIGHT, self.image_height);
        gl.pixel_storei(Gl::UNPACK_SKIP_PIXELS, self.skip_pixels);
        gl.pixel_storei(Gl::UNPACK_SKIP_ROWS, self.skip_rows);
        gl.pixel_storei(Gl::UNPACK_SKIP_IMAGES, self.skip_images);
        gl.pixel_storei(Gl::UNPACK_FLIP_Y_WEBGL, self.flip_y);
    }
}

struct Scratch<'a> {
    gl: &'a Gl,
    buffer: WebGlBuffer,
}

impl<'a> Scratch<'a> {
    fn new(gl: &'a Gl) -> Result<Self> {
        Ok(Self {
            gl,
            buffer: gl.create_buffer().ok_or(Error::OutOfMemory)?,
        })
    }
}

impl Drop for Scratch<'_> {
    fn drop(&mut self) {
        self.gl.delete_buffer(Some(&self.buffer));
    }
}

impl BrowserEncoder {
    /// Performs the D2/D2-array depth PBO workaround after generic geometry
    /// has been validated. `false` means the ordinary upload path must handle it.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn copy_depth_pbo_to_texture(
        &mut self,
        source: &BrowserBuffer,
        offset: u64,
        bytes_per_row: u32,
        rows_per_image: u32,
        destination: &BrowserTexture,
        texture: &WebGlTexture,
        mip: u32,
        origin: [u32; 3],
        extent: [u32; 3],
        target: u32,
        format: u32,
        kind: u32,
        row_bytes: u64,
    ) -> Result<bool> {
        let texture_binding = match target {
            Gl::TEXTURE_2D => Gl::TEXTURE_BINDING_2D,
            Gl::TEXTURE_2D_ARRAY => Gl::TEXTURE_BINDING_2D_ARRAY,
            _ => return Ok(false),
        };
        if !matches!(
            destination.desc.format,
            GlFormat::Depth16Unorm | GlFormat::Depth32Float
        ) {
            return Ok(false);
        }
        let layer_stride = u64::from(bytes_per_row)
            .checked_mul(u64::from(rows_per_image))
            .ok_or(Error::InvalidRange)?;
        let packed_size = row_bytes
            .checked_mul(u64::from(extent[1]))
            .and_then(|size| size.checked_mul(u64::from(extent[2])))
            .ok_or(Error::InvalidRange)?;
        if packed_size > i32::MAX as u64 {
            return Err(Error::InvalidRange);
        }

        let mut inner = Ok(());
        let outer = self.with_context("depth_pbo_buffer_to_texture", |raw| {
            inner = (|| -> Result<()> {
                let scratch = Scratch::new(raw)?;
                let state = State::capture(raw, target, texture_binding)?;
                raw.bind_buffer(Gl::COPY_WRITE_BUFFER, Some(&scratch.buffer));
                raw.buffer_data_with_i32(
                    Gl::COPY_WRITE_BUFFER,
                    packed_size as i32,
                    Gl::STREAM_DRAW,
                );
                raw.bind_buffer(Gl::COPY_READ_BUFFER, Some(&source.raw));
                for layer in 0..extent[2] {
                    for row in 0..extent[1] {
                        raw.copy_buffer_sub_data_with_f64_and_f64_and_f64(
                            Gl::COPY_READ_BUFFER,
                            Gl::COPY_WRITE_BUFFER,
                            (offset
                                + u64::from(layer) * layer_stride
                                + u64::from(row) * u64::from(bytes_per_row))
                                as f64,
                            (u64::from(layer) * u64::from(extent[1]) * row_bytes
                                + u64::from(row) * row_bytes) as f64,
                            row_bytes as f64,
                        );
                    }
                }
                raw.bind_texture(target, Some(texture));
                raw.bind_buffer(Gl::PIXEL_UNPACK_BUFFER, Some(&scratch.buffer));
                raw.pixel_storei(Gl::UNPACK_ALIGNMENT, 1);
                raw.pixel_storei(Gl::UNPACK_ROW_LENGTH, 0);
                raw.pixel_storei(Gl::UNPACK_IMAGE_HEIGHT, 0);
                raw.pixel_storei(Gl::UNPACK_SKIP_PIXELS, 0);
                raw.pixel_storei(Gl::UNPACK_SKIP_ROWS, 0);
                raw.pixel_storei(Gl::UNPACK_SKIP_IMAGES, 0);
                raw.pixel_storei(Gl::UNPACK_FLIP_Y_WEBGL, 0);
                let result = if target == Gl::TEXTURE_2D {
                    raw.tex_sub_image_2d_with_i32_and_i32_and_u32_and_type_and_f64(
                        target,
                        mip as i32,
                        origin[0] as i32,
                        origin[1] as i32,
                        extent[0] as i32,
                        extent[1] as i32,
                        format,
                        kind,
                        0.0,
                    )
                } else {
                    raw.tex_sub_image_3d_with_f64(
                        target,
                        mip as i32,
                        origin[0] as i32,
                        origin[1] as i32,
                        origin[2] as i32,
                        extent[0] as i32,
                        extent[1] as i32,
                        extent[2] as i32,
                        format,
                        kind,
                        0.0,
                    )
                };
                result.map_err(|_| {
                    Error::Backend("WebGL2 depth PBO upload threw an exception".into())
                })?;
                drop(state);
                drop(scratch);
                Ok(())
            })();
        });
        outer?;
        inner?;
        Ok(true)
    }
}
