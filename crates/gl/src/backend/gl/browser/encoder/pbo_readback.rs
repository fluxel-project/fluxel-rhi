//! Direct texture-to-buffer transfer using WebGL2 pixel-pack buffers.
//! Each row is read into its final GPU offset, preserving Core's upper-left
//! order and leaving all row and image padding untouched.
use super::super::owner::{BrowserBuffer, BrowserEncoder, BrowserTexture};
use crate::backend::gl::api::{GlFamilyApi as _, GlTextureDimension, GlTextureUsage};
use fluxel_rhi_core::api::{Error, Result};
use wasm_bindgen::JsCast;
use web_sys::{WebGl2RenderingContext as Gl, WebGlBuffer, WebGlFramebuffer};

impl BrowserEncoder {
    pub(crate) fn copy_texture_to_buffer(
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
        let texture = source.texture()?;
        if !source.desc.usage.contains(GlTextureUsage::COPY_SOURCE) || source.desc.sample_count != 1
        {
            return Err(Error::InvalidState);
        }
        let encoding = super::super::ordinary_format::encoding(source.desc.format)
            .ok_or(Error::Unsupported("WebGL2 pixel-pack readback format"))?;
        let pixel_bytes = u64::from(encoding.bytes);
        let bound = source.desc.mip_extent(mip).ok_or(Error::InvalidRange)?;
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
            || u64::from(bytes_per_row) < u64::from(extent[0]) * pixel_bytes
            || rows_per_image < extent[1]
        {
            return Err(Error::InvalidRange);
        }
        match source.desc.dimension {
            GlTextureDimension::D2 if origin[2] == 0 && extent[2] == 1 => {}
            GlTextureDimension::D2Array | GlTextureDimension::D3 => {}
            GlTextureDimension::D2 => return Err(Error::InvalidRange),
            _ => return Err(Error::Unsupported("WebGL2 pixel-pack texture dimension")),
        }
        let layer_stride = u64::from(bytes_per_row) * u64::from(rows_per_image);
        let end = offset
            .checked_add(
                layer_stride
                    .checked_mul(u64::from(extent[2] - 1))
                    .ok_or(Error::InvalidRange)?,
            )
            .and_then(|v| v.checked_add(u64::from(bytes_per_row) * u64::from(extent[1] - 1)))
            .and_then(|v| v.checked_add(u64::from(extent[0]) * pixel_bytes))
            .ok_or(Error::InvalidRange)?;
        if end > destination.size || end > (1u64 << 53) {
            return Err(Error::InvalidRange);
        }
        {
            let state = self.owner.state.borrow();
            let fact = state
                .context
                .discovery()
                .formats()
                .get(source.desc.format, 1);
            if !state
                .context
                .discovery()
                .formats()
                .exact_buffer_readback(source.desc.format)
                || !fact.is_some_and(|v| v.renderable)
            {
                return Err(Error::Unsupported("WebGL2 pixel-pack framebuffer format"));
            }
        }
        let mut inner = Ok(());
        let outer = self.with_context("texture_to_pixel_pack_buffer", |raw| {
            let operation = || -> Result<()> {
                let previous_framebuffer = raw
                    .get_parameter(Gl::READ_FRAMEBUFFER_BINDING)
                    .map_err(|_| Error::InvalidState)?;
                let previous_buffer = raw
                    .get_parameter(Gl::PIXEL_PACK_BUFFER_BINDING)
                    .map_err(|_| Error::InvalidState)?;
                let previous_copy_read = raw
                    .get_parameter(Gl::COPY_READ_BUFFER_BINDING)
                    .map_err(|_| Error::InvalidState)?;
                let previous_copy_write = raw
                    .get_parameter(Gl::COPY_WRITE_BUFFER_BINDING)
                    .map_err(|_| Error::InvalidState)?;
                let number = |parameter| {
                    raw.get_parameter(parameter)
                        .ok()
                        .and_then(|v| v.as_f64())
                        .map(|v| v as i32)
                        .ok_or(Error::InvalidState)
                };
                let alignment = number(Gl::PACK_ALIGNMENT)?;
                let row_length = number(Gl::PACK_ROW_LENGTH)?;
                let skip_rows = number(Gl::PACK_SKIP_ROWS)?;
                let skip_pixels = number(Gl::PACK_SKIP_PIXELS)?;
                // Some ANGLE implementations require whole-texel alignment
                // even where GL only requires alignment to the pixel type.
                let scratch_needed = offset % pixel_bytes != 0
                    || u64::from(bytes_per_row) % pixel_bytes != 0
                    || layer_stride % pixel_bytes != 0;
                let row_bytes = u64::from(extent[0]) * pixel_bytes;
                let row_size = i32::try_from(row_bytes).map_err(|_| Error::InvalidRange)?;
                let framebuffer = raw.create_framebuffer().ok_or(Error::OutOfMemory)?;
                let scratch = if scratch_needed {
                    match raw.create_buffer() {
                        Some(buffer) => Some(buffer),
                        None => {
                            raw.delete_framebuffer(Some(&framebuffer));
                            return Err(Error::OutOfMemory);
                        }
                    }
                } else {
                    None
                };
                raw.bind_framebuffer(Gl::READ_FRAMEBUFFER, Some(&framebuffer));
                raw.bind_buffer(
                    Gl::PIXEL_PACK_BUFFER,
                    scratch.as_ref().or(Some(&destination.raw)),
                );
                if scratch.is_some() {
                    raw.buffer_data_with_i32(Gl::PIXEL_PACK_BUFFER, row_size, Gl::STREAM_READ);
                }
                raw.pixel_storei(Gl::PACK_ALIGNMENT, 1);
                raw.pixel_storei(Gl::PACK_ROW_LENGTH, 0);
                raw.pixel_storei(Gl::PACK_SKIP_ROWS, 0);
                raw.pixel_storei(Gl::PACK_SKIP_PIXELS, 0);
                let mut result = Ok(());
                for layer in 0..extent[2] {
                    if source.desc.dimension == GlTextureDimension::D2 {
                        raw.framebuffer_texture_2d(
                            Gl::READ_FRAMEBUFFER,
                            Gl::COLOR_ATTACHMENT0,
                            Gl::TEXTURE_2D,
                            Some(texture),
                            mip as i32,
                        );
                    } else {
                        raw.framebuffer_texture_layer(
                            Gl::READ_FRAMEBUFFER,
                            Gl::COLOR_ATTACHMENT0,
                            Some(texture),
                            mip as i32,
                            (origin[2] + layer) as i32,
                        );
                    }
                    if raw.check_framebuffer_status(Gl::READ_FRAMEBUFFER)
                        != Gl::FRAMEBUFFER_COMPLETE
                    {
                        result = Err(Error::Backend(
                            "WebGL2 pixel-pack framebuffer incomplete".into(),
                        ));
                        break;
                    }
                    raw.read_buffer(Gl::COLOR_ATTACHMENT0);
                    for row in 0..extent[1] {
                        let position = offset
                            + u64::from(layer) * layer_stride
                            + u64::from(row) * u64::from(bytes_per_row);
                        let y = origin[1] + row;
                        if raw
                            .read_pixels_with_f64(
                                origin[0] as i32,
                                y as i32,
                                extent[0] as i32,
                                1,
                                encoding.pixel,
                                encoding.ty,
                                if scratch_needed { 0.0 } else { position as f64 },
                            )
                            .is_err()
                        {
                            result = Err(Error::Backend(
                                "WebGL2 pixel-pack readPixels threw an exception".into(),
                            ));
                            break;
                        }
                        if let Some(scratch) = &scratch {
                            raw.bind_buffer(Gl::COPY_READ_BUFFER, Some(scratch));
                            raw.bind_buffer(Gl::COPY_WRITE_BUFFER, Some(&destination.raw));
                            raw.copy_buffer_sub_data_with_f64_and_f64_and_f64(
                                Gl::COPY_READ_BUFFER,
                                Gl::COPY_WRITE_BUFFER,
                                0.0,
                                position as f64,
                                row_bytes as f64,
                            );
                        }
                    }
                    if result.is_err() {
                        break;
                    }
                }
                raw.pixel_storei(Gl::PACK_ALIGNMENT, alignment);
                raw.pixel_storei(Gl::PACK_ROW_LENGTH, row_length);
                raw.pixel_storei(Gl::PACK_SKIP_ROWS, skip_rows);
                raw.pixel_storei(Gl::PACK_SKIP_PIXELS, skip_pixels);
                raw.bind_buffer(
                    Gl::PIXEL_PACK_BUFFER,
                    previous_buffer.dyn_ref::<WebGlBuffer>(),
                );
                raw.bind_framebuffer(
                    Gl::READ_FRAMEBUFFER,
                    previous_framebuffer.dyn_ref::<WebGlFramebuffer>(),
                );
                if let Some(scratch) = &scratch {
                    raw.bind_buffer(
                        Gl::COPY_READ_BUFFER,
                        previous_copy_read.dyn_ref::<WebGlBuffer>(),
                    );
                    raw.bind_buffer(
                        Gl::COPY_WRITE_BUFFER,
                        previous_copy_write.dyn_ref::<WebGlBuffer>(),
                    );
                    raw.delete_buffer(Some(scratch));
                }
                raw.delete_framebuffer(Some(&framebuffer));
                result
            };
            inner = operation();
        });
        outer.and(inner)
    }
}
