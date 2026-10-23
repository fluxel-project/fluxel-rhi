//! Per-format compressed block image-copy and byte readback probe.
use super::*;

impl GlowProbes<'_> {
    /// Verify an actual compressed block survives the native image-copy and PBO route.
    pub(super) fn probe_compressed_alias_read(
        &self,
        format: crate::backend::gl::api::GlFormat,
        integer: crate::backend::gl::api::GlFormat,
        entry: usize,
    ) -> ProbeAnswer {
        use crate::backend::gl::api::{
            GlCompressedColorSpace as C, GlCompressedTextureFamily as F, GlFormat,
        };
        use glow::HasContext as _;
        if self.fatal_error.get() {
            return ProbeAnswer::ContextLost;
        }
        let Some(info) = format.compressed_info() else {
            return ProbeAnswer::Unavailable;
        };
        let Some(internal) = super::super::super::resource::native_texture_format(format) else {
            return ProbeAnswer::Unavailable;
        };
        let Some(integer_internal) = super::super::super::resource::native_texture_format(integer)
        else {
            return ProbeAnswer::Unavailable;
        };
        let Some((external, kind)) = super::super::super::exec_copy::upload_encoding(integer)
        else {
            return ProbeAnswer::Unavailable;
        };
        let Some(&ptr) = self.copy_image_entries.get(entry) else {
            return ProbeAnswer::Unavailable;
        };
        let Some(unmap_buffer) = self.unmap_buffer else {
            return ProbeAnswer::Unavailable;
        };
        if ptr.is_null() {
            return ProbeAnswer::Unavailable;
        }
        let mut block = vec![0u8; info.block_bytes as usize];
        match info.family {
            F::Astc => {
                block.copy_from_slice(&match format {
                    GlFormat::Astc {
                        color_space: C::Hdr,
                        ..
                    } => [
                        0xfc, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0, 0x3c, 0, 0, 0, 0, 0,
                        0x3c,
                    ],
                    _ => [
                        0xfc, 0xfd, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0,
                        0xff, 0xff,
                    ],
                });
            }
            _ => {
                for (i, b) in block.iter_mut().enumerate() {
                    *b = [0xf0, 0, 0, 1, 0x44, 0x11, 0x55, 0x11][i % 8];
                }
                if matches!(format, GlFormat::Bc6hRgbUfloat | GlFormat::Bc6hRgbSfloat) {
                    block.copy_from_slice(
                        &(3u128
                            | (450u128 << 5)
                            | (400u128 << 15)
                            | (450u128 << 35)
                            | (400u128 << 45))
                            .to_le_bytes(),
                    );
                }
                if matches!(format, GlFormat::Bc7RgbaUnorm | GlFormat::Bc7RgbaSrgb) {
                    block.copy_from_slice(
                        &((1u128 << 6) | (127u128 << 7) | (127u128 << 14)).to_le_bytes(),
                    );
                }
            }
        }
        let mut scratch = ProbeScratch::new(self.gl);
        // SAFETY: the current context owns all scratch objects and the loaded
        // entry has the standard core/EXT/OES CopyImageSubData ABI. The map
        // range exactly covers one integer texel and is unmapped before deletion.
        unsafe {
            let copy: unsafe extern "system" fn(
                u32,
                u32,
                i32,
                i32,
                i32,
                i32,
                u32,
                u32,
                i32,
                i32,
                i32,
                i32,
                i32,
                i32,
                i32,
            ) = core::mem::transmute(ptr);
            let Ok(src) = self.gl.create_texture() else {
                let _ = self.take_error();
                return if self.fatal_error.get() {
                    ProbeAnswer::ContextLost
                } else {
                    ProbeAnswer::Unavailable
                };
            };
            scratch.textures.push(src);
            let Ok(dst) = self.gl.create_texture() else {
                let _ = self.take_error();
                return if self.fatal_error.get() {
                    ProbeAnswer::ContextLost
                } else {
                    ProbeAnswer::Unavailable
                };
            };
            scratch.textures.push(dst);
            let Ok(fbo) = self.gl.create_framebuffer() else {
                let _ = self.take_error();
                return if self.fatal_error.get() {
                    ProbeAnswer::ContextLost
                } else {
                    ProbeAnswer::Unavailable
                };
            };
            scratch.framebuffer = Some(fbo);
            let Ok(pbo) = self.gl.create_buffer() else {
                let _ = self.take_error();
                return if self.fatal_error.get() {
                    ProbeAnswer::ContextLost
                } else {
                    ProbeAnswer::Unavailable
                };
            };
            scratch.buffers.push(pbo);
            self.gl.bind_buffer(glow::PIXEL_UNPACK_BUFFER, None);
            self.gl.bind_texture(glow::TEXTURE_2D, Some(src));
            self.gl.tex_storage_2d(
                glow::TEXTURE_2D,
                1,
                internal,
                info.block_width as i32,
                info.block_height as i32,
            );
            self.gl.compressed_tex_sub_image_2d(
                glow::TEXTURE_2D,
                0,
                0,
                0,
                info.block_width as i32,
                info.block_height as i32,
                internal,
                glow::CompressedPixelUnpackData::Slice(&block),
            );
            self.gl.bind_texture(glow::TEXTURE_2D, Some(dst));
            self.gl
                .tex_storage_2d(glow::TEXTURE_2D, 1, integer_internal, 1, 1);
            self.gl.bind_framebuffer(glow::FRAMEBUFFER, Some(fbo));
            self.gl.framebuffer_texture_2d(
                glow::FRAMEBUFFER,
                glow::COLOR_ATTACHMENT0,
                glow::TEXTURE_2D,
                Some(dst),
                0,
            );
            let complete =
                self.gl.check_framebuffer_status(glow::FRAMEBUFFER) == glow::FRAMEBUFFER_COMPLETE;
            let pack = self.gl.get_parameter_i32(glow::PACK_ALIGNMENT);
            self.gl.pixel_store_i32(glow::PACK_ALIGNMENT, 1);
            self.gl.bind_buffer(glow::PIXEL_PACK_BUFFER, Some(pbo));
            self.gl.buffer_data_size(
                glow::PIXEL_PACK_BUFFER,
                block.len() as i32,
                glow::STREAM_READ,
            );
            let mut failed = self.take_error() || !complete;
            if !failed {
                copy(
                    src.0.get(),
                    glow::TEXTURE_2D,
                    0,
                    0,
                    0,
                    0,
                    dst.0.get(),
                    glow::TEXTURE_2D,
                    0,
                    0,
                    0,
                    0,
                    info.block_width as i32,
                    info.block_height as i32,
                    1,
                );
                failed = self.take_error();
            }
            if !failed {
                self.gl.read_pixels(
                    0,
                    0,
                    1,
                    1,
                    external,
                    kind,
                    glow::PixelPackData::BufferOffset(0),
                );
                failed = self.take_error();
            }
            if !failed {
                let mapped = self.gl.map_buffer_range(
                    glow::PIXEL_PACK_BUFFER,
                    0,
                    block.len() as i32,
                    glow::MAP_READ_BIT,
                );
                if mapped.is_null() {
                    failed = true;
                } else {
                    failed = core::slice::from_raw_parts(mapped, block.len()) != block;
                    // `glow` intentionally discards GLboolean here. Raw block
                    // evidence is trustworthy only after the driver reports
                    // the mapped PBO contents were not corrupted.
                    if unmap_buffer(glow::PIXEL_PACK_BUFFER) == 0 {
                        failed = true;
                    }
                }
                failed |= self.take_error();
            }
            self.gl.bind_buffer(glow::PIXEL_PACK_BUFFER, None);
            self.gl.pixel_store_i32(glow::PACK_ALIGNMENT, pack);
            if self.fatal_error.get() {
                ProbeAnswer::ContextLost
            } else if failed {
                ProbeAnswer::Failed
            } else {
                ProbeAnswer::Passed
            }
        }
    }
}
