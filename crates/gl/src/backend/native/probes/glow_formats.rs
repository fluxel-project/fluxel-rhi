//! Texture, framebuffer, and format operation probes for the current context.
use super::*;

impl GlowProbes<'_> {
    /// Upload distinct finite UNORM codes and compare the native readback bytes.
    pub(super) fn probe_reads_unorm_format(
        &self,
        format: crate::backend::api::GlFormat,
    ) -> ProbeAnswer {
        use crate::backend::api::GlFormat as F;
        use glow::HasContext as _;
        let (Some(internal), Some((external, kind)), Some(pixel)) = (
            super::super::super::resource::native_texture_format(format),
            super::super::super::exec_copy::upload_encoding(format),
            super::super::super::exec_copy::upload_pixel_format(format),
        ) else {
            return ProbeAnswer::Unavailable;
        };
        let data: Vec<u8> = match format {
            F::R8Unorm => vec![0, 63, 128, 255],
            F::Rg8Unorm => vec![0, 255, 63, 191, 128, 17, 255, 0],
            F::R16Unorm | F::Rg16Unorm | F::Rgba16Unorm => {
                let count = pixel.bytes_per_pixel() as usize * 4 / 2;
                (0..count)
                    .flat_map(|i| {
                        [0u16, 16385, 32769, 65535, 1193, 50123, 2, 40000][i % 8].to_ne_bytes()
                    })
                    .collect()
            }
            F::Rgb10a2Unorm => [0u32, 0x4008_03ff, 0x9555_55aa, u32::MAX]
                .into_iter()
                .flat_map(u32::to_ne_bytes)
                .collect(),
            _ => return ProbeAnswer::Unavailable,
        };
        let mut scratch = ProbeScratch::new(self.gl);
        // SAFETY: discovery owns the current context, scratch owns each object,
        // and the transfer buffers cover four tightly packed pixels.
        unsafe {
            let Ok(texture) = self.gl.create_texture() else {
                return ProbeAnswer::Unavailable;
            };
            scratch.textures.push(texture);
            let Ok(fbo) = self.gl.create_framebuffer() else {
                return ProbeAnswer::Unavailable;
            };
            scratch.framebuffer = Some(fbo);
            let pack = self.gl.get_parameter_i32(glow::PACK_ALIGNMENT);
            let unpack = self.gl.get_parameter_i32(glow::UNPACK_ALIGNMENT);
            self.gl.bind_buffer(glow::PIXEL_UNPACK_BUFFER, None);
            self.gl.bind_buffer(glow::PIXEL_PACK_BUFFER, None);
            self.gl.pixel_store_i32(glow::PACK_ALIGNMENT, 1);
            self.gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 1);
            self.gl.bind_texture(glow::TEXTURE_2D, Some(texture));
            self.gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                internal as i32,
                4,
                1,
                0,
                external,
                kind,
                glow::PixelUnpackData::Slice(Some(&data)),
            );
            self.gl.bind_framebuffer(glow::FRAMEBUFFER, Some(fbo));
            self.gl.framebuffer_texture_2d(
                glow::FRAMEBUFFER,
                glow::COLOR_ATTACHMENT0,
                glow::TEXTURE_2D,
                Some(texture),
                0,
            );
            let mut result = vec![0; data.len()];
            let complete =
                self.gl.check_framebuffer_status(glow::FRAMEBUFFER) == glow::FRAMEBUFFER_COMPLETE;
            if complete {
                self.gl.read_pixels(
                    0,
                    0,
                    4,
                    1,
                    external,
                    kind,
                    glow::PixelPackData::Slice(Some(&mut result)),
                );
            }
            let failed = self.take_error();
            self.gl.pixel_store_i32(glow::PACK_ALIGNMENT, pack);
            self.gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, unpack);
            if complete && !failed && result == data {
                ProbeAnswer::Passed
            } else {
                ProbeAnswer::Failed
            }
        }
    }
    pub(super) fn probe_multisample_array_attachment(
        &self,
        format: u32,
        attachment: u32,
        samples: u32,
        profile: GlFamilyProfile,
        extensions: &GlExtensionSet,
    ) -> ProbeAnswer {
        use glow::HasContext as _;
        let Some(allocate) = self.multisample_array.select(profile, extensions) else {
            return ProbeAnswer::Unavailable;
        };
        let mut scratch = ProbeScratch::new(self.gl);
        // SAFETY: the entry belongs to this context and scratch retains all
        // objects until the exact count and layer attachment are checked.
        unsafe {
            let Ok(texture) = self.gl.create_texture() else {
                let _ = self.take_error();
                return if self.fatal_error.get() {
                    ProbeAnswer::ContextLost
                } else {
                    ProbeAnswer::Unavailable
                };
            };
            scratch.textures.push(texture);
            self.gl
                .bind_texture(glow::TEXTURE_2D_MULTISAMPLE_ARRAY, Some(texture));
            allocate(
                glow::TEXTURE_2D_MULTISAMPLE_ARRAY,
                samples as i32,
                format,
                4,
                4,
                2,
                1,
            );
            let mut failed = self.take_error();
            if !failed {
                let actual = self.gl.get_tex_level_parameter_i32(
                    glow::TEXTURE_2D_MULTISAMPLE_ARRAY,
                    0,
                    glow::TEXTURE_SAMPLES,
                );
                failed = self.take_error() || actual != samples as i32;
            }
            if !failed {
                if let Ok(fbo) = self.gl.create_framebuffer() {
                    scratch.framebuffer = Some(fbo);
                    self.gl.bind_framebuffer(glow::FRAMEBUFFER, Some(fbo));
                    self.gl.framebuffer_texture_layer(
                        glow::FRAMEBUFFER,
                        attachment,
                        Some(texture),
                        0,
                        1,
                    );
                    if attachment != glow::COLOR_ATTACHMENT0 {
                        self.gl.draw_buffers(&[glow::NONE]);
                        self.gl.read_buffer(glow::NONE);
                    }
                    let complete = self.gl.check_framebuffer_status(glow::FRAMEBUFFER)
                        == glow::FRAMEBUFFER_COMPLETE;
                    failed = self.take_error() || !complete;
                } else {
                    let _ = self.take_error();
                    failed = true;
                }
            }
            self.gl
                .bind_texture(glow::TEXTURE_2D_MULTISAMPLE_ARRAY, None);
            if self.fatal_error.get() {
                ProbeAnswer::ContextLost
            } else if failed {
                ProbeAnswer::Failed
            } else {
                ProbeAnswer::Passed
            }
        }
    }
    pub(super) fn probe_multisample_renderbuffer_attachment(
        &self,
        format: u32,
        attachment: u32,
        samples: u32,
    ) -> ProbeAnswer {
        use glow::HasContext as _;
        let mut scratch = ProbeScratch::new(self.gl);
        // SAFETY: discovery owns the current context and scratch cleans up all
        // objects; both the allocation and exact sample count are queried.
        unsafe {
            let Ok(rb) = self.gl.create_renderbuffer() else {
                let _ = self.take_error();
                return if self.fatal_error.get() {
                    ProbeAnswer::ContextLost
                } else {
                    ProbeAnswer::Unavailable
                };
            };
            scratch.renderbuffer = Some(rb);
            self.gl.bind_renderbuffer(glow::RENDERBUFFER, Some(rb));
            self.gl.renderbuffer_storage_multisample(
                glow::RENDERBUFFER,
                samples as i32,
                format,
                4,
                4,
            );
            let mut failed = self.take_error();
            if !failed {
                let actual = self
                    .gl
                    .get_renderbuffer_parameter_i32(glow::RENDERBUFFER, glow::RENDERBUFFER_SAMPLES);
                failed = self.take_error() || actual != samples as i32;
            }
            if !failed {
                if let Ok(fbo) = self.gl.create_framebuffer() {
                    scratch.framebuffer = Some(fbo);
                    self.gl.bind_framebuffer(glow::FRAMEBUFFER, Some(fbo));
                    self.gl.framebuffer_renderbuffer(
                        glow::FRAMEBUFFER,
                        attachment,
                        glow::RENDERBUFFER,
                        Some(rb),
                    );
                    if attachment != glow::COLOR_ATTACHMENT0 {
                        self.gl.draw_buffers(&[glow::NONE]);
                        self.gl.read_buffer(glow::NONE);
                    }
                    let complete = self.gl.check_framebuffer_status(glow::FRAMEBUFFER)
                        == glow::FRAMEBUFFER_COMPLETE;
                    failed = self.take_error() || !complete;
                } else {
                    let _ = self.take_error();
                    failed = true;
                }
            }
            if self.fatal_error.get() {
                ProbeAnswer::ContextLost
            } else if failed {
                ProbeAnswer::Failed
            } else {
                ProbeAnswer::Passed
            }
        }
    }
    pub(super) fn probe_multisample_texture_attachment(
        &self,
        format: u32,
        attachment: u32,
        samples: u32,
        immutable: bool,
    ) -> ProbeAnswer {
        use glow::HasContext as _;
        let mut scratch = ProbeScratch::new(self.gl);
        // SAFETY: discovery holds the context exclusively; scratch retains all
        // objects through the allocation, exact-count query and attachment test.
        unsafe {
            let Ok(texture) = self.gl.create_texture() else {
                let _ = self.take_error();
                return if self.fatal_error.get() {
                    ProbeAnswer::ContextLost
                } else {
                    ProbeAnswer::Unavailable
                };
            };
            scratch.textures.push(texture);
            self.gl
                .bind_texture(glow::TEXTURE_2D_MULTISAMPLE, Some(texture));
            if immutable {
                self.gl.tex_storage_2d_multisample(
                    glow::TEXTURE_2D_MULTISAMPLE,
                    samples as i32,
                    format,
                    4,
                    4,
                    true,
                );
            } else {
                self.gl.tex_image_2d_multisample(
                    glow::TEXTURE_2D_MULTISAMPLE,
                    samples as i32,
                    format as i32,
                    4,
                    4,
                    true,
                );
            }
            let mut failed = self.take_error();
            if !failed {
                let actual = self.gl.get_tex_level_parameter_i32(
                    glow::TEXTURE_2D_MULTISAMPLE,
                    0,
                    glow::TEXTURE_SAMPLES,
                );
                failed = self.take_error() || actual != samples as i32;
            }
            if !failed {
                if let Ok(fbo) = self.gl.create_framebuffer() {
                    scratch.framebuffer = Some(fbo);
                    self.gl.bind_framebuffer(glow::FRAMEBUFFER, Some(fbo));
                    self.gl.framebuffer_texture_2d(
                        glow::FRAMEBUFFER,
                        attachment,
                        glow::TEXTURE_2D_MULTISAMPLE,
                        Some(texture),
                        0,
                    );
                    if attachment != glow::COLOR_ATTACHMENT0 {
                        self.gl.draw_buffers(&[glow::NONE]);
                        self.gl.read_buffer(glow::NONE);
                    }
                    let complete = self.gl.check_framebuffer_status(glow::FRAMEBUFFER)
                        == glow::FRAMEBUFFER_COMPLETE;
                    failed = self.take_error() || !complete;
                } else {
                    let _ = self.take_error();
                    failed = true;
                }
            }
            self.gl.bind_texture(glow::TEXTURE_2D_MULTISAMPLE, None);
            if self.fatal_error.get() {
                ProbeAnswer::ContextLost
            } else if failed {
                ProbeAnswer::Failed
            } else {
                ProbeAnswer::Passed
            }
        }
    }
    pub(super) fn probe_reads_integer_format(
        &self,
        format: u32,
        external: u32,
        kind: u32,
        bytes: u32,
    ) -> ProbeAnswer {
        use glow::HasContext as _;
        let mut scratch = ProbeScratch::new(self.gl);
        // SAFETY: discovery owns the current context; all scratch resources
        // are restored/deleted before returning, including failure paths.
        unsafe {
            let Ok(texture) = self.gl.create_texture() else {
                return ProbeAnswer::Unavailable;
            };
            scratch.textures.push(texture);
            self.gl.bind_texture(glow::TEXTURE_2D, Some(texture));
            self.gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                format as i32,
                1,
                1,
                0,
                external,
                kind,
                glow::PixelUnpackData::Slice(None),
            );
            if self.take_error() {
                return ProbeAnswer::Failed;
            }
            let Ok(fbo) = self.gl.create_framebuffer() else {
                return ProbeAnswer::Unavailable;
            };
            scratch.framebuffer = Some(fbo);
            self.gl.bind_framebuffer(glow::FRAMEBUFFER, Some(fbo));
            self.gl.framebuffer_texture_2d(
                glow::FRAMEBUFFER,
                glow::COLOR_ATTACHMENT0,
                glow::TEXTURE_2D,
                Some(texture),
                0,
            );
            if self.gl.check_framebuffer_status(glow::FRAMEBUFFER) != glow::FRAMEBUFFER_COMPLETE {
                return ProbeAnswer::Failed;
            }
            let Ok(buffer) = self.gl.create_buffer() else {
                return ProbeAnswer::Unavailable;
            };
            scratch.buffers.push(buffer);
            self.gl.bind_buffer(glow::PIXEL_PACK_BUFFER, Some(buffer));
            self.gl
                .buffer_data_size(glow::PIXEL_PACK_BUFFER, bytes as i32, glow::STREAM_READ);
            self.gl.read_pixels(
                0,
                0,
                1,
                1,
                external,
                kind,
                glow::PixelPackData::BufferOffset(0),
            );
            let failed = self.take_error();
            self.gl.bind_buffer(glow::PIXEL_PACK_BUFFER, None);
            if failed {
                ProbeAnswer::Failed
            } else {
                ProbeAnswer::Passed
            }
        }
    }
    pub(super) fn probe_binds_image_format(
        &self,
        format: u32,
        qualifier: &str,
        image_type: &str,
        desktop: bool,
    ) -> ProbeAnswer {
        use glow::HasContext as _;
        let version = if desktop {
            "#version 420\n"
        } else {
            "#version 310 es\nprecision highp float;\nprecision highp int;\n"
        };
        let precision = if desktop { "" } else { "highp " };
        let source = format!(
            "{version}layout({qualifier}, binding=0) readonly uniform {precision}{image_type} src;\nlayout({qualifier}, binding=1) writeonly uniform {precision}{image_type} dst;\nvoid main() {{ imageStore(dst, ivec2(0), imageLoad(src, ivec2(0))); }}\n"
        );
        let vertex = if desktop {
            DESKTOP_IMAGE_VERTEX
        } else {
            ES_VERTEX
        };
        let mut scratch = ProbeScratch::new(self.gl);
        if !self.link_program(
            &[
                (glow::VERTEX_SHADER, vertex),
                (glow::FRAGMENT_SHADER, &source),
            ],
            &mut scratch,
        ) {
            return ProbeAnswer::Failed;
        }
        // SAFETY: the profile gate established image operations and these
        // temporary textures belong to the current context.
        unsafe {
            for unit in 0..2 {
                let Ok(texture) = self.gl.create_texture() else {
                    return ProbeAnswer::Unavailable;
                };
                scratch.textures.push(texture);
                self.gl.bind_texture(glow::TEXTURE_2D, Some(texture));
                self.gl.tex_storage_2d(glow::TEXTURE_2D, 1, format, 1, 1);
                if self.take_error() {
                    return ProbeAnswer::Failed;
                }
                self.gl.bind_image_texture(
                    unit,
                    Some(texture),
                    0,
                    false,
                    0,
                    if unit == 0 {
                        glow::READ_ONLY
                    } else {
                        glow::WRITE_ONLY
                    },
                    format,
                );
                if self.take_error() {
                    return ProbeAnswer::Failed;
                }
            }
        }
        ProbeAnswer::Passed
    }
    pub(super) fn probe_attachment_completes(
        &self,
        internal_format: u32,
        upload_format: u32,
        upload_type: u32,
        attachment: u32,
    ) -> ProbeAnswer {
        use glow::HasContext as _;
        let mut scratch = ProbeScratch::new(self.gl);
        // SAFETY: current-context contract; every parameter is a closed
        // registry constant and the upload is a 4x4 empty image. The scratch
        // guard unbinds and deletes everything on every return path.
        unsafe {
            let Ok(texture) = self.gl.create_texture() else {
                let _ = self.take_error();
                return ProbeAnswer::Unavailable;
            };
            scratch.textures.push(texture);
            self.gl.bind_texture(glow::TEXTURE_2D, Some(texture));
            self.gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                internal_format as i32,
                4,
                4,
                0,
                upload_format,
                upload_type,
                glow::PixelUnpackData::Slice(None),
            );
            let Ok(framebuffer) = self.gl.create_framebuffer() else {
                let _ = self.take_error();
                return ProbeAnswer::Unavailable;
            };
            scratch.framebuffer = Some(framebuffer);
            self.gl
                .bind_framebuffer(glow::FRAMEBUFFER, Some(framebuffer));
            self.gl.framebuffer_texture_2d(
                glow::FRAMEBUFFER,
                attachment,
                glow::TEXTURE_2D,
                Some(texture),
                0,
            );
            let status = self.gl.check_framebuffer_status(glow::FRAMEBUFFER);
            let errored = self.take_error();
            let answer = if errored {
                ProbeAnswer::Unavailable
            } else if status == glow::FRAMEBUFFER_COMPLETE {
                ProbeAnswer::Passed
            } else {
                ProbeAnswer::Failed
            };
            drop(scratch);
            let _ = self.take_error();
            answer
        }
    }

    pub(super) fn probe_renderbuffer_attachment_completes(
        &self,
        internal_format: u32,
        samples: u32,
    ) -> ProbeAnswer {
        use glow::HasContext as _;
        let mut scratch = ProbeScratch::new(self.gl);
        // SAFETY: current-context contract; closed registry constants only.
        // The scratch guard unbinds and deletes everything on every return.
        unsafe {
            let Ok(renderbuffer) = self.gl.create_renderbuffer() else {
                let _ = self.take_error();
                return ProbeAnswer::Unavailable;
            };
            scratch.renderbuffer = Some(renderbuffer);
            self.gl
                .bind_renderbuffer(glow::RENDERBUFFER, Some(renderbuffer));
            self.gl.renderbuffer_storage_multisample(
                glow::RENDERBUFFER,
                samples as i32,
                internal_format,
                4,
                4,
            );
            let Ok(framebuffer) = self.gl.create_framebuffer() else {
                let _ = self.take_error();
                return ProbeAnswer::Unavailable;
            };
            scratch.framebuffer = Some(framebuffer);
            self.gl
                .bind_framebuffer(glow::FRAMEBUFFER, Some(framebuffer));
            self.gl.framebuffer_renderbuffer(
                glow::FRAMEBUFFER,
                glow::DEPTH_ATTACHMENT,
                glow::RENDERBUFFER,
                Some(renderbuffer),
            );
            let status = self.gl.check_framebuffer_status(glow::FRAMEBUFFER);
            let errored = self.take_error();
            let answer = if errored {
                ProbeAnswer::Unavailable
            } else if status == glow::FRAMEBUFFER_COMPLETE {
                ProbeAnswer::Passed
            } else {
                ProbeAnswer::Failed
            };
            drop(scratch);
            let _ = self.take_error();
            answer
        }
    }
    /// Exercises a layered 3D image at mip 1 through completion and readback.
    pub(crate) fn storage_3d_nonzero_mip(
        &self,
        desktop: bool,
    ) -> Result<bool, super::super::super::discovery::NativeDiscoveryError> {
        let result = self.storage_3d_nonzero_mip_inner(desktop);
        let _ = self.take_error();
        if self.fatal_error.get() {
            Err(
                super::super::super::discovery::NativeDiscoveryError::QueryFailed(
                    "3D storage probe context lost or out of memory",
                ),
            )
        } else {
            Ok(result)
        }
    }

    fn storage_3d_nonzero_mip_inner(&self, desktop: bool) -> bool {
        use glow::HasContext as _;
        let source = if desktop {
            "#version 430\nlayout(local_size_x=1) in; layout(rgba8, binding=0) writeonly uniform image3D dst; void main(){ imageStore(dst, ivec3(0,0,0), vec4(0.5,0.25,0.0,1.0)); imageStore(dst, ivec3(0,0,1), vec4(0.25,0.5,0.0,1.0)); }"
        } else {
            "#version 310 es\nprecision highp float; precision highp int; layout(local_size_x=1) in; layout(rgba8, binding=0) writeonly uniform highp image3D dst; void main(){ imageStore(dst, ivec3(0,0,0), vec4(0.5,0.25,0.0,1.0)); imageStore(dst, ivec3(0,0,1), vec4(0.25,0.5,0.0,1.0)); }"
        };
        let mut scratch = ProbeScratch::new(self.gl);
        if !self.link_program(&[(glow::COMPUTE_SHADER, source)], &mut scratch) {
            return false;
        }
        // SAFETY: discovery holds this context exclusively. All temporary GL
        // objects are owned by scratch and completion precedes their deletion.
        unsafe {
            let Ok(texture) = self.gl.create_texture() else {
                return false;
            };
            scratch.textures.push(texture);
            self.gl.bind_texture(glow::TEXTURE_3D, Some(texture));
            self.gl
                .tex_storage_3d(glow::TEXTURE_3D, 2, glow::RGBA8, 8, 8, 4);
            self.gl
                .bind_image_texture(0, Some(texture), 1, true, 0, glow::WRITE_ONLY, glow::RGBA8);
            self.gl.use_program(scratch.program);
            self.gl.dispatch_compute(1, 1, 1);
            self.gl.memory_barrier(glow::FRAMEBUFFER_BARRIER_BIT);
            self.gl.finish();
            if self.take_error() {
                self.gl.bind_texture(glow::TEXTURE_3D, None);
                return false;
            }
            let Ok(fbo) = self.gl.create_framebuffer() else {
                self.gl.bind_texture(glow::TEXTURE_3D, None);
                return false;
            };
            scratch.framebuffer = Some(fbo);
            self.gl.bind_framebuffer(glow::FRAMEBUFFER, Some(fbo));
            let mut ok = true;
            for (layer, expected) in [(0, [128, 64, 0, 255]), (1, [64, 128, 0, 255])] {
                self.gl.framebuffer_texture_layer(
                    glow::FRAMEBUFFER,
                    glow::COLOR_ATTACHMENT0,
                    Some(texture),
                    1,
                    layer,
                );
                let mut pixel = [0u8; 4];
                let complete = self.gl.check_framebuffer_status(glow::FRAMEBUFFER)
                    == glow::FRAMEBUFFER_COMPLETE;
                if complete {
                    self.gl.read_pixels(
                        0,
                        0,
                        1,
                        1,
                        glow::RGBA,
                        glow::UNSIGNED_BYTE,
                        glow::PixelPackData::Slice(Some(&mut pixel)),
                    );
                }
                let errored = self.take_error();
                ok &= complete && !errored && pixel == expected;
            }
            self.gl.bind_texture(glow::TEXTURE_3D, None);
            ok
        }
    }
}
