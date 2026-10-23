//! Glow-backed native capability probes.
//!
//! The caller has already made one exact WGL/EGL context current and keeps it
//! current for this entire synchronous probe. This module owns only temporary
//! probe objects and deletes them before returning.

use super::*;

/// glow-backed probe execution over the already-current context.
pub(crate) struct GlowProbes<'a> {
    gl: &'a glow::Context,
    get_query_iv: Option<GetQueryivFn>,
    copy_image_symbols: [bool; 3],
    compressed_get_symbol: bool,
    multiview: Option<super::super::multiview::NativeMultiview>,
    multisample_array: super::super::multisample_array::MultisampleArrayFns,
    fatal_error: std::cell::Cell<bool>,
}

impl<'a> GlowProbes<'a> {
    /// Creates the probe executor for a current context.
    ///
    /// `get_query_iv` must come from the same Host loader that built `gl`.
    pub(crate) fn new(
        gl: &'a glow::Context,
        get_query_iv: Option<GetQueryivFn>,
        copy_image_symbols: [bool; 3],
        compressed_get_symbol: bool,
        multiview: Option<super::super::multiview::NativeMultiview>,
        multisample_array: super::super::multisample_array::MultisampleArrayFns,
    ) -> Self {
        Self {
            gl,
            get_query_iv,
            copy_image_symbols,
            compressed_get_symbol,
            multiview,
            multisample_array,
            fatal_error: std::cell::Cell::new(false),
        }
    }

    fn take_error(&self) -> bool {
        use glow::HasContext as _;
        // SAFETY: upheld by GlowProbes::new's current-context contract.
        let mut failed = false;
        // Drain distinct pending GL errors so a preceding validation failure
        // cannot hide an OOM or a lost context from the caller.
        for _ in 0..16 {
            let error = unsafe { self.gl.get_error() };
            if error == glow::NO_ERROR {
                break;
            }
            failed = true;
            if error == glow::OUT_OF_MEMORY || error == glow::CONTEXT_LOST {
                self.fatal_error.set(true);
            }
        }
        failed
    }

    /// Exercises a layered 3D image at mip 1 through actual completion and
    /// readback. Binding alone misses drivers that fail UAV creation at dispatch.
    pub(crate) fn storage_3d_nonzero_mip(
        &self,
        desktop: bool,
    ) -> Result<bool, super::super::discovery::NativeDiscoveryError> {
        let result = self.storage_3d_nonzero_mip_inner(desktop);
        let _ = self.take_error();
        if self.fatal_error.get() {
            Err(super::super::discovery::NativeDiscoveryError::QueryFailed(
                "3D storage probe context lost or out of memory",
            ))
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

/// Scratch objects of one probe, unbound and deleted on every drop path.
struct ProbeScratch<'a> {
    gl: &'a glow::Context,
    program: Option<glow::Program>,
    vertex_array: Option<glow::VertexArray>,
    buffers: Vec<glow::NativeBuffer>,
    textures: Vec<glow::NativeTexture>,
    renderbuffer: Option<glow::NativeRenderbuffer>,
    framebuffer: Option<glow::NativeFramebuffer>,
    /// Indexed bindings that must be unbound before the objects are deleted.
    storage_bound: bool,
}

impl<'a> ProbeScratch<'a> {
    fn new(gl: &'a glow::Context) -> Self {
        Self {
            gl,
            program: None,
            vertex_array: None,
            buffers: Vec::new(),
            textures: Vec::new(),
            renderbuffer: None,
            framebuffer: None,
            storage_bound: false,
        }
    }
}

impl Drop for ProbeScratch<'_> {
    fn drop(&mut self) {
        use glow::HasContext as _;
        let gl = self.gl;
        // SAFETY: every handle was created on this current context during the
        // probe; unbinding precedes deletion on all paths, including unwinds.
        unsafe {
            if self.program.is_some() {
                gl.use_program(None);
            }
            if self.vertex_array.is_some() {
                gl.bind_vertex_array(None);
            }
            gl.bind_buffer(glow::ARRAY_BUFFER, None);
            gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, None);
            gl.bind_buffer(glow::DRAW_INDIRECT_BUFFER, None);
            gl.bind_buffer(glow::DISPATCH_INDIRECT_BUFFER, None);
            if self.storage_bound {
                gl.bind_buffer_base(glow::SHADER_STORAGE_BUFFER, 0, None);
            }
            // Deleting each texture below also detaches it from image units.
            // Avoid a redundant bind-image-to-zero call: GLES drivers may
            // reject that form even though image-unit ownership is released
            // when the texture object is deleted.
            gl.bind_texture(glow::TEXTURE_2D, None);
            if self.renderbuffer.is_some() {
                gl.bind_renderbuffer(glow::RENDERBUFFER, None);
            }
            if self.framebuffer.is_some() {
                gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            }
            for buffer in self.buffers.drain(..) {
                gl.delete_buffer(buffer);
            }
            for texture in self.textures.drain(..) {
                gl.delete_texture(texture);
            }
            if let Some(renderbuffer) = self.renderbuffer.take() {
                gl.delete_renderbuffer(renderbuffer);
            }
            if let Some(framebuffer) = self.framebuffer.take() {
                gl.delete_framebuffer(framebuffer);
            }
            if let Some(vertex_array) = self.vertex_array.take() {
                gl.delete_vertex_array(vertex_array);
            }
            if let Some(program) = self.program.take() {
                gl.delete_program(program);
            }
        }
    }
}

impl NativeGlProbes for GlowProbes<'_> {
    fn multisample_array_attachment(
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
    fn multisample_renderbuffer_attachment(
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
    fn multisample_texture_attachment(
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
    fn reads_integer_format(
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
    fn binds_image_format(
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
    fn multiview_attachment(&self) -> ProbeAnswer {
        match self.multiview {
            Some(entry) => unsafe { super::super::multiview::probe(self.gl, entry) },
            None => ProbeAnswer::Unavailable,
        }
    }
    fn copy_image_symbols(&self) -> [bool; 3] {
        self.copy_image_symbols
    }
    fn compressed_get_symbol(&self) -> bool {
        self.compressed_get_symbol
    }
    fn query_counter_bits(&self) -> Option<u32> {
        let get_query_iv = self.get_query_iv?;
        let mut bits: i32 = 0;
        // SAFETY: the entry point came from the same Host loader as the current
        // context and is called while discovery keeps that context current, as
        // `GetQueryivFn` documents.
        unsafe { get_query_iv(glow::TIMESTAMP, glow::QUERY_COUNTER_BITS, &mut bits) };
        let errored = self.take_error();
        if errored || bits <= 0 {
            return None;
        }
        u32::try_from(bits).ok()
    }

    fn attachment_completes(
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

    fn renderbuffer_attachment_completes(&self, internal_format: u32, samples: u32) -> ProbeAnswer {
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

    fn links_trivial_program(&self, stages: &[(u32, &'static str)]) -> ProbeAnswer {
        let mut scratch = ProbeScratch::new(self.gl);
        let linked = self.link_program(stages, &mut scratch);
        if linked {
            ProbeAnswer::Passed
        } else {
            ProbeAnswer::Failed
        }
    }

    fn dispatches_compute(&self, source: &'static str) -> ProbeAnswer {
        use glow::HasContext as _;
        let mut scratch = ProbeScratch::new(self.gl);
        let stages = [(glow::COMPUTE_SHADER, source)];
        if !self.link_active_program(&stages, &mut scratch) {
            return ProbeAnswer::Failed;
        }
        // SAFETY: current-context contract; a one-work-group empty dispatch,
        // with the compute program this just made current.
        let errored = unsafe {
            self.gl.dispatch_compute(1, 1, 1);
            self.take_error()
        };
        if errored {
            ProbeAnswer::Failed
        } else {
            ProbeAnswer::Passed
        }
    }

    fn binds_shader_storage(&self, source: &'static str) -> ProbeAnswer {
        use glow::HasContext as _;
        let mut scratch = ProbeScratch::new(self.gl);
        let vertex = if is_es_shader(source) {
            ES_VERTEX
        } else {
            DESKTOP_STORAGE_VERTEX
        };
        let stages = [
            (glow::VERTEX_SHADER, vertex),
            (glow::FRAGMENT_SHADER, source),
        ];
        if !self.link_program(&stages, &mut scratch) {
            return ProbeAnswer::Failed;
        }
        // SAFETY: current-context contract; the route gate guarantees the
        // SSBO target and indexed binding point are valid on this context.
        let errored = unsafe {
            let buffer = self.gl.create_buffer();
            let Ok(buffer) = buffer else {
                return ProbeAnswer::Unavailable;
            };
            scratch.buffers.push(buffer);
            self.gl.bind_buffer(glow::ARRAY_BUFFER, Some(buffer));
            self.gl
                .buffer_data_u8_slice(glow::ARRAY_BUFFER, &[0; 16], glow::STATIC_DRAW);
            self.gl
                .bind_buffer_base(glow::SHADER_STORAGE_BUFFER, 0, Some(buffer));
            scratch.storage_bound = true;
            self.take_error()
        };
        if errored {
            ProbeAnswer::Failed
        } else {
            ProbeAnswer::Passed
        }
    }

    fn binds_image_load_store(&self, source: &'static str) -> ProbeAnswer {
        use glow::HasContext as _;
        let mut scratch = ProbeScratch::new(self.gl);
        let vertex = if is_es_shader(source) {
            ES_VERTEX
        } else {
            DESKTOP_IMAGE_VERTEX
        };
        let stages = [
            (glow::VERTEX_SHADER, vertex),
            (glow::FRAGMENT_SHADER, source),
        ];
        if !self.link_program(&stages, &mut scratch) {
            return ProbeAnswer::Failed;
        }
        // SAFETY: current-context contract; the route gate guarantees image
        // units and the rgba8 image format are valid on this context.
        let mut errored = false;
        unsafe {
            for unit in 0..2u32 {
                let texture = self.gl.create_texture();
                let Ok(texture) = texture else {
                    return ProbeAnswer::Unavailable;
                };
                scratch.textures.push(texture);
                self.gl.bind_texture(glow::TEXTURE_2D, Some(texture));
                self.gl
                    .tex_storage_2d(glow::TEXTURE_2D, 1, glow::RGBA8, 4, 4);
                errored |= self.gl.get_error() != glow::NO_ERROR;
                self.gl.bind_image_texture(
                    unit,
                    Some(texture),
                    0,
                    false,
                    0,
                    glow::READ_ONLY,
                    glow::RGBA8,
                );
                errored |= self.gl.get_error() != glow::NO_ERROR;
            }
        }
        if errored {
            ProbeAnswer::Failed
        } else {
            ProbeAnswer::Passed
        }
    }

    fn issues_indirect_draw(&self, indexed: bool, stages: &[(u32, &'static str)]) -> ProbeAnswer {
        use glow::HasContext as _;
        let mut scratch = ProbeScratch::new(self.gl);
        if !self.link_active_program(stages, &mut scratch) {
            return ProbeAnswer::Failed;
        }
        // SAFETY: current-context contract. A zeroed `count` draw record is a
        // defined no-op, so the probe cannot rasterize into any framebuffer.
        let errored = unsafe {
            let vertex_array = self.gl.create_vertex_array();
            let Ok(vertex_array) = vertex_array else {
                return ProbeAnswer::Unavailable;
            };
            scratch.vertex_array = Some(vertex_array);
            self.gl.bind_vertex_array(Some(vertex_array));
            let command = self.gl.create_buffer();
            let Ok(command) = command else {
                return ProbeAnswer::Unavailable;
            };
            scratch.buffers.push(command);
            // The record is read from the indirect target, not from the array
            // buffer: `glDrawArraysIndirect` and `glDrawElementsIndirect` never
            // look at the `GL_ARRAY_BUFFER` binding, so a record bound there is
            // a record the draw never sees.
            self.gl
                .bind_buffer(glow::DRAW_INDIRECT_BUFFER, Some(command));
            // An indexed record is twenty bytes -- `count`, `instanceCount`,
            // `firstIndex`, `baseVertex`, `baseInstance` -- and the driver reads
            // the whole record before it knows the count is zero. Sizing the
            // store for the non-indexed shape would therefore leave the indexed
            // read running past the end of it, which is a defect the wrong bind
            // target was hiding.
            let record: &[u8] = if indexed { &[0; 20] } else { &[0; 16] };
            self.gl
                .buffer_data_u8_slice(glow::DRAW_INDIRECT_BUFFER, record, glow::STATIC_DRAW);
            if indexed {
                let elements = self.gl.create_buffer();
                let Ok(elements) = elements else {
                    return ProbeAnswer::Unavailable;
                };
                scratch.buffers.push(elements);
                self.gl
                    .bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(elements));
                self.gl.buffer_data_u8_slice(
                    glow::ELEMENT_ARRAY_BUFFER,
                    &[0; 20],
                    glow::STATIC_DRAW,
                );
                self.gl
                    .draw_elements_indirect_offset(glow::TRIANGLES, glow::UNSIGNED_INT, 0);
            } else {
                self.gl.draw_arrays_indirect_offset(glow::TRIANGLES, 0);
            }
            self.take_error()
        };
        if errored {
            ProbeAnswer::Failed
        } else {
            ProbeAnswer::Passed
        }
    }

    fn issues_indirect_dispatch(&self, compute_source: &'static str) -> ProbeAnswer {
        use glow::HasContext as _;
        let mut scratch = ProbeScratch::new(self.gl);
        let stages = [(glow::COMPUTE_SHADER, compute_source)];
        if !self.link_active_program(&stages, &mut scratch) {
            return ProbeAnswer::Failed;
        }
        // SAFETY: current-context contract; the empty shader makes the single
        // work group a no-op on every profile.
        let errored = unsafe {
            let command = self.gl.create_buffer();
            let Ok(command) = command else {
                return ProbeAnswer::Unavailable;
            };
            scratch.buffers.push(command);
            // `glDispatchComputeIndirect` reads its three work-group counts from
            // the dispatch-indirect target; an array-buffer binding leaves it
            // with no command at all.
            self.gl
                .bind_buffer(glow::DISPATCH_INDIRECT_BUFFER, Some(command));
            self.gl.buffer_data_u8_slice(
                glow::DISPATCH_INDIRECT_BUFFER,
                &[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                glow::STATIC_DRAW,
            );
            self.gl.dispatch_compute_indirect(0);
            self.take_error()
        };
        if errored {
            ProbeAnswer::Failed
        } else {
            ProbeAnswer::Passed
        }
    }
}

fn is_es_shader(source: &str) -> bool {
    source
        .lines()
        .next()
        .is_some_and(|version| version.trim_end().ends_with(" es"))
}

impl GlowProbes<'_> {
    /// Compiles and links one program, retaining it in `scratch` for cleanup.
    /// Returns `false` when any stage or the link failed, and `Unavailable`
    /// answers are surfaced by the caller as a failed probe.
    fn link_program(&self, stages: &[(u32, &str)], scratch: &mut ProbeScratch) -> bool {
        use glow::HasContext as _;
        // SAFETY: current-context contract; compile/link state is per-context
        // and every created shader is deleted by the scratch guard.
        unsafe {
            let Ok(program) = self.gl.create_program() else {
                return false;
            };
            scratch.program = Some(program);
            let mut shaders = Vec::new();
            for (stage, source) in stages {
                let Ok(shader) = self.gl.create_shader(*stage) else {
                    for shader in shaders {
                        self.gl.delete_shader(shader);
                    }
                    return false;
                };
                shaders.push(shader);
                self.gl.shader_source(shader, source);
                self.gl.compile_shader(shader);
                if !self.gl.get_shader_compile_status(shader) || self.take_error() {
                    for shader in shaders {
                        self.gl.delete_shader(shader);
                    }
                    return false;
                }
                self.gl.attach_shader(program, shader);
            }
            self.gl.link_program(program);
            let linked = self.gl.get_program_link_status(program) && !self.take_error();
            for shader in shaders {
                self.gl.detach_shader(program, shader);
                self.gl.delete_shader(shader);
            }
            linked
        }
    }

    /// Links one program and leaves it current.
    ///
    /// Every probe that issues a draw or a dispatch needs an active program:
    /// the driver raises `GL_INVALID_OPERATION` for a draw or dispatch with no
    /// program current for the stage it uses, so a successful link is not
    /// enough.  This exists as one call so the requirement is stated once
    /// instead of remembered per probe -- three of them used to omit it and
    /// answered `Failed` on every real driver while the bind-only probes beside
    /// them passed, which is what made `compute`, `indirect-draw` and
    /// `indirect-dispatch` permanently false on real hardware.
    fn link_active_program(
        &self,
        stages: &[(u32, &'static str)],
        scratch: &mut ProbeScratch,
    ) -> bool {
        use glow::HasContext as _;
        if !self.link_program(stages, scratch) {
            return false;
        }
        let Some(program) = scratch.program else {
            return false;
        };
        // SAFETY: current-context contract.  The program was linked on this
        // context a moment ago and stays owned by `scratch`, whose `Drop`
        // clears the binding before deleting it.
        unsafe { self.gl.use_program(Some(program)) };
        true
    }
}
#[cfg(test)]
mod tests;
