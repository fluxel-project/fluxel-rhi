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
}

impl<'a> GlowProbes<'a> {
    /// Creates the probe executor for a current context.
    ///
    /// `get_query_iv` must come from the same Host loader that built `gl`.
    pub(crate) fn new(gl: &'a glow::Context, get_query_iv: Option<GetQueryivFn>) -> Self {
        Self { gl, get_query_iv }
    }

    fn take_error(&self) -> bool {
        use glow::HasContext as _;
        // SAFETY: upheld by GlowProbes::new's current-context contract.
        unsafe { self.gl.get_error() != glow::NO_ERROR }
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
    image_bound: bool,
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
            image_bound: false,
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
            if self.image_bound {
                gl.bind_image_texture(0, None, 0, false, 0, glow::READ_ONLY, glow::RGBA8);
                gl.bind_image_texture(1, None, 0, false, 0, glow::READ_ONLY, glow::RGBA8);
            }
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
        let stages = [(glow::FRAGMENT_SHADER, source)];
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
        let stages = [(glow::FRAGMENT_SHADER, source)];
        if !self.link_program(&stages, &mut scratch) {
            return ProbeAnswer::Failed;
        }
        // SAFETY: current-context contract; the route gate guarantees image
        // units and the rgba8 image format are valid on this context.
        let errored = unsafe {
            for unit in 0..2u32 {
                let texture = self.gl.create_texture();
                let Ok(texture) = texture else {
                    return ProbeAnswer::Unavailable;
                };
                scratch.textures.push(texture);
                self.gl.bind_texture(glow::TEXTURE_2D, Some(texture));
                self.gl.tex_image_2d(
                    glow::TEXTURE_2D,
                    0,
                    glow::RGBA8 as i32,
                    4,
                    4,
                    0,
                    glow::RGBA,
                    glow::UNSIGNED_BYTE,
                    glow::PixelUnpackData::Slice(None),
                );
                self.gl.bind_image_texture(
                    unit,
                    Some(texture),
                    0,
                    false,
                    0,
                    glow::READ_ONLY,
                    glow::RGBA8,
                );
            }
            scratch.image_bound = true;
            self.take_error()
        };
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

impl GlowProbes<'_> {
    /// Compiles and links one program, retaining it in `scratch` for cleanup.
    /// Returns `false` when any stage or the link failed, and `Unavailable`
    /// answers are surfaced by the caller as a failed probe.
    fn link_program(&self, stages: &[(u32, &'static str)], scratch: &mut ProbeScratch) -> bool {
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
                    return false;
                };
                shaders.push(shader);
                self.gl.shader_source(shader, source);
                self.gl.compile_shader(shader);
                if !self.gl.get_shader_compile_status(shader) || self.take_error() {
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
