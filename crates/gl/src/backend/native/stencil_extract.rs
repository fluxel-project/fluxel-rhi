//! Convert stencil array subresources to an integer sampled mirror.
//! The source remains authoritative; the extraction preserves the caller's GL state.
//! Some GLES implementations address stencil array subresources inconsistently:
//! direct integer fetch works for the base mip, while stencil testing works for
//! non-base mips. Extract each plane into R8UI using the appropriate path, so
//! shader-visible mip and layer coordinates retain their ordinary meaning.
use super::provider::NativeGlProvider;
use crate::backend::api::{GlError, GlTextureDesc, GlTextureDimension};
use crate::backend::state::{ScopedRawAccess, StateEvent};
use glow::HasContext as _;

/// Context-owned fixed objects for stencil-array extraction.
///
/// The utility is deliberately outside Fluxel's resource tables: it is a
/// private implementation detail and is invalid with the provider generation.
/// Its shader source and uniform interface never vary by extraction, so warm
/// calls only update uniform values and attachments.
pub(super) struct StencilExtractUtility {
    program: glow::NativeProgram,
    vao: glow::NativeVertexArray,
    fbo: glow::NativeFramebuffer,
    src: glow::NativeUniformLocation,
    layer: glow::NativeUniformLocation,
    value: glow::NativeUniformLocation,
    direct: glow::NativeUniformLocation,
}

impl StencilExtractUtility {
    /// Deletes objects while the owning context is still current.
    ///
    /// Context-loss reset merely drops the handles: issuing deletes there is
    /// not valid.  Provider disposal calls this helper before that reset.
    pub(super) unsafe fn delete(&self, gl: &glow::Context) {
        // SAFETY: the caller guarantees this exact utility's context is
        // current and remains live for all three deletions.
        unsafe {
            gl.delete_framebuffer(self.fbo);
            gl.delete_vertex_array(self.vao);
            gl.delete_program(self.program);
        }
    }
}

impl NativeGlProvider {
    /// # Safety
    /// The provider's exact context is current during normal device teardown.
    pub(crate) unsafe fn clear_stencil_extract_utility(&mut self) {
        if let Some(utility) = self.stencil_extract_utility.take() {
            // SAFETY: caller holds the owning context current.
            unsafe { utility.delete(&self.gl) };
        }
    }

    pub(super) fn extract_stencil_array(
        &mut self,
        src: glow::NativeTexture,
        source: GlTextureDesc,
        dst: glow::NativeTexture,
        view: GlTextureDesc,
        base_mip: u32,
        base_layer: u32,
    ) -> Result<(), GlError> {
        const OP: &str = "extract stencil array";
        if !self.requires_stencil_extract()
            || source.dimension != GlTextureDimension::D2Array
            || !matches!(
                view.dimension,
                GlTextureDimension::D2 | GlTextureDimension::D2Array
            )
            || source.sample_count != 1
            || view.sample_count != 1
            || base_mip
                .checked_add(view.mip_level_count)
                .is_none_or(|v| v > source.mip_level_count)
            || base_layer
                .checked_add(view.extent.depth_or_layers)
                .is_none_or(|v| v > source.extent.depth_or_layers)
        {
            return Err(GlError::Unsupported {
                operation: OP,
                reason: "stencil extraction shape",
            });
        }
        let utility = self.ensure_stencil_extract_utility()?;
        let (program, vao, fbo, src_loc, layer_loc, value_loc, direct_loc) = (
            utility.program,
            utility.vao,
            utility.fbo,
            utility.src,
            utility.layer,
            utility.value,
            utility.direct,
        );
        // SAFETY: all names belong to this context. This private utility pass
        // canonical-cleans its scratch bindings then invalidates the CPU state
        // mirror; its caller reinstalls the active pass and complete pipeline
        // packet before issuing the next draw.  It never snapshots GL state.
        unsafe {
            let gl = &self.gl;
            let caps = [
                glow::DEPTH_TEST,
                glow::STENCIL_TEST,
                glow::CULL_FACE,
                glow::SCISSOR_TEST,
                glow::DITHER,
                glow::RASTERIZER_DISCARD,
                glow::SAMPLE_COVERAGE,
                glow::SAMPLE_ALPHA_TO_COVERAGE,
                glow::POLYGON_OFFSET_FILL,
            ];
            let has_extension = |name| {
                self.discovery
                    .extensions()
                    .raw_reported_names()
                    .any(|item| item == name)
            };
            let indexed = if has_extension("GL_OES_draw_buffers_indexed") {
                self.indexed_blend.oes
            } else if has_extension("GL_EXT_draw_buffers_indexed") {
                self.indexed_blend.ext
            } else {
                None
            };
            let indexed_core = matches!(self.discovery.context().profile(), crate::backend::api::GlFamilyProfile::Embedded { major, minor } if (major, minor) >= (3, 2));
            let sample_mask = matches!(self.discovery.context().profile(), crate::backend::api::GlFamilyProfile::Embedded { major, minor } if (major, minor) >= (3, 1));
            gl.active_texture(glow::TEXTURE0);
            gl.bind_texture(glow::TEXTURE_2D_ARRAY, Some(src));
            gl.tex_parameter_i32(glow::TEXTURE_2D_ARRAY, 0x90EA, glow::STENCIL_INDEX as i32);
            gl.bind_sampler(0, None);
            let result = (|| -> Result<(), GlError> {
                for cap in caps {
                    gl.disable(cap);
                }
                if sample_mask {
                    gl.disable(glow::SAMPLE_MASK);
                }
                if let Some(fns) = indexed {
                    (fns.disable)(glow::BLEND, 0);
                    (fns.mask)(0, 1, 1, 1, 1);
                } else if indexed_core {
                    gl.disable_draw_buffer(glow::BLEND, 0);
                    gl.color_mask_draw_buffer(0, true, true, true, true);
                } else {
                    gl.disable(glow::BLEND);
                    gl.color_mask(true, true, true, true);
                }
                gl.stencil_mask(0);
                gl.use_program(Some(program));
                gl.bind_vertex_array(Some(vao));
                gl.uniform_1_i32(Some(&src_loc), 0);
                gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, Some(fbo));
                gl.draw_buffers(&[glow::COLOR_ATTACHMENT0]);
                for mip in 0..view.mip_level_count {
                    let extent = view.mip_extent(mip).expect("validated mip");
                    gl.viewport(0, 0, extent.width as i32, extent.height as i32);
                    for layer in 0..view.extent.depth_or_layers {
                        if view.dimension == GlTextureDimension::D2 {
                            gl.framebuffer_texture_2d(
                                glow::DRAW_FRAMEBUFFER,
                                glow::COLOR_ATTACHMENT0,
                                glow::TEXTURE_2D,
                                Some(dst),
                                mip as i32,
                            );
                        } else {
                            gl.framebuffer_texture_layer(
                                glow::DRAW_FRAMEBUFFER,
                                glow::COLOR_ATTACHMENT0,
                                Some(dst),
                                mip as i32,
                                layer as i32,
                            );
                        }
                        let abs_mip = base_mip + mip;
                        let abs_layer = base_layer + layer;
                        if abs_mip == 0 {
                            gl.bind_texture(glow::TEXTURE_2D_ARRAY, Some(src));
                            gl.framebuffer_texture_layer(
                                glow::DRAW_FRAMEBUFFER,
                                glow::DEPTH_STENCIL_ATTACHMENT,
                                None,
                                0,
                                0,
                            );
                            gl.disable(glow::STENCIL_TEST);
                            gl.uniform_1_i32(Some(&direct_loc), 1);
                            gl.uniform_1_i32(Some(&layer_loc), abs_layer as i32);
                        } else {
                            gl.bind_texture(glow::TEXTURE_2D_ARRAY, None);
                            gl.framebuffer_texture_layer(
                                glow::DRAW_FRAMEBUFFER,
                                glow::DEPTH_STENCIL_ATTACHMENT,
                                Some(src),
                                abs_mip as i32,
                                abs_layer as i32,
                            );
                            gl.enable(glow::STENCIL_TEST);
                            gl.uniform_1_i32(Some(&direct_loc), 0);
                        }
                        // Both attachment descriptors and their selected
                        // mip/layer ranges were validated before this utility
                        // is entered. Avoid a per-plane completeness query on
                        // the warm path; a driver rejection is quarantined by
                        // the deferred frame-boundary observation.
                        if abs_mip == 0 {
                            gl.draw_arrays(glow::TRIANGLES, 0, 3);
                        } else {
                            for value in 0..=255 {
                                gl.stencil_func(glow::EQUAL, value, 255);
                                gl.uniform_1_u32(Some(&value_loc), value as u32);
                                gl.draw_arrays(glow::TRIANGLES, 0, 3);
                            }
                        }
                    }
                }
                Ok(())
            })();
            gl.bind_texture(glow::TEXTURE_2D_ARRAY, Some(src));
            gl.tex_parameter_i32(glow::TEXTURE_2D_ARRAY, 0x90EA, glow::DEPTH_COMPONENT as i32);
            gl.bind_texture(glow::TEXTURE_2D_ARRAY, None);
            gl.bind_sampler(0, None);
            gl.use_program(None);
            gl.bind_vertex_array(None);
            gl.bind_framebuffer(glow::READ_FRAMEBUFFER, None);
            gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, None);
            self.current_program = None;
            self.current_native_program = None;
            self.bound_vertex_array = None;
            self.context_state
                .event(StateEvent::ScopedRawAccess(ScopedRawAccess::all()));
            result.and_then(|()| self.defer_driver_error(OP))
        }
    }

    fn ensure_stencil_extract_utility(&mut self) -> Result<&StencilExtractUtility, GlError> {
        const OP: &str = "extract stencil array";
        if self.stencil_extract_utility.is_none() {
            // SAFETY: the provider owns the current context. These names are
            // retained only for this provider generation and deleted at normal
            // provider disposal.
            unsafe {
                let compile = |kind, source: &str| -> Result<glow::NativeShader, GlError> {
                    let shader =
                        self.gl
                            .create_shader(kind)
                            .map_err(|message| GlError::Driver {
                                operation: OP,
                                message,
                            })?;
                    self.gl.shader_source(shader, source);
                    self.gl.compile_shader(shader);
                    if !self.gl.get_shader_compile_status(shader) {
                        let message = self.gl.get_shader_info_log(shader);
                        self.gl.delete_shader(shader);
                        return Err(GlError::Driver {
                            operation: OP,
                            message,
                        });
                    }
                    Ok(shader)
                };
                let vs = compile(
                    glow::VERTEX_SHADER,
                    "#version 300 es\nvoid main(){ vec2 p=vec2((gl_VertexID<<1)&2,gl_VertexID&2); gl_Position=vec4(p*2.0-1.0,0.0,1.0); }",
                )?;
                let fs = match compile(
                    glow::FRAGMENT_SHADER,
                    "#version 300 es\nprecision highp float; precision highp int; uniform highp usampler2DArray src; uniform int layer; uniform uint value; uniform bool direct; layout(location=0) out highp uint out_value; void main(){ out_value=direct ? texelFetch(src,ivec3(ivec2(gl_FragCoord.xy),layer),0).r : value; }",
                ) {
                    Ok(shader) => shader,
                    Err(error) => {
                        self.gl.delete_shader(vs);
                        return Err(error);
                    }
                };
                let program = match self.gl.create_program() {
                    Ok(program) => program,
                    Err(message) => {
                        self.gl.delete_shader(vs);
                        self.gl.delete_shader(fs);
                        return Err(GlError::Driver {
                            operation: OP,
                            message,
                        });
                    }
                };
                self.gl.attach_shader(program, vs);
                self.gl.attach_shader(program, fs);
                self.gl.link_program(program);
                self.gl.delete_shader(vs);
                self.gl.delete_shader(fs);
                if !self.gl.get_program_link_status(program) {
                    let message = self.gl.get_program_info_log(program);
                    self.gl.delete_program(program);
                    return Err(GlError::Driver {
                        operation: OP,
                        message,
                    });
                }
                let required_uniform = |name| {
                    self.gl
                        .get_uniform_location(program, name)
                        .ok_or_else(|| GlError::Driver {
                            operation: OP,
                            message: format!("stencil extraction uniform {name} optimized out"),
                        })
                };
                let (src, layer, value, direct) = match (
                    required_uniform("src"),
                    required_uniform("layer"),
                    required_uniform("value"),
                    required_uniform("direct"),
                ) {
                    (Ok(src), Ok(layer), Ok(value), Ok(direct)) => (src, layer, value, direct),
                    _ => {
                        self.gl.delete_program(program);
                        return Err(GlError::Driver {
                            operation: OP,
                            message: "stencil extraction utility uniform optimized out".into(),
                        });
                    }
                };
                let vao = match self.gl.create_vertex_array() {
                    Ok(vao) => vao,
                    Err(message) => {
                        self.gl.delete_program(program);
                        return Err(GlError::Driver {
                            operation: OP,
                            message,
                        });
                    }
                };
                let fbo = match self.gl.create_framebuffer() {
                    Ok(fbo) => fbo,
                    Err(message) => {
                        self.gl.delete_vertex_array(vao);
                        self.gl.delete_program(program);
                        return Err(GlError::Driver {
                            operation: OP,
                            message,
                        });
                    }
                };
                self.stencil_extract_utility = Some(StencilExtractUtility {
                    program,
                    vao,
                    fbo,
                    src,
                    layer,
                    value,
                    direct,
                });
            }
        }
        Ok(self
            .stencil_extract_utility
            .as_ref()
            .expect("stencil extraction utility inserted"))
    }
}
