//! Convert stencil array subresources to an integer sampled mirror.
//! The source remains authoritative; the extraction preserves the caller's GL state.
//! Some GLES implementations address stencil array subresources inconsistently:
//! direct integer fetch works for the base mip, while stencil testing works for
//! non-base mips. Extract each plane into R8UI using the appropriate path, so
//! shader-visible mip and layer coordinates retain their ordinary meaning.
use super::provider::NativeGlProvider;
use crate::backend::gl::api::{GlError, GlTextureDesc, GlTextureDimension};
use glow::HasContext as _;
use std::num::NonZeroU32;

impl NativeGlProvider {
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
        // SAFETY: all names belong to this context. Every mutated GL binding,
        // enable and dynamic state is restored below, including error paths.
        unsafe {
            let gl = &self.gl;
            let read = gl.get_parameter_i32(glow::READ_FRAMEBUFFER_BINDING) as u32;
            let draw = gl.get_parameter_i32(glow::DRAW_FRAMEBUFFER_BINDING) as u32;
            let old_program = gl.get_parameter_i32(glow::CURRENT_PROGRAM) as u32;
            let old_vao = gl.get_parameter_i32(glow::VERTEX_ARRAY_BINDING) as u32;
            let old_active = gl.get_parameter_i32(glow::ACTIVE_TEXTURE) as u32;
            let mut viewport = [0; 4];
            gl.get_parameter_i32_slice(glow::VIEWPORT, &mut viewport);
            // OES/EXT indexed draw-buffer state defines the non-indexed
            // COLOR_WRITEMASK and BLEND queries as draw buffer zero. Only
            // index zero is changed below when indexed entry points exist.
            let mut color_mask = [0; 4];
            gl.get_parameter_i32_slice(glow::COLOR_WRITEMASK, &mut color_mask);
            let front = [
                gl.get_parameter_i32(glow::STENCIL_FUNC),
                gl.get_parameter_i32(glow::STENCIL_REF),
                gl.get_parameter_i32(glow::STENCIL_VALUE_MASK),
                gl.get_parameter_i32(glow::STENCIL_WRITEMASK),
            ];
            let back = [
                gl.get_parameter_i32(glow::STENCIL_BACK_FUNC),
                gl.get_parameter_i32(glow::STENCIL_BACK_REF),
                gl.get_parameter_i32(glow::STENCIL_BACK_VALUE_MASK),
                gl.get_parameter_i32(glow::STENCIL_BACK_WRITEMASK),
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
            let indexed_core = matches!(self.discovery.context().profile(), crate::backend::gl::api::GlFamilyProfile::Embedded { major, minor } if (major, minor) >= (3, 2));
            let old_blend = gl.is_enabled(glow::BLEND);
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
            let enabled = caps.map(|cap| gl.is_enabled(cap));
            let sample_mask = matches!(self.discovery.context().profile(), crate::backend::gl::api::GlFamilyProfile::Embedded { major, minor } if (major, minor) >= (3, 1))
                .then(|| gl.is_enabled(glow::SAMPLE_MASK));
            gl.active_texture(glow::TEXTURE0);
            let old_tex = gl.get_parameter_i32(glow::TEXTURE_BINDING_2D_ARRAY) as u32;
            let old_sampler = gl.get_parameter_i32(glow::SAMPLER_BINDING) as u32;
            gl.bind_texture(glow::TEXTURE_2D_ARRAY, Some(src));
            let old_mode = gl.get_tex_parameter_i32(glow::TEXTURE_2D_ARRAY, 0x90EA);
            gl.tex_parameter_i32(glow::TEXTURE_2D_ARRAY, 0x90EA, glow::STENCIL_INDEX as i32);
            gl.bind_sampler(0, None);
            let result = (|| -> Result<(), GlError> {
                let shader = |kind, code: &str| -> Result<glow::NativeShader, GlError> {
                    let shader = gl.create_shader(kind).map_err(|message| GlError::Driver {
                        operation: OP,
                        message,
                    })?;
                    gl.shader_source(shader, code);
                    gl.compile_shader(shader);
                    if !gl.get_shader_compile_status(shader) {
                        let message = gl.get_shader_info_log(shader);
                        gl.delete_shader(shader);
                        return Err(GlError::Driver {
                            operation: OP,
                            message,
                        });
                    }
                    Ok(shader)
                };
                let vs = shader(
                    glow::VERTEX_SHADER,
                    "#version 300 es\nvoid main(){ vec2 p=vec2((gl_VertexID<<1)&2,gl_VertexID&2); gl_Position=vec4(p*2.0-1.0,0.0,1.0); }",
                )?;
                let fs = match shader(
                    glow::FRAGMENT_SHADER,
                    "#version 300 es\nprecision highp float; precision highp int; uniform highp usampler2DArray src; uniform int layer; uniform uint value; uniform bool direct; layout(location=0) out highp uint out_value; void main(){ out_value=direct ? texelFetch(src,ivec3(ivec2(gl_FragCoord.xy),layer),0).r : value; }",
                ) {
                    Ok(v) => v,
                    Err(e) => {
                        gl.delete_shader(vs);
                        return Err(e);
                    }
                };
                let program = match gl.create_program() {
                    Ok(v) => v,
                    Err(message) => {
                        gl.delete_shader(vs);
                        gl.delete_shader(fs);
                        return Err(GlError::Driver {
                            operation: OP,
                            message,
                        });
                    }
                };
                gl.attach_shader(program, vs);
                gl.attach_shader(program, fs);
                gl.link_program(program);
                gl.delete_shader(vs);
                gl.delete_shader(fs);
                if !gl.get_program_link_status(program) {
                    let message = gl.get_program_info_log(program);
                    gl.delete_program(program);
                    return Err(GlError::Driver {
                        operation: OP,
                        message,
                    });
                }
                let vao = match gl.create_vertex_array() {
                    Ok(v) => v,
                    Err(message) => {
                        gl.delete_program(program);
                        return Err(GlError::Driver {
                            operation: OP,
                            message,
                        });
                    }
                };
                let fbo = match gl.create_framebuffer() {
                    Ok(v) => v,
                    Err(message) => {
                        gl.delete_vertex_array(vao);
                        gl.delete_program(program);
                        return Err(GlError::Driver {
                            operation: OP,
                            message,
                        });
                    }
                };
                for cap in caps {
                    gl.disable(cap);
                }
                if sample_mask.is_some() {
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
                gl.uniform_1_i32(gl.get_uniform_location(program, "src").as_ref(), 0);
                let layer_loc = gl.get_uniform_location(program, "layer");
                let value_loc = gl.get_uniform_location(program, "value");
                let direct_loc = gl.get_uniform_location(program, "direct");
                gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, Some(fbo));
                gl.draw_buffers(&[glow::COLOR_ATTACHMENT0]);
                let mut outcome = Ok(());
                'mips: for mip in 0..view.mip_level_count {
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
                            gl.uniform_1_i32(direct_loc.as_ref(), 1);
                            gl.uniform_1_i32(layer_loc.as_ref(), abs_layer as i32);
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
                            gl.uniform_1_i32(direct_loc.as_ref(), 0);
                        }
                        if gl.check_framebuffer_status(glow::DRAW_FRAMEBUFFER)
                            != glow::FRAMEBUFFER_COMPLETE
                        {
                            outcome = Err(GlError::Unsupported {
                                operation: OP,
                                reason: "stencil extraction framebuffer incomplete",
                            });
                            break 'mips;
                        }
                        if abs_mip == 0 {
                            gl.draw_arrays(glow::TRIANGLES, 0, 3);
                        } else {
                            for value in 0..=255 {
                                gl.stencil_func(glow::EQUAL, value, 255);
                                gl.uniform_1_u32(value_loc.as_ref(), value as u32);
                                gl.draw_arrays(glow::TRIANGLES, 0, 3);
                            }
                        }
                        if let Err(e) = self.driver_error(OP) {
                            outcome = Err(e);
                            break 'mips;
                        }
                    }
                }
                gl.delete_framebuffer(fbo);
                gl.delete_vertex_array(vao);
                gl.delete_program(program);
                outcome
            })();
            gl.bind_texture(glow::TEXTURE_2D_ARRAY, Some(src));
            gl.tex_parameter_i32(glow::TEXTURE_2D_ARRAY, 0x90EA, old_mode);
            gl.bind_texture(
                glow::TEXTURE_2D_ARRAY,
                NonZeroU32::new(old_tex).map(glow::NativeTexture),
            );
            gl.bind_sampler(0, NonZeroU32::new(old_sampler).map(glow::NativeSampler));
            gl.active_texture(old_active);
            gl.bind_framebuffer(
                glow::READ_FRAMEBUFFER,
                NonZeroU32::new(read).map(glow::NativeFramebuffer),
            );
            gl.bind_framebuffer(
                glow::DRAW_FRAMEBUFFER,
                NonZeroU32::new(draw).map(glow::NativeFramebuffer),
            );
            gl.use_program(NonZeroU32::new(old_program).map(glow::NativeProgram));
            gl.bind_vertex_array(NonZeroU32::new(old_vao).map(glow::NativeVertexArray));
            gl.viewport(viewport[0], viewport[1], viewport[2], viewport[3]);
            if let Some(fns) = indexed {
                (fns.mask)(
                    0,
                    u8::from(color_mask[0] != 0),
                    u8::from(color_mask[1] != 0),
                    u8::from(color_mask[2] != 0),
                    u8::from(color_mask[3] != 0),
                );
                if old_blend {
                    (fns.enable)(glow::BLEND, 0);
                } else {
                    (fns.disable)(glow::BLEND, 0);
                }
            } else if indexed_core {
                gl.color_mask_draw_buffer(
                    0,
                    color_mask[0] != 0,
                    color_mask[1] != 0,
                    color_mask[2] != 0,
                    color_mask[3] != 0,
                );
                if old_blend {
                    gl.enable_draw_buffer(glow::BLEND, 0);
                } else {
                    gl.disable_draw_buffer(glow::BLEND, 0);
                }
            } else {
                gl.color_mask(
                    color_mask[0] != 0,
                    color_mask[1] != 0,
                    color_mask[2] != 0,
                    color_mask[3] != 0,
                );
                if old_blend {
                    gl.enable(glow::BLEND);
                } else {
                    gl.disable(glow::BLEND);
                }
            }
            gl.stencil_func_separate(glow::FRONT, front[0] as u32, front[1], front[2] as u32);
            gl.stencil_mask_separate(glow::FRONT, front[3] as u32);
            gl.stencil_func_separate(glow::BACK, back[0] as u32, back[1], back[2] as u32);
            gl.stencil_mask_separate(glow::BACK, back[3] as u32);
            if let Some(was_enabled) = sample_mask {
                if was_enabled {
                    gl.enable(glow::SAMPLE_MASK);
                } else {
                    gl.disable(glow::SAMPLE_MASK);
                }
            }
            for (cap, was_enabled) in caps.into_iter().zip(enabled) {
                if was_enabled {
                    gl.enable(cap);
                } else {
                    gl.disable(cap);
                }
            }
            result.and_then(|()| self.driver_error(OP))
        }
    }
}
