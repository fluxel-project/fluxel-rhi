//! Native fixed-raster and draw execution.
//!
//! Indexed blend and write masks use desktop GL 4.0 or GLES 3.2 core.
//! Older embedded contexts use the global state path when targets agree.

use super::exec_vertex::{indexed_draw_offset, indexed_draw_span, indexed_draw_type};
use super::provider::{ActiveRaster, NativeGlProvider};
use crate::backend::gl::api::{
    GlCullMode, GlDepthStencilState, GlDrawCommand, GlError, GlFamilyApi as _, GlFamilyProfile,
    GlFrontFace, GlMultiDraw, GlMultiDrawApi, GlRasterCommandApi, GlRasterPipeline, GlRasterState,
    GlRasterValidationInfo, GlScissorRect, GlStencilFaceState, GlVertexBufferMetadata, GlViewport,
    issue_single_draws,
};
use crate::backend::gl::state::RasterPipelineDiff;
#[path = "exec_raster/lowering.rs"]
mod lowering;
#[path = "exec_raster/rect.rs"]
mod rect;
pub(super) use lowering::topology_mode;
use lowering::{blend_factor, blend_operation, compare_function, stencil_operation};
use rect::native_rect;

impl NativeGlProvider {
    pub(crate) fn draw_direct_advanced(
        &mut self,
        draw: GlDrawCommand,
        base_vertex: i32,
        first_instance: u32,
    ) -> Result<(), GlError> {
        use crate::backend::gl::api::{GlVertexApi as _, GlVertexStepMode};
        const OP: &str = "draw-raster";
        self.assert_ready(OP)?;
        if base_vertex != 0 && self.base_vertex_draw.is_none() {
            return Err(GlError::Unsupported {
                operation: OP,
                reason: "base vertex draw entry point is unavailable",
            });
        }
        let program = self
            .raster
            .as_ref()
            .ok_or_else(|| Self::validation(OP, "no raster pipeline is installed"))?
            .program;
        let bound = self
            .bound_vertex_array
            .ok_or_else(|| Self::validation(OP, "no vertex array is bound"))?;
        let array = self.vertex_array(OP, bound)?;
        let original = array.bindings.clone();
        let index = array.index;
        let mut shifted = original.clone();
        for binding in &mut shifted {
            if let Some(layout) = array.layout.buffers.iter().find(|layout| {
                layout.slot == binding.slot && layout.step_mode == GlVertexStepMode::Instance
            }) {
                binding.offset = binding
                    .offset
                    .checked_add(u64::from(first_instance) * u64::from(layout.stride))
                    .ok_or_else(|| Self::validation(OP, "instance offset overflow"))?;
            }
        }
        if first_instance != 0 {
            self.bind_vertex_array(bound, &shifted, index)?;
        }
        let result = self
            .set_push_uniforms(
                program,
                &[(
                    naga::back::glsl::FIRST_INSTANCE_BINDING.to_owned(),
                    super::push_uniform::GlPushUniformValue::Uint(vec![first_instance]),
                )],
            )
            .and_then(|()| self.draw_direct_impl(draw, base_vertex));
        let restore = if first_instance != 0 {
            self.bind_vertex_array(bound, &original, index)
        } else {
            Ok(())
        };
        result.and(restore)
    }

    fn draw_direct_impl(&mut self, draw: GlDrawCommand, base_vertex: i32) -> Result<(), GlError> {
        use glow::HasContext as _;
        const OP: &str = "draw-raster";
        self.assert_ready(OP)?;
        if self.pass.is_none() {
            return Err(Self::validation(OP, "no active render pass"));
        }
        let (program, topology) = {
            let raster = self
                .raster
                .as_ref()
                .ok_or_else(|| Self::validation(OP, "no raster pipeline is installed"))?;
            (raster.program, raster.topology)
        };
        // A compute install or a link may have left another program current since
        // this pipeline was installed, so re-asserting the program is what makes
        // this draw the one the caller asked for rather than the one that happens
        // to be selected.
        self.ensure_program(OP, program)?;
        // The vertex array is *not* re-asserted from the pipeline: the geometry
        // domain reconciles inputs between the install and this draw and, under
        // the uncached execution mode, replaces the array on every request, so
        // the install-time identity is a dead object by now.  The provider's
        // record of what the driver holds is the fact this draw needs, and it is
        // also the array whose index binding and layout the draw must read.
        let bound = self
            .bound_vertex_array
            .ok_or_else(|| Self::validation(OP, "no vertex array is bound"))?;
        let mode = topology_mode(topology);
        let vertex_array = self.vertex_array(OP, bound)?;
        let index = vertex_array.index;
        let bindings = vertex_array.bindings.clone();
        let layout = vertex_array.layout.clone();
        let mut metadata = Vec::with_capacity(bindings.len() + usize::from(index.is_some()));
        for binding in &bindings {
            let entry = self.buffer(OP, binding.buffer)?;
            metadata.push(GlVertexBufferMetadata {
                buffer: binding.buffer,
                byte_length: entry.1.size,
                usage: entry.1.usage,
            });
        }
        if let Some(index) = index {
            let entry = self.buffer(OP, index.buffer)?;
            metadata.push(GlVertexBufferMetadata {
                buffer: index.buffer,
                byte_length: entry.1.size,
                usage: entry.1.usage,
            });
        }
        layout
            .validate_bindings(&bindings, index, &metadata, self.context_stamp())
            .map_err(|_| Self::validation(OP, "vertex bindings are incomplete or invalid"))?;
        // SAFETY: current-context contract; counts, instances, and index spans
        // are checked against the recorded allocations before any draw call.
        unsafe {
            match draw {
                GlDrawCommand::NonIndexed(draw) => {
                    if draw.vertex_count == 0 || draw.instance_count == 0 {
                        return Err(Self::validation(
                            OP,
                            "draw count and instances must be nonzero",
                        ));
                    }
                    let first = i32::try_from(draw.first_vertex)
                        .map_err(|_| Self::validation(OP, "first vertex exceeds i32"))?;
                    let count = i32::try_from(draw.vertex_count)
                        .map_err(|_| Self::validation(OP, "vertex count exceeds i32"))?;
                    let instances = i32::try_from(draw.instance_count)
                        .map_err(|_| Self::validation(OP, "instance count exceeds i32"))?;
                    if instances == 1 {
                        self.gl.draw_arrays(mode, first, count);
                    } else {
                        self.gl.draw_arrays_instanced(mode, first, count, instances);
                    }
                }
                GlDrawCommand::Indexed(draw) => {
                    if draw.index_count == 0 || draw.instance_count == 0 {
                        return Err(Self::validation(
                            OP,
                            "draw count and instances must be nonzero",
                        ));
                    }
                    let Some(index) = index else {
                        return Err(Self::validation(
                            OP,
                            "indexed draw requires a bound index buffer",
                        ));
                    };
                    // Bound checks use the allocation table, so an out-of-range
                    // draw rejects before the driver can silently ignore it.
                    let byte_length = self.buffer(OP, index.buffer)?.1.size;
                    let span = indexed_draw_span(index, draw.first_index, draw.index_count)
                        .ok_or_else(|| Self::validation(OP, "indexed draw span overflows"))?;
                    if span > byte_length {
                        return Err(Self::validation(OP, "indexed draw leaves the index buffer"));
                    }
                    let offset = indexed_draw_offset(index, draw.first_index)
                        .and_then(|value| isize::try_from(value).ok())
                        .ok_or_else(|| {
                            Self::validation(OP, "index offset exceeds native GLintptr")
                        })?;
                    let count = i32::try_from(draw.index_count)
                        .map_err(|_| Self::validation(OP, "index count exceeds i32"))?;
                    let instances = i32::try_from(draw.instance_count)
                        .map_err(|_| Self::validation(OP, "instance count exceeds i32"))?;
                    if base_vertex != 0 {
                        self.base_vertex_draw
                            .expect("entry point checked before drawing")(
                            mode,
                            count,
                            indexed_draw_type(index),
                            offset as usize as *const core::ffi::c_void,
                            instances,
                            base_vertex,
                        );
                    } else if instances == 1 {
                        if let Some(draw_elements) = self.draw_functions.elements {
                            draw_elements(
                                mode,
                                count,
                                indexed_draw_type(index),
                                offset as usize as *const core::ffi::c_void,
                            );
                        } else {
                            self.gl.draw_elements(
                                mode,
                                count,
                                indexed_draw_type(index),
                                i32::try_from(offset).map_err(|_| {
                                    Self::validation(
                                        OP,
                                        "wide DrawElements entry point unavailable",
                                    )
                                })?,
                            );
                        }
                    } else {
                        if let Some(draw_elements_instanced) =
                            self.draw_functions.elements_instanced
                        {
                            draw_elements_instanced(
                                mode,
                                count,
                                indexed_draw_type(index),
                                offset as usize as *const core::ffi::c_void,
                                instances,
                            );
                        } else {
                            self.gl.draw_elements_instanced(
                                mode,
                                count,
                                indexed_draw_type(index),
                                i32::try_from(offset).map_err(|_| {
                                    Self::validation(
                                        OP,
                                        "wide DrawElementsInstanced entry point unavailable",
                                    )
                                })?,
                                instances,
                            );
                        }
                    }
                }
            }
        }
        // Direct draw errors remain sticky until the raster-pass observation
        // boundary. Pass end/abort reports them and poisons uncertain work.
        Ok(())
    }
}

impl GlRasterCommandApi for NativeGlProvider {
    fn set_raster_pipeline(&mut self, pipeline: &GlRasterPipeline) -> Result<(), GlError> {
        const OP: &str = "set-raster-pipeline";
        let packet = self.raster_state_packet(pipeline)?;
        let diff = self.context_state.prepare_dynamic_pipeline(packet);
        if !diff.is_empty() {
            if let Err(error) = self.apply_raster_pipeline_diff(pipeline, diff) {
                self.context_state.pipeline_failed();
                return Err(error);
            }
            self.context_state.commit_pipeline(packet);
        }
        // The direct Layer-2 API still installs its declared VAO. The v13
        // driver instead lets the independent geometry domain bind it after
        // the pipeline diff, because per-draw buffers may differ.
        self.ensure_vertex_array(OP, pipeline.vertex_array)?;
        // The fixed GL state may already match a pipeline from an earlier
        // pass, but `begin_raster` clears this pass-local draw authority. Keep
        // the selected program/topology for draw validation even on an exact
        // ContextState cache hit.
        self.raster = Some(ActiveRaster {
            program: pipeline.program,
            topology: pipeline.state.topology,
            depth_stencil: pipeline.state.depth_stencil,
        });
        Ok(())
    }

    fn draw_raster(&mut self, draw: GlDrawCommand) -> Result<(), GlError> {
        self.draw_direct_advanced(draw, 0, 0)
    }

    fn set_viewport(&mut self, viewport: GlViewport) -> Result<(), GlError> {
        use glow::HasContext as _;
        let height = self
            .pass
            .as_ref()
            .ok_or_else(|| Self::validation("set-viewport", "no active render pass"))?
            .height;
        let rect = native_rect(
            "set-viewport",
            height,
            viewport.x,
            viewport.y,
            viewport.width,
            viewport.height,
        )?;
        // SAFETY: the active pass proves the framebuffer height; native_rect
        // converts the public upper-left rectangle before the current-context call.
        unsafe {
            self.gl.viewport(rect.0, rect.1, rect.2, rect.3);
            self.gl.depth_range_f32(
                f32::from_bits(viewport.min_depth),
                f32::from_bits(viewport.max_depth),
            );
        }
        self.driver_error("set-viewport")
    }

    fn set_scissor(&mut self, scissor: Option<GlScissorRect>) -> Result<(), GlError> {
        use glow::HasContext as _;
        let height = self
            .pass
            .as_ref()
            .ok_or_else(|| Self::validation("set-scissor", "no active render pass"))?
            .height;
        // SAFETY: the active pass supplies the framebuffer height used for the
        // public-to-GL Y conversion, and this provider owns the current context.
        unsafe {
            match scissor {
                Some(s) => {
                    let r = native_rect("set-scissor", height, s.x, s.y, s.width, s.height)?;
                    self.gl.enable(glow::SCISSOR_TEST);
                    self.gl.scissor(r.0, r.1, r.2, r.3);
                }
                None => self.gl.disable(glow::SCISSOR_TEST),
            };
        }
        self.driver_error("set-scissor")
    }
    fn set_blend_constant(&mut self, value: [f32; 4]) -> Result<(), GlError> {
        use glow::HasContext as _;
        self.assert_ready("set-blend-constant")?;
        if self.pass.is_none() || value.iter().any(|v| !v.is_finite()) {
            return Err(Self::validation(
                "set-blend-constant",
                "invalid pass or blend constant",
            ));
        }
        // SAFETY: the provider owns the current context and values are finite.
        unsafe {
            self.gl.blend_color(value[0], value[1], value[2], value[3]);
        }
        self.context_state.pipeline_failed();
        self.driver_error("set-blend-constant")
    }

    fn set_stencil_reference(&mut self, value: u32) -> Result<(), GlError> {
        self.assert_ready("set-stencil-reference")?;
        let state = self
            .raster
            .as_ref()
            .and_then(|r| r.depth_stencil)
            .ok_or_else(|| {
                Self::validation(
                    "set-stencil-reference",
                    "no depth stencil pipeline is installed",
                )
            })?;
        // SAFETY: the provider owns the current context and the face states are from the installed pipeline.
        unsafe {
            self.apply_stencil_face(glow::FRONT, &state.stencil_front, value);
            self.apply_stencil_face(glow::BACK, &state.stencil_back, value);
        }
        self.context_state.pipeline_failed();
        self.driver_error("set-stencil-reference")
    }
}

/// A batch always takes the single-draw route on the native families.
///
/// The native contract proves no combined batch command, so rather than
/// branching on a capability the provider decomposes every batch into the
/// validated single-draw path.
impl GlMultiDrawApi for NativeGlProvider {
    fn multi_draw(&mut self, command: &GlMultiDraw) -> Result<(), GlError> {
        const OP: &str = "multi-draw";
        self.assert_ready(OP)?;
        issue_single_draws(self, command)
    }
}

impl NativeGlProvider {
    /// Lowers only the fixed-state leaves selected by the shared context
    /// authority.  A caller commits the corresponding packet only after this
    /// returns successfully; an identity hit never reaches this method.
    pub(super) fn apply_raster_pipeline_diff(
        &mut self,
        pipeline: &GlRasterPipeline,
        diff: RasterPipelineDiff,
    ) -> Result<(), GlError> {
        const OP: &str = "apply-raster-pipeline-diff";
        self.assert_ready(OP)?;
        // The pass record is the authority for this exact bound framebuffer.
        // Do not retain these values in `ActiveRaster`: a subsequent pass can
        // have a different attachment extent or a resized default drawable.
        let (pass_width, pass_height, pass_samples) = self
            .pass
            .as_ref()
            .map(|pass| (pass.width, pass.height, pass.samples))
            .ok_or_else(|| Self::validation(OP, "no active render pass"))?;
        self.program(OP, pipeline.program)?;
        pipeline
            .state
            .validate(GlRasterValidationInfo {
                width: pass_width,
                height: pass_height,
                max_samples: self.discovery.limits().max_samples,
                max_color_targets: self.discovery.limits().max_color_attachments,
            })
            .map_err(|_| Self::validation(OP, "invalid raster state"))?;
        if pipeline.state.multisample.sample_count != pass_samples {
            return Err(Self::validation(
                OP,
                "pipeline sample count does not match the active framebuffer",
            ));
        }
        // SAFETY: `assert_ready` proves this provider's context is current on
        // its owner thread; the pass, program, and full raster state were
        // validated above. A driver error clears `self.raster` below so failed
        // partial state is never advertised as an installed pipeline.
        let applied = unsafe {
            if diff.program {
                // ContextState, not the provider's observed current program,
                // decided this fixed pipeline leaf must be emitted.
                self.select_program(OP, pipeline.program)?;
            }
            if diff.raster {
                self.apply_rasterization(pipeline.state.rasterization)?;
                self.apply_raster_block(OP, &pipeline.state, pass_height)?;
            }
            if diff.depth_stencil {
                self.apply_depth_stencil_block(OP, &pipeline.state)?;
            }
            if diff.blend {
                self.apply_blend_block(OP, &pipeline.state)?;
            }
            if diff.multisample {
                self.apply_multisample_block(OP, &pipeline.state)?;
            }
            Ok(())
        };
        if let Err(error) = applied.and_then(|()| self.driver_error(OP)) {
            self.raster = None;
            return Err(error);
        }
        Ok(())
    }

    /// Applies one complete validated raster state in a fixed order.
    ///
    /// # Safety
    ///
    /// The provider context must be current on its owner thread. `state` and
    /// `framebuffer_height` must already match the active pass; this helper
    /// changes fixed state only and does not own a framebuffer or pass.
    unsafe fn apply_raster_block(
        &self,
        op: &'static str,
        state: &GlRasterState,
        framebuffer_height: u32,
    ) -> Result<(), GlError> {
        use glow::HasContext as _;
        let viewport = &state.viewport;
        let viewport = native_rect(
            op,
            framebuffer_height,
            viewport.x,
            viewport.y,
            viewport.width,
            viewport.height,
        )?;
        let scissor = state
            .scissor
            .map(|rect| {
                native_rect(
                    op,
                    framebuffer_height,
                    rect.x,
                    rect.y,
                    rect.width,
                    rect.height,
                )
            })
            .transpose()?;
        // SAFETY: the enclosing helper's current-context contract holds and
        // every value was lowered from the validated fixed raster state.
        unsafe {
            // `native_rect` reflects public upper-left Y into GL lower-left Y
            // against this pass's live framebuffer height.
            self.gl
                .viewport(viewport.0, viewport.1, viewport.2, viewport.3);
            self.gl.depth_range_f32(
                f32::from_bits(state.viewport.min_depth),
                f32::from_bits(state.viewport.max_depth),
            );
            match scissor {
                Some(scissor) => {
                    self.gl.enable(glow::SCISSOR_TEST);
                    self.gl.scissor(scissor.0, scissor.1, scissor.2, scissor.3);
                }
                None => self.gl.disable(glow::SCISSOR_TEST),
            }
            match state.cull_mode {
                GlCullMode::None => self.gl.disable(glow::CULL_FACE),
                GlCullMode::Front => {
                    self.gl.enable(glow::CULL_FACE);
                    self.gl.cull_face(glow::FRONT);
                }
                GlCullMode::Back => {
                    self.gl.enable(glow::CULL_FACE);
                    self.gl.cull_face(glow::BACK);
                }
            }
            // Pipeline lowering accounts for the vertex-space Y reflection.
            // Culling and gl_FrontFacing must use the same native winding.
            self.gl.front_face(match state.front_face {
                GlFrontFace::Clockwise => glow::CW,
                GlFrontFace::CounterClockwise => glow::CCW,
            });
        }
        Ok(())
    }

    /// # Safety
    ///
    /// The provider context must be current and `state` must be the validated
    /// state for the currently active pass. This changes fixed state only.
    unsafe fn apply_depth_stencil_block(
        &self,
        op: &'static str,
        state: &GlRasterState,
    ) -> Result<(), GlError> {
        match &state.depth_stencil {
            // SAFETY: forwarded from this helper's current-context contract;
            // the enclosing state validation authenticated these fields.
            Some(depth_stencil) => unsafe { self.apply_depth_stencil(op, depth_stencil) },
            None => {
                use glow::HasContext as _;
                // SAFETY: this branch only disables state on the current
                // context and has no object handle or allocation precondition.
                unsafe {
                    self.gl.disable(glow::DEPTH_TEST);
                    self.gl.disable(glow::STENCIL_TEST);
                    self.set_polygon_offset_enabled(false);
                }
                Ok(())
            }
        }
    }

    /// # Safety
    ///
    /// The provider context must be current and `state` must originate from a
    /// validated raster pipeline. It neither validates nor owns attachments.
    unsafe fn apply_depth_stencil(
        &self,
        op: &'static str,
        state: &GlDepthStencilState,
    ) -> Result<(), GlError> {
        use glow::HasContext as _;
        // SAFETY: forwarded current-context and validated-state requirements;
        // all GLenum values are selected by closed lowering helpers.
        unsafe {
            self.gl.enable(glow::DEPTH_TEST);
            self.gl.depth_mask(state.depth_write_enabled);
            self.gl.depth_func(compare_function(state.depth_compare));
            // Depth bias travels as exact bits: [slope factor, units, clamp].
            let (factor, units, clamp) = (
                f32::from_bits(state.depth_bias[0]),
                f32::from_bits(state.depth_bias[1]),
                f32::from_bits(state.depth_bias[2]),
            );
            if clamp != 0.0 && self.polygon_offset_clamp.is_none() {
                return Err(GlError::Unsupported {
                    operation: op,
                    reason: "depth-bias clamp requires native polygon-offset-clamp support",
                });
            }
            if factor == 0.0 && units == 0.0 {
                self.set_polygon_offset_enabled(false);
            } else {
                self.set_polygon_offset_enabled(true);
                if let Some(set_clamped) = self.polygon_offset_clamp.filter(|_| clamp != 0.0) {
                    set_clamped(factor, units, clamp);
                } else {
                    // EXT/core PolygonOffset resets the clamp to zero, so a
                    // pipeline switch also restores ordinary bias semantics.
                    self.gl.polygon_offset(factor, units);
                }
            }
            self.gl.enable(glow::STENCIL_TEST);
            self.apply_stencil_face(glow::FRONT, &state.stencil_front, state.stencil_reference);
            self.apply_stencil_face(glow::BACK, &state.stencil_back, state.stencil_reference);
        }
        Ok(())
    }

    /// # Safety
    ///
    /// The provider context must be current; `face` is a closed GL front/back
    /// target and `state` belongs to a validated pipeline.
    unsafe fn apply_stencil_face(&self, face: u32, state: &GlStencilFaceState, reference: u32) {
        use glow::HasContext as _;
        // SAFETY: forwarded current-context contract and closed enum lowering.
        unsafe {
            self.gl.stencil_func_separate(
                face,
                compare_function(state.compare),
                reference as i32,
                state.read_mask,
            );
            self.gl.stencil_mask_separate(face, state.write_mask);
            self.gl.stencil_op_separate(
                face,
                stencil_operation(state.fail_op),
                stencil_operation(state.depth_fail_op),
                stencil_operation(state.pass_op),
            );
        }
    }

    /// # Safety
    ///
    /// The provider context must be current and `state` must have passed the
    /// common raster validation. This only mutates blend registers.
    unsafe fn apply_blend_block(
        &self,
        op: &'static str,
        state: &GlRasterState,
    ) -> Result<(), GlError> {
        use glow::HasContext as _;
        // SAFETY: forwarded current-context contract; all factors, operations,
        // masks, and float bits were validated or lowered from closed enums.
        unsafe {
            let Some(first) = state.color_targets.first() else {
                return Ok(());
            };
            // Desktop 4.0 provides indexed blend and color masks. ES core
            // exposes these from 3.2; older ES requires extension entry points.
            let indexed = match self.discovery.context().profile() {
                GlFamilyProfile::Desktop { major, .. } => major >= 4,
                GlFamilyProfile::Embedded { major, minor } => {
                    major > 3 || (major == 3 && minor >= 2)
                }
                _ => false,
            };
            let extension = if !indexed
                && matches!(
                    self.discovery.context().profile(),
                    GlFamilyProfile::Embedded { .. }
                ) {
                let has = |name| {
                    self.discovery
                        .extensions()
                        .raw_reported_names()
                        .any(|reported| reported == name)
                };
                (has("GL_OES_draw_buffers_indexed")
                    .then_some(self.indexed_blend.oes)
                    .flatten())
                .or_else(|| {
                    has("GL_EXT_draw_buffers_indexed")
                        .then_some(self.indexed_blend.ext)
                        .flatten()
                })
            } else {
                None
            };
            if let Some(functions) = extension {
                for (i, target) in state.color_targets.iter().enumerate() {
                    let i = i as u32;
                    (functions.mask)(
                        i,
                        u8::from(target.write_mask & 1 != 0),
                        u8::from(target.write_mask & 2 != 0),
                        u8::from(target.write_mask & 4 != 0),
                        u8::from(target.write_mask & 8 != 0),
                    );
                    if let Some(blend) = target.blend {
                        (functions.enable)(glow::BLEND, i);
                        (functions.func)(
                            i,
                            blend_factor(blend.color.src_factor),
                            blend_factor(blend.color.dst_factor),
                            blend_factor(blend.alpha.src_factor),
                            blend_factor(blend.alpha.dst_factor),
                        );
                        (functions.equation)(
                            i,
                            blend_operation(blend.color.operation),
                            blend_operation(blend.alpha.operation),
                        );
                    } else {
                        (functions.disable)(glow::BLEND, i);
                    }
                }
                self.gl.blend_color(
                    f32::from_bits(state.blend_constant[0]),
                    f32::from_bits(state.blend_constant[1]),
                    f32::from_bits(state.blend_constant[2]),
                    f32::from_bits(state.blend_constant[3]),
                );
                return Ok(());
            }
            if indexed {
                for (i, target) in state.color_targets.iter().enumerate() {
                    let i = i as u32;
                    self.gl.color_mask_draw_buffer(
                        i,
                        target.write_mask & 1 != 0,
                        target.write_mask & 2 != 0,
                        target.write_mask & 4 != 0,
                        target.write_mask & 8 != 0,
                    );
                    if let Some(blend) = target.blend {
                        self.gl.enable_draw_buffer(glow::BLEND, i);
                        self.gl.blend_func_separate_draw_buffer(
                            i,
                            blend_factor(blend.color.src_factor),
                            blend_factor(blend.color.dst_factor),
                            blend_factor(blend.alpha.src_factor),
                            blend_factor(blend.alpha.dst_factor),
                        );
                        self.gl.blend_equation_separate_draw_buffer(
                            i,
                            blend_operation(blend.color.operation),
                            blend_operation(blend.alpha.operation),
                        );
                    } else {
                        self.gl.disable_draw_buffer(glow::BLEND, i);
                    }
                }
                self.gl.blend_color(
                    f32::from_bits(state.blend_constant[0]),
                    f32::from_bits(state.blend_constant[1]),
                    f32::from_bits(state.blend_constant[2]),
                    f32::from_bits(state.blend_constant[3]),
                );
                return Ok(());
            }
            if state
                .color_targets
                .iter()
                .any(|target| target.write_mask != first.write_mask || target.blend != first.blend)
            {
                return Err(GlError::Unsupported {
                    operation: op,
                    reason: "independent blend requires indexed draw-buffer state",
                });
            }
            self.gl.color_mask(
                first.write_mask & 0b0001 != 0,
                first.write_mask & 0b0010 != 0,
                first.write_mask & 0b0100 != 0,
                first.write_mask & 0b1000 != 0,
            );
            match first.blend {
                Some(blend) => {
                    self.gl.enable(glow::BLEND);
                    self.gl.blend_func_separate(
                        blend_factor(blend.color.src_factor),
                        blend_factor(blend.color.dst_factor),
                        blend_factor(blend.alpha.src_factor),
                        blend_factor(blend.alpha.dst_factor),
                    );
                    self.gl.blend_equation_separate(
                        blend_operation(blend.color.operation),
                        blend_operation(blend.alpha.operation),
                    );
                    self.gl.blend_color(
                        f32::from_bits(state.blend_constant[0]),
                        f32::from_bits(state.blend_constant[1]),
                        f32::from_bits(state.blend_constant[2]),
                        f32::from_bits(state.blend_constant[3]),
                    );
                }
                None => self.gl.disable(glow::BLEND),
            }
        }
        Ok(())
    }

    /// # Safety
    ///
    /// The provider context must be current and `state` must be validated for
    /// the active pass. This helper only changes coverage state.
    unsafe fn apply_multisample_block(
        &self,
        op: &'static str,
        state: &GlRasterState,
    ) -> Result<(), GlError> {
        use glow::HasContext as _;
        // SAFETY: forwarded current-context contract; the mask entry is loaded
        // from this same context only when its core version supports it.
        unsafe {
            if let Some(sample_mask) = self.sample_mask_entry {
                if state.multisample.sample_mask == u64::MAX {
                    self.gl.disable(glow::SAMPLE_MASK);
                } else {
                    // Core carries 64 coverage bits. Initialize every active
                    // native word so borrowed-context high words cannot leak.
                    for word in 0..state.multisample.sample_count.div_ceil(32) {
                        let mask = if word < 2 {
                            (state.multisample.sample_mask >> (word * 32)) as u32
                        } else {
                            u32::MAX
                        };
                        sample_mask(word, mask);
                    }
                    self.gl.enable(glow::SAMPLE_MASK);
                }
            } else if state.multisample.sample_mask != u64::MAX {
                return Err(GlError::Unsupported {
                    operation: op,
                    reason: "sample mask requires a native GL/GLES sample-mask entry point",
                });
            }
            // GLES keeps multisample rasterization enabled for multisampled
            // targets and does not define the desktop GL_MULTISAMPLE toggle.
            if !matches!(
                self.discovery.context().profile(),
                GlFamilyProfile::Embedded { .. }
            ) {
                self.gl.enable(glow::MULTISAMPLE);
            }
            if state.multisample.alpha_to_coverage_enabled {
                self.gl.enable(glow::SAMPLE_ALPHA_TO_COVERAGE);
            } else {
                self.gl.disable(glow::SAMPLE_ALPHA_TO_COVERAGE);
            }
        }
        Ok(())
    }
}
