//! Native fixed-raster and draw execution.
//!
//! The core GL 4.x/GLES 3.x families blend and mask color globally like
//! WebGL2: independent per-attachment state needs `EXT_draw_buffers_indexed`,
//! so mismatched per-target declarations reject with a structured error
//! instead of being silently collapsed to global state.

use super::exec_vertex::{indexed_draw_offset, indexed_draw_span, indexed_draw_type};
use super::provider::{ActiveRaster, NativeGlProvider};
use crate::backend::gl::api::{
    GlCullMode, GlDepthStencilState, GlDrawCommand, GlError, GlFamilyApi as _, GlFrontFace,
    GlMultiDraw, GlMultiDrawApi, GlRasterCommandApi, GlRasterPipeline, GlRasterState,
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

impl GlRasterCommandApi for NativeGlProvider {
    fn set_raster_pipeline(&mut self, pipeline: &GlRasterPipeline) -> Result<(), GlError> {
        const OP: &str = "set-raster-pipeline";
        let packet = self.raster_state_packet(pipeline)?;
        self.context_state
            .register_pipeline(packet)
            .map_err(|error| {
                Self::validation(
                    OP,
                    match error {
                        crate::backend::gl::state::ContextStateError::UnknownPipeline => {
                            "native raster pipeline was not registered"
                        }
                        crate::backend::gl::state::ContextStateError::PipelineIdentityConflict => {
                            "native raster pipeline identity conflicts with fixed state"
                        }
                    },
                )
            })?;
        let diff = self
            .context_state
            .prepare_pipeline(packet)
            .map_err(|error| {
                Self::validation(
                    OP,
                    match error {
                        crate::backend::gl::state::ContextStateError::UnknownPipeline => {
                            "native raster pipeline was not registered"
                        }
                        crate::backend::gl::state::ContextStateError::PipelineIdentityConflict => {
                            "native raster pipeline identity conflicts with fixed state"
                        }
                    },
                )
            })?;
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
        });
        Ok(())
    }

    fn draw_raster(&mut self, draw: GlDrawCommand) -> Result<(), GlError> {
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
                        .and_then(|value| i32::try_from(value).ok())
                        .ok_or_else(|| Self::validation(OP, "index offset exceeds i32"))?;
                    let count = i32::try_from(draw.index_count)
                        .map_err(|_| Self::validation(OP, "index count exceeds i32"))?;
                    let instances = i32::try_from(draw.instance_count)
                        .map_err(|_| Self::validation(OP, "instance count exceeds i32"))?;
                    if instances == 1 {
                        self.gl
                            .draw_elements(mode, count, indexed_draw_type(index), offset);
                    } else {
                        self.gl.draw_elements_instanced(
                            mode,
                            count,
                            indexed_draw_type(index),
                            offset,
                            instances,
                        );
                    }
                }
            }
        }
        self.driver_error(OP)
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
            // `gl_FrontFacing` and fixed-function culling use this same native
            // selection. The Core front-face declaration is therefore passed
            // through unchanged; viewport-origin conversion only changes the
            // rectangle origin, not the vertex winding in GL window space.
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
                    self.gl.disable(glow::POLYGON_OFFSET_FILL);
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
            // The shared semantic exposes no polygon-offset clamp; desktop
            // expressiveness would require EXT_polygon_offset_clamp, which is
            // not a typed route, so a nonzero clamp rejects everywhere.
            let (factor, units, clamp) = (
                f32::from_bits(state.depth_bias[0]),
                f32::from_bits(state.depth_bias[1]),
                f32::from_bits(state.depth_bias[2]),
            );
            if clamp != 0.0 {
                return Err(Self::validation(
                    op,
                    "depth-bias clamp has no typed expression in this contract",
                ));
            }
            if factor == 0.0 && units == 0.0 {
                self.gl.disable(glow::POLYGON_OFFSET_FILL);
            } else {
                self.gl.enable(glow::POLYGON_OFFSET_FILL);
                self.gl.polygon_offset(factor, units);
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
                return Err(Self::validation(
                    op,
                    "raster state declares no color target for the active pass",
                ));
            };
            // Independent per-attachment blend/write-mask is not expressible
            // in the common GL core semantic; refuse to collapse differing
            // declarations silently (EXT_draw_buffers_indexed is not a typed
            // route in this contract).
            if state
                .color_targets
                .iter()
                .any(|target| target.write_mask != first.write_mask || target.blend != first.blend)
            {
                return Err(Self::validation(
                    op,
                    "independent per-attachment blend/write-mask is unavailable in the common semantic",
                ));
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
        // SAFETY: forwarded current-context contract; the capability checks
        // below reject the sample-mask operation glow cannot express.
        unsafe {
            // Multisampling is framebuffer state; only coverage knobs live in
            // this canonical leaf. glow has no typed `SampleMaski` route.
            if state.multisample.sample_mask != u32::MAX {
                return Err(Self::validation(
                    op,
                    "explicit sample masks have no typed command in this provider",
                ));
            }
            self.gl.enable(glow::MULTISAMPLE);
            if state.multisample.alpha_to_coverage_enabled {
                self.gl.enable(glow::SAMPLE_ALPHA_TO_COVERAGE);
            } else {
                self.gl.disable(glow::SAMPLE_ALPHA_TO_COVERAGE);
            }
        }
        Ok(())
    }
}
