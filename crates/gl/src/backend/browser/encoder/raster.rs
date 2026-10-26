//! Immediate WebGL2 raster-pass and fixed-pipeline state.
//!
//! The browser encoder owns temporary FBO objects through a pass. Every method
//! here mutates the owner context before returning; it stores no command stream.

use fluxel_rhi_core::api::{ClearColor, Error, Result};
use web_sys::WebGl2RenderingContext as Gl;

use super::super::format_map;
use super::super::owner::{
    BrowserEncoder, BrowserOffscreenPass, BrowserRasterProgram, BrowserTexture, BrowserVertexArray,
};

impl BrowserEncoder {
    /// Returns whether this encoder currently owns an FBO-backed raster pass.
    ///
    /// The two variants have different publication rules: a surface/depth pass
    /// blits its temporary color into framebuffer zero, while an offscreen pass
    /// leaves its caller-owned texture in place and only retires the FBO.
    pub(crate) const fn has_owned_framebuffer_pass(&self) -> bool {
        self.surface_depth_pass.is_some() || self.offscreen_pass.is_some()
    }
    pub(crate) const fn has_surface_depth_pass(&self) -> bool {
        self.surface_depth_pass.is_some()
    }
    /// Installs depth, stencil and polygon offset state immediately.
    pub(crate) fn set_depth_state(
        &mut self,
        state: Option<crate::backend::api::GlDepthStencilState>,
    ) -> Result<()> {
        self.require_raster()?;
        let mut clamp_command = None;
        if state.is_some_and(|s| f32::from_bits(s.depth_bias[2]) != 0.0) {
            self.with_context("acquire_polygon_offset_clamp", |raw| {
                clamp_command = super::super::polygon_offset::acquire(raw);
            })?;
            if clamp_command.is_none() {
                return Err(Error::Unsupported("WebGL2 polygon offset clamp"));
            }
        }
        self.stencil_state = state;
        let mut command_failed = false;
        self.with_context("set_depth_state", |raw| match state {
            None => {
                raw.disable(Gl::DEPTH_TEST);
                raw.disable(Gl::STENCIL_TEST);
                raw.disable(Gl::POLYGON_OFFSET_FILL);
                super::super::rasterization::line_offset(raw, false);
            }
            Some(state) => {
                raw.enable(Gl::DEPTH_TEST);
                raw.depth_mask(state.depth_write_enabled);
                raw.depth_func(compare(state.depth_compare));
                let factor = f32::from_bits(state.depth_bias[0]);
                let units = f32::from_bits(state.depth_bias[1]);
                if factor == 0.0 && units == 0.0 {
                    raw.disable(Gl::POLYGON_OFFSET_FILL);
                    super::super::rasterization::line_offset(raw, false);
                } else {
                    raw.enable(Gl::POLYGON_OFFSET_FILL);
                    super::super::rasterization::line_offset(raw, true);
                    command_failed = super::super::polygon_offset::apply(
                        raw,
                        clamp_command.as_ref(),
                        factor,
                        units,
                        f32::from_bits(state.depth_bias[2]),
                    )
                    .is_err();
                }
                raw.enable(Gl::STENCIL_TEST);
                stencil_face(raw, Gl::FRONT, state.stencil_front, state.stencil_reference);
                stencil_face(raw, Gl::BACK, state.stencil_back, state.stencil_reference);
            }
        })?;
        if command_failed {
            return Err(Error::Backend(
                "WebGL polygon offset clamp call failed".into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn set_blend_constant(&mut self, color: [f32; 4]) -> Result<()> {
        self.require_raster()?;
        self.with_context("set_blend_constant", |raw| {
            raw.blend_color(color[0], color[1], color[2], color[3])
        })
    }

    pub(crate) fn set_stencil_reference(&mut self, reference: u32) -> Result<()> {
        self.require_raster()?;
        let mut state = self.stencil_state.ok_or(Error::InvalidState)?;
        state.stencil_reference = reference;
        self.stencil_state = Some(state);
        self.with_context("set_stencil_reference", |raw| {
            stencil_face(raw, Gl::FRONT, state.stencil_front, reference);
            stencil_face(raw, Gl::BACK, state.stencil_back, reference);
        })
    }
    /// Begins a single-sample pass directly against a caller-owned color
    /// texture, with an optional caller-owned Depth32 texture.
    ///
    /// WebGL needs an FBO name to attach such textures, but there is no
    /// deferred render-pass object: the FBO is bound and cleared before this
    /// method returns.  The pass record retains only the wrappers required to
    /// make the borrow valid until `end_offscreen_raster` or abort cleanup.
    pub(crate) fn begin_offscreen_raster(
        &mut self,
        color: &BrowserTexture,
        depth: Option<&BrowserTexture>,
        extent: [u32; 2],
        color_clear: Option<ClearColor>,
        depth_clear: Option<f32>,
    ) -> Result<()> {
        self.same_owner(&color.owner)?;
        if let Some(depth) = depth {
            self.same_owner(&depth.owner)?;
        }
        if extent.contains(&0)
            || !matches!(
                color.desc.dimension,
                crate::backend::api::GlTextureDimension::D2
            )
            || color.desc.extent.width != extent[0]
            || color.desc.extent.height != extent[1]
            || color.desc.extent.depth_or_layers != 1
            || color.desc.mip_level_count != 1
            || color.desc.sample_count != 1
            || !color
                .desc
                .usage
                .contains(crate::backend::api::GlTextureUsage::RENDER_ATTACHMENT)
            || !matches!(color.desc.format, crate::backend::api::GlFormat::Rgba8Unorm)
        {
            return Err(Error::InvalidRange);
        }
        if let Some(depth) = depth {
            if !matches!(
                depth.desc.format,
                crate::backend::api::GlFormat::Depth32Float
            ) || !matches!(
                depth.desc.dimension,
                crate::backend::api::GlTextureDimension::D2
            ) || depth.desc.extent.width != extent[0]
                || depth.desc.extent.height != extent[1]
                || depth.desc.extent.depth_or_layers != 1
                || depth.desc.mip_level_count != 1
                || depth.desc.sample_count != 1
                || !depth
                    .desc
                    .usage
                    .contains(crate::backend::api::GlTextureUsage::RENDER_ATTACHMENT)
            {
                return Err(Error::InvalidRange);
            }
        }
        let framebuffer = {
            let mut state = self.owner.state.borrow_mut();
            state.ensure_encoder(self.epoch)?;
            state.ensure_active()?;
            if state.raster_extent.is_some()
                || self.surface_depth_pass.is_some()
                || self.offscreen_pass.is_some()
            {
                return Err(Error::InvalidState);
            }
            let raw = &state.context.raw;
            let Some(framebuffer) = raw.create_framebuffer() else {
                return Err(Self::allocation_error(&mut state, Error::OutOfMemory));
            };
            raw.bind_framebuffer(Gl::FRAMEBUFFER, Some(&framebuffer));
            raw.framebuffer_texture_2d(
                Gl::FRAMEBUFFER,
                Gl::COLOR_ATTACHMENT0,
                Gl::TEXTURE_2D,
                color.raw.as_ref(),
                0,
            );
            if let Some(depth) = depth {
                raw.framebuffer_texture_2d(
                    Gl::FRAMEBUFFER,
                    Gl::DEPTH_ATTACHMENT,
                    Gl::TEXTURE_2D,
                    depth.raw.as_ref(),
                    0,
                );
            }
            if raw.check_framebuffer_status(Gl::FRAMEBUFFER) != Gl::FRAMEBUFFER_COMPLETE {
                raw.bind_framebuffer(Gl::FRAMEBUFFER, None);
                raw.delete_framebuffer(Some(&framebuffer));
                return Err(Self::allocation_error(
                    &mut state,
                    Error::Backend("WebGL2 offscreen framebuffer is incomplete".into()),
                ));
            }
            framebuffer
        };
        if let Err(error) = self.with_context("begin_offscreen_raster", |raw| {
            raw.viewport(0, 0, extent[0] as i32, extent[1] as i32);
            // Clear the full new pass extent instead of inheriting a prior
            // immediate pass's scissor and color write mask.
            raw.disable(Gl::SCISSOR_TEST);
            raw.color_mask(true, true, true, true);
            if let Some(ClearColor::Float(value)) = color_clear {
                raw.clear_color(value[0], value[1], value[2], value[3]);
            }
            let mut mask = if matches!(color_clear, Some(ClearColor::Float(_))) {
                Gl::COLOR_BUFFER_BIT
            } else {
                0
            };
            if let Some(value) = depth_clear {
                raw.depth_mask(true);
                raw.clear_depth(value);
                mask |= Gl::DEPTH_BUFFER_BIT;
            }
            if mask != 0 {
                raw.clear(mask);
            }
        }) {
            let mut state = self.owner.state.borrow_mut();
            state.context.raw.bind_framebuffer(Gl::FRAMEBUFFER, None);
            state.context.raw.delete_framebuffer(Some(&framebuffer));
            return Err(error);
        }
        self.raster_scissor = None;
        self.owner.state.borrow_mut().raster_extent = Some(extent);
        self.offscreen_pass = Some(BrowserOffscreenPass {
            framebuffer,
            colors: vec![color.clone()],
            depth: depth.cloned(),
            extent,
            discard: Vec::new(),
            resolves: Vec::new(),
            surface_color: None,
            surface_slot: None,
            surface_srgb: false,
            render_area: fluxel_rhi_core::api::Rect {
                x: 0,
                y: 0,
                width: extent[0],
                height: extent[1],
            },
        });
        Ok(())
    }

    /// Ends an offscreen pass and releases only the FBO that this encoder
    /// allocated. The color/depth allocations remain caller-owned and are not
    /// invalidated merely because this pass ended.
    pub(crate) fn end_offscreen_raster(&mut self) -> Result<()> {
        let pass = self.offscreen_pass.take().ok_or(Error::InvalidState)?;
        let result = (|| {
            for (slot, target, mip, layer, format) in &pass.resolves {
                let (scratch, intermediate) = {
                    let mut state = self.owner.state.borrow_mut();
                    let raw = &state.context.raw;
                    let Some(scratch) = raw.create_framebuffer() else {
                        return Err(Self::allocation_error(&mut state, Error::OutOfMemory));
                    };
                    let intermediate = if target.is_none() {
                        match raw.create_texture() {
                            Some(texture) => Some(texture),
                            None => {
                                raw.delete_framebuffer(Some(&scratch));
                                return Err(Self::allocation_error(&mut state, Error::OutOfMemory));
                            }
                        }
                    } else {
                        None
                    };
                    (scratch, intermediate)
                };
                let mut conversion = Ok(());
                let result =
                    self.with_surface_color_context("resolve_offscreen_color", |raw, cache| {
                        raw.bind_framebuffer(Gl::FRAMEBUFFER, Some(&scratch));
                        if let Some(target) = target {
                            super::attachments::attach(
                                raw,
                                Gl::COLOR_ATTACHMENT0,
                                target,
                                *mip,
                                *layer,
                            );
                        } else if let Some(texture) = &intermediate {
                            raw.bind_texture(Gl::TEXTURE_2D, Some(texture));
                            let internal = format_map::internal_format(*format)
                                .expect("admitted source format");
                            raw.tex_storage_2d(
                                Gl::TEXTURE_2D,
                                1,
                                internal,
                                pass.extent[0] as i32,
                                pass.extent[1] as i32,
                            );
                            raw.framebuffer_texture_2d(
                                Gl::FRAMEBUFFER,
                                Gl::COLOR_ATTACHMENT0,
                                Gl::TEXTURE_2D,
                                Some(texture),
                                0,
                            );
                        }
                        raw.bind_framebuffer(Gl::READ_FRAMEBUFFER, Some(&pass.framebuffer));
                        raw.read_buffer(Gl::COLOR_ATTACHMENT0 + *slot);
                        raw.bind_framebuffer(Gl::DRAW_FRAMEBUFFER, Some(&scratch));
                        raw.disable(Gl::SCISSOR_TEST);
                        let r = pass.render_area;
                        let x0 = r.x as i32;
                        let x1 = (r.x + r.width) as i32;
                        let y0 = r.y as i32;
                        let y1 = (r.y + r.height) as i32;
                        raw.blit_framebuffer(
                            x0,
                            y0,
                            x1,
                            y1,
                            x0,
                            y0,
                            x1,
                            y1,
                            Gl::COLOR_BUFFER_BIT,
                            Gl::NEAREST,
                        );
                        if intermediate.is_some() {
                            // Default framebuffer storage can have a different native
                            // component count; first resolve into an exact format.
                            raw.bind_framebuffer(Gl::READ_FRAMEBUFFER, Some(&scratch));
                            raw.read_buffer(Gl::COLOR_ATTACHMENT0);
                            raw.bind_framebuffer(Gl::DRAW_FRAMEBUFFER, None);
                            if *format == crate::backend::api::GlFormat::Rgba8Srgb {
                                conversion = super::surface_color::convert(
                                    raw,
                                    cache,
                                    intermediate.as_ref().expect("surface resolve texture"),
                                    None,
                                    0,
                                    pass.extent,
                                    r,
                                    true,
                                );
                            } else {
                                raw.blit_framebuffer(
                                    x0,
                                    y0,
                                    x1,
                                    y1,
                                    x0,
                                    pass.extent[1] as i32 - y0,
                                    x1,
                                    pass.extent[1] as i32 - y1,
                                    Gl::COLOR_BUFFER_BIT,
                                    Gl::NEAREST,
                                );
                            }
                        }
                        if let Some(texture) = &intermediate {
                            raw.delete_texture(Some(texture));
                        }
                        raw.delete_framebuffer(Some(&scratch));
                        raw.bind_framebuffer(Gl::FRAMEBUFFER, Some(&pass.framebuffer));
                    });
                let result = result.and(conversion);
                if result.is_err() {
                    let state = self.owner.state.borrow();
                    if let Some(texture) = &intermediate {
                        state.context.raw.delete_texture(Some(texture));
                    }
                    state.context.raw.delete_framebuffer(Some(&scratch));
                }
                result?;
            }
            if let Some(slot) = pass.surface_slot {
                let mut conversion = Ok(());
                self.with_surface_color_context("publish_offscreen_surface", |raw, cache| {
                    raw.disable(Gl::SCISSOR_TEST);
                    if pass.surface_srgb {
                        let full = fluxel_rhi_core::api::Rect {
                            x: 0,
                            y: 0,
                            width: pass.extent[0],
                            height: pass.extent[1],
                        };
                        conversion = super::surface_color::convert(
                            raw,
                            cache,
                            pass.surface_color.as_ref().expect("surface texture"),
                            None,
                            0,
                            pass.extent,
                            full,
                            true,
                        );
                    } else {
                        raw.bind_framebuffer(Gl::READ_FRAMEBUFFER, Some(&pass.framebuffer));
                        raw.read_buffer(Gl::COLOR_ATTACHMENT0 + slot);
                        raw.bind_framebuffer(Gl::DRAW_FRAMEBUFFER, None);
                        raw.blit_framebuffer(
                            0,
                            pass.extent[1] as i32,
                            pass.extent[0] as i32,
                            0,
                            0,
                            0,
                            pass.extent[0] as i32,
                            pass.extent[1] as i32,
                            Gl::COLOR_BUFFER_BIT,
                            Gl::NEAREST,
                        );
                    }
                    raw.bind_framebuffer(Gl::FRAMEBUFFER, Some(&pass.framebuffer));
                })?;
                conversion?;
            }
            self.with_context("release_offscreen_framebuffer", |raw| {
                if !pass.discard.is_empty() {
                    let attachments = js_sys::Array::from_iter(
                        pass.discard
                            .iter()
                            .map(|v| wasm_bindgen::JsValue::from_f64(f64::from(*v))),
                    );
                    let _ = raw.invalidate_framebuffer(Gl::FRAMEBUFFER, &attachments);
                }
                raw.bind_framebuffer(Gl::FRAMEBUFFER, None);
                raw.delete_framebuffer(Some(&pass.framebuffer));
                if let Some(color) = &pass.surface_color {
                    raw.delete_texture(Some(color));
                }
            })
        })();
        if result.is_ok() {
            self.owner.state.borrow_mut().raster_extent = None;
        } else {
            self.destroy_offscreen_pass(pass);
        }
        result
    }

    /// Aborts an offscreen pass without changing the caller-owned attachments.
    pub(crate) fn abort_offscreen_raster(&mut self) {
        let Some(pass) = self.offscreen_pass.take() else {
            return;
        };
        self.destroy_offscreen_pass(pass);
    }

    fn destroy_offscreen_pass(&mut self, pass: BrowserOffscreenPass) {
        // Do not delete `color` or `depth`: they are Core resources retained
        // only for the active-pass lifetime.  The framebuffer is the only
        // object this encoder created and therefore the only object it owns.
        let mut state = self.owner.state.borrow_mut();
        state.context.raw.bind_framebuffer(Gl::FRAMEBUFFER, None);
        state
            .context
            .raw
            .delete_framebuffer(Some(&pass.framebuffer));
        if let Some(color) = &pass.surface_color {
            state.context.raw.delete_texture(Some(color));
        }
        state.raster_extent = None;
    }

    /// Applies static WebGL2 raster state on pipeline selection. This remains
    /// immediate context state; no draw payload is cached by the encoder.
    pub(crate) fn set_fixed_raster_state(
        &mut self,
        state: &crate::backend::api::GlRasterState,
    ) -> Result<()> {
        use crate::backend::api::{GlBlendFactor, GlBlendOperation, GlCullMode, GlFrontFace};
        self.require_raster()?;
        let mut rasterization = Ok(());
        self.with_context("set_rasterization", |raw| {
            rasterization = super::super::rasterization::apply(raw, state.rasterization);
        })?;
        rasterization.map_err(Error::Unsupported)?;
        let target = state.color_targets.first();
        if state.multisample.sample_mask != u64::MAX {
            return Err(Error::Unsupported("WebGL2 fixed raster state"));
        }
        let factor = |value| match value {
            GlBlendFactor::Zero => Gl::ZERO,
            GlBlendFactor::One => Gl::ONE,
            GlBlendFactor::Src => Gl::SRC_COLOR,
            GlBlendFactor::OneMinusSrc => Gl::ONE_MINUS_SRC_COLOR,
            GlBlendFactor::SrcAlpha => Gl::SRC_ALPHA,
            GlBlendFactor::OneMinusSrcAlpha => Gl::ONE_MINUS_SRC_ALPHA,
            GlBlendFactor::Dst => Gl::DST_COLOR,
            GlBlendFactor::OneMinusDst => Gl::ONE_MINUS_DST_COLOR,
            GlBlendFactor::DstAlpha => Gl::DST_ALPHA,
            GlBlendFactor::OneMinusDstAlpha => Gl::ONE_MINUS_DST_ALPHA,
            GlBlendFactor::Constant => Gl::CONSTANT_COLOR,
            GlBlendFactor::OneMinusConstant => Gl::ONE_MINUS_CONSTANT_COLOR,
            GlBlendFactor::SrcAlphaSaturated => Gl::SRC_ALPHA_SATURATE,
            GlBlendFactor::Src1 => 0x88F9,
            GlBlendFactor::OneMinusSrc1 => 0x88FA,
            GlBlendFactor::Src1Alpha => 0x8589,
            GlBlendFactor::OneMinusSrc1Alpha => 0x88FB,
        };
        let operation = |value| match value {
            GlBlendOperation::Add => Gl::FUNC_ADD,
            GlBlendOperation::Subtract => Gl::FUNC_SUBTRACT,
            GlBlendOperation::ReverseSubtract => Gl::FUNC_REVERSE_SUBTRACT,
            GlBlendOperation::Min => Gl::MIN,
            GlBlendOperation::Max => Gl::MAX,
        };
        let independent =
            target.is_some_and(|first| state.color_targets.iter().any(|t| t != first));
        if independent {
            let mut state_guard = self.owner.state.borrow_mut();
            state_guard.ensure_encoder(self.epoch)?;
            state_guard.ensure_active()?;
            // Extension methods issue GL calls immediately and can partially
            // mutate indexed state before a later call rejects.
            self.issued = true;
            super::super::indexed_blend::apply(&state_guard.context.raw, &state.color_targets)
                .map_err(Error::Unsupported)?;
        }
        self.with_context("set_fixed_raster_state", |raw| {
            match state.cull_mode {
                GlCullMode::None => raw.disable(Gl::CULL_FACE),
                GlCullMode::Front => {
                    raw.enable(Gl::CULL_FACE);
                    raw.cull_face(Gl::FRONT);
                }
                GlCullMode::Back => {
                    raw.enable(Gl::CULL_FACE);
                    raw.cull_face(Gl::BACK);
                }
            }
            raw.front_face(match state.front_face {
                GlFrontFace::Clockwise => Gl::CW,
                GlFrontFace::CounterClockwise => Gl::CCW,
            });
            if let Some(target) = target.filter(|_| !independent) {
                raw.color_mask(
                    target.write_mask & 1 != 0,
                    target.write_mask & 2 != 0,
                    target.write_mask & 4 != 0,
                    target.write_mask & 8 != 0,
                );
                match target.blend {
                    Some(blend) => {
                        raw.enable(Gl::BLEND);
                        raw.blend_func_separate(
                            factor(blend.color.src_factor),
                            factor(blend.color.dst_factor),
                            factor(blend.alpha.src_factor),
                            factor(blend.alpha.dst_factor),
                        );
                        raw.blend_equation_separate(
                            operation(blend.color.operation),
                            operation(blend.alpha.operation),
                        );
                        raw.blend_color(
                            f32::from_bits(state.blend_constant[0]),
                            f32::from_bits(state.blend_constant[1]),
                            f32::from_bits(state.blend_constant[2]),
                            f32::from_bits(state.blend_constant[3]),
                        );
                    }
                    None => raw.disable(Gl::BLEND),
                }
            }
            raw.blend_color(
                f32::from_bits(state.blend_constant[0]),
                f32::from_bits(state.blend_constant[1]),
                f32::from_bits(state.blend_constant[2]),
                f32::from_bits(state.blend_constant[3]),
            );
            if state.multisample.alpha_to_coverage_enabled {
                raw.enable(Gl::SAMPLE_ALPHA_TO_COVERAGE);
            } else {
                raw.disable(Gl::SAMPLE_ALPHA_TO_COVERAGE);
            }
        })
    }
}

fn compare(value: crate::backend::api::GlDepthCompareFunction) -> u32 {
    use crate::backend::api::GlDepthCompareFunction::*;
    match value {
        Never => Gl::NEVER,
        Less => Gl::LESS,
        Equal => Gl::EQUAL,
        LessEqual => Gl::LEQUAL,
        Greater => Gl::GREATER,
        NotEqual => Gl::NOTEQUAL,
        GreaterEqual => Gl::GEQUAL,
        Always => Gl::ALWAYS,
    }
}
fn stencil_face(
    raw: &Gl,
    face: u32,
    state: crate::backend::api::GlStencilFaceState,
    reference: u32,
) {
    use crate::backend::api::GlStencilOperation::*;
    let op = |v| match v {
        Keep => Gl::KEEP,
        Zero => Gl::ZERO,
        Replace => Gl::REPLACE,
        IncrementClamp => Gl::INCR,
        DecrementClamp => Gl::DECR,
        Invert => Gl::INVERT,
        IncrementWrap => Gl::INCR_WRAP,
        DecrementWrap => Gl::DECR_WRAP,
    };
    raw.stencil_func_separate(
        face,
        compare(state.compare),
        reference as i32,
        state.read_mask,
    );
    raw.stencil_mask_separate(face, state.write_mask);
    raw.stencil_op_separate(
        face,
        op(state.fail_op),
        op(state.depth_fail_op),
        op(state.pass_op),
    );
}
