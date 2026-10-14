//! Immediate WebGL2 raster-pass and fixed-pipeline state.
//!
//! The browser encoder owns temporary FBO objects through a pass. Every method
//! here mutates the owner context before returning; it stores no command stream.

use fluxel_rhi_core::api::{ClearColor, Error, Result};
use web_sys::WebGl2RenderingContext as Gl;

use super::super::format_map;
use super::super::owner::{
    BrowserEncoder, BrowserOffscreenPass, BrowserRasterProgram, BrowserSurfaceDepthPass,
    BrowserTexture, BrowserVertexArray,
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
    /// Applies the depth subset required by the current GL raster baseline.
    /// Stencil remains rejected by pipeline lowering; WebGL state is changed
    /// here, during immediate pipeline selection, rather than inferred at draw.
    pub(crate) fn set_depth_state(
        &mut self,
        state: Option<crate::backend::gl::api::GlDepthStencilState>,
    ) -> Result<()> {
        self.require_raster()?;
        self.with_context("set_depth_state", |raw| match state {
            None => {
                raw.disable(Gl::DEPTH_TEST);
                raw.disable(Gl::STENCIL_TEST);
            }
            Some(state) => {
                raw.enable(Gl::DEPTH_TEST);
                raw.depth_mask(state.depth_write_enabled);
                raw.depth_func(match state.depth_compare {
                    crate::backend::gl::api::GlDepthCompareFunction::Never => Gl::NEVER,
                    crate::backend::gl::api::GlDepthCompareFunction::Less => Gl::LESS,
                    crate::backend::gl::api::GlDepthCompareFunction::Equal => Gl::EQUAL,
                    crate::backend::gl::api::GlDepthCompareFunction::LessEqual => Gl::LEQUAL,
                    crate::backend::gl::api::GlDepthCompareFunction::Greater => Gl::GREATER,
                    crate::backend::gl::api::GlDepthCompareFunction::NotEqual => Gl::NOTEQUAL,
                    crate::backend::gl::api::GlDepthCompareFunction::GreaterEqual => Gl::GEQUAL,
                    crate::backend::gl::api::GlDepthCompareFunction::Always => Gl::ALWAYS,
                });
                // Pipeline creation admits only the disabled stencil subset;
                // the temporary Depth32 attachment has no stencil plane.
                raw.disable(Gl::STENCIL_TEST);
            }
        })
    }
    /// Starts a surface pass through an owned FBO when the Core pass supplies
    /// a Depth32 texture. Framebuffer zero cannot attach that texture, so the
    /// temporary color image is blitted back during `end_surface_depth_raster`.
    pub(crate) fn begin_surface_depth_raster(
        &mut self,
        extent: [u32; 2],
        color_clear: ClearColor,
        depth_clear: Option<f32>,
        depth: &BrowserTexture,
    ) -> Result<()> {
        self.same_owner(&depth.owner)?;
        if extent.contains(&0)
            || !matches!(
                depth.desc.format,
                crate::backend::gl::api::GlFormat::Depth32Float
            )
            || depth.desc.extent.width != extent[0]
            || depth.desc.extent.height != extent[1]
            || depth.desc.extent.depth_or_layers != 1
            || depth.desc.mip_level_count != 1
            || depth.desc.sample_count != 1
        {
            return Err(Error::InvalidRange);
        }
        let internal = format_map::internal_format(crate::backend::gl::api::GlFormat::Rgba8Unorm)
            .ok_or(Error::Unsupported("WebGL2 intermediate color format"))?;
        let (framebuffer, color) = {
            let mut state = self.owner.state.borrow_mut();
            state.ensure_encoder(self.epoch)?;
            state.ensure_active()?;
            if state.raster_extent.is_some() || self.surface_depth_pass.is_some() {
                return Err(Error::InvalidState);
            }
            let raw = &state.context.raw;
            let framebuffer = raw.create_framebuffer().ok_or(Error::OutOfMemory)?;
            let Some(color) = raw.create_texture() else {
                raw.delete_framebuffer(Some(&framebuffer));
                return Err(Error::OutOfMemory);
            };
            raw.bind_texture(Gl::TEXTURE_2D, Some(&color));
            raw.tex_storage_2d(
                Gl::TEXTURE_2D,
                1,
                internal,
                extent[0] as i32,
                extent[1] as i32,
            );
            raw.bind_framebuffer(Gl::FRAMEBUFFER, Some(&framebuffer));
            raw.framebuffer_texture_2d(
                Gl::FRAMEBUFFER,
                Gl::COLOR_ATTACHMENT0,
                Gl::TEXTURE_2D,
                Some(&color),
                0,
            );
            raw.framebuffer_texture_2d(
                Gl::FRAMEBUFFER,
                Gl::DEPTH_ATTACHMENT,
                Gl::TEXTURE_2D,
                Some(&depth.raw),
                0,
            );
            if raw.check_framebuffer_status(Gl::FRAMEBUFFER) != Gl::FRAMEBUFFER_COMPLETE
                || raw.get_error() != Gl::NO_ERROR
            {
                raw.bind_framebuffer(Gl::FRAMEBUFFER, None);
                raw.delete_texture(Some(&color));
                raw.delete_framebuffer(Some(&framebuffer));
                return Err(Error::Backend(
                    "WebGL2 surface depth framebuffer is incomplete".into(),
                ));
            }
            (framebuffer, color)
        };
        let clear_depth = depth_clear;
        if let Err(error) = self.with_context("begin_surface_depth_raster", |raw| {
            raw.viewport(0, 0, extent[0] as i32, extent[1] as i32);
            // Native GL disables scissor before render-pass clears. WebGL
            // must do the same because clear is otherwise clipped by the
            // previous pass's immediate scissor rectangle.
            raw.disable(Gl::SCISSOR_TEST);
            // A previous pass may have disabled depth writes. Clearing the
            // attached Depth32 image must not inherit that mutable state.
            raw.depth_mask(true);
            raw.color_mask(true, true, true, true);
            match color_clear {
                ClearColor::Float(value) => raw.clear_color(value[0], value[1], value[2], value[3]),
                _ => {}
            }
            let mut mask = Gl::COLOR_BUFFER_BIT;
            if let Some(value) = clear_depth {
                raw.clear_depth(value);
                mask |= Gl::DEPTH_BUFFER_BIT;
            }
            raw.clear(mask);
        }) {
            // The FBO was fully owned by this encoder, so an immediate begin
            // failure still has a safe owner-thread cleanup path.
            let mut state = self.owner.state.borrow_mut();
            state.context.raw.bind_framebuffer(Gl::FRAMEBUFFER, None);
            state.context.raw.delete_texture(Some(&color));
            state.context.raw.delete_framebuffer(Some(&framebuffer));
            return Err(error);
        }
        self.owner.state.borrow_mut().raster_extent = Some(extent);
        self.surface_depth_pass = Some(BrowserSurfaceDepthPass {
            framebuffer,
            color,
            depth: depth.clone(),
            extent,
        });
        Ok(())
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
                crate::backend::gl::api::GlTextureDimension::D2
            )
            || color.desc.extent.width != extent[0]
            || color.desc.extent.height != extent[1]
            || color.desc.extent.depth_or_layers != 1
            || color.desc.mip_level_count != 1
            || color.desc.sample_count != 1
            || !color
                .desc
                .usage
                .contains(crate::backend::gl::api::GlTextureUsage::RENDER_ATTACHMENT)
            || !matches!(
                color.desc.format,
                crate::backend::gl::api::GlFormat::Rgba8Unorm
            )
        {
            return Err(Error::InvalidRange);
        }
        if let Some(depth) = depth {
            if !matches!(
                depth.desc.format,
                crate::backend::gl::api::GlFormat::Depth32Float
            ) || !matches!(
                depth.desc.dimension,
                crate::backend::gl::api::GlTextureDimension::D2
            ) || depth.desc.extent.width != extent[0]
                || depth.desc.extent.height != extent[1]
                || depth.desc.extent.depth_or_layers != 1
                || depth.desc.mip_level_count != 1
                || depth.desc.sample_count != 1
                || !depth
                    .desc
                    .usage
                    .contains(crate::backend::gl::api::GlTextureUsage::RENDER_ATTACHMENT)
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
            let framebuffer = raw.create_framebuffer().ok_or(Error::OutOfMemory)?;
            raw.bind_framebuffer(Gl::FRAMEBUFFER, Some(&framebuffer));
            raw.framebuffer_texture_2d(
                Gl::FRAMEBUFFER,
                Gl::COLOR_ATTACHMENT0,
                Gl::TEXTURE_2D,
                Some(&color.raw),
                0,
            );
            if let Some(depth) = depth {
                raw.framebuffer_texture_2d(
                    Gl::FRAMEBUFFER,
                    Gl::DEPTH_ATTACHMENT,
                    Gl::TEXTURE_2D,
                    Some(&depth.raw),
                    0,
                );
            }
            if raw.check_framebuffer_status(Gl::FRAMEBUFFER) != Gl::FRAMEBUFFER_COMPLETE
                || raw.get_error() != Gl::NO_ERROR
            {
                raw.bind_framebuffer(Gl::FRAMEBUFFER, None);
                raw.delete_framebuffer(Some(&framebuffer));
                return Err(Error::Backend(
                    "WebGL2 offscreen framebuffer is incomplete".into(),
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
        self.owner.state.borrow_mut().raster_extent = Some(extent);
        self.offscreen_pass = Some(BrowserOffscreenPass {
            framebuffer,
            color: color.clone(),
            depth: depth.cloned(),
            extent,
        });
        Ok(())
    }

    /// Resolves the immediate temporary FBO into the acquired default target
    /// and tears down only objects created by `begin_surface_depth_raster`.
    pub(crate) fn end_surface_depth_raster(&mut self) -> Result<()> {
        let pass = self.surface_depth_pass.take().ok_or(Error::InvalidState)?;
        let result = (|| {
            self.with_context("bind_surface_depth_read_framebuffer", |raw| {
                raw.bind_framebuffer(Gl::READ_FRAMEBUFFER, Some(&pass.framebuffer));
            })?;
            self.with_context("bind_surface_depth_draw_framebuffer", |raw| {
                raw.bind_framebuffer(Gl::DRAW_FRAMEBUFFER, None);
            })?;
            self.with_context("blit_surface_depth_color", |raw| {
                raw.blit_framebuffer(
                    0,
                    0,
                    pass.extent[0] as i32,
                    pass.extent[1] as i32,
                    0,
                    0,
                    pass.extent[0] as i32,
                    pass.extent[1] as i32,
                    Gl::COLOR_BUFFER_BIT,
                    Gl::NEAREST,
                )
            })?;
            self.with_context("release_surface_depth_framebuffer", |raw| {
                raw.bind_framebuffer(Gl::FRAMEBUFFER, None);
                raw.delete_texture(Some(&pass.color));
                raw.delete_framebuffer(Some(&pass.framebuffer));
            })
        })();
        // `with_context` can reject before it invokes its closure, for example
        // after context loss. Repeating WebGL deletion after an issued close is
        // harmless; leaving the only owned handles behind is not.
        if result.is_err() {
            self.destroy_surface_depth_pass(pass);
        } else {
            self.owner.state.borrow_mut().raster_extent = None;
        }
        result
    }

    /// Drops a failed surface-depth pass without copying its partial color
    /// image into the acquired default framebuffer.
    pub(crate) fn abort_surface_depth_raster(&mut self) {
        let Some(pass) = self.surface_depth_pass.take() else {
            return;
        };
        self.destroy_surface_depth_pass(pass);
    }

    /// Ends an offscreen pass and releases only the FBO that this encoder
    /// allocated. The color/depth allocations remain caller-owned and are not
    /// invalidated merely because this pass ended.
    pub(crate) fn end_offscreen_raster(&mut self) -> Result<()> {
        let pass = self.offscreen_pass.take().ok_or(Error::InvalidState)?;
        let result = self.with_context("release_offscreen_framebuffer", |raw| {
            raw.bind_framebuffer(Gl::FRAMEBUFFER, None);
            raw.delete_framebuffer(Some(&pass.framebuffer));
        });
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

    fn destroy_surface_depth_pass(&mut self, pass: BrowserSurfaceDepthPass) {
        // It is called only after a synchronous owner call returned; no owner
        // borrow can legitimately remain live here. A regular borrow makes a
        // cleanup failure visible instead of silently dropping raw handles.
        let mut state = self.owner.state.borrow_mut();
        state.context.raw.bind_framebuffer(Gl::FRAMEBUFFER, None);
        state.context.raw.delete_texture(Some(&pass.color));
        state
            .context
            .raw
            .delete_framebuffer(Some(&pass.framebuffer));
        state.raster_extent = None;
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
        state.raster_extent = None;
    }

    /// Applies static WebGL2 raster state on pipeline selection. This remains
    /// immediate context state; no draw payload is cached by the encoder.
    pub(crate) fn set_fixed_raster_state(
        &mut self,
        state: &crate::backend::gl::api::GlRasterState,
    ) -> Result<()> {
        use crate::backend::gl::api::{GlBlendFactor, GlBlendOperation, GlCullMode, GlFrontFace};
        self.require_raster()?;
        let target = state.color_targets.first().ok_or(Error::InvalidState)?;
        if state.color_targets.len() != 1 || state.multisample.sample_mask != u32::MAX {
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
        };
        let operation = |value| match value {
            GlBlendOperation::Add => Gl::FUNC_ADD,
            GlBlendOperation::Subtract => Gl::FUNC_SUBTRACT,
            GlBlendOperation::ReverseSubtract => Gl::FUNC_REVERSE_SUBTRACT,
            GlBlendOperation::Min => Gl::MIN,
            GlBlendOperation::Max => Gl::MAX,
        };
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
            if state.multisample.alpha_to_coverage_enabled {
                raw.enable(Gl::SAMPLE_ALPHA_TO_COVERAGE);
            } else {
                raw.disable(Gl::SAMPLE_ALPHA_TO_COVERAGE);
            }
        })
    }
}
