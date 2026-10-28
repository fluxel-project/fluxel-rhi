//! Native framebuffer, render-pass, and blit execution.
//!
//! Pass load/clear uses `glClearBuffer*`, which honors scissor and color,
//! depth, and stencil write masks. A scoped driver workaround splits a logical
//! color clear into two disjoint write masks.
//! Pass begin disables scissor, enables writes for cleared attachments, and
//! invalidates cached pipeline state so the next pipeline restores its masks.
//! Resolve stays an
//! explicit `glBlitFramebuffer` from the multisample source, per the shared
//! framebuffer contract.

use super::provider::{NativeFramebuffer, NativeGlProvider};
use super::resource::{depth_attachment_point, has_stencil_plane};
use crate::backend::api::{
    FramebufferId, GlAttachmentTarget, GlBlitMask, GlBlitRegion, GlDepthStencilAttachment, GlError,
    GlFamilyApi as _, GlFilterMode, GlFormat, GlFramebufferApi, GlFramebufferDescriptor, GlLoadOp,
    GlPassAttachmentView, GlRenderPassDescriptor, GlRenderTarget, GlStoreOp, GlTextureDimension,
    GlTextureView,
};
use fluxel_rhi_core::api::{ClearColor, ColorAttachmentClear, Rect};

const fn blit_mask_bits(masks: GlBlitMask) -> u32 {
    let mut bits = 0;
    if masks.color {
        bits |= glow::COLOR_BUFFER_BIT;
    }
    if masks.depth {
        bits |= glow::DEPTH_BUFFER_BIT;
    }
    if masks.stencil {
        bits |= glow::STENCIL_BUFFER_BIT;
    }
    bits
}

const fn blit_filter(filter: GlFilterMode) -> u32 {
    match filter {
        GlFilterMode::Nearest => glow::NEAREST,
        GlFilterMode::Linear => glow::LINEAR,
    }
}

/// The attachment constant of one `glDrawBuffers` selection entry.
pub(super) const fn draw_buffer_constant(index: u32) -> u32 {
    glow::COLOR_ATTACHMENT0 + index
}

/// `(width, height, sample_count)` facts of a stored descriptor.
pub(super) fn framebuffer_shape(descriptor: &GlFramebufferDescriptor) -> (u32, u32, u32) {
    descriptor
        .color_attachments
        .iter()
        .flatten()
        .next()
        .copied()
        .or(descriptor.depth_stencil_attachment)
        .map(|view| (view.width, view.height, view.sample_count))
        .unwrap_or((0, 0, 1))
}

const fn has_depth_plane(format: GlFormat) -> bool {
    matches!(
        format,
        GlFormat::Depth16Unorm
            | GlFormat::Depth24Unorm
            | GlFormat::Depth24PlusStencil8
            | GlFormat::Depth32Float
            | GlFormat::Depth32FloatStencil8
    )
}

const fn color_clear_matches(format: GlFormat, value: ClearColor) -> bool {
    match format {
        GlFormat::R8Sint
        | GlFormat::Rg8Sint
        | GlFormat::Rgba8Sint
        | GlFormat::R16Sint
        | GlFormat::Rg16Sint
        | GlFormat::Rgba16Sint
        | GlFormat::R32Sint
        | GlFormat::Rg32Sint
        | GlFormat::Rgba32Sint => matches!(value, ClearColor::Sint(_)),
        format if format.is_integer_color() => matches!(value, ClearColor::Uint(_)),
        _ => matches!(value, ClearColor::Float(_)),
    }
}

fn clear_native_rect(rect: Rect) -> Result<(i32, i32, i32, i32), GlError> {
    Ok((
        i32::try_from(rect.x).map_err(|_| {
            NativeGlProvider::validation("clear-active-rect", "rectangle x exceeds GLint")
        })?,
        i32::try_from(rect.y).map_err(|_| {
            NativeGlProvider::validation("clear-active-rect", "rectangle y exceeds GLint")
        })?,
        i32::try_from(rect.width).map_err(|_| {
            NativeGlProvider::validation("clear-active-rect", "rectangle width exceeds GLsizei")
        })?,
        i32::try_from(rect.height).map_err(|_| {
            NativeGlProvider::validation("clear-active-rect", "rectangle height exceeds GLsizei")
        })?,
    ))
}

impl NativeGlProvider {
    pub(crate) fn create_multiview_framebuffer(
        &mut self,
        descriptor: &GlFramebufferDescriptor,
    ) -> Result<FramebufferId, GlError> {
        self.create_framebuffer_mode(descriptor, true)
    }

    fn create_framebuffer_mode(
        &mut self,
        descriptor: &GlFramebufferDescriptor,
        multiview: bool,
    ) -> Result<FramebufferId, GlError> {
        use glow::HasContext as _;
        const OP: &str = "create-framebuffer";
        self.assert_ready(OP)?;
        for view in descriptor
            .color_attachments
            .iter()
            .flatten()
            .copied()
            .chain(descriptor.depth_stencil_attachment)
        {
            self.validate_attachment(OP, view)?;
            if multiview {
                self.validate_multiview_attachment(OP, view)?;
            }
        }
        let limits = self.discovery.limits();
        descriptor
            .validate(
                limits.max_color_attachments,
                limits.max_draw_buffers,
                self.context_stamp(),
            )
            .map_err(|_| Self::validation(OP, "invalid framebuffer descriptor"))?;
        // A descriptor that asks for several views per attachment is refused
        // here, before the framebuffer object exists, unless this context
        // proved a view count that can serve it.
        descriptor
            .validate_multiview(self.discovery.max_multiview_view_count())
            .map_err(|_| Self::validation(OP, "multiview view count is not proved"))?;
        // SAFETY: current-context contract; the framebuffer is deleted on
        // every error path before this function returns.
        let raw = unsafe { self.gl.create_framebuffer() }.map_err(|message| GlError::Driver {
            operation: OP,
            message,
        })?;
        let attached = self.attach_all(OP, raw, descriptor, multiview);
        // Depth-only framebuffers must not keep a draw buffer selected.
        // SAFETY: `raw` was allocated by this current provider and is bound by
        // `attach_all` on this same context. Descriptor validation fixed the
        // attachment and draw-buffer indices; the explicit unbind below runs
        // even if completeness or the driver error check fails.
        let result = unsafe {
            if descriptor.color_attachments.iter().all(Option::is_none) {
                self.gl.draw_buffers(&[glow::NONE]);
                self.gl.read_buffer(glow::NONE);
            } else if !descriptor.draw_buffers.is_empty() {
                let constants: Vec<u32> = descriptor
                    .draw_buffers
                    .iter()
                    .map(|index| {
                        if *index == u32::MAX {
                            glow::NONE
                        } else {
                            draw_buffer_constant(*index)
                        }
                    })
                    .collect();
                self.gl.draw_buffers(&constants);
                self.gl.read_buffer(
                    descriptor
                        .color_attachments
                        .iter()
                        .position(Option::is_some)
                        .map_or(glow::NONE, |i| draw_buffer_constant(i as u32)),
                );
            }
            attached.and_then(|()| self.require_complete(OP))
        };
        let result = result.and_then(|()| self.driver_error(OP));
        // SAFETY: see above; restore the default framebuffer binding so no
        // half-validated target stays selected on any path.
        // SAFETY: the active pass authenticated the target and attachments
        // before this mutation. Clear values come from the validated descriptor,
        // and the error path immediately restores framebuffer zero.
        unsafe {
            self.gl.bind_framebuffer(glow::FRAMEBUFFER, None);
        }
        if let Err(error) = result {
            // SAFETY: current-context contract.
            unsafe { self.gl.delete_framebuffer(raw) };
            return Err(error);
        }
        let slot = match self.slot(OP) {
            Ok(slot) => slot,
            Err(error) => {
                // SAFETY: this scratch framebuffer is still owned here.
                unsafe { self.gl.delete_framebuffer(raw) };
                return Err(error);
            }
        };
        let id = FramebufferId::new(self.context_stamp(), slot, 0);
        self.framebuffers.insert(
            id,
            NativeFramebuffer {
                generation: id.generation,
                raw,
                descriptor: descriptor.clone(),
                multiview,
            },
        );
        Ok(id)
    }
}

impl GlFramebufferApi for NativeGlProvider {
    fn create_framebuffer(
        &mut self,
        descriptor: &GlFramebufferDescriptor,
    ) -> Result<FramebufferId, GlError> {
        let multiview = descriptor
            .color_attachments
            .iter()
            .flatten()
            .chain(descriptor.depth_stencil_attachment.iter())
            .any(|view| view.layer_count > 1);
        self.create_framebuffer_mode(descriptor, multiview)
    }

    fn destroy_framebuffer(&mut self, framebuffer: FramebufferId) -> Result<(), GlError> {
        use glow::HasContext as _;
        const OP: &str = "destroy-framebuffer";
        self.assert_ready(OP)?;
        self.framebuffer(OP, framebuffer)?;
        let entry = self
            .framebuffers
            .remove(&framebuffer)
            .ok_or_else(|| Self::validation(OP, "framebuffer disappeared"))?;
        // SAFETY: current-context contract; liveness was checked first.
        unsafe { self.gl.delete_framebuffer(entry.raw) };
        self.defer_readonly_error(OP)
    }

    fn begin_render_pass(&mut self, descriptor: &GlRenderPassDescriptor) -> Result<(), GlError> {
        use glow::HasContext as _;
        const OP: &str = "begin-render-pass";
        self.assert_ready(OP)?;
        if self.pass.is_some() {
            return Err(Self::validation(OP, "render pass already active"));
        }
        let (record, shape) = match descriptor.target {
            GlRenderTarget::Offscreen(framebuffer) => {
                let record = self.framebuffer(OP, framebuffer)?;
                (Some(record), framebuffer_shape(&record.descriptor))
            }
            GlRenderTarget::Default(target) => {
                if target.width == 0 || target.height == 0 || target.sample_count == 0 {
                    return Err(Self::validation(
                        OP,
                        "default framebuffer has an empty shape",
                    ));
                }
                (None, (target.width, target.height, target.sample_count))
            }
        };
        descriptor
            .validate(
                record.map(|record| &record.descriptor),
                self.discovery.limits().max_color_attachments,
                self.discovery.limits().max_draw_buffers,
                self.context_stamp(),
            )
            .map_err(|_| {
                Self::validation(OP, "render pass does not match its framebuffer descriptor")
            })?;
        descriptor
            .validate_multiview(self.discovery.max_multiview_view_count())
            .map_err(|_| Self::validation(OP, "multiview view count is not proved"))?;
        for attachment in descriptor.color_attachments.iter().flatten() {
            if let GlPassAttachmentView::Allocated(view) = attachment.view {
                self.validate_attachment(OP, view)?;
            }
            if let Some(resolve) = attachment.resolve_target {
                // Resolving stays an explicit blit; the pass only proves the
                // resolve target is live and structurally valid.
                self.validate_attachment(OP, resolve)?;
            }
        }
        if let Some(depth_stencil) = &descriptor.depth_stencil_attachment {
            self.validate_attachment(OP, depth_stencil.view)?;
            if depth_stencil.stencil_load == GlLoadOp::Clear
                && !has_stencil_plane(depth_stencil.view.format)
            {
                return Err(Self::validation(
                    OP,
                    "attachment has no stencil plane to clear",
                ));
            }
        }
        let partial = if let Some(area) = descriptor.render_area {
            if area.width == 0
                || area.height == 0
                || area.x.checked_add(area.width).is_none_or(|x| x > shape.0)
                || area.y.checked_add(area.height).is_none_or(|y| y > shape.1)
            {
                return Err(Self::validation(OP, "render area exceeds framebuffer"));
            }
            area.x != 0 || area.y != 0 || area.width != shape.0 || area.height != shape.1
        } else {
            false
        };
        let raw = record.map(|record| record.raw);
        let draw_buffers = record.map(|record| record.descriptor.draw_buffers.clone());
        let discard_color: Vec<bool> = descriptor
            .color_attachments
            .iter()
            .map(|attachment| !partial && attachment.is_some_and(|a| a.store == GlStoreOp::Discard))
            .collect();
        let discard_depth_stencil =
            descriptor
                .depth_stencil_attachment
                .as_ref()
                .map(|attachment| {
                    (
                        !partial && attachment.depth_store == GlStoreOp::Discard,
                        !partial && attachment.stencil_store == GlStoreOp::Discard,
                    )
                });

        // Clear setup mutates fixed GL state independently of the cached
        // pipeline. Invalidate before mutation, including on an error path,
        // so the next pipeline reapplies scissor and write masks in full.
        self.context_state.pipeline_failed();
        // SAFETY: current-context contract; the framebuffer is live and the
        // clears cover the requested render area exactly once per channel.
        unsafe {
            self.gl.bind_framebuffer(glow::FRAMEBUFFER, raw);
            // Desktop GL requires explicit linear-to-sRGB conversion for sRGB
            // attachments. GLES performs that conversion without this switch.
            if matches!(
                self.discovery.context().profile(),
                crate::backend::api::GlFamilyProfile::Desktop { .. }
            ) {
                let srgb = descriptor
                    .color_attachments
                    .iter()
                    .flatten()
                    .any(|attachment| {
                        (match attachment.view {
                            GlPassAttachmentView::Allocated(view) => view.format,
                            GlPassAttachmentView::DefaultColor(target) => target.color_format,
                        }) == crate::backend::api::GlFormat::Rgba8Srgb
                    });
                if srgb {
                    self.gl.enable(glow::FRAMEBUFFER_SRGB);
                } else {
                    self.gl.disable(glow::FRAMEBUFFER_SRGB);
                }
            }
            if let Some(draw_buffers) = draw_buffers.filter(|buffers| !buffers.is_empty()) {
                let constants: Vec<u32> = draw_buffers
                    .iter()
                    .map(|index| {
                        if *index == u32::MAX {
                            glow::NONE
                        } else {
                            draw_buffer_constant(*index)
                        }
                    })
                    .collect();
                self.gl.draw_buffers(&constants);
            }
            // ClearBuffer obeys the same scissor and write masks as Clear.
            // Shared GL uses one global color mask for all attachments.
            if let Some(area) = descriptor.render_area {
                self.gl.enable(glow::SCISSOR_TEST);
                self.gl.scissor(
                    area.x as i32,
                    area.y as i32,
                    area.width as i32,
                    area.height as i32,
                );
            } else {
                self.gl.disable(glow::SCISSOR_TEST);
            }
            if descriptor
                .color_attachments
                .iter()
                .flatten()
                .any(|a| a.load == GlLoadOp::Clear)
            {
                self.gl.color_mask(true, true, true, true);
            }
            if let Some(attachment) = &descriptor.depth_stencil_attachment {
                if attachment.depth_load == GlLoadOp::Clear {
                    self.gl.depth_mask(true);
                }
                if attachment.stencil_load == GlLoadOp::Clear
                    && has_stencil_plane(attachment.view.format)
                {
                    self.gl.stencil_mask(u32::MAX);
                }
            }
            for (index, attachment) in descriptor.color_attachments.iter().enumerate() {
                if let Some(attachment) = attachment.filter(|a| a.load == GlLoadOp::Clear) {
                    match attachment.clear {
                        crate::backend::api::GlColorClearValue::Float(v) => {
                            let needs_split = matches!(attachment.view, GlPassAttachmentView::Allocated(view)
                                if matches!(view.target, GlAttachmentTarget::Texture(id) if self.texture(OP, id).is_ok_and(|(_, desc)| desc.dimension == GlTextureDimension::D2Array && desc.sample_count > 1))
                                && view.format == crate::backend::api::GlFormat::Rgba8Unorm)
                                && self
                                    .discovery
                                    .context()
                                    .version()
                                    .contains("ANGLE 2.1.22762")
                                && self
                                    .discovery
                                    .context()
                                    .renderer()
                                    .contains("AMD Radeon 780M Graphics (0x00001900) Direct3D11");
                            // This ANGLE/D3D11 implementation can skip a repeated full
                            // clear when successive FBOs select different MSAA layers.
                            // Separate channel masks avoid its full-clear path while
                            // preserving scissor and the logical all-channel clear.
                            if needs_split {
                                self.gl.color_mask(true, true, true, false);
                                self.gl.clear_buffer_f32_slice(
                                    glow::COLOR,
                                    index as u32,
                                    &v.map(f32::from_bits),
                                );
                                self.gl.color_mask(false, false, false, true);
                                self.gl.clear_buffer_f32_slice(
                                    glow::COLOR,
                                    index as u32,
                                    &v.map(f32::from_bits),
                                );
                                self.gl.color_mask(true, true, true, true);
                            } else {
                                self.gl.clear_buffer_f32_slice(
                                    glow::COLOR,
                                    index as u32,
                                    &v.map(f32::from_bits),
                                );
                            }
                        }
                        crate::backend::api::GlColorClearValue::Uint(v) => self
                            .gl
                            .clear_buffer_u32_slice(glow::COLOR, index as u32, &v),
                        crate::backend::api::GlColorClearValue::Sint(v) => self
                            .gl
                            .clear_buffer_i32_slice(glow::COLOR, index as u32, &v),
                    }
                }
            }
            if let Some(attachment) = &descriptor.depth_stencil_attachment {
                let depth_clears = attachment.depth_load == GlLoadOp::Clear;
                let stencil_clears = attachment.stencil_load == GlLoadOp::Clear
                    && has_stencil_plane(attachment.view.format);
                match (depth_clears, stencil_clears) {
                    (true, true) => self.gl.clear_buffer_depth_stencil(
                        glow::DEPTH_STENCIL,
                        0,
                        f32::from_bits(attachment.clear.depth),
                        attachment.clear.stencil as i32,
                    ),
                    (true, false) => self.gl.clear_buffer_f32_slice(
                        glow::DEPTH,
                        0,
                        &[f32::from_bits(attachment.clear.depth)],
                    ),
                    (false, true) => self.gl.clear_buffer_i32_slice(
                        glow::STENCIL,
                        0,
                        &[attachment.clear.stencil as i32],
                    ),
                    (false, false) => {}
                }
            }
        }
        self.defer_driver_error(OP)?;
        self.pass = Some(super::provider::ActivePass {
            target: descriptor.target,
            width: shape.0,
            height: shape.1,
            samples: shape.2,
            color_formats: descriptor
                .color_attachments
                .iter()
                .map(|attachment| {
                    attachment.map(|attachment| match attachment.view {
                        GlPassAttachmentView::Allocated(view) => view.format,
                        GlPassAttachmentView::DefaultColor(target) => target.color_format,
                    })
                })
                .collect(),
            depth_stencil_format: descriptor
                .depth_stencil_attachment
                .map(|attachment| attachment.view.format),
            discard_color,
            discard_depth_stencil,
        });
        Ok(())
    }

    /// Clears selected active-pass attachments and restores the complete
    /// pipeline packet, including scissor and write masks, before returning.
    fn clear_active_rect(
        &mut self,
        rect: Rect,
        colors: &[ColorAttachmentClear],
        depth: Option<f32>,
        stencil: Option<u32>,
        restore: Option<&crate::backend::api::GlRasterPipeline>,
    ) -> Result<(), GlError> {
        use crate::backend::api::GlRasterCommandApi as _;
        use glow::HasContext as _;
        const OP: &str = "clear-active-rect";
        self.assert_ready(OP)?;
        let pass = self
            .pass
            .as_ref()
            .ok_or_else(|| Self::validation(OP, "no active render pass"))?;
        let right = rect
            .x
            .checked_add(rect.width)
            .ok_or_else(|| Self::validation(OP, "rectangle overflow"))?;
        let bottom = rect
            .y
            .checked_add(rect.height)
            .ok_or_else(|| Self::validation(OP, "rectangle overflow"))?;
        if rect.width == 0 || rect.height == 0 || right > pass.width || bottom > pass.height {
            return Err(Self::validation(OP, "rectangle outside active render pass"));
        }
        let color_formats = pass.color_formats.clone();
        let depth_stencil_format = pass.depth_stencil_format;
        for clear in colors {
            let format = color_formats
                .get(clear.attachment as usize)
                .and_then(|format| *format)
                .ok_or_else(|| Self::validation(OP, "missing color attachment"))?;
            if !color_clear_matches(format, clear.value) {
                return Err(Self::validation(
                    OP,
                    "color clear value does not match attachment",
                ));
            }
        }
        if (depth.is_some() || stencil.is_some()) && depth_stencil_format.is_none() {
            return Err(Self::validation(OP, "missing depth/stencil attachment"));
        }
        if depth.is_some_and(|_| !has_depth_plane(depth_stencil_format.unwrap())) {
            return Err(Self::validation(OP, "active attachment has no depth plane"));
        }
        if stencil.is_some_and(|_| !has_stencil_plane(depth_stencil_format.unwrap())) {
            return Err(Self::validation(
                OP,
                "active attachment has no stencil plane",
            ));
        }
        let native = clear_native_rect(rect)?;
        // SAFETY: the active pass keeps the framebuffer bound, native_rect
        // converted upper-left Core coordinates, and all clear values were
        // validated against the active attachment formats above.
        unsafe {
            self.gl.enable(glow::SCISSOR_TEST);
            self.gl.scissor(native.0, native.1, native.2, native.3);
            for clear in colors {
                self.gl
                    .color_mask_draw_buffer(clear.attachment, true, true, true, true);
                match clear.value {
                    ClearColor::Float(value) => {
                        self.gl
                            .clear_buffer_f32_slice(glow::COLOR, clear.attachment, &value)
                    }
                    ClearColor::Uint(value) => {
                        self.gl
                            .clear_buffer_u32_slice(glow::COLOR, clear.attachment, &value)
                    }
                    ClearColor::Sint(value) => {
                        self.gl
                            .clear_buffer_i32_slice(glow::COLOR, clear.attachment, &value)
                    }
                }
            }
            self.gl.depth_mask(true);
            self.gl.stencil_mask(u32::MAX);
            match (depth, stencil) {
                (Some(depth), Some(stencil)) => self.gl.clear_buffer_depth_stencil(
                    glow::DEPTH_STENCIL,
                    0,
                    depth,
                    stencil as i32,
                ),
                (Some(depth), None) => self.gl.clear_buffer_f32_slice(glow::DEPTH, 0, &[depth]),
                (None, Some(stencil)) => {
                    self.gl
                        .clear_buffer_i32_slice(glow::STENCIL, 0, &[stencil as i32])
                }
                (None, None) => {}
            }
        }
        let clear_result = self.defer_driver_error(OP);
        // The clear mutates masks and scissor behind the cached pipeline
        // packet. Invalidate first, then replay the caller's packet so the
        // next draw observes exactly its pre-clear state. Without a bound
        // pipeline there is no RHI-visible mutable packet to restore; the
        // first later bind will replay its complete state after invalidation.
        self.context_state.pipeline_failed();
        match restore {
            Some(restore) => clear_result.and(self.set_raster_pipeline(restore)),
            None => clear_result,
        }
    }

    fn end_render_pass(&mut self) -> Result<(), GlError> {
        use glow::HasContext as _;
        const OP: &str = "end-render-pass";
        self.assert_ready(OP)?;
        let Some(pass) = self.pass.take() else {
            return Err(Self::validation(OP, "no active render pass"));
        };
        self.raster = None;
        let raw = match pass.target {
            GlRenderTarget::Offscreen(framebuffer) => Some(self.framebuffer(OP, framebuffer)?.raw),
            GlRenderTarget::Default(_) => None,
        };
        let mut invalidate: Vec<u32> = pass
            .discard_color
            .iter()
            .enumerate()
            .filter(|(_, discard)| **discard)
            .map(|(index, _)| match pass.target {
                GlRenderTarget::Default(_) => glow::COLOR,
                _ => draw_buffer_constant(index as u32),
            })
            .collect();
        if let Some((discard_depth, discard_stencil)) = pass.discard_depth_stencil {
            if discard_depth {
                invalidate.push(if matches!(pass.target, GlRenderTarget::Default(_)) {
                    glow::DEPTH
                } else {
                    glow::DEPTH_ATTACHMENT
                });
            }
            if discard_stencil {
                invalidate.push(if matches!(pass.target, GlRenderTarget::Default(_)) {
                    glow::STENCIL
                } else {
                    glow::STENCIL_ATTACHMENT
                });
            }
        }
        // SAFETY: current-context contract; glInvalidateFramebuffer is issued
        // only where the profile's core supplies it (ES 3.x, desktop 4.3+);
        // desktop cores before 4.3 keep store semantics, which is the honest
        // fallback because no invalidate command exists to call.
        unsafe {
            self.gl.bind_framebuffer(glow::FRAMEBUFFER, raw);
            let result = if invalidate.is_empty()
                || !supports_framebuffer_invalidate(self.discovery.context().profile())
            {
                Ok(())
            } else {
                self.gl
                    .invalidate_framebuffer(glow::FRAMEBUFFER, &invalidate);
                self.defer_driver_error(OP)
            };
            self.gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            // A raster pass is not a submission boundary: frames can contain
            // many passes. Preserve diagnostics for presentation/fence
            // publication rather than reading GL's sticky flag here.
            result.and_then(|()| self.defer_driver_error(OP))
        }
    }

    fn blit_framebuffer(
        &mut self,
        source: FramebufferId,
        destination: FramebufferId,
        region: GlBlitRegion,
        filter: GlFilterMode,
        masks: GlBlitMask,
    ) -> Result<(), GlError> {
        use glow::HasContext as _;
        const OP: &str = "blit-framebuffer";
        self.assert_ready(OP)?;
        if source == destination {
            return Err(Self::validation(
                OP,
                "blit source and destination are identical",
            ));
        }
        if masks.is_empty() {
            return Err(Self::validation(
                OP,
                "blit selects no color/depth/stencil plane",
            ));
        }
        region
            .validate()
            .map_err(|_| Self::validation(OP, "invalid blit region"))?;
        let source_shape = self
            .framebuffer(OP, source)
            .map(|record| framebuffer_shape(&record.descriptor))?;
        let destination_shape = self
            .framebuffer(OP, destination)
            .map(|record| framebuffer_shape(&record.descriptor))?;
        let within = |offset: [u32; 2], extent: [u32; 2], shape: (u32, u32, u32)| {
            offset[0]
                .checked_add(extent[0])
                .is_some_and(|end| end <= shape.0)
                && offset[1]
                    .checked_add(extent[1])
                    .is_some_and(|end| end <= shape.1)
        };
        if !within(region.src_offset, region.src_extent, source_shape) {
            return Err(Self::validation(OP, "blit source leaves its framebuffer"));
        }
        if !within(region.dst_offset, region.dst_extent, destination_shape) {
            return Err(Self::validation(
                OP,
                "blit destination leaves its framebuffer",
            ));
        }
        // The shared contract: multisampled targets only accept nearest.
        if filter != GlFilterMode::Nearest && (source_shape.2 > 1 || destination_shape.2 > 1) {
            return Err(Self::validation(
                OP,
                "multisampled blit targets only accept nearest filtering",
            ));
        }
        // Depth/stencil planes accept nearest filtering, including scaling.
        if (masks.depth || masks.stencil) && filter != GlFilterMode::Nearest {
            return Err(Self::validation(
                OP,
                "depth/stencil blits require nearest filtering",
            ));
        }
        let source_raw = self.framebuffer(OP, source)?.raw;
        let destination_record = self.framebuffer(OP, destination)?;
        let destination_raw = destination_record.raw;
        let destination_srgb = destination_record
            .descriptor
            .color_attachments
            .iter()
            .flatten()
            .any(|view| view.format == crate::backend::api::GlFormat::Rgba8Srgb);
        self.context_state.pipeline_failed();
        // SAFETY: current-context contract; both framebuffers are live and
        // every bounds/filter rule was validated before the blit.
        unsafe {
            self.gl.disable(glow::SCISSOR_TEST);
            if matches!(
                self.discovery.context().profile(),
                crate::backend::api::GlFamilyProfile::Desktop { .. }
            ) {
                if destination_srgb {
                    self.gl.enable(glow::FRAMEBUFFER_SRGB);
                } else {
                    self.gl.disable(glow::FRAMEBUFFER_SRGB);
                }
            }
            self.gl
                .bind_framebuffer(glow::READ_FRAMEBUFFER, Some(source_raw));
            self.gl
                .bind_framebuffer(glow::DRAW_FRAMEBUFFER, Some(destination_raw));
            self.gl.blit_framebuffer(
                region.src_offset[0] as i32,
                region.src_offset[1] as i32,
                (region.src_offset[0] + region.src_extent[0]) as i32,
                (region.src_offset[1] + region.src_extent[1]) as i32,
                region.dst_offset[0] as i32,
                region.dst_offset[1] as i32,
                (region.dst_offset[0] + region.dst_extent[0]) as i32,
                (region.dst_offset[1] + region.dst_extent[1]) as i32,
                blit_mask_bits(masks),
                blit_filter(filter),
            );
            let result = self.defer_driver_error(OP);
            self.gl.bind_framebuffer(glow::READ_FRAMEBUFFER, None);
            self.gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, None);
            result
        }
    }
}

/// The GL attachment target of one live texture allocation.
///
/// A multisample texture is a different GL object class with its own target and
/// its own attachment entry point, so the target has to come from the
/// allocation's sample count rather than from the attachment descriptor, which
/// describes a view and not the storage behind it.
pub(super) const fn attachment_target(sample_count: u32) -> u32 {
    if sample_count > 1 {
        glow::TEXTURE_2D_MULTISAMPLE
    } else {
        glow::TEXTURE_2D
    }
}

/// `glInvalidateFramebuffer` is core ES 3.x and desktop 4.3+.
pub(super) fn supports_framebuffer_invalidate(
    profile: crate::backend::api::GlFamilyProfile,
) -> bool {
    match profile {
        crate::backend::api::GlFamilyProfile::Embedded { .. } => true,
        crate::backend::api::GlFamilyProfile::Desktop { major, minor } => major > 4 || minor >= 3,
        crate::backend::api::GlFamilyProfile::WebGl2 => true,
    }
}

#[cfg(test)]
#[path = "../../tests/backend/native/framebuffer_rect.rs"]
mod tests;
