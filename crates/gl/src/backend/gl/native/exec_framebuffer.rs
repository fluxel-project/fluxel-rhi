//! Native framebuffer, render-pass, and blit execution.
//!
//! Pass load/clear executes exactly once per pass through the `glClearBuffer*`
//! commands, which honor scissor and color, depth, and stencil write masks.
//! Pass begin disables scissor, enables writes for cleared attachments, and
//! invalidates cached pipeline state so the next pipeline restores its masks.
//! Resolve stays an
//! explicit `glBlitFramebuffer` from the multisample source, per the shared
//! framebuffer contract.

use super::provider::{NativeFramebuffer, NativeGlProvider};
use super::resource::{depth_attachment_point, has_stencil_plane};
use crate::backend::gl::api::{
    FramebufferId, GlAttachmentTarget, GlBlitMask, GlBlitRegion, GlDepthStencilAttachment, GlError,
    GlFamilyApi as _, GlFilterMode, GlFramebufferApi, GlFramebufferDescriptor, GlLoadOp,
    GlPassAttachmentView, GlRenderPassDescriptor, GlRenderTarget, GlStoreOp, GlTextureDimension,
    GlTextureView,
};

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
        .first()
        .copied()
        .or(descriptor.depth_stencil_attachment)
        .map(|view| (view.width, view.height, view.sample_count))
        .unwrap_or((0, 0, 1))
}

impl GlFramebufferApi for NativeGlProvider {
    fn create_framebuffer(
        &mut self,
        descriptor: &GlFramebufferDescriptor,
    ) -> Result<FramebufferId, GlError> {
        use glow::HasContext as _;
        const OP: &str = "create-framebuffer";
        self.assert_ready(OP)?;
        for view in descriptor
            .color_attachments
            .iter()
            .copied()
            .chain(descriptor.depth_stencil_attachment)
        {
            self.validate_attachment(OP, view)?;
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
        let attached = self.attach_all(OP, raw, descriptor);
        // Depth-only framebuffers must not keep a draw buffer selected.
        // SAFETY: `raw` was allocated by this current provider and is bound by
        // `attach_all` on this same context. Descriptor validation fixed the
        // attachment and draw-buffer indices; the explicit unbind below runs
        // even if completeness or the driver error check fails.
        let result = unsafe {
            if descriptor.color_attachments.is_empty() {
                self.gl.draw_buffers(&[glow::NONE]);
            } else if !descriptor.draw_buffers.is_empty() {
                let constants: Vec<u32> = descriptor
                    .draw_buffers
                    .iter()
                    .map(|index| draw_buffer_constant(*index))
                    .collect();
                self.gl.draw_buffers(&constants);
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
        let slot = self.slot(OP)?;
        let id = FramebufferId::new(self.context_stamp(), slot, 0);
        self.framebuffers.insert(
            id,
            NativeFramebuffer {
                generation: id.generation,
                raw,
                descriptor: descriptor.clone(),
            },
        );
        Ok(id)
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
        self.driver_error(OP)
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
        for attachment in &descriptor.color_attachments {
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
        let raw = record.map(|record| record.raw);
        let draw_buffers = record.map(|record| record.descriptor.draw_buffers.clone());
        let discard_color: Vec<bool> = descriptor
            .color_attachments
            .iter()
            .map(|attachment| attachment.store == GlStoreOp::Discard)
            .collect();
        let discard_depth_stencil =
            descriptor
                .depth_stencil_attachment
                .as_ref()
                .map(|attachment| {
                    (
                        attachment.depth_store == GlStoreOp::Discard,
                        attachment.stencil_store == GlStoreOp::Discard,
                    )
                });

        // Clear setup mutates fixed GL state independently of the cached
        // pipeline. Invalidate before mutation, including on an error path,
        // so the next pipeline reapplies scissor and write masks in full.
        self.context_state.pipeline_failed();
        // SAFETY: current-context contract; the framebuffer is live and the
        // clears are total exactly once per pass begin.
        unsafe {
            self.gl.bind_framebuffer(glow::FRAMEBUFFER, raw);
            if let Some(draw_buffers) = draw_buffers.filter(|buffers| !buffers.is_empty()) {
                let constants: Vec<u32> = draw_buffers
                    .iter()
                    .map(|index| draw_buffer_constant(*index))
                    .collect();
                self.gl.draw_buffers(&constants);
            }
            // ClearBuffer obeys the same scissor and write masks as Clear.
            // Shared GL uses one global color mask for all attachments.
            self.gl.disable(glow::SCISSOR_TEST);
            if descriptor
                .color_attachments
                .iter()
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
                if attachment.load == GlLoadOp::Clear {
                    let clear = &attachment.clear;
                    self.gl.clear_buffer_f32_slice(
                        glow::COLOR,
                        index as u32,
                        &[
                            f32::from_bits(clear.red),
                            f32::from_bits(clear.green),
                            f32::from_bits(clear.blue),
                            f32::from_bits(clear.alpha),
                        ],
                    );
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
        if let Err(error) = self.driver_error(OP) {
            // SAFETY: current-context contract.
            unsafe { self.gl.bind_framebuffer(glow::FRAMEBUFFER, None) };
            return Err(error);
        }
        self.pass = Some(super::provider::ActivePass {
            target: descriptor.target,
            width: shape.0,
            height: shape.1,
            samples: shape.2,
            discard_color,
            discard_depth_stencil,
        });
        Ok(())
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
            .map(|(index, _)| draw_buffer_constant(index as u32))
            .collect();
        if let Some((discard_depth, discard_stencil)) = pass.discard_depth_stencil {
            if discard_depth {
                invalidate.push(glow::DEPTH_ATTACHMENT);
            }
            if discard_stencil {
                invalidate.push(glow::STENCIL_ATTACHMENT);
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
                self.driver_error(OP)
            };
            self.gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            result
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
        // Depth/stencil planes never scale and never filter.
        if (masks.depth || masks.stencil)
            && (filter != GlFilterMode::Nearest || region.src_extent != region.dst_extent)
        {
            return Err(Self::validation(
                OP,
                "depth/stencil blits require nearest filtering and identical extents",
            ));
        }
        let source_raw = self.framebuffer(OP, source)?.raw;
        let destination_raw = self.framebuffer(OP, destination)?.raw;
        // SAFETY: current-context contract; both framebuffers are live and
        // every bounds/filter rule was validated before the blit.
        unsafe {
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
            let result = self.driver_error(OP);
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
    profile: crate::backend::gl::api::GlFamilyProfile,
) -> bool {
    match profile {
        crate::backend::gl::api::GlFamilyProfile::Embedded { .. } => true,
        crate::backend::gl::api::GlFamilyProfile::Desktop { major, minor } => {
            major > 4 || minor >= 3
        }
        crate::backend::gl::api::GlFamilyProfile::WebGl2 => true,
    }
}
