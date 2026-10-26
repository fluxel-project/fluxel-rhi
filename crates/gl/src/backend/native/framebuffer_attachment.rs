//! Framebuffer attachment validation and default-framebuffer presentation.
//!
//! Render-pass construction lives in `exec_framebuffer`. This module contains
//! the storage-facing half: validation and attachment issue paths whose
//! correctness depends on the live resource tables rather than pass load/store
//! policy. Keeping it separate makes the native ownership boundary explicit.

use super::resource::depth_attachment_point;
use super::{
    exec_framebuffer::{attachment_target, draw_buffer_constant, framebuffer_shape},
    provider::NativeGlProvider,
};
use crate::backend::api::{
    FramebufferId, GlAttachmentTarget, GlError, GlFamilyApi as _, GlFramebufferDescriptor,
    GlTextureDimension, GlTextureView,
};
impl NativeGlProvider {
    /// Copies or resolves a complete owned color framebuffer into the
    /// authenticated WGL/EGL default framebuffer. The default target remains a
    /// typed acquired-frame fact; it is never entered into the owned FBO table.
    pub(crate) fn blit_framebuffer_to_default(
        &mut self,
        source: FramebufferId,
        target: crate::backend::api::GlDefaultFramebufferTarget,
    ) -> Result<(), GlError> {
        self.blit_framebuffer_slot_to_default(source, 0, target)
    }

    /// Copies the acquired default color into an owned single-sample framebuffer.
    pub(crate) fn blit_default_to_framebuffer(
        &mut self,
        target: crate::backend::api::GlDefaultFramebufferTarget,
        destination: FramebufferId,
    ) -> Result<(), GlError> {
        self.blit_default_to_framebuffer_slot(target, destination, 0)
    }

    /// Copies default color into an owned FBO, reversing the window Y axis.
    /// Core texture row zero is GL y=0; the drawable top row is GL y=height-1.
    pub(crate) fn blit_default_to_framebuffer_slot(
        &mut self,
        target: crate::backend::api::GlDefaultFramebufferTarget,
        destination: FramebufferId,
        slot: u32,
    ) -> Result<(), GlError> {
        use glow::HasContext as _;
        const OP: &str = "blit-default-to-color-slot";
        self.assert_ready(OP)?;
        let record = self.framebuffer(OP, destination)?;
        let (width, height, samples) = framebuffer_shape(&record.descriptor);
        let view = record
            .descriptor
            .color_attachments
            .get(slot as usize)
            .copied()
            .flatten();
        if target.context != self.context_stamp()
            || target.sample_count != 1
            || samples != 1
            || width != target.width
            || height != target.height
            || view.is_none_or(|v| v.format != target.color_format)
        {
            return Err(Self::validation(
                OP,
                "default and selected destination attachment do not match",
            ));
        }
        let raw = record.raw;
        let mapping: Vec<_> = record
            .descriptor
            .draw_buffers
            .iter()
            .map(|i| {
                if *i == u32::MAX {
                    glow::NONE
                } else {
                    draw_buffer_constant(*i)
                }
            })
            .collect();
        self.context_state.pipeline_failed();
        // SAFETY: both framebuffer bindings and the selected attachment were authenticated above.
        unsafe {
            self.gl.disable(glow::SCISSOR_TEST);
            if matches!(
                self.discovery.context().profile(),
                crate::backend::api::GlFamilyProfile::Desktop { .. }
            ) {
                if target.color_format == crate::backend::api::GlFormat::Rgba8Srgb {
                    self.gl.enable(glow::FRAMEBUFFER_SRGB);
                } else {
                    self.gl.disable(glow::FRAMEBUFFER_SRGB);
                }
            }
            self.gl.bind_framebuffer(glow::READ_FRAMEBUFFER, None);
            self.gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, Some(raw));
            let mut selected = vec![glow::NONE; slot as usize + 1];
            selected[slot as usize] = draw_buffer_constant(slot);
            self.gl.draw_buffers(&selected);
            self.gl.blit_framebuffer(
                0,
                height as i32,
                width as i32,
                0,
                0,
                0,
                width as i32,
                height as i32,
                glow::COLOR_BUFFER_BIT,
                glow::NEAREST,
            );
            let result = self.defer_driver_error(OP);
            if mapping.is_empty() {
                self.gl.draw_buffers(&[glow::COLOR_ATTACHMENT0]);
            } else {
                self.gl.draw_buffers(&mapping);
            }
            self.gl.bind_framebuffer(glow::READ_FRAMEBUFFER, None);
            self.gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, None);
            result.and_then(|()| self.defer_driver_error(OP))
        }
    }

    /// Publishes one selected owned color attachment to the default framebuffer.
    pub(crate) fn blit_framebuffer_slot_to_default(
        &mut self,
        source: FramebufferId,
        slot: u32,
        target: crate::backend::api::GlDefaultFramebufferTarget,
    ) -> Result<(), GlError> {
        self.blit_framebuffer_slot_to_default_region(source, slot, target, None)
    }

    /// Publishes a selected upper-left render area, reversing the window Y axis.
    /// The source must be single-sample: resolve before flipping a surface.
    pub(crate) fn blit_framebuffer_slot_to_default_region(
        &mut self,
        source: FramebufferId,
        slot: u32,
        target: crate::backend::api::GlDefaultFramebufferTarget,
        area: Option<crate::backend::api::GlScissorRect>,
    ) -> Result<(), GlError> {
        use glow::HasContext as _;
        const OP: &str = "blit-color-slot-to-default";
        self.assert_ready(OP)?;
        let record = self.framebuffer(OP, source)?;
        let (width, height, samples) = framebuffer_shape(&record.descriptor);
        let view = record
            .descriptor
            .color_attachments
            .get(slot as usize)
            .copied()
            .flatten();
        if target.context != self.context_stamp()
            || target.sample_count != 1
            || samples != 1
            || width != target.width
            || height != target.height
            || view.is_none_or(|v| v.format != target.color_format)
        {
            return Err(Self::validation(
                OP,
                "selected source and default framebuffer do not match",
            ));
        }
        let area = area.unwrap_or(crate::backend::api::GlScissorRect {
            x: 0,
            y: 0,
            width,
            height,
        });
        if area.width == 0
            || area.height == 0
            || area.x.checked_add(area.width).is_none_or(|x| x > width)
            || area.y.checked_add(area.height).is_none_or(|y| y > height)
        {
            return Err(Self::validation(OP, "resolve region exceeds framebuffer"));
        }
        let x0 = area.x as i32;
        let x1 = (area.x + area.width) as i32;
        let y0 = area.y as i32;
        let y1 = (area.y + area.height) as i32;
        let raw = record.raw;
        let first = record
            .descriptor
            .color_attachments
            .iter()
            .position(Option::is_some)
            .map_or(glow::NONE, |i| draw_buffer_constant(i as u32));
        self.context_state.pipeline_failed();
        // SAFETY: both framebuffer bindings and the selected attachment were authenticated above.
        unsafe {
            self.gl.disable(glow::SCISSOR_TEST);
            if matches!(
                self.discovery.context().profile(),
                crate::backend::api::GlFamilyProfile::Desktop { .. }
            ) {
                if target.color_format == crate::backend::api::GlFormat::Rgba8Srgb {
                    self.gl.enable(glow::FRAMEBUFFER_SRGB);
                } else {
                    self.gl.disable(glow::FRAMEBUFFER_SRGB);
                }
            }
            self.gl.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(raw));
            self.gl.read_buffer(draw_buffer_constant(slot));
            self.gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, None);
            self.gl.blit_framebuffer(
                x0,
                y0,
                x1,
                y1,
                x0,
                height as i32 - y0,
                x1,
                height as i32 - y1,
                glow::COLOR_BUFFER_BIT,
                glow::NEAREST,
            );
            let result = self.defer_driver_error(OP);
            self.gl.read_buffer(first);
            self.gl.bind_framebuffer(glow::READ_FRAMEBUFFER, None);
            self.gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, None);
            result.and_then(|()| self.defer_driver_error(OP))
        }
    }

    /// Attaches every validated view of a descriptor to one fresh FBO.
    pub(super) fn attach_all(
        &self,
        op: &'static str,
        raw: glow::NativeFramebuffer,
        descriptor: &GlFramebufferDescriptor,
        multiview: bool,
    ) -> Result<(), GlError> {
        use glow::HasContext as _;
        // SAFETY: current-context contract; every attachment was validated.
        unsafe {
            self.gl.bind_framebuffer(glow::FRAMEBUFFER, Some(raw));
            for (index, view) in descriptor.color_attachments.iter().enumerate() {
                if let Some(view) = view {
                    self.attach_one(op, draw_buffer_constant(index as u32), *view, multiview)?;
                }
            }
            if let Some(view) = descriptor.depth_stencil_attachment {
                let point = match view.aspect {
                    crate::backend::api::GlTextureAspect::DepthOnly => glow::DEPTH_ATTACHMENT,
                    crate::backend::api::GlTextureAspect::StencilOnly => glow::STENCIL_ATTACHMENT,
                    crate::backend::api::GlTextureAspect::All => {
                        depth_attachment_point(view.format).ok_or_else(|| {
                            Self::validation(op, "depth attachment format has no attachment point")
                        })?
                    }
                    crate::backend::api::GlTextureAspect::Color => {
                        return Err(Self::validation(op, "color aspect on depth attachment"));
                    }
                };
                self.attach_one(op, point, view, multiview)?;
            }
        }
        Ok(())
    }

    /// Attaches one validated view at one attachment point of the bound FBO.
    ///
    /// The attach entry point is a property of the allocation, not of the
    /// descriptor. A texture is attached through the target its own sample count
    /// selects, because a multisample texture is a different GL object class;
    /// a renderbuffer is attached through the renderbuffer entry point, because
    /// it is not a texture at all. Attaching either through the other's entry
    /// point is an error the driver reports only after the framebuffer has
    /// already been half-populated, so the choice has to follow the storage.
    fn attach_one(
        &self,
        op: &'static str,
        point: u32,
        view: GlTextureView,
        multiview: bool,
    ) -> Result<(), GlError> {
        use glow::HasContext as _;
        match view.target {
            GlAttachmentTarget::Texture(texture) => {
                let (raw, desc) = self.texture(op, texture)?;
                // SAFETY: current-context contract; the framebuffer is bound and
                // the view was validated against this exact allocation.
                unsafe {
                    if multiview {
                        let entry = self.multiview.ok_or(GlError::Unsupported {
                            operation: op,
                            reason: "OVR multiview entry unavailable",
                        })?;
                        entry.attach(
                            point,
                            raw,
                            view.mip_level as i32,
                            view.array_layer as i32,
                            view.layer_count as i32,
                        );
                        return Ok(());
                    }
                    match desc.dimension {
                        GlTextureDimension::D2 => self.gl.framebuffer_texture_2d(
                            glow::FRAMEBUFFER,
                            point,
                            attachment_target(desc.sample_count),
                            Some(raw),
                            view.mip_level as i32,
                        ),
                        GlTextureDimension::D2Array | GlTextureDimension::D3 => {
                            self.gl.framebuffer_texture_layer(
                                glow::FRAMEBUFFER,
                                point,
                                Some(raw),
                                view.mip_level as i32,
                                view.array_layer as i32,
                            )
                        }
                        _ => {
                            return Err(GlError::Unsupported {
                                operation: op,
                                reason: "attachment texture dimension is unavailable",
                            });
                        }
                    }
                }
                Ok(())
            }
            GlAttachmentTarget::Renderbuffer(renderbuffer) => {
                let (raw, _) = self.renderbuffer(op, renderbuffer)?;
                // SAFETY: current-context contract; the framebuffer is bound and
                // the view was validated against this exact allocation.
                unsafe {
                    self.gl.framebuffer_renderbuffer(
                        glow::FRAMEBUFFER,
                        point,
                        glow::RENDERBUFFER,
                        Some(raw),
                    );
                }
                Ok(())
            }
        }
    }

    /// Validates the OVR mode even when it names a single view.
    pub(super) fn validate_multiview_attachment(
        &self,
        op: &'static str,
        view: GlTextureView,
    ) -> Result<(), GlError> {
        let GlAttachmentTarget::Texture(texture) = view.target else {
            return Err(Self::validation(op, "multiview requires array textures"));
        };
        let (_, desc) = self.texture(op, texture)?;
        if desc.dimension != GlTextureDimension::D2Array
            || desc.sample_count != 1
            || self.multiview.is_none()
            || self.discovery.max_multiview_view_count() < 2
            || view.layer_count > self.discovery.max_multiview_view_count()
            || view.mip_level > i32::MAX as u32
            || view.array_layer > i32::MAX as u32
            || view.layer_count > i32::MAX as u32
        {
            return Err(Self::validation(op, "invalid multiview array attachment"));
        }
        Ok(())
    }

    /// Validates one attachment view against the live allocation it names.
    ///
    /// Both storage classes run the same rule sequence, so a descriptor is
    /// rejected for the same reason whichever one backs it: the view must name
    /// a live allocation of the same context, agree with it on format and
    /// sample count, address storage that exists, and be renderable here. The
    /// two arms differ only in what "storage that exists" means. A texture
    /// addresses a mip level and may span layers; a renderbuffer has neither,
    /// so its only addressable view is level 0, layer 0, of exactly one layer,
    /// and any other coordinate is rejected as a layered attachment rather than
    /// silently attaching level 0 and rendering into storage the caller did not
    /// name.
    pub(super) fn validate_attachment(
        &self,
        op: &'static str,
        view: GlTextureView,
    ) -> Result<(), GlError> {
        match view.target {
            GlAttachmentTarget::Texture(texture) => {
                let (_, desc) = self.texture(op, texture)?;
                if desc.format != view.format {
                    return Err(Self::validation(
                        op,
                        "attachment view format does not match the allocation",
                    ));
                }
                let Some(mip) = desc.mip_extent(view.mip_level) else {
                    return Err(Self::validation(op, "attachment mip level is invalid"));
                };
                if view.width != mip.width || view.height != mip.height {
                    return Err(Self::validation(
                        op,
                        "attachment view extent does not match the allocation extent",
                    ));
                }
                if (view.layer_count > 1
                    && (desc.dimension != GlTextureDimension::D2Array
                        || desc.sample_count != 1
                        || self.multiview.is_none()
                        || view.layer_count > self.discovery.max_multiview_view_count()
                        || view.mip_level > i32::MAX as u32
                        || view.array_layer > i32::MAX as u32
                        || view.layer_count > i32::MAX as u32))
                    || (desc.dimension == GlTextureDimension::D2
                        && (view.array_layer != 0 || view.layer_count != 1))
                    || (matches!(
                        desc.dimension,
                        GlTextureDimension::D2Array | GlTextureDimension::D3
                    ) && view
                        .array_layer
                        .checked_add(view.layer_count)
                        .is_none_or(|end| end > mip.depth_or_layers))
                    || !matches!(
                        desc.dimension,
                        GlTextureDimension::D2
                            | GlTextureDimension::D2Array
                            | GlTextureDimension::D3
                    )
                {
                    return Err(GlError::Unsupported {
                        operation: op,
                        reason: "layered attachments are not part of this framebuffer slice",
                    });
                }
                if desc.sample_count != view.sample_count {
                    return Err(Self::validation(
                        op,
                        "attachment sample count does not match the allocation",
                    ));
                }
                let kind = if desc.sample_count > 1 && desc.dimension == GlTextureDimension::D2Array
                {
                    crate::backend::api::GlFormatResourceKind::MultisampleArray
                } else {
                    crate::backend::api::GlFormatResourceKind::Texture
                };
                let facts = self
                    .discovery
                    .formats()
                    .get_for(kind, view.format, view.sample_count);
                if facts.is_none_or(|facts| !facts.renderable) {
                    return Err(GlError::Unsupported {
                        operation: op,
                        reason: "attachment format lacks renderable evidence on this context",
                    });
                }
                Ok(())
            }
            GlAttachmentTarget::Renderbuffer(renderbuffer) => {
                let (_, desc) = self.renderbuffer(op, renderbuffer)?;
                if desc.format != view.format {
                    return Err(Self::validation(
                        op,
                        "attachment view format does not match the allocation",
                    ));
                }
                if view.mip_level != 0 {
                    return Err(Self::validation(op, "attachment mip level is invalid"));
                }
                if view.width != desc.width || view.height != desc.height {
                    return Err(Self::validation(
                        op,
                        "attachment view extent does not match the allocation extent",
                    ));
                }
                if view.array_layer != 0 || view.layer_count != 1 {
                    return Err(GlError::Unsupported {
                        operation: op,
                        reason: "layered attachments are not part of this framebuffer slice",
                    });
                }
                if desc.samples != view.sample_count {
                    return Err(Self::validation(
                        op,
                        "attachment sample count does not match the allocation",
                    ));
                }
                let facts = self.discovery.formats().get_for(
                    crate::backend::api::GlFormatResourceKind::Renderbuffer,
                    view.format,
                    view.sample_count,
                );
                if facts.is_none_or(|facts| !facts.renderable) {
                    return Err(GlError::Unsupported {
                        operation: op,
                        reason: "attachment format lacks renderable evidence on this context",
                    });
                }
                Ok(())
            }
        }
    }
}
