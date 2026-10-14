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
use crate::backend::gl::api::{
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
        target: crate::backend::gl::api::GlDefaultFramebufferTarget,
    ) -> Result<(), GlError> {
        use glow::HasContext as _;
        const OP: &str = "blit-framebuffer-to-default";
        self.assert_ready(OP)?;
        if target.context != self.context_stamp()
            || target.width == 0
            || target.height == 0
            || target.sample_count != 1
        {
            return Err(Self::validation(
                OP,
                "invalid acquired default framebuffer target",
            ));
        }
        let record = self.framebuffer(OP, source)?;
        let (width, height, samples) = framebuffer_shape(&record.descriptor);
        let source_format = record
            .descriptor
            .color_attachments
            .first()
            .map(|view| view.format);
        if width != target.width
            || height != target.height
            || samples == 0
            || source_format != Some(target.color_format)
        {
            return Err(Self::validation(
                OP,
                "owned color framebuffer does not match the acquired default framebuffer",
            ));
        }
        // SAFETY: both bindings are valid for this owner context. FBO zero is
        // selected only after the acquired frame was authenticated by Phase A.
        unsafe {
            self.gl
                .bind_framebuffer(glow::READ_FRAMEBUFFER, Some(record.raw));
            self.gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, None);
            // Equal extents and NEAREST permit the native multisample resolve
            // case as well as an ordinary same-size copy.
            self.gl.blit_framebuffer(
                0,
                0,
                width as i32,
                height as i32,
                0,
                0,
                width as i32,
                height as i32,
                glow::COLOR_BUFFER_BIT,
                glow::NEAREST,
            );
            let result = self.driver_error(OP);
            self.gl.bind_framebuffer(glow::READ_FRAMEBUFFER, None);
            self.gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, None);
            result
        }
    }

    /// Attaches every validated view of a descriptor to one fresh FBO.
    pub(super) fn attach_all(
        &self,
        op: &'static str,
        raw: glow::NativeFramebuffer,
        descriptor: &GlFramebufferDescriptor,
    ) -> Result<(), GlError> {
        use glow::HasContext as _;
        // SAFETY: current-context contract; every attachment was validated.
        unsafe {
            self.gl.bind_framebuffer(glow::FRAMEBUFFER, Some(raw));
            for (index, view) in descriptor.color_attachments.iter().enumerate() {
                self.attach_one(op, draw_buffer_constant(index as u32), *view)?;
            }
            if let Some(view) = descriptor.depth_stencil_attachment {
                let point = depth_attachment_point(view.format).ok_or_else(|| {
                    Self::validation(op, "depth attachment format has no attachment point")
                })?;
                self.attach_one(op, point, view)?;
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
    fn attach_one(&self, op: &'static str, point: u32, view: GlTextureView) -> Result<(), GlError> {
        use glow::HasContext as _;
        match view.target {
            GlAttachmentTarget::Texture(texture) => {
                let (raw, desc) = self.texture(op, texture)?;
                // SAFETY: current-context contract; the framebuffer is bound and
                // the view was validated against this exact allocation.
                unsafe {
                    self.gl.framebuffer_texture_2d(
                        glow::FRAMEBUFFER,
                        point,
                        attachment_target(desc.sample_count),
                        Some(raw),
                        view.mip_level as i32,
                    );
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
                if view.array_layer != 0 || desc.dimension != GlTextureDimension::D2 {
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
                let facts = self.discovery.formats().get_for(
                    crate::backend::gl::api::GlFormatResourceKind::Texture,
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
                    crate::backend::gl::api::GlFormatResourceKind::Renderbuffer,
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
