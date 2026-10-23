//! Surface/depth attachment lowering for immediate GL raster passes.
//!
//! Default framebuffer color cannot share an FBO with a Core-owned depth
//! texture. The caller therefore retains the depth wrapper in encoder-local
//! pass state while native/browser owners render to a temporary color target.

use crate::backend::gl::backend::types::{Gl, GlTextureView};
use fluxel_rhi_core::api::{self, AttachmentView, Error, LoadOp, Result};

pub(super) fn validate_aspects(
    attachment: &api::DepthStencilAttachment<'_, Gl>,
    view: &GlTextureView,
) -> Result<()> {
    if (attachment.depth.is_some() && view.aspects.0 & api::TextureAspects::DEPTH.0 == 0)
        || (attachment.stencil.is_some() && view.aspects.0 & api::TextureAspects::STENCIL.0 == 0)
    {
        return Err(Error::InvalidRange);
    }
    Ok(())
}

pub(super) fn native_aspect(view: &GlTextureView) -> crate::backend::gl::api::GlTextureAspect {
    match view.aspects {
        api::TextureAspects::DEPTH => crate::backend::gl::api::GlTextureAspect::DepthOnly,
        api::TextureAspects::STENCIL => crate::backend::gl::api::GlTextureAspect::StencilOnly,
        _ => crate::backend::gl::api::GlTextureAspect::All,
    }
}

#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
pub(super) fn lower_native_view(
    attachment: Option<&api::DepthStencilAttachment<'_, Gl>>,
) -> Result<Option<GlTextureView>> {
    let Some(attachment) = attachment else {
        return Ok(None);
    };
    let AttachmentView::Texture(view) = attachment.view else {
        return Err(Error::Unsupported("GL surface depth attachment"));
    };
    // SAFETY: the returned wrapper stays in GlEncoder through FBO teardown.
    let view = unsafe { view.native_ref() }.clone();
    validate_aspects(attachment, &view)?;
    Ok(Some(view))
}

#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
pub(super) fn validate_native_surface_depth_view(
    view: &GlTextureView,
    target: crate::backend::gl::api::GlDefaultFramebufferTarget,
) -> Result<()> {
    // This check happens before allocating the temporary color attachment. The
    // provider repeats it while constructing the FBO, but rejecting here keeps
    // a caller mismatch from creating short-lived native objects first.
    if view.native.width != target.width
        || view.native.height != target.height
        || view.native.mip_level != 0
        || view.native.layer_count != 1
        || view.native.sample_count != 1
    {
        return Err(Error::InvalidRange);
    }
    Ok(())
}

#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
pub(super) fn lower_native_attachment(
    attachment: &api::DepthStencilAttachment<'_, Gl>,
    view: crate::backend::gl::api::GlTextureView,
) -> Result<crate::backend::gl::api::GlDepthStencilAttachment> {
    let (depth_load, depth_store, depth_clear) = match attachment.depth {
        Some((LoadOp::Clear(value), store))
            if value.is_finite() && (0.0..=1.0).contains(&value) =>
        {
            (
                crate::backend::gl::api::GlLoadOp::Clear,
                super::lower_store(store),
                value,
            )
        }
        Some((LoadOp::Clear(_), _)) => return Err(Error::InvalidRange),
        Some((_, store)) => (
            crate::backend::gl::api::GlLoadOp::Load,
            super::lower_store(store),
            0.0,
        ),
        None => (
            crate::backend::gl::api::GlLoadOp::Load,
            crate::backend::gl::api::GlStoreOp::Store,
            0.0,
        ),
    };
    let (stencil_load, stencil_store, stencil_clear) = match attachment.stencil {
        Some((LoadOp::Clear(value), store)) => (
            crate::backend::gl::api::GlLoadOp::Clear,
            super::lower_store(store),
            value,
        ),
        Some((_, store)) => (
            crate::backend::gl::api::GlLoadOp::Load,
            super::lower_store(store),
            0,
        ),
        None => (
            crate::backend::gl::api::GlLoadOp::Load,
            crate::backend::gl::api::GlStoreOp::Store,
            0,
        ),
    };
    Ok(crate::backend::gl::api::GlDepthStencilAttachment {
        view,
        depth_load,
        depth_store,
        stencil_load,
        stencil_store,
        clear: crate::backend::gl::api::GlDepthStencilClearValue {
            depth: depth_clear.to_bits(),
            stencil: stencil_clear,
        },
    })
}

#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
pub(super) fn begin_browser_surface_raster(
    encoder: &mut crate::backend::gl::backend::types::GlEncoder,
    target: crate::backend::gl::api::GlDefaultFramebufferTarget,
    color: &api::ColorAttachment<'_, Gl>,
    depth_attachment: Option<&api::DepthStencilAttachment<'_, Gl>>,
    render_area: api::Rect,
    timestamp_writes: Option<&api::PassTimestampWrites<'_, Gl>>,
) -> Result<()> {
    use crate::backend::gl::browser::BrowserDepthAttachment;
    super::render_area(render_area, [target.width, target.height])?;
    if let Some(attachment) = depth_attachment {
        let AttachmentView::Texture(view) = attachment.view else {
            return Err(Error::InvalidRange);
        };
        // SAFETY: the descriptor borrows the view throughout this call.
        let view = unsafe { view.native_ref() };
        validate_aspects(attachment, view)?;
        if view.layer_count != 1
            || [view.width, view.height] != [target.width, target.height]
            || view.sample_count != 1
        {
            return Err(Error::InvalidRange);
        }
    }
    let timestamps = super::super::query::prepare_pass_timestamps(encoder, timestamp_writes)?;
    super::super::query::start_pass_timestamps(encoder, timestamps)?;
    let browser = encoder.browser.as_mut().ok_or(Error::InvalidState)?;
    let result = if let Some(attachment) = depth_attachment {
        let AttachmentView::Texture(view) = attachment.view else {
            return Err(Error::InvalidRange);
        };
        // SAFETY: the pass descriptor borrows the view during this call and the browser retains the texture.
        let view = unsafe { view.native_ref() };
        if view.layer_count != 1
            || [view.width, view.height] != [target.width, target.height]
            || view.sample_count != 1
        {
            return Err(Error::InvalidRange);
        }
        browser.begin_surface_depth_raster(
            [target.width, target.height],
            target.color_format,
            color.load,
            BrowserDepthAttachment {
                texture: &view.texture.browser,
                aspect: native_aspect(view),
                mip_level: view.mip_level,
                layer: view.array_layer,
                depth: attachment.depth,
                stencil: attachment.stencil,
            },
            render_area,
        )
    } else {
        let clear = match color.load {
            LoadOp::Clear(value) => Some(value),
            LoadOp::Load | LoadOp::Discard => None,
        };
        browser.begin_default_raster([target.width, target.height], clear, render_area)
    };
    if let Err(error) = result {
        encoder.indeterminate = true;
        return Err(error);
    }
    Ok(())
}
