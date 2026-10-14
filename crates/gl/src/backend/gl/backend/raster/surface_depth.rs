//! Surface/depth attachment lowering for immediate GL raster passes.
//!
//! Default framebuffer color cannot share an FBO with a Core-owned depth
//! texture. The caller therefore retains the depth wrapper in encoder-local
//! pass state while native/browser owners render to a temporary color target.

use crate::backend::gl::backend::types::{Gl, GlTextureView};
use fluxel_rhi_core::api::{self, AttachmentView, ClearColor, Error, LoadOp, Result};

#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
pub(super) fn lower_native_view(
    attachment: Option<&api::DepthStencilAttachment<'_, Gl>>,
) -> Result<Option<GlTextureView>> {
    let Some(attachment) = attachment else {
        return Ok(None);
    };
    if attachment.depth_read_only || attachment.stencil_read_only || attachment.stencil.is_some() {
        return Err(Error::Unsupported("GL surface depth-stencil mode"));
    }
    let AttachmentView::Texture(view) = attachment.view else {
        return Err(Error::Unsupported("GL surface depth attachment"));
    };
    // SAFETY: the returned wrapper stays in GlEncoder through FBO teardown.
    let view = unsafe { view.native_ref() }.clone();
    if view.native.format != crate::backend::gl::api::GlFormat::Depth32Float {
        return Err(Error::Unsupported("GL Depth32Float attachment"));
    }
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
    let Some((load, store)) = attachment.depth else {
        return Err(Error::Unsupported("GL stencil-only surface pass"));
    };
    if attachment.stencil.is_some() || attachment.depth_read_only || attachment.stencil_read_only {
        return Err(Error::Unsupported("GL surface depth-stencil mode"));
    }
    let clear = match load {
        LoadOp::Clear(value) if value.is_finite() && (0.0..=1.0).contains(&value) => value,
        LoadOp::Clear(_) => return Err(Error::InvalidRange),
        LoadOp::Load => 0.0,
        LoadOp::Discard => return Err(Error::Unsupported("GL depth load discard")),
    };
    Ok(crate::backend::gl::api::GlDepthStencilAttachment {
        view,
        depth_load: match load {
            LoadOp::Load => crate::backend::gl::api::GlLoadOp::Load,
            LoadOp::Clear(_) => crate::backend::gl::api::GlLoadOp::Clear,
            LoadOp::Discard => unreachable!(),
        },
        depth_store: super::lower_store(store),
        stencil_load: crate::backend::gl::api::GlLoadOp::Load,
        stencil_store: crate::backend::gl::api::GlStoreOp::Store,
        clear: crate::backend::gl::api::GlDepthStencilClearValue {
            depth: clear.to_bits(),
            stencil: 0,
        },
    })
}

#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
pub(super) fn lower_browser_view(
    attachment: Option<&api::DepthStencilAttachment<'_, Gl>>,
) -> Result<Option<GlTextureView>> {
    let Some(attachment) = attachment else {
        return Ok(None);
    };
    if attachment.depth_read_only
        || attachment.stencil_read_only
        || attachment.stencil.is_some()
        || attachment.depth.is_none()
    {
        return Err(Error::Unsupported("WebGL2 surface depth-stencil mode"));
    }
    let AttachmentView::Texture(view) = attachment.view else {
        return Err(Error::Unsupported("WebGL2 surface depth attachment"));
    };
    // SAFETY: the encoder owns this clone until blit and FBO deletion.
    let view = unsafe { view.native_ref() }.clone();
    if !matches!(
        view.texture.desc.format,
        crate::backend::gl::api::GlFormat::Depth32Float
    ) || view.texture.desc.extent.depth_or_layers != 1
        || view.texture.desc.mip_level_count != 1
    {
        return Err(Error::Unsupported("WebGL2 Depth32Float surface attachment"));
    }
    Ok(Some(view))
}

#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
pub(super) fn begin_browser_surface_raster(
    encoder: &mut crate::backend::gl::backend::types::GlEncoder,
    target: crate::backend::gl::api::GlDefaultFramebufferTarget,
    color: &api::ColorAttachment<'_, Gl>,
    depth_attachment: Option<&api::DepthStencilAttachment<'_, Gl>>,
) -> Result<()> {
    let clear = match color.load {
        LoadOp::Clear(value) => Some(value),
        LoadOp::Load => None,
        LoadOp::Discard => return Err(Error::Unsupported("GL color load discard")),
    };
    let browser = encoder.browser.as_mut().ok_or(Error::InvalidState)?;
    let result = if let Some(depth) = lower_browser_view(depth_attachment)? {
        let ClearColor::Float(color_clear) =
            clear.ok_or(Error::Unsupported("WebGL2 surface depth color load"))?
        else {
            return Err(Error::Unsupported("WebGL2 integer surface clear"));
        };
        let depth_clear = depth_attachment
            .and_then(|attachment| attachment.depth)
            .map(|(load, _)| match load {
                LoadOp::Clear(value) if value.is_finite() && (0.0..=1.0).contains(&value) => {
                    Ok(Some(value))
                }
                LoadOp::Load => Ok(None),
                LoadOp::Clear(_) => Err(Error::InvalidRange),
                LoadOp::Discard => Err(Error::Unsupported("WebGL2 depth load discard")),
            })
            .transpose()?
            .flatten();
        browser.begin_surface_depth_raster(
            [target.width, target.height],
            ClearColor::Float(color_clear),
            depth_clear,
            &depth.texture.browser,
        )
    } else {
        browser.begin_default_raster([target.width, target.height], clear)
    };
    if let Err(error) = result {
        // The browser call may have bound/cleared framebuffer state before the
        // driver reports an error, so this immediate prefix is not reusable.
        encoder.indeterminate = true;
        return Err(error);
    }
    Ok(())
}
