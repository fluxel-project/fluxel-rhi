//! Immediate offscreen framebuffer creation for native GL and WebGL2.
//!
//! Helpers validate borrowed attachment views, issue pass begin directly, and
//! leave only the active pass lifetime on the encoder.

use super::*;

/// Attaches the caller's single-sample WebGL textures to a short-lived FBO and
/// begins the pass immediately.  The FBO is encoder-owned; texture wrappers are
/// retained by the browser encoder only until matching end/abort cleanup.
#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
pub(super) fn begin_browser_offscreen_raster(
    encoder: &mut GlEncoder,
    desc: &RasterPassDesc<'_, Gl>,
    color: &api::ColorAttachment<'_, Gl>,
) -> Result<()> {
    let AttachmentView::Texture(color_view) = color.view else {
        return Err(Error::Unsupported("WebGL2 offscreen color attachment view"));
    };
    if color.resolve_target.is_some() {
        return Err(Error::Unsupported("WebGL2 offscreen resolve target"));
    }
    // SAFETY: Core holds the view borrow for this call and the encoder clones
    // the browser wrapper into its active-pass lifetime record before return.
    let color_view = unsafe { color_view.native_ref() }.clone();
    let color_texture = &color_view.texture.browser;
    let extent = [
        color_view.texture.desc.extent.width,
        color_view.texture.desc.extent.height,
    ];
    let clear = match color.load {
        LoadOp::Clear(value) => Some(value),
        LoadOp::Load => None,
        LoadOp::Discard => return Err(Error::Unsupported("WebGL2 offscreen color load discard")),
    };
    if matches!(
        clear,
        Some(api::ClearColor::Uint(_) | api::ClearColor::Sint(_))
    ) {
        return Err(Error::Unsupported("WebGL2 integer offscreen clear"));
    }
    let depth = lower_browser_view(desc.depth_stencil.as_ref())?;
    let depth_texture = depth.as_ref().map(|view| &view.texture.browser);
    let depth_clear = desc
        .depth_stencil
        .as_ref()
        .and_then(|attachment| attachment.depth)
        .map(|(load, _)| match load {
            LoadOp::Clear(value) if value.is_finite() && (0.0..=1.0).contains(&value) => {
                Ok(Some(value))
            }
            LoadOp::Load => Ok(None),
            LoadOp::Clear(_) => Err(Error::InvalidRange),
            LoadOp::Discard => Err(Error::Unsupported("WebGL2 offscreen depth load discard")),
        })
        .transpose()?
        .flatten();
    let browser = encoder.browser.as_mut().ok_or(Error::InvalidState)?;
    if let Err(error) =
        browser.begin_offscreen_raster(color_texture, depth_texture, extent, clear, depth_clear)
    {
        // The browser may have bound or cleared an FBO before it reports an
        // error. This immediate prefix cannot safely be treated as a rejection.
        encoder.indeterminate = true;
        return Err(error);
    }
    encoder.begin_raster_state(extent);
    encoder.raster_pipeline = None;
    encoder.vertex_bindings.clear();
    encoder.index_binding = None;
    encoder.issued = true;
    encoder.occlusion_query_set = desc.occlusion_query_set.map(|set| {
        // SAFETY: the browser encoder retains the cloned query-set owner until
        // the matching immediate end; Core keeps the descriptor borrowed here.
        unsafe { set.native_ref() }.clone()
    });
    Ok(())
}

/// Attaches the caller's texture views to a temporary FBO and begins the pass
/// now. The FBO name is retired by `end_raster`/`abort_raster`; no pass packet
/// or texture ownership is retained beyond the active GL pass.
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
pub(super) fn begin_native_offscreen_raster(
    encoder: &mut GlEncoder,
    desc: &RasterPassDesc<'_, Gl>,
    color: &api::ColorAttachment<'_, Gl>,
) -> Result<()> {
    use crate::backend::gl::api::GlFramebufferApi as _;

    let AttachmentView::Texture(color_view) = color.view else {
        return Err(Error::Unsupported("GL offscreen color attachment view"));
    };
    // SAFETY: begin_raster borrows the view for validation and copies only its
    // immutable native attachment identity; GL commands execute synchronously.
    let color_view = unsafe { color_view.native_ref() }.native;
    if color_view.layer_count != 1 {
        return Err(Error::Unsupported(
            "GL offscreen attachment sample/layer count",
        ));
    }
    let resolve = match color.resolve_target.as_ref() {
        None if color_view.sample_count == 1 => None,
        Some(AttachmentView::Surface(surface)) if color_view.sample_count > 1 => {
            // SAFETY: the acquired image is borrowed for the pass; the token
            // lets end_raster reject publication after reconfigure/abandon.
            let native = unsafe { surface.native_ref() };
            let target = native.default_target()?;
            if target.width != color_view.width
                || target.height != color_view.height
                || target.color_format != color_view.format
            {
                return Err(Error::InvalidRange);
            }
            Some(crate::backend::gl::backend::types::GlOffscreenResolve {
                target,
                surface_token: native.target_token(),
            })
        }
        Some(AttachmentView::Texture(_)) => {
            return Err(Error::Unsupported("GL texture resolve target"));
        }
        Some(AttachmentView::Surface(_)) => {
            return Err(Error::Unsupported(
                "GL surface resolve requires multisampling",
            ));
        }
        None => return Err(Error::Unsupported("GL multisample resolve target")),
    };
    if resolve.is_some() && !matches!(color.store, api::StoreOp::Discard) {
        // The GL adapter currently implements the surface resolve as one
        // framebuffer blit. Preserving both source and resolved output is
        // outside this example's adopted Core usage.
        return Err(Error::Unsupported("GL multisample resolve with Store"));
    }
    let depth_view = surface_depth::lower_native_view(desc.depth_stencil.as_ref())?;
    if depth_view.as_ref().is_some_and(|depth| {
        depth.native.width != color_view.width
            || depth.native.height != color_view.height
            || depth.native.sample_count != color_view.sample_count
            || depth.native.layer_count != color_view.layer_count
    }) {
        return Err(Error::InvalidRange);
    }
    let depth_attachment = match (&depth_view, desc.depth_stencil.as_ref()) {
        (Some(view), Some(attachment)) => Some(surface_depth::lower_native_attachment(
            attachment,
            view.native,
        )?),
        (None, None) => None,
        _ => return Err(Error::InvalidState),
    };
    let color_attachments = vec![crate::backend::gl::api::GlColorAttachment {
        view: crate::backend::gl::api::GlPassAttachmentView::Allocated(color_view),
        resolve_target: None,
        load: lower_load(color.load)?,
        // The source samples must survive EndRenderPass until the explicit
        // GL framebuffer blit resolves them into the acquired default target.
        store: if resolve.is_some() {
            crate::backend::gl::api::GlStoreOp::Store
        } else {
            lower_store(color.store)
        },
        clear: lower_clear(color.load)?,
    }];
    let framebuffer_desc = crate::backend::gl::api::GlFramebufferDescriptor {
        color_attachments: vec![color_view],
        depth_stencil_attachment: depth_view.map(|view| view.native),
        draw_buffers: vec![0],
    };
    let result = encoder
        .state
        .with_native("begin GL offscreen raster pass", |provider| {
            let framebuffer = provider.create_framebuffer(&framebuffer_desc)?;
            let descriptor = crate::backend::gl::api::GlRenderPassDescriptor {
                target: crate::backend::gl::api::GlRenderTarget::Offscreen(framebuffer),
                color_attachments,
                depth_stencil_attachment: depth_attachment,
            };
            if let Err(error) = provider.begin_render_pass(&descriptor) {
                let _ = provider.destroy_framebuffer(framebuffer);
                return Err(error);
            }
            Ok(framebuffer)
        })
        .map_err(native_error);
    match result {
        Ok(framebuffer) => {
            encoder.offscreen_framebuffer = Some(framebuffer);
            encoder.offscreen_resolve = resolve;
            encoder.begin_raster_state([color_view.width, color_view.height]);
            encoder.raster_pipeline = None;
            encoder.vertex_bindings.clear();
            encoder.index_binding = None;
            encoder.issued = true;
            encoder.occlusion_query_set = desc.occlusion_query_set.map(|set| {
                // SAFETY: Core keeps the pass descriptor and query set alive
                // while this helper immediately starts the native FBO pass.
                unsafe { set.native_ref() }.clone()
            });
            Ok(())
        }
        Err(error) => {
            encoder.indeterminate = true;
            Err(error)
        }
    }
}
