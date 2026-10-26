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
) -> Result<()> {
    use crate::backend::browser::{
        BrowserColorAttachment, BrowserDepthAttachment, BrowserResolveAttachment,
    };
    let mut colors = Vec::with_capacity(desc.colors.len());
    let mut extent = None;
    let mut samples = None;
    let mut has_surface = false;
    let mut surface_token = None;
    for color in desc.colors {
        let Some(color) = color else {
            colors.push(None);
            continue;
        };
        let AttachmentView::Texture(view) = color.view else {
            let AttachmentView::Surface(surface) = &color.view else {
                unreachable!()
            };
            if has_surface || color.resolve_target.is_some() {
                return Err(Error::InvalidRange);
            }
            // SAFETY: the borrowed image lease is checked before allocating browser objects.
            let image = unsafe { surface.native_ref() };
            let target = image.default_target()?;
            surface_token = Some(image.target_token());
            let size = [target.width, target.height];
            if extent.is_some_and(|v| v != size) || samples.is_some_and(|v| v != 1) {
                return Err(Error::InvalidRange);
            }
            extent = Some(size);
            samples = Some(1);
            has_surface = true;
            colors.push(Some(BrowserColorAttachment {
                texture: None,
                surface_format: Some(target.color_format),
                mip_level: 0,
                layer: 0,
                load: color.load,
                store: color.store,
                resolve: None,
            }));
            continue;
        };
        // SAFETY: Core keeps each view borrowed while the immediate pass begins.
        let view = unsafe { view.native_ref() };
        if view.layer_count != 1
            || extent.is_some_and(|v| v != [view.width, view.height])
            || samples.is_some_and(|v| v != view.sample_count)
        {
            return Err(Error::InvalidRange);
        }
        extent = Some([view.width, view.height]);
        samples = Some(view.sample_count);
        let resolve = match &color.resolve_target {
            None => None,
            Some(AttachmentView::Texture(target)) => {
                // SAFETY: Core borrows the resolve view until this begin call completes.
                let target = unsafe { target.native_ref() };
                if view.sample_count <= 1
                    || target.sample_count != 1
                    || target.layer_count != 1
                    || [target.width, target.height] != [view.width, view.height]
                    || target.texture.desc.format != view.texture.desc.format
                {
                    return Err(Error::InvalidRange);
                }
                Some(BrowserResolveAttachment {
                    texture: Some(&target.texture.browser),
                    mip_level: target.mip_level,
                    layer: target.array_layer,
                })
            }
            Some(AttachmentView::Surface(surface)) => {
                if has_surface || view.sample_count <= 1 {
                    return Err(Error::InvalidRange);
                }
                // SAFETY: Core borrows the surface lease for this immediate call.
                let image = unsafe { surface.native_ref() };
                let target = image.default_target()?;
                if [target.width, target.height] != [view.width, view.height]
                    || target.color_format != view.texture.desc.format
                {
                    return Err(Error::InvalidRange);
                }
                has_surface = true;
                surface_token = Some(image.target_token());
                Some(BrowserResolveAttachment {
                    texture: None,
                    mip_level: 0,
                    layer: 0,
                })
            }
        };
        colors.push(Some(BrowserColorAttachment {
            texture: Some(&view.texture.browser),
            surface_format: None,
            mip_level: view.mip_level,
            layer: view.array_layer,
            load: color.load,
            store: color.store,
            resolve,
        }));
    }
    let depth = if let Some(attachment) = &desc.depth_stencil {
        let AttachmentView::Texture(view) = attachment.view else {
            return Err(Error::InvalidRange);
        };
        // SAFETY: the descriptor retains this view during the call.
        let view = unsafe { view.native_ref() };
        super::surface_depth::validate_aspects(attachment, view)?;
        if view.layer_count != 1
            || extent.is_some_and(|v| v != [view.width, view.height])
            || samples.is_some_and(|v| v != view.sample_count)
        {
            return Err(Error::InvalidRange);
        }
        extent = Some([view.width, view.height]);
        Some(BrowserDepthAttachment {
            aspect: super::surface_depth::native_aspect(view),
            texture: &view.texture.browser,
            mip_level: view.mip_level,
            layer: view.array_layer,
            depth: attachment.depth,
            stencil: attachment.stencil,
        })
    } else {
        None
    };
    let extent = extent.ok_or(Error::InvalidRange)?;
    render_area(desc.render_area, extent)?;
    let timestamps =
        super::super::query::prepare_pass_timestamps(encoder, desc.timestamp_writes.as_ref())?;
    super::super::query::start_pass_timestamps(encoder, timestamps)?;
    let result = encoder
        .browser
        .as_mut()
        .ok_or(Error::InvalidState)?
        .begin_offscreen_attachments(&colors, depth, extent, desc.render_area);
    if let Err(error) = result {
        encoder.indeterminate = true;
        return Err(error);
    }
    encoder.browser_surface_token = surface_token;
    encoder.begin_raster_state(extent);
    encoder.set_raster_read_only(desc);
    encoder.raster_pipeline = None;
    encoder.vertex_bindings.clear();
    encoder.index_binding = None;
    encoder.issued = true;
    encoder.occlusion_query_set = desc
        .occlusion_query_set
        .map(|set| unsafe { set.native_ref() }.clone());
    Ok(())
}

#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
pub(super) fn begin_native_offscreen_raster(
    encoder: &mut GlEncoder,
    desc: &RasterPassDesc<'_, Gl>,
) -> Result<()> {
    use crate::backend::api as gl;
    use crate::backend::api::{GlFramebufferApi as _, GlResourceApi as _};
    let mut extent = None;
    let mut samples = None;
    let mut views = Vec::with_capacity(desc.colors.len());
    let mut colors = Vec::with_capacity(desc.colors.len());
    let mut resolve = None;
    let mut texture_resolves = Vec::new();
    let mut surface_color = None;
    for (slot, color) in desc.colors.iter().enumerate() {
        let Some(color) = color else {
            views.push(None);
            colors.push(None);
            continue;
        };
        let AttachmentView::Texture(view) = color.view else {
            let AttachmentView::Surface(surface) = &color.view else {
                unreachable!()
            };
            if surface_color.is_some() || resolve.is_some() || color.resolve_target.is_some() {
                return Err(Error::InvalidRange);
            }
            // SAFETY: Core borrows the surface image for this call; its lease token is retained.
            let surface = unsafe { surface.native_ref() };
            let target = surface.default_target()?;
            let size = [target.width, target.height];
            if extent.is_some_and(|old| old != size) || samples.is_some_and(|old| old != 1) {
                return Err(Error::InvalidRange);
            }
            extent = Some(size);
            samples = Some(1);
            surface_color = Some((
                slot,
                target,
                surface.target_token(),
                color.load,
                color.store,
            ));
            views.push(None);
            colors.push(None);
            continue;
        };
        // SAFETY: the public descriptor borrows the view for this immediate call.
        let view = unsafe { view.native_ref() }.native;
        if view.layer_count != 1 {
            return Err(Error::InvalidRange);
        }
        let size = [view.width, view.height];
        if extent.is_some_and(|old| old != size)
            || samples.is_some_and(|old| old != view.sample_count)
        {
            return Err(Error::InvalidRange);
        }
        extent = Some(size);
        samples = Some(view.sample_count);
        if let Some(target) = &color.resolve_target {
            if let AttachmentView::Texture(target) = target {
                // SAFETY: both attachment views are borrowed by Core for this pass.
                let target = unsafe { target.native_ref() }.native;
                if view.sample_count <= 1
                    || target.sample_count != 1
                    || target.layer_count != 1
                    || target.format != view.format
                    || [target.width, target.height] != size
                {
                    return Err(Error::InvalidRange);
                }
                texture_resolves.push((view, target, None));
            } else {
                let AttachmentView::Surface(surface) = target else {
                    unreachable!();
                };
                if surface_color.is_some() || resolve.is_some() || view.sample_count <= 1 {
                    return Err(Error::Unsupported("GL surface resolve attachment"));
                }
                // SAFETY: the surface borrow remains valid through this call and its token is retained.
                let surface = unsafe { surface.native_ref() };
                let target = surface.default_target()?;
                if [target.width, target.height] != size || target.color_format != view.format {
                    return Err(Error::InvalidRange);
                }
                resolve = Some(super::super::types::GlOffscreenResolve {
                    source: view,
                    target,
                    surface_token: surface.target_token(),
                    render_area: None,
                });
            }
        }
        views.push(Some(view));
        colors.push(Some(gl::GlColorAttachment {
            view: gl::GlPassAttachmentView::Allocated(view),
            resolve_target: None,
            load: lower_load(color.load)?,
            store: if color.resolve_target.is_some() {
                gl::GlStoreOp::Store
            } else {
                lower_store(color.store)
            },
            clear: lower_clear(color.load)?,
        }));
    }
    let depth_view = surface_depth::lower_native_view(desc.depth_stencil.as_ref())?;
    if let Some(depth) = &depth_view {
        let view = depth.native;
        if view.layer_count != 1
            || extent.is_some_and(|size| size != [view.width, view.height])
            || samples.is_some_and(|count| count != view.sample_count)
        {
            return Err(Error::InvalidRange);
        }
        extent = Some([view.width, view.height]);
    }
    let extent = extent.ok_or(Error::InvalidRange)?;
    let render_area = render_area(desc.render_area, extent)?;
    for (_, _, area) in &mut texture_resolves {
        *area = render_area;
    }
    if let Some(resolve) = &mut resolve {
        resolve.render_area = render_area;
    }
    let depth = match (&depth_view, desc.depth_stencil.as_ref()) {
        (Some(view), Some(attachment)) => Some(surface_depth::lower_native_attachment(
            attachment,
            view.native,
        )?),
        (None, None) => None,
        _ => return Err(Error::InvalidState),
    };
    let draw_buffers = views
        .iter()
        .enumerate()
        .map(|(i, view)| {
            if view.is_some() || surface_color.as_ref().is_some_and(|surface| surface.0 == i) {
                i as u32
            } else {
                u32::MAX
            }
        })
        .collect();
    let depth_native = depth_view.as_ref().map(|view| view.native);
    let timestamps =
        super::super::query::prepare_pass_timestamps(encoder, desc.timestamp_writes.as_ref())?;
    super::super::query::start_pass_timestamps(encoder, timestamps)?;
    let result = encoder
        .state
        .with_native("begin GL offscreen raster pass", |provider| {
            let mut surface_resource = None;
            if let Some((slot, target, _, load, _store)) = &surface_color {
                let texture = provider.create_texture_resource(gl::GlTextureDesc {
                    dimension: gl::GlTextureDimension::D2,
                    extent: gl::GlExtent3d {
                        width: target.width,
                        height: target.height,
                        depth_or_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    format: target.color_format,
                    usage: gl::GlTextureUsage::RENDER_ATTACHMENT,
                })?;
                let view = gl::GlTextureView {
                    aspect: gl::GlTextureAspect::All,
                    target: gl::GlAttachmentTarget::Texture(texture),
                    format: target.color_format,
                    mip_level: 0,
                    array_layer: 0,
                    layer_count: 1,
                    width: target.width,
                    height: target.height,
                    sample_count: 1,
                };
                views[*slot] = Some(view);
                colors[*slot] = Some(gl::GlColorAttachment {
                    view: gl::GlPassAttachmentView::Allocated(view),
                    resolve_target: None,
                    load: lower_load(*load).expect("validated color load"),
                    store: gl::GlStoreOp::Store,
                    clear: lower_clear(*load).expect("validated color clear"),
                });
                surface_resource = Some(texture);
            }
            let framebuffer_desc = gl::GlFramebufferDescriptor {
                color_attachments: views,
                depth_stencil_attachment: depth_native,
                draw_buffers,
            };
            let framebuffer = match provider.create_framebuffer(&framebuffer_desc) {
                Ok(value) => value,
                Err(error) => {
                    if let Some(texture) = surface_resource {
                        let _ = provider.destroy_texture_resource(texture);
                    }
                    return Err(error);
                }
            };
            if let Some((slot, target, _, load, _)) = &surface_color {
                if matches!(load, LoadOp::Load) || render_area.is_some() {
                    if let Err(error) = provider.blit_default_to_framebuffer_slot(
                        *target,
                        framebuffer,
                        *slot as u32,
                    ) {
                        let _ = provider.destroy_framebuffer(framebuffer);
                        if let Some(texture) = surface_resource {
                            let _ = provider.destroy_texture_resource(texture);
                        }
                        return Err(error);
                    }
                }
            }
            let descriptor = gl::GlRenderPassDescriptor {
                render_area,
                target: gl::GlRenderTarget::Offscreen(framebuffer),
                color_attachments: colors,
                depth_stencil_attachment: depth,
            };
            if let Err(error) = provider.begin_render_pass(&descriptor) {
                let _ = provider.destroy_framebuffer(framebuffer);
                if let Some(texture) = surface_resource {
                    let _ = provider.destroy_texture_resource(texture);
                }
                return Err(error);
            }
            Ok((framebuffer, surface_resource))
        })
        .map_err(native_error);
    match result {
        Ok((framebuffer, surface_resource)) => {
            if let (Some((slot, target, token, _, _)), Some(color)) =
                (surface_color, surface_resource)
            {
                encoder.surface_depth_pass = Some(super::super::types::GlSurfaceDepthPass {
                    target,
                    surface_token: token,
                    framebuffer,
                    color,
                    depth: depth_view,
                    color_slot: slot as u32,
                    render_area,
                });
            } else {
                encoder.offscreen_framebuffer = Some(framebuffer);
            }
            encoder.offscreen_resolve = resolve;
            encoder.native_texture_resolves = texture_resolves;
            encoder.begin_raster_state(extent);
            encoder.set_raster_read_only(desc);
            encoder.raster_pipeline = None;
            encoder.vertex_bindings.clear();
            encoder.index_binding = None;
            encoder.issued = true;
            encoder.occlusion_query_set = desc
                .occlusion_query_set
                .map(|set| unsafe { set.native_ref() }.clone());
            Ok(())
        }
        Err(error) => {
            encoder.indeterminate = true;
            Err(error)
        }
    }
}
