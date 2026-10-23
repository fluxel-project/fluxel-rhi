//! Native OVR multiview immediate raster-pass construction.
//!
//! The selected Core mask is a consecutive interval. Each borrowed view may
//! start at a different array layer; adding the mask's first bit selects the
//! corresponding interval in that view. The encoder owns the temporary FBO.

use super::super::*;
use crate::backend::gl::api as gl;
use crate::backend::gl::api::GlFramebufferApi as _;

fn selected(mut view: gl::GlTextureView, first: u32, count: u32) -> Result<gl::GlTextureView> {
    if count == 0
        || first
            .checked_add(count)
            .is_none_or(|end| end > view.layer_count)
    {
        return Err(Error::InvalidRange);
    }
    view.array_layer = view
        .array_layer
        .checked_add(first)
        .ok_or(Error::InvalidRange)?;
    view.layer_count = count;
    Ok(view)
}

pub(super) fn begin(
    encoder: &mut GlEncoder,
    desc: &RasterPassDesc<'_, Gl>,
    first: u32,
    count: u32,
) -> Result<()> {
    // OVR_multiview leaves timer query results undefined while a multiview
    // draw framebuffer participates in the query interval.
    if desc.timestamp_writes.is_some() {
        return Err(Error::Unsupported("GL multiview timestamp queries"));
    }
    let mut extent = None;
    let mut samples = None;
    let mut views = Vec::with_capacity(desc.colors.len());
    let mut colors = Vec::with_capacity(desc.colors.len());
    for color in desc.colors {
        let Some(color) = color else {
            views.push(None);
            colors.push(None);
            continue;
        };
        if color.resolve_target.is_some() {
            return Err(Error::Unsupported("GL multiview resolve"));
        }
        let AttachmentView::Texture(view) = color.view else {
            return Err(Error::Unsupported("GL multiview surface attachment"));
        };
        // SAFETY: the descriptor borrows the view during this immediate call;
        // the encoder retains the attachment wrappers for the active pass.
        let view = selected(unsafe { view.native_ref() }.native, first, count)?;
        let size = [view.width, view.height];
        if extent.is_some_and(|old| old != size)
            || samples.is_some_and(|old| old != view.sample_count)
        {
            return Err(Error::InvalidRange);
        }
        extent = Some(size);
        samples = Some(view.sample_count);
        views.push(Some(view));
        colors.push(Some(gl::GlColorAttachment {
            view: gl::GlPassAttachmentView::Allocated(view),
            resolve_target: None,
            load: lower_load(color.load)?,
            store: lower_store(color.store),
            clear: lower_clear(color.load)?,
        }));
    }
    let depth_view = surface_depth::lower_native_view(desc.depth_stencil.as_ref())?;
    let depth_native = match &depth_view {
        Some(view) => {
            let view = selected(view.native, first, count)?;
            if extent.is_some_and(|old| old != [view.width, view.height])
                || samples.is_some_and(|old| old != view.sample_count)
            {
                return Err(Error::InvalidRange);
            }
            extent = Some([view.width, view.height]);
            Some(view)
        }
        None => None,
    };
    let depth = match (desc.depth_stencil.as_ref(), depth_native) {
        (Some(attachment), Some(view)) => {
            Some(surface_depth::lower_native_attachment(attachment, view)?)
        }
        (None, None) => None,
        _ => return Err(Error::InvalidState),
    };
    let extent = extent.ok_or(Error::InvalidRange)?;
    let render_area = render_area(desc.render_area, extent)?;
    let draw_buffers = views
        .iter()
        .enumerate()
        .map(|(slot, view)| {
            if view.is_some() {
                slot as u32
            } else {
                u32::MAX
            }
        })
        .collect();
    let timestamps = super::super::super::query::prepare_pass_timestamps(
        encoder,
        desc.timestamp_writes.as_ref(),
    )?;
    super::super::super::query::start_pass_timestamps(encoder, timestamps)?;
    let result = encoder
        .state
        .with_native("begin GL multiview raster pass", |provider| {
            let framebuffer =
                provider.create_multiview_framebuffer(&gl::GlFramebufferDescriptor {
                    color_attachments: views,
                    depth_stencil_attachment: depth_native,
                    draw_buffers,
                })?;
            let pass = gl::GlRenderPassDescriptor {
                render_area,
                target: gl::GlRenderTarget::Offscreen(framebuffer),
                color_attachments: colors,
                depth_stencil_attachment: depth,
            };
            if let Err(error) = provider.begin_render_pass(&pass) {
                let _ = provider.destroy_framebuffer(framebuffer);
                return Err(error);
            }
            Ok(framebuffer)
        })
        .map_err(native_error);
    match result {
        Ok(framebuffer) => {
            encoder.offscreen_framebuffer = Some(framebuffer);
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
