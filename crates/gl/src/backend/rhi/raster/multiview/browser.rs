//! Lowers a contiguous Core multiview pass to WebGL array attachments.
//!
//! The selected view index is relative to the supplied texture view. The
//! browser boundary checks the WebGL-specific common base-layer rule.
use super::super::*;
use crate::backend::browser::{BrowserColorAttachment, BrowserDepthAttachment};

pub(super) fn begin(
    encoder: &mut GlEncoder,
    desc: &RasterPassDesc<'_, Gl>,
    first: u32,
    count: u32,
) -> Result<()> {
    let end = first.checked_add(count).ok_or(Error::InvalidRange)?;
    if count == 0 {
        return Err(Error::InvalidRange);
    }
    let mut extent = None;
    let mut common_base = None;
    let mut colors = Vec::with_capacity(desc.colors.len());
    for color in desc.colors {
        let Some(color) = color else {
            colors.push(None);
            continue;
        };
        if color.resolve_target.is_some() {
            return Err(Error::Unsupported("WebGL2 multiview resolve"));
        }
        let AttachmentView::Texture(view) = color.view else {
            return Err(Error::Unsupported("WebGL2 multiview surface"));
        };
        // SAFETY: Core borrows every view for the duration of this begin call.
        let view = unsafe { view.native_ref() };
        let size = [view.width, view.height];
        if view.texture.desc.dimension != crate::backend::api::GlTextureDimension::D2Array {
            return Err(Error::Unsupported(
                "WebGL2 multiview requires array textures",
            ));
        }
        if view.sample_count != 1 {
            return Err(Error::Unsupported("WebGL2 multisampled multiview"));
        }
        if view.layer_count < end || extent.is_some_and(|v| v != size) {
            return Err(Error::InvalidRange);
        }
        let layer = view
            .array_layer
            .checked_add(first)
            .ok_or(Error::InvalidRange)?;
        if common_base.is_some_and(|base| base != layer) {
            return Err(Error::Unsupported(
                "WebGL2 multiview attachment base layers must match",
            ));
        }
        common_base = Some(layer);
        extent = Some(size);
        colors.push(Some(BrowserColorAttachment {
            texture: Some(&view.texture.browser),
            surface_format: None,
            mip_level: view.mip_level,
            layer,
            load: color.load,
            store: color.store,
            resolve: None,
        }));
    }
    let depth = if let Some(attachment) = &desc.depth_stencil {
        let AttachmentView::Texture(view) = attachment.view else {
            return Err(Error::Unsupported("WebGL2 multiview surface depth"));
        };
        // SAFETY: Core borrows the depth/stencil view for this begin call.
        let view = unsafe { view.native_ref() };
        super::super::surface_depth::validate_aspects(attachment, view)?;
        let size = [view.width, view.height];
        if view.texture.desc.dimension != crate::backend::api::GlTextureDimension::D2Array {
            return Err(Error::Unsupported(
                "WebGL2 multiview requires array textures",
            ));
        }
        if view.sample_count != 1 {
            return Err(Error::Unsupported("WebGL2 multisampled multiview"));
        }
        if view.layer_count < end || extent.is_some_and(|v| v != size) {
            return Err(Error::InvalidRange);
        }
        let layer = view
            .array_layer
            .checked_add(first)
            .ok_or(Error::InvalidRange)?;
        if common_base.is_some_and(|base| base != layer) {
            return Err(Error::Unsupported(
                "WebGL2 multiview attachment base layers must match",
            ));
        }
        extent = Some(size);
        Some(BrowserDepthAttachment {
            texture: &view.texture.browser,
            aspect: super::super::surface_depth::native_aspect(view),
            mip_level: view.mip_level,
            layer,
            depth: attachment.depth,
            stencil: attachment.stencil,
        })
    } else {
        None
    };
    let extent = extent.ok_or(Error::InvalidRange)?;
    render_area(desc.render_area, extent)?;
    let timestamps = super::super::super::query::prepare_pass_timestamps(
        encoder,
        desc.timestamp_writes.as_ref(),
    )?;
    super::super::super::query::start_pass_timestamps(encoder, timestamps)?;
    let result = encoder
        .browser
        .as_mut()
        .ok_or(Error::InvalidState)?
        .begin_multiview_attachments(&colors, depth, extent, desc.render_area, count);
    if let Err(error) = result {
        encoder.indeterminate = true;
        return Err(error);
    }
    encoder.begin_raster_state(extent);
    encoder.set_raster_read_only(desc);
    encoder.raster_pipeline = None;
    encoder.vertex_bindings.clear();
    encoder.index_binding = None;
    encoder.issued = true;
    encoder.occlusion_query_set = desc.occlusion_query_set.map(|set| {
        // SAFETY: Core borrows the query set until this begin call returns.
        unsafe { set.native_ref() }.clone()
    });
    Ok(())
}
