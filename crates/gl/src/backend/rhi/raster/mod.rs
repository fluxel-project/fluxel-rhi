//! Immediate raster-pass lowering for the OpenGL backend.
//!
//! A pass owns the provider's active framebuffer state until `end_raster`; each
//! accepted verb mutates that state directly, so pass validity is not draw replay.

use std::ops::Range;

use fluxel_rhi_core::api::{
    self, AttachmentView, Error, IndexFormat, LoadOp, RasterPassDesc, Rect, Result, Viewport,
};
use fluxel_rhi_core::backend::RasterBackend;

use super::Gl;
use super::spi::native_error;
use super::types::{GlBuffer, GlEncoder, GlRasterPipeline};

mod attachment;
pub(crate) mod binding;
pub(crate) mod cube_mirror;
mod draw;
mod dynamic;
mod finish;
pub(crate) mod layout;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
pub(crate) mod resolve;
mod surface_depth;
use attachment::{lower_clear, lower_load, lower_store};
#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
use binding::bind_browser_group;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
use binding::bind_native_group;
use binding::validate_dynamic_offsets;
use draw::issue_draw;
use dynamic::{set_scissor, set_viewport};
use layout::matches as bind_group_layout_matches;
#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
use surface_depth::begin_browser_surface_raster;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
use surface_depth::{
    lower_native_attachment, lower_native_view, validate_native_surface_depth_view,
};

mod multiview;
mod offscreen;
mod vertex;
#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
use offscreen::begin_browser_offscreen_raster;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
use offscreen::begin_native_offscreen_raster;

/// Re-establish the active raster target and the complete CPU-owned state
/// packet after a private native mirror pass.  Mirror helpers intentionally
/// do not snapshot GL state; they invalidate it, and this replay is the one
/// canonical restoration point before the next immediate draw.
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
pub(crate) fn restore_after_native_mirror(
    encoder: &mut GlEncoder,
    pipeline: &GlRasterPipeline,
) -> Result<()> {
    encoder
        .state
        .with_native("restore GL raster target after mirror", |provider| {
            provider.restore_active_pass_framebuffer()
        })
        .map_err(native_error)?;
    <Gl as RasterBackend>::set_pipeline(encoder, pipeline)
}

impl RasterBackend for Gl {
    fn begin_raster(encoder: &mut GlEncoder, desc: &RasterPassDesc<'_, Self>) -> Result<()> {
        if desc.secondary_inheritance.is_some() {
            return Err(Error::Unsupported("GL baseline raster-pass feature"));
        }
        if let Some(depth) = &desc.depth_stencil {
            if depth.depth_read_only
                && depth.depth.is_some_and(|(load, store)| {
                    !matches!(load, LoadOp::Load) || !matches!(store, api::StoreOp::Store)
                })
                || depth.stencil_read_only
                    && depth.stencil.is_some_and(|(load, store)| {
                        !matches!(load, LoadOp::Load) || !matches!(store, api::StoreOp::Store)
                    })
            {
                return Err(Error::InvalidState);
            }
        }
        let occlusion_query_set = desc.occlusion_query_set.map(|set| {
            // SAFETY: Core borrowed this QuerySet for this begin call; the
            // encoder clones the native owner so it remains live through the
            // matching immediate end of this raster pass.
            unsafe { set.native_ref() }.clone()
        });
        if let Some(set) = occlusion_query_set.as_ref() {
            if !std::rc::Rc::ptr_eq(&encoder.state, &set.state)
                || !matches!(set.kind, api::QueryKind::Occlusion)
            {
                return Err(Error::WrongDevice);
            }
        }
        if desc.multiview_mask.is_some() {
            return multiview::begin(encoder, desc);
        }
        let surface_color = match desc.colors {
            [Some(color)] if matches!(color.view, AttachmentView::Surface(_)) => Some(color),
            _ => None,
        };
        let Some(color) = surface_color else {
            #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
            return begin_native_offscreen_raster(encoder, desc);
            #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
            return begin_browser_offscreen_raster(encoder, desc);
            #[cfg(not(any(
                all(windows, feature = "native-gl-wgl"),
                feature = "native-gles-egl",
                all(target_arch = "wasm32", feature = "webgl2")
            )))]
            return Err(Error::Unsupported("GL offscreen raster target"));
        };
        let AttachmentView::Surface(surface) = &color.view else {
            unreachable!()
        };
        if color.resolve_target.is_some() {
            return Err(Error::Unsupported("GL resolve into a surface pass"));
        }
        // SAFETY: Core keeps this acquired image borrowed for the entire pass;
        // we read immutable generation facts only. Native depth passes retain a
        // private token because their temporary FBO outlives this borrow.
        #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
        let surface_token = unsafe { surface.native_ref() }.target_token();
        let target = unsafe { surface.native_ref() }.default_target()?;
        #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
        {
            // Every canvas pass uses the common texture row order. Publication
            // reverses Y once at the default framebuffer boundary.
            return begin_browser_offscreen_raster(encoder, desc);
        }
        let attachment = crate::backend::api::GlColorAttachment {
            view: crate::backend::api::GlPassAttachmentView::DefaultColor(target),
            resolve_target: None,
            load: lower_load(color.load)?,
            store: lower_store(color.store),
            clear: lower_clear(color.load)?,
        };
        let native = crate::backend::api::GlRenderPassDescriptor {
            render_area: render_area(desc.render_area, [target.width, target.height])?,
            target: crate::backend::api::GlRenderTarget::Default(target),
            color_attachments: vec![Some(attachment)],
            depth_stencil_attachment: None,
        };
        #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
        {
            use crate::backend::api::GlFramebufferApi as _;
            use crate::backend::api::GlResourceApi as _;
            let depth = lower_native_view(desc.depth_stencil.as_ref())?;
            if let Some(depth) = depth.as_ref() {
                validate_native_surface_depth_view(depth, target)?;
            }
            let depth_pass = match (&depth, desc.depth_stencil.as_ref()) {
                (Some(view), Some(attachment)) => {
                    Some(lower_native_attachment(attachment, view.native)?)
                }
                (None, None) => None,
                _ => return Err(Error::InvalidState),
            };
            let preload_color = matches!(color.load, LoadOp::Load) || native.render_area.is_some();
            let timestamps =
                super::query::prepare_pass_timestamps(encoder, desc.timestamp_writes.as_ref())?;
            super::query::start_pass_timestamps(encoder, timestamps)?;
            let result = encoder
                .state
                .with_native("begin GL raster pass", |provider| {
                    {
                        let color_desc = crate::backend::api::GlTextureDesc {
                            dimension: crate::backend::api::GlTextureDimension::D2,
                            extent: crate::backend::api::GlExtent3d {
                                width: target.width,
                                height: target.height,
                                depth_or_layers: 1,
                            },
                            mip_level_count: 1,
                            sample_count: 1,
                            format: target.color_format,
                            usage: crate::backend::api::GlTextureUsage::RENDER_ATTACHMENT,
                        };
                        let color = provider.create_texture_resource(color_desc)?;
                        let color_view = crate::backend::api::GlTextureView {
                            target: crate::backend::api::GlAttachmentTarget::Texture(color),
                            format: target.color_format,
                            aspect: crate::backend::api::GlTextureAspect::All,
                            mip_level: 0,
                            array_layer: 0,
                            layer_count: 1,
                            width: target.width,
                            height: target.height,
                            sample_count: 1,
                        };
                        let framebuffer = provider.create_framebuffer(
                            &crate::backend::api::GlFramebufferDescriptor {
                                color_attachments: vec![Some(color_view)],
                                depth_stencil_attachment: depth.as_ref().map(|depth| depth.native),
                                draw_buffers: vec![0],
                            },
                        );
                        let framebuffer = match framebuffer {
                            Ok(value) => value,
                            Err(error) => {
                                let _ = provider.destroy_texture_resource(color);
                                return Err(error);
                            }
                        };
                        if preload_color {
                            if let Err(error) =
                                provider.blit_default_to_framebuffer(target, framebuffer)
                            {
                                let _ = provider.destroy_framebuffer(framebuffer);
                                let _ = provider.destroy_texture_resource(color);
                                return Err(error);
                            }
                        }
                        let mut color_attachment = attachment;
                        // The temporary source must survive until publication.
                        color_attachment.store = crate::backend::api::GlStoreOp::Store;
                        color_attachment.view =
                            crate::backend::api::GlPassAttachmentView::Allocated(color_view);
                        let pass = crate::backend::api::GlRenderPassDescriptor {
                            render_area: native.render_area,
                            target: crate::backend::api::GlRenderTarget::Offscreen(framebuffer),
                            color_attachments: vec![Some(color_attachment)],
                            depth_stencil_attachment: depth_pass,
                        };
                        if let Err(error) = provider.begin_render_pass(&pass) {
                            let _ = provider.destroy_framebuffer(framebuffer);
                            let _ = provider.destroy_texture_resource(color);
                            return Err(error);
                        }
                        encoder.surface_depth_pass = Some(super::types::GlSurfaceDepthPass {
                            target,
                            surface_token: surface_token.clone(),
                            framebuffer,
                            color,
                            depth: depth.clone(),
                            color_slot: 0,
                            render_area: native.render_area,
                        });
                        Ok(())
                    }
                })
                .map_err(native_error);
            match result {
                Ok(()) => {
                    encoder.begin_raster_state([target.width, target.height]);
                    encoder.set_raster_read_only(desc);
                    encoder.raster_pipeline = None;
                    encoder.vertex_bindings.clear();
                    encoder.index_binding = None;
                    encoder.issued = true;
                    encoder.occlusion_query_set = occlusion_query_set.clone();
                    Ok(())
                }
                Err(error) => {
                    encoder.indeterminate = true;
                    Err(error)
                }
            }
        }
        #[cfg(not(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl")))]
        {
            #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
            {
                // SAFETY: the borrowed acquired image supplies a read-only lease proof.
                let token = unsafe { surface.native_ref() }.target_token();
                begin_browser_surface_raster(
                    encoder,
                    target,
                    color,
                    desc.depth_stencil.as_ref(),
                    desc.render_area,
                    desc.timestamp_writes.as_ref(),
                )?;
                encoder.browser_surface_token = Some(token);
                encoder.begin_raster_state([target.width, target.height]);
                encoder.set_raster_read_only(desc);
                encoder.raster_pipeline = None;
                encoder.vertex_bindings.clear();
                encoder.index_binding = None;
                encoder.issued = true;
                encoder.occlusion_query_set = occlusion_query_set.clone();
                Ok(())
            }
            #[cfg(not(all(target_arch = "wasm32", feature = "webgl2")))]
            {
                let _ = (encoder, native);
                Err(Error::Unsupported("GL platform context"))
            }
        }
    }

    fn set_pipeline(encoder: &mut GlEncoder, pipeline: &GlRasterPipeline) -> Result<()> {
        encoder.raster_extent.ok_or(Error::InvalidState)?;
        if pipeline.multiview != encoder.raster_multiview.map(|(count, _)| count) {
            return Err(Error::InvalidState);
        }
        if let Some(depth) = pipeline.raster.depth_stencil {
            let writes_stencil = |face: crate::backend::api::GlStencilFaceState| {
                use crate::backend::api::GlStencilOperation::Keep;
                face.write_mask != 0
                    && (face.fail_op != Keep || face.depth_fail_op != Keep || face.pass_op != Keep)
            };
            if encoder.raster_depth_read_only && depth.depth_write_enabled
                || encoder.raster_stencil_read_only
                    && (writes_stencil(depth.stencil_front) || writes_stencil(depth.stencil_back))
            {
                return Err(Error::InvalidState);
            }
        }
        #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
        {
            let viewport = encoder
                .raster_viewport
                .expect("active raster pass has a viewport");
            let mut native = pipeline.native_with_dynamic_state(
                viewport,
                encoder.raster_scissor,
                encoder.raster_blend_constant,
                encoder.raster_stencil_reference,
            );
            normalize_inactive_color_slots(&mut native.state, &encoder.raster_color_slots);
            use crate::backend::api::GlRasterCommandApi as _;
            let result = encoder
                .state
                .with_native("set GL raster pipeline", |provider| {
                    provider.set_raster_pipeline(&native)?;
                    Ok(())
                })
                .map_err(native_error);
            match result {
                Ok(()) => {
                    encoder.raster_pipeline = Some(pipeline.clone());
                    let groups = binding::compatible_groups(encoder, &pipeline.layout);
                    for (slot, group, offsets) in groups {
                        bind_native_group(
                            encoder,
                            &pipeline.layout,
                            &pipeline.reflection,
                            pipeline.program,
                            &pipeline.mip_uniforms,
                            slot,
                            &group,
                            &offsets,
                        )?;
                    }
                    multiview::install(encoder, &pipeline)?;
                    super::push::upload_raster(encoder)?;
                    encoder.issued = true;
                    Ok(())
                }
                Err(error) => {
                    encoder.indeterminate = true;
                    Err(error)
                }
            }
        }
        #[cfg(not(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl")))]
        {
            #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
            {
                let browser = encoder.browser.as_mut().ok_or(Error::InvalidState)?;
                // Each operation changes the current WebGL context immediately.
                // A later failure therefore makes the immediate prefix
                // indeterminate instead of pretending that pipeline selection
                // was an all-or-nothing CPU-side cache update.
                let result = (|| {
                    browser.set_raster_program(&pipeline.browser_program)?;
                    browser.set_browser_vertex_array(&pipeline.browser_vertex_array)?;
                    let mut state = pipeline.raster.clone();
                    normalize_inactive_color_slots(&mut state, &encoder.raster_color_slots);
                    state.blend_constant = encoder.raster_blend_constant.map(f32::to_bits);
                    if let Some(depth) = &mut state.depth_stencil {
                        depth.stencil_reference = encoder.raster_stencil_reference;
                    }
                    browser.set_fixed_raster_state(&state)?;
                    browser.set_depth_state(state.depth_stencil)?;
                    Ok(())
                })();
                if let Err(error) = result {
                    encoder.indeterminate = true;
                    return Err(error);
                }
                encoder.raster_pipeline = Some(pipeline.clone());
                let groups = binding::compatible_groups(encoder, &pipeline.layout);
                for (slot, group, offsets) in groups {
                    bind_browser_group(encoder, pipeline, slot, &group, &offsets)?;
                }
                multiview::install(encoder, &pipeline)?;
                super::push::upload_raster(encoder)?;
                encoder.issued = true;
                Ok(())
            }
            #[cfg(not(all(target_arch = "wasm32", feature = "webgl2")))]
            {
                let _ = (encoder, pipeline);
                Err(Error::Unsupported("GL platform context"))
            }
        }
    }

    fn set_bind_group(
        encoder: &mut GlEncoder,
        index: u32,
        group: &super::types::GlBindGroup,
        dynamic_offsets: &[u32],
    ) -> Result<()> {
        let pipeline = encoder
            .raster_pipeline
            .as_ref()
            .ok_or(Error::InvalidState)?
            .clone();
        let Some(expected) = pipeline.layout.groups.get(index as usize) else {
            return Err(Error::InvalidRange);
        };
        if !std::rc::Rc::ptr_eq(&encoder.state, &group.layout.state)
            || !bind_group_layout_matches(expected, &group.layout)
        {
            return Err(Error::Unsupported("GL bind group selection/layout"));
        }
        validate_dynamic_offsets(group, dynamic_offsets)?;
        #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
        return bind_native_group(
            encoder,
            &pipeline.layout,
            &pipeline.reflection,
            pipeline.program,
            &pipeline.mip_uniforms,
            index,
            group,
            dynamic_offsets,
        );
        #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
        return bind_browser_group(encoder, &pipeline, index, group, dynamic_offsets);
        #[cfg(not(any(
            all(windows, feature = "native-gl-wgl"),
            feature = "native-gles-egl",
            all(target_arch = "wasm32", feature = "webgl2")
        )))]
        {
            let _ = (encoder, pipeline, index, group, dynamic_offsets);
            Err(Error::Unsupported("GL platform context"))
        }
    }

    fn set_vertex_buffer(
        encoder: &mut GlEncoder,
        slot: u32,
        buffer: &GlBuffer,
        offset: u64,
        size: u64,
    ) -> Result<()> {
        vertex::set_vertex_buffer(encoder, slot, buffer, offset, size)
    }

    fn set_index_buffer(
        encoder: &mut GlEncoder,
        buffer: &GlBuffer,
        offset: u64,
        size: u64,
        format: IndexFormat,
    ) -> Result<()> {
        vertex::set_index_buffer(encoder, buffer, offset, size, format)
    }
    fn viewport(encoder: &mut GlEncoder, viewport: Viewport) -> Result<()> {
        set_viewport(encoder, viewport)
    }

    fn scissor(encoder: &mut GlEncoder, scissor: Rect) -> Result<()> {
        set_scissor(encoder, scissor)
    }
    fn blend_constant(encoder: &mut GlEncoder, value: [f32; 4]) -> Result<()> {
        dynamic::set_blend_constant(encoder, value)
    }
    fn stencil_reference(encoder: &mut GlEncoder, value: u32) -> Result<()> {
        dynamic::set_stencil_reference(encoder, value)
    }

    fn draw(encoder: &mut GlEncoder, vertices: Range<u32>, instances: Range<u32>) -> Result<()> {
        let pipeline = encoder.raster_pipeline.clone().ok_or(Error::InvalidState)?;
        let vertex_count = vertices
            .end
            .checked_sub(vertices.start)
            .ok_or(Error::InvalidRange)?;
        let instance_count = instances
            .end
            .checked_sub(instances.start)
            .ok_or(Error::InvalidRange)?;
        let draw =
            crate::backend::api::GlDrawCommand::NonIndexed(crate::backend::api::GlNonIndexedDraw {
                first_vertex: vertices.start,
                vertex_count,
                instance_count,
            });
        issue_draw(encoder, &pipeline, draw, 0, instances.start)
    }

    fn draw_indexed(
        encoder: &mut GlEncoder,
        indices: Range<u32>,
        base_vertex: i32,
        instances: Range<u32>,
    ) -> Result<()> {
        let pipeline = encoder.raster_pipeline.clone().ok_or(Error::InvalidState)?;
        let index_count = indices
            .end
            .checked_sub(indices.start)
            .ok_or(Error::InvalidRange)?;
        let instance_count = instances
            .end
            .checked_sub(instances.start)
            .ok_or(Error::InvalidRange)?;
        let draw =
            crate::backend::api::GlDrawCommand::Indexed(crate::backend::api::GlIndexedDraw {
                first_index: indices.start,
                index_count,
                instance_count,
            });
        issue_draw(encoder, &pipeline, draw, base_vertex, instances.start)
    }

    fn end_raster(encoder: &mut GlEncoder) -> Result<()> {
        if encoder.occlusion_query_active {
            return Err(Error::InvalidState);
        }
        #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
        {
            use crate::backend::api::GlFramebufferApi as _;
            use crate::backend::api::GlResourceApi as _;
            let intermediate = encoder.surface_depth_pass.take();
            let offscreen_framebuffer = encoder.offscreen_framebuffer.take();
            let offscreen_resolve = encoder.offscreen_resolve.take();
            let texture_resolves = std::mem::take(&mut encoder.native_texture_resolves);
            let stale_surface = intermediate
                .as_ref()
                .is_some_and(|pass| !pass.surface_token.is_current())
                || offscreen_resolve
                    .as_ref()
                    .is_some_and(|resolve| !resolve.surface_token.is_current());
            let result = encoder
                .state
                .with_native("end GL raster pass", |provider| {
                    let pass_result = provider.end_render_pass();
                    let pass_result =
                        pass_result.and_then(|()| resolve::textures(provider, &texture_resolves));
                    if let Some(pass) = &intermediate {
                        // Cleanup is mandatory even when ending the pass fails. The
                        // original end error remains the observable failure; a
                        // temporary FBO/color pair must never be stranded merely
                        // because it cannot be published.
                        if let Err(error) = pass_result {
                            let _ = provider.destroy_framebuffer(pass.framebuffer);
                            let _ = provider.destroy_texture_resource(pass.color);
                            return Err(error);
                        }
                        // A reconfigure invalidates the acquired lease. End
                        // and retire the temporary objects, but never publish
                        // their pixels into a newer default framebuffer.
                        let publish = if stale_surface {
                            Ok(())
                        } else {
                            provider.blit_framebuffer_slot_to_default_region(
                                pass.framebuffer,
                                pass.color_slot,
                                pass.target,
                                pass.render_area,
                            )
                        };
                        let drop_framebuffer = provider.destroy_framebuffer(pass.framebuffer);
                        let drop_color = provider.destroy_texture_resource(pass.color);
                        publish.and(drop_framebuffer).and(drop_color)?;
                    } else if let Some(framebuffer) = offscreen_framebuffer {
                        if let Err(error) = pass_result {
                            let _ = provider.destroy_framebuffer(framebuffer);
                            return Err(error);
                        }
                        let publish = match &offscreen_resolve {
                            Some(resolve) if !stale_surface => resolve::surface(
                                provider,
                                resolve.source,
                                resolve.target,
                                resolve.render_area,
                            ),
                            Some(_) | None => Ok(()),
                        };
                        let drop_framebuffer = provider.destroy_framebuffer(framebuffer);
                        publish.and(drop_framebuffer)?;
                    } else {
                        pass_result?;
                    }
                    Ok(())
                })
                .map_err(native_error);
            let result = if stale_surface && result.is_ok() {
                Err(Error::SurfaceOutdated)
            } else {
                result
            };
            encoder.end_raster_state();
            encoder.raster_pipeline = None;
            match result {
                Ok(()) => {
                    encoder.occlusion_query_set = None;
                    super::query::finish_pass_timestamps(encoder)
                }
                Err(error) => {
                    encoder.indeterminate = true;
                    Err(error)
                }
            }
        }
        #[cfg(not(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl")))]
        {
            #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
            {
                let stale = encoder
                    .browser_surface_token
                    .take()
                    .is_some_and(|token| !token.is_current());
                let browser = encoder.browser.as_mut().ok_or(Error::InvalidState)?;
                if stale {
                    // Even though the stale lease determines the returned
                    // error, consume a deferred direct-draw error before
                    // abandoning the pass so close_immediate reports the
                    // issued prefix as indeterminate rather than cleanly
                    // rejected.
                    if browser
                        .observe_raster_boundary("abort stale WebGL raster")
                        .is_err()
                    {
                        encoder.indeterminate = true;
                    }
                    if browser.has_surface_depth_pass() {
                        browser.abort_surface_depth_raster();
                    } else if browser.has_owned_framebuffer_pass() {
                        browser.abort_offscreen_raster();
                    } else {
                        let _ = browser.end_raster();
                    }
                    encoder.end_raster_state();
                    encoder.raster_pipeline = None;
                    encoder.occlusion_query_set = None;
                    return Err(Error::SurfaceOutdated);
                }
                let result = if browser.has_surface_depth_pass() {
                    browser.end_surface_depth_raster()
                } else if browser.has_owned_framebuffer_pass() {
                    browser.end_offscreen_raster()
                } else {
                    browser.end_raster()
                };
                encoder.end_raster_state();
                encoder.raster_pipeline = None;
                encoder.occlusion_query_set = None;
                if result.is_err() {
                    encoder.indeterminate = true;
                }
                result.and_then(|()| super::query::finish_pass_timestamps(encoder))
            }
            #[cfg(not(all(target_arch = "wasm32", feature = "webgl2")))]
            Err(Error::Unsupported("GL platform context"))
        }
    }

    fn abort_raster(encoder: &mut GlEncoder) {
        finish::abort(encoder);
    }
}

/// Validates the upper-left Core render area against the active attachments.
pub(super) fn render_area(
    rect: Rect,
    extent: [u32; 2],
) -> Result<Option<crate::backend::api::GlScissorRect>> {
    if rect.width == 0
        || rect.height == 0
        || rect
            .x
            .checked_add(rect.width)
            .is_none_or(|end| end > extent[0])
        || rect
            .y
            .checked_add(rect.height)
            .is_none_or(|end| end > extent[1])
    {
        return Err(Error::InvalidRange);
    }
    if rect.x == 0 && rect.y == 0 && rect.width == extent[0] && rect.height == extent[1] {
        return Ok(None);
    }
    Ok(Some(crate::backend::api::GlScissorRect {
        x: rect.x,
        y: rect.y,
        width: rect.width,
        height: rect.height,
    }))
}

/// Empty framebuffer slots cannot receive writes. Match their fixed state to a
/// live slot so holes do not require an indexed-blend extension.
fn normalize_inactive_color_slots(state: &mut crate::backend::api::GlRasterState, slots: &[bool]) {
    let first = state
        .color_targets
        .iter()
        .enumerate()
        .find_map(|(i, value)| slots.get(i).copied().unwrap_or(false).then_some(*value));
    if let Some(first) = first {
        for (i, value) in state.color_targets.iter_mut().enumerate() {
            if !slots.get(i).copied().unwrap_or(false) {
                *value = first;
            }
        }
    }
}
