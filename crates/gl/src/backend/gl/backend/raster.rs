//! Immediate raster-pass lowering for the OpenGL backend.
//!
//! A pass owns the provider's active framebuffer state until `end_raster`; each
//! accepted verb mutates that state directly, so pass validity is not draw replay.

use std::ops::Range;

use fluxel_rhi_core::api::{
    self, AttachmentView, Error, IndexFormat, LoadOp, RasterBackend, RasterPassDesc, Rect, Result,
    Viewport,
};

use super::Gl;
use super::spi::native_error;
use super::types::{GlBuffer, GlEncoder, GlRasterPipeline};

mod attachment;
pub(crate) mod binding;
mod draw;
mod dynamic;
mod finish;
pub(crate) mod layout;
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

impl RasterBackend for Gl {
    fn begin_raster(encoder: &mut GlEncoder, desc: &RasterPassDesc<'_, Self>) -> Result<()> {
        if desc.timestamp_writes.is_some()
            || desc.secondary_inheritance.is_some()
            || desc.multiview_mask.is_some()
        {
            return Err(Error::Unsupported("GL baseline raster-pass feature"));
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
        let [Some(color)]: &[Option<api::ColorAttachment<'_, Gl>>] = desc.colors else {
            return Err(Error::Unsupported(
                "one GL default-framebuffer color attachment",
            ));
        };
        let AttachmentView::Surface(surface) = &color.view else {
            #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
            return begin_native_offscreen_raster(encoder, desc, color);
            #[cfg(not(any(
                all(windows, feature = "native-gl-wgl"),
                feature = "native-gles-egl"
            )))]
            return Err(Error::Unsupported("GL offscreen raster target"));
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
        let attachment = crate::backend::gl::api::GlColorAttachment {
            view: crate::backend::gl::api::GlPassAttachmentView::DefaultColor(target),
            resolve_target: None,
            load: lower_load(color.load)?,
            store: lower_store(color.store),
            clear: lower_clear(color.load)?,
        };
        let native = crate::backend::gl::api::GlRenderPassDescriptor {
            target: crate::backend::gl::api::GlRenderTarget::Default(target),
            color_attachments: vec![attachment],
            depth_stencil_attachment: None,
        };
        #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
        {
            use crate::backend::gl::api::GlFramebufferApi as _;
            use crate::backend::gl::api::GlResourceApi as _;
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
            if depth.is_some() && matches!(color.load, LoadOp::Load) {
                // The temporary color target has no contents until this pass;
                // preserving framebuffer zero would require an explicit preload.
                return Err(Error::Unsupported("GL surface depth color load"));
            }
            let result = encoder
                .state
                .with_native("begin GL raster pass", |provider| {
                    if let Some(depth) = depth.as_ref() {
                        let color_desc = crate::backend::gl::api::GlTextureDesc {
                            dimension: crate::backend::gl::api::GlTextureDimension::D2,
                            extent: crate::backend::gl::api::GlExtent3d {
                                width: target.width,
                                height: target.height,
                                depth_or_layers: 1,
                            },
                            mip_level_count: 1,
                            sample_count: 1,
                            format: target.color_format,
                            usage: crate::backend::gl::api::GlTextureUsage::RENDER_ATTACHMENT,
                        };
                        let color = provider.create_texture_resource(color_desc)?;
                        let color_view = crate::backend::gl::api::GlTextureView {
                            target: crate::backend::gl::api::GlAttachmentTarget::Texture(color),
                            format: target.color_format,
                            mip_level: 0,
                            array_layer: 0,
                            layer_count: 1,
                            width: target.width,
                            height: target.height,
                            sample_count: 1,
                        };
                        let framebuffer = provider.create_framebuffer(
                            &crate::backend::gl::api::GlFramebufferDescriptor {
                                color_attachments: vec![color_view],
                                depth_stencil_attachment: Some(depth.native),
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
                        let mut color_attachment = attachment;
                        color_attachment.view =
                            crate::backend::gl::api::GlPassAttachmentView::Allocated(color_view);
                        let pass = crate::backend::gl::api::GlRenderPassDescriptor {
                            target: crate::backend::gl::api::GlRenderTarget::Offscreen(framebuffer),
                            color_attachments: vec![color_attachment],
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
                        });
                        Ok(())
                    } else {
                        provider.begin_render_pass(&native)
                    }
                })
                .map_err(native_error);
            match result {
                Ok(()) => {
                    encoder.raster_extent = Some([target.width, target.height]);
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
                begin_browser_surface_raster(encoder, target, color, desc.depth_stencil.as_ref())?;
                encoder.raster_extent = Some([target.width, target.height]);
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
        let [width, height] = encoder.raster_extent.ok_or(Error::InvalidState)?;
        #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
        {
            let native = pipeline.native_with_viewport(width, height);
            use crate::backend::gl::api::GlRasterCommandApi as _;
            let result = encoder
                .state
                .with_native("set GL raster pipeline", |provider| {
                    provider.set_raster_pipeline(&native)
                })
                .map_err(native_error);
            match result {
                Ok(()) => {
                    encoder.raster_pipeline = Some(pipeline.clone());
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
                    browser.set_fixed_raster_state(&pipeline.raster)?;
                    browser.set_depth_state(pipeline.raster.depth_stencil)?;
                    Ok(())
                })();
                if let Err(error) = result {
                    encoder.indeterminate = true;
                    return Err(error);
                }
                let _ = (width, height);
                encoder.raster_pipeline = Some(pipeline.clone());
                encoder.issued = true;
                Ok(())
            }
            #[cfg(not(all(target_arch = "wasm32", feature = "webgl2")))]
            {
                let _ = (encoder, pipeline, width, height);
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
            index,
            group,
            dynamic_offsets,
        );
        #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
        return bind_browser_group(encoder, &pipeline, index, group, dynamic_offsets);
    }

    fn set_vertex_buffer(
        encoder: &mut GlEncoder,
        slot: u32,
        buffer: &GlBuffer,
        offset: u64,
        size: u64,
    ) -> Result<()> {
        if offset
            .checked_add(size)
            .is_none_or(|end| end > buffer.desc.size)
        {
            return Err(Error::InvalidRange);
        }
        let pipeline = encoder
            .raster_pipeline
            .as_ref()
            .ok_or(Error::InvalidState)?;
        #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
        {
            let binding = crate::backend::gl::api::GlVertexBufferBinding {
                slot,
                buffer: buffer.id,
                offset,
            };
            use crate::backend::gl::api::GlVertexApi as _;
            let result = encoder
                .state
                .with_native("set GL vertex binding", |provider| {
                    provider.set_vertex_binding(pipeline.vertex_array, binding)
                })
                .map_err(native_error);
            match result {
                Ok(()) => {
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
                let layout = pipeline
                    .vertex_layout
                    .buffers
                    .iter()
                    .find(|layout| layout.slot == slot)
                    .ok_or(Error::InvalidRange)?;
                let result = (|| -> Result<()> {
                    browser.set_browser_vertex_array(&pipeline.browser_vertex_array)?;
                    for attribute in pipeline
                        .vertex_layout
                        .attributes
                        .iter()
                        .filter(|attribute| attribute.buffer_slot == slot)
                    {
                        browser.set_vertex_attribute(&buffer.browser, layout, attribute, offset)?;
                    }
                    Ok(())
                })();
                if result.is_ok() {
                    encoder.issued = true;
                } else {
                    encoder.indeterminate = true;
                }
                result
            }
            #[cfg(not(all(target_arch = "wasm32", feature = "webgl2")))]
            {
                let _ = (pipeline, slot, buffer, offset);
                Err(Error::Unsupported("GL platform context"))
            }
        }
    }

    fn set_index_buffer(
        encoder: &mut GlEncoder,
        buffer: &GlBuffer,
        offset: u64,
        size: u64,
        format: IndexFormat,
    ) -> Result<()> {
        if offset
            .checked_add(size)
            .is_none_or(|end| end > buffer.desc.size)
        {
            return Err(Error::InvalidRange);
        }
        let pipeline = encoder
            .raster_pipeline
            .as_ref()
            .ok_or(Error::InvalidState)?;
        #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
        {
            let binding = crate::backend::gl::api::GlIndexBinding {
                buffer: buffer.id,
                offset,
                format: match format {
                    IndexFormat::Uint16 => crate::backend::gl::api::GlIndexFormat::Uint16,
                    IndexFormat::Uint32 => crate::backend::gl::api::GlIndexFormat::Uint32,
                },
            };
            use crate::backend::gl::api::GlVertexApi as _;
            let result = encoder
                .state
                .with_native("set GL index binding", |provider| {
                    provider.set_index_binding(pipeline.vertex_array, Some(binding))
                })
                .map_err(native_error);
            match result {
                Ok(()) => {
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
                if offset > i32::MAX as u64 {
                    return Err(Error::InvalidRange);
                }
                let browser = encoder.browser.as_mut().ok_or(Error::InvalidState)?;
                let result = (|| -> Result<()> {
                    browser.set_browser_vertex_array(&pipeline.browser_vertex_array)?;
                    browser.set_index_buffer(&buffer.browser)?;
                    Ok(())
                })();
                if result.is_err() {
                    encoder.indeterminate = true;
                    return result;
                }
                encoder.browser_index = Some((offset, format));
                encoder.issued = true;
                Ok(())
            }
            #[cfg(not(all(target_arch = "wasm32", feature = "webgl2")))]
            {
                let _ = (pipeline, buffer, offset, format);
                Err(Error::Unsupported("GL platform context"))
            }
        }
    }

    fn viewport(encoder: &mut GlEncoder, viewport: Viewport) -> Result<()> {
        set_viewport(encoder, viewport)
    }

    fn scissor(encoder: &mut GlEncoder, scissor: Rect) -> Result<()> {
        set_scissor(encoder, scissor)
    }
    fn blend_constant(_: &mut GlEncoder, _: [f32; 4]) -> Result<()> {
        Err(Error::Unsupported("GL dynamic blend constant"))
    }
    fn stencil_reference(_: &mut GlEncoder, _: u32) -> Result<()> {
        Err(Error::Unsupported("GL dynamic stencil reference"))
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
        let draw = crate::backend::gl::api::GlDrawCommand::NonIndexed(
            crate::backend::gl::api::GlNonIndexedDraw {
                first_vertex: vertices.start,
                vertex_count,
                instance_count,
            },
        );
        issue_draw(encoder, &pipeline, draw)
    }

    fn draw_indexed(
        encoder: &mut GlEncoder,
        indices: Range<u32>,
        base_vertex: i32,
        instances: Range<u32>,
    ) -> Result<()> {
        if base_vertex != 0 {
            return Err(Error::Unsupported("GL base-vertex draw"));
        }
        let pipeline = encoder.raster_pipeline.clone().ok_or(Error::InvalidState)?;
        let index_count = indices
            .end
            .checked_sub(indices.start)
            .ok_or(Error::InvalidRange)?;
        let instance_count = instances
            .end
            .checked_sub(instances.start)
            .ok_or(Error::InvalidRange)?;
        let draw = crate::backend::gl::api::GlDrawCommand::Indexed(
            crate::backend::gl::api::GlIndexedDraw {
                first_index: indices.start,
                index_count,
                instance_count,
            },
        );
        issue_draw(encoder, &pipeline, draw)
    }

    fn end_raster(encoder: &mut GlEncoder) -> Result<()> {
        if encoder.occlusion_query_active {
            return Err(Error::InvalidState);
        }
        #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
        {
            use crate::backend::gl::api::GlFramebufferApi as _;
            use crate::backend::gl::api::GlResourceApi as _;
            let intermediate = encoder.surface_depth_pass.take();
            let offscreen_framebuffer = encoder.offscreen_framebuffer.take();
            let offscreen_resolve = encoder.offscreen_resolve.take();
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
                            provider.blit_framebuffer_to_default(pass.framebuffer, pass.target)
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
                            Some(resolve) if !stale_surface => {
                                provider.blit_framebuffer_to_default(framebuffer, resolve.target)
                            }
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
            encoder.raster_extent = None;
            encoder.raster_pipeline = None;
            match result {
                Ok(()) => {
                    encoder.occlusion_query_set = None;
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
                let result = if browser.has_surface_depth_pass() {
                    browser.end_surface_depth_raster()
                } else {
                    browser.end_raster()
                };
                encoder.raster_extent = None;
                encoder.raster_pipeline = None;
                encoder.occlusion_query_set = None;
                if result.is_err() {
                    encoder.indeterminate = true;
                }
                result
            }
            #[cfg(not(all(target_arch = "wasm32", feature = "webgl2")))]
            Err(Error::Unsupported("GL platform context"))
        }
    }

    fn abort_raster(encoder: &mut GlEncoder) {
        finish::abort(encoder);
    }
}

/// Attaches the caller's texture views to a temporary FBO and begins the pass
/// now. The FBO name is retired by `end_raster`/`abort_raster`; no pass packet
/// or texture ownership is retained beyond the active GL pass.
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
fn begin_native_offscreen_raster(
    encoder: &mut GlEncoder,
    desc: &RasterPassDesc<'_, Gl>,
    color: &api::ColorAttachment<'_, Gl>,
) -> Result<()> {
    use crate::backend::gl::api::{GlFramebufferApi as _, GlResourceApi as _};

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
            Some(super::types::GlOffscreenResolve {
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
            encoder.raster_extent = Some([color_view.width, color_view.height]);
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
