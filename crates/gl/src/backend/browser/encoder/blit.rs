//! Immediate framebuffer blits between caller-owned texture mip slices.
//!
//! WebGL2 has no `glBlitNamedFramebuffer`; this adapter creates two short-lived
//! FBOs, issues one native blit, and deletes them before returning. No command
//! payload or FBO survives this call.

use fluxel_rhi_core::api::{Error, Result};
use wasm_bindgen::JsCast;
use web_sys::{WebGl2RenderingContext as Gl, WebGlFramebuffer};

use super::super::owner::{BrowserEncoder, BrowserTexture};

impl BrowserEncoder {
    pub(crate) fn blit_texture_regions(
        &mut self,
        source: &BrowserTexture,
        source_mip: u32,
        source_layer: u32,
        source_origin: [u32; 2],
        source_extent: [u32; 2],
        destination: &BrowserTexture,
        destination_mip: u32,
        destination_layer: u32,
        destination_origin: [u32; 2],
        destination_extent: [u32; 2],
        layers: u32,
        aspect: crate::backend::api::GlTextureAspect,
        filter: crate::backend::api::GlFilterMode,
    ) -> Result<()> {
        self.same_owner(&source.owner)?;
        self.same_owner(&destination.owner)?;
        let src = source.desc;
        let dst = destination.desc;
        // A standalone S8 allocation is a WebGL renderbuffer, not a texture.
        // It has one mip/layer and is admitted solely for typed stencil
        // attachment blits. Do not generalize this to other renderbuffers:
        // their Core copy contract may require resolves, sampling, or planes
        // which framebuffer_renderbuffer cannot provide.
        let standalone_stencil_renderbuffers = src.format
            == crate::backend::api::GlFormat::Stencil8
            && dst.format == crate::backend::api::GlFormat::Stencil8
            && src.sample_count == 1
            && dst.sample_count == 1
            && source.raw.is_none()
            && destination.raw.is_none()
            && source.renderbuffer.is_some()
            && destination.renderbuffer.is_some();
        if !standalone_stencil_renderbuffers
            && (source.renderbuffer.is_some() || destination.renderbuffer.is_some())
        {
            return Err(Error::Unsupported("WebGL2 renderbuffer blit format"));
        }
        let source_raw = (!standalone_stencil_renderbuffers)
            .then(|| source.texture())
            .transpose()?;
        let destination_raw = (!standalone_stencil_renderbuffers)
            .then(|| destination.texture())
            .transpose()?;
        let src_extent = src.mip_extent(source_mip).ok_or(Error::InvalidRange)?;
        let dst_extent = dst.mip_extent(destination_mip).ok_or(Error::InvalidRange)?;
        if !matches!(
            src.dimension,
            crate::backend::api::GlTextureDimension::D2
                | crate::backend::api::GlTextureDimension::D2Array
                | crate::backend::api::GlTextureDimension::D3
        ) || !matches!(
            dst.dimension,
            crate::backend::api::GlTextureDimension::D2
                | crate::backend::api::GlTextureDimension::D2Array
                | crate::backend::api::GlTextureDimension::D3
        ) || src.sample_count != 1
            || dst.sample_count != 1
            || !src.format.blit_compatible(dst.format, aspect)
            || (aspect == crate::backend::api::GlTextureAspect::Color
                && !src.format.is_integer_color()
                && !src.format.is_float_color()
                && dst.format.is_float_color())
            || self
                .owner
                .discovery_snapshot()
                .formats()
                .get_for(
                    if standalone_stencil_renderbuffers {
                        crate::backend::api::GlFormatResourceKind::Renderbuffer
                    } else {
                        crate::backend::api::GlFormatResourceKind::Texture
                    },
                    src.format,
                    1,
                )
                .is_none_or(|facts| !facts.renderable)
            || self
                .owner
                .discovery_snapshot()
                .formats()
                .get_for(
                    if standalone_stencil_renderbuffers {
                        crate::backend::api::GlFormatResourceKind::Renderbuffer
                    } else {
                        crate::backend::api::GlFormatResourceKind::Texture
                    },
                    dst.format,
                    1,
                )
                .is_none_or(|facts| !facts.renderable)
            || (filter == crate::backend::api::GlFilterMode::Linear
                && (src.format.is_integer_color()
                    || aspect != crate::backend::api::GlTextureAspect::Color))
            || !src
                .usage
                .contains(crate::backend::api::GlTextureUsage::COPY_SOURCE)
            || !dst
                .usage
                .contains(crate::backend::api::GlTextureUsage::COPY_DESTINATION)
            || source_extent.contains(&0)
            || destination_extent.contains(&0)
            || layers == 0
            || source_layer
                .checked_add(layers)
                .is_none_or(|end| end > src_extent.depth_or_layers)
            || destination_layer
                .checked_add(layers)
                .is_none_or(|end| end > dst_extent.depth_or_layers)
            || source_origin[0]
                .checked_add(source_extent[0])
                .is_none_or(|end| end > src_extent.width)
            || source_origin[1]
                .checked_add(source_extent[1])
                .is_none_or(|end| end > src_extent.height)
            || destination_origin[0]
                .checked_add(destination_extent[0])
                .is_none_or(|end| end > dst_extent.width)
            || destination_origin[1]
                .checked_add(destination_extent[1])
                .is_none_or(|end| end > dst_extent.height)
            || (same_storage(source, destination)
                && source_mip == destination_mip
                && source_layer < destination_layer + layers
                && destination_layer < source_layer + layers)
        {
            return Err(Error::Unsupported("WebGL2 texture blit shape or usage"));
        }
        let src_x0 = i32::try_from(source_origin[0]).map_err(|_| Error::InvalidRange)?;
        let src_x1 =
            i32::try_from(source_origin[0] + source_extent[0]).map_err(|_| Error::InvalidRange)?;
        let src_y0 = i32::try_from(source_origin[1]).map_err(|_| Error::InvalidRange)?;
        let src_y1 =
            i32::try_from(source_origin[1] + source_extent[1]).map_err(|_| Error::InvalidRange)?;
        let dst_x0 = i32::try_from(destination_origin[0]).map_err(|_| Error::InvalidRange)?;
        let dst_x1 = i32::try_from(destination_origin[0] + destination_extent[0])
            .map_err(|_| Error::InvalidRange)?;
        let dst_y0 = i32::try_from(destination_origin[1]).map_err(|_| Error::InvalidRange)?;
        let dst_y1 = i32::try_from(destination_origin[1] + destination_extent[1])
            .map_err(|_| Error::InvalidRange)?;

        let mut state = self.owner.state.borrow_mut();
        state.ensure_encoder(self.epoch)?;
        state.ensure_active()?;
        let raw = &state.context.raw;
        let previous_read = raw
            .get_parameter(Gl::READ_FRAMEBUFFER_BINDING)
            .map_err(|_| Error::InvalidState)?;
        let previous_draw = raw
            .get_parameter(Gl::DRAW_FRAMEBUFFER_BINDING)
            .map_err(|_| Error::InvalidState)?;
        let previous_scissor = raw.is_enabled(Gl::SCISSOR_TEST);
        use crate::backend::api::{GlFormat, GlTextureAspect};
        let attachment = match src.format {
            GlFormat::Depth24PlusStencil8 | GlFormat::Depth32FloatStencil8 => {
                Gl::DEPTH_STENCIL_ATTACHMENT
            }
            _ => match aspect {
                GlTextureAspect::Color => Gl::COLOR_ATTACHMENT0,
                GlTextureAspect::DepthOnly => Gl::DEPTH_ATTACHMENT,
                GlTextureAspect::StencilOnly => Gl::STENCIL_ATTACHMENT,
                GlTextureAspect::All => unreachable!("validated aspect"),
            },
        };
        let mask = match aspect {
            GlTextureAspect::Color => Gl::COLOR_BUFFER_BIT,
            GlTextureAspect::DepthOnly => Gl::DEPTH_BUFFER_BIT,
            GlTextureAspect::StencilOnly => Gl::STENCIL_BUFFER_BIT,
            GlTextureAspect::All => unreachable!("validated aspect"),
        };
        let Some(src_fbo) = raw.create_framebuffer() else {
            return Err(Error::OutOfMemory);
        };
        let Some(dst_fbo) = raw.create_framebuffer() else {
            raw.delete_framebuffer(Some(&src_fbo));
            return Err(Error::OutOfMemory);
        };
        let mut destination_complete = true;
        for layer in 0..layers {
            raw.bind_framebuffer(Gl::FRAMEBUFFER, Some(&src_fbo));
            if standalone_stencil_renderbuffers {
                raw.framebuffer_renderbuffer(
                    Gl::FRAMEBUFFER,
                    attachment,
                    Gl::RENDERBUFFER,
                    source.renderbuffer.as_ref(),
                );
            } else if src.dimension == crate::backend::api::GlTextureDimension::D2 {
                raw.framebuffer_texture_2d(
                    Gl::FRAMEBUFFER,
                    attachment,
                    Gl::TEXTURE_2D,
                    source_raw,
                    source_mip as i32,
                );
            } else {
                raw.framebuffer_texture_layer(
                    Gl::FRAMEBUFFER,
                    attachment,
                    source_raw,
                    source_mip as i32,
                    (source_layer + layer) as i32,
                );
            }
            if aspect != GlTextureAspect::Color {
                raw.draw_buffers(&js_sys::Array::of1(&Gl::NONE.into()));
                raw.read_buffer(Gl::NONE);
            }
            let source_complete =
                raw.check_framebuffer_status(Gl::FRAMEBUFFER) == Gl::FRAMEBUFFER_COMPLETE;
            if !source_complete {
                destination_complete = false;
                break;
            }
            raw.bind_framebuffer(Gl::FRAMEBUFFER, Some(&dst_fbo));
            if standalone_stencil_renderbuffers {
                raw.framebuffer_renderbuffer(
                    Gl::FRAMEBUFFER,
                    attachment,
                    Gl::RENDERBUFFER,
                    destination.renderbuffer.as_ref(),
                );
            } else if dst.dimension == crate::backend::api::GlTextureDimension::D2 {
                raw.framebuffer_texture_2d(
                    Gl::FRAMEBUFFER,
                    attachment,
                    Gl::TEXTURE_2D,
                    destination_raw,
                    destination_mip as i32,
                );
            } else {
                raw.framebuffer_texture_layer(
                    Gl::FRAMEBUFFER,
                    attachment,
                    destination_raw,
                    destination_mip as i32,
                    (destination_layer + layer) as i32,
                );
            }
            if aspect != GlTextureAspect::Color {
                raw.draw_buffers(&js_sys::Array::of1(&Gl::NONE.into()));
                raw.read_buffer(Gl::NONE);
            }
            if raw.check_framebuffer_status(Gl::FRAMEBUFFER) != Gl::FRAMEBUFFER_COMPLETE {
                destination_complete = false;
                break;
            }
            // A prior immediate raster pass may leave SCISSOR_TEST enabled.
            // GL applies scissor to framebuffer blits, so the private transfer
            // must not inherit that unrelated draw state.
            raw.disable(Gl::SCISSOR_TEST);
            raw.bind_framebuffer(Gl::READ_FRAMEBUFFER, Some(&src_fbo));
            raw.bind_framebuffer(Gl::DRAW_FRAMEBUFFER, Some(&dst_fbo));
            raw.blit_framebuffer(
                src_x0,
                src_y0,
                src_x1,
                src_y1,
                dst_x0,
                dst_y0,
                dst_x1,
                dst_y1,
                mask,
                match filter {
                    crate::backend::api::GlFilterMode::Nearest => Gl::NEAREST,
                    crate::backend::api::GlFilterMode::Linear => Gl::LINEAR,
                },
            );
            self.issued = true;
        }
        raw.bind_framebuffer(
            Gl::READ_FRAMEBUFFER,
            previous_read.dyn_ref::<WebGlFramebuffer>(),
        );
        raw.bind_framebuffer(
            Gl::DRAW_FRAMEBUFFER,
            previous_draw.dyn_ref::<WebGlFramebuffer>(),
        );
        if previous_scissor {
            raw.enable(Gl::SCISSOR_TEST);
        } else {
            raw.disable(Gl::SCISSOR_TEST);
        }
        raw.delete_framebuffer(Some(&dst_fbo));
        raw.delete_framebuffer(Some(&src_fbo));
        if state.context.raw.is_context_lost() {
            state.mark_lost();
            return Err(Error::DeviceLost);
        }
        let error = raw.get_error();
        if error != Gl::NO_ERROR {
            return Err(Error::Backend(format!(
                "WebGL2 texture blit failed with GL error {error}"
            )));
        }
        if !destination_complete {
            return Err(Error::Backend(
                "WebGL2 texture blit framebuffer is incomplete".into(),
            ));
        }
        Ok(())
    }
}

/// Texture and renderbuffer handles are distinct WebGL object types. Compare
/// the actual object identity so self-overlap remains rejected for S8
/// renderbuffers as it already is for texture mip/layer ranges.
fn same_storage(source: &BrowserTexture, destination: &BrowserTexture) -> bool {
    match (
        source.raw.as_ref(),
        destination.raw.as_ref(),
        source.renderbuffer.as_ref(),
        destination.renderbuffer.as_ref(),
    ) {
        (Some(source), Some(destination), None, None) => js_sys::Object::is(
            <web_sys::WebGlTexture as AsRef<wasm_bindgen::JsValue>>::as_ref(source),
            <web_sys::WebGlTexture as AsRef<wasm_bindgen::JsValue>>::as_ref(destination),
        ),
        (None, None, Some(source), Some(destination)) => js_sys::Object::is(
            <web_sys::WebGlRenderbuffer as AsRef<wasm_bindgen::JsValue>>::as_ref(source),
            <web_sys::WebGlRenderbuffer as AsRef<wasm_bindgen::JsValue>>::as_ref(destination),
        ),
        _ => false,
    }
}
