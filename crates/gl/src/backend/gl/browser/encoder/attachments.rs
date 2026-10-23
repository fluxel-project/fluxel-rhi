//! Immediate WebGL framebuffer attachment and per-attachment load operations.
use super::super::owner::{BrowserEncoder, BrowserOffscreenPass, BrowserTexture};
use crate::backend::gl::api::{GlFormat, GlTextureDimension};
use fluxel_rhi_core::api::{ClearColor, Error, LoadOp, Rect, Result, StoreOp};
use js_sys::Array;
use wasm_bindgen::JsValue;
use web_sys::WebGl2RenderingContext as Gl;

pub(crate) struct BrowserColorAttachment<'a> {
    pub texture: Option<&'a BrowserTexture>,
    pub surface_format: Option<GlFormat>,
    pub mip_level: u32,
    pub layer: u32,
    pub load: LoadOp<ClearColor>,
    pub store: StoreOp,
    pub resolve: Option<BrowserResolveAttachment<'a>>,
}
pub(crate) struct BrowserResolveAttachment<'a> {
    pub texture: Option<&'a BrowserTexture>,
    pub mip_level: u32,
    pub layer: u32,
}
pub(crate) struct BrowserDepthAttachment<'a> {
    pub texture: &'a BrowserTexture,
    pub aspect: crate::backend::gl::api::GlTextureAspect,
    pub mip_level: u32,
    pub layer: u32,
    pub depth: Option<(LoadOp<f32>, StoreOp)>,
    pub stencil: Option<(LoadOp<u32>, StoreOp)>,
}
fn array(values: &[u32]) -> JsValue {
    Array::from_iter(values.iter().map(|v| JsValue::from_f64(f64::from(*v)))).into()
}
pub(super) fn attach(raw: &Gl, attachment: u32, texture: &BrowserTexture, mip: u32, layer: u32) {
    if let Some(renderbuffer) = &texture.renderbuffer {
        raw.framebuffer_renderbuffer(
            Gl::FRAMEBUFFER,
            attachment,
            Gl::RENDERBUFFER,
            Some(renderbuffer),
        );
        return;
    }
    let Some(texture_raw) = texture.raw.as_ref() else {
        return;
    };
    match texture.desc.dimension {
        GlTextureDimension::D2 if texture.desc.extent.depth_or_layers == 1 => raw
            .framebuffer_texture_2d(
                Gl::FRAMEBUFFER,
                attachment,
                Gl::TEXTURE_2D,
                Some(texture_raw),
                mip as i32,
            ),
        _ => raw.framebuffer_texture_layer(
            Gl::FRAMEBUFFER,
            attachment,
            Some(texture_raw),
            mip as i32,
            layer as i32,
        ),
    }
}
pub(super) fn depth_attachment(format: GlFormat) -> u32 {
    match format {
        GlFormat::Depth24PlusStencil8 | GlFormat::Depth32FloatStencil8 => {
            Gl::DEPTH_STENCIL_ATTACHMENT
        }
        GlFormat::Stencil8 => Gl::STENCIL_ATTACHMENT,
        _ => Gl::DEPTH_ATTACHMENT,
    }
}
impl BrowserEncoder {
    pub(crate) fn begin_offscreen_attachments(
        &mut self,
        colors: &[Option<BrowserColorAttachment<'_>>],
        depth: Option<BrowserDepthAttachment<'_>>,
        extent: [u32; 2],
        render_area: Rect,
    ) -> Result<()> {
        self.begin_attachments_impl(colors, depth, extent, render_area, None)
    }

    /// Starts a pass over consecutive array layers using OVR_multiview2.
    pub(crate) fn begin_multiview_attachments(
        &mut self,
        colors: &[Option<BrowserColorAttachment<'_>>],
        depth: Option<BrowserDepthAttachment<'_>>,
        extent: [u32; 2],
        render_area: Rect,
        view_count: u32,
    ) -> Result<()> {
        self.begin_attachments_impl(colors, depth, extent, render_area, Some(view_count))
    }

    fn begin_attachments_impl(
        &mut self,
        colors: &[Option<BrowserColorAttachment<'_>>],
        depth: Option<BrowserDepthAttachment<'_>>,
        extent: [u32; 2],
        render_area: Rect,
        view_count: Option<u32>,
    ) -> Result<()> {
        if render_area
            .x
            .checked_add(render_area.width)
            .is_none_or(|x| x > extent[0])
            || render_area
                .y
                .checked_add(render_area.height)
                .is_none_or(|y| y > extent[1])
            || render_area.width == 0
            || render_area.height == 0
        {
            return Err(Error::InvalidRange);
        }
        if extent.contains(&0) || (colors.iter().all(Option::is_none) && depth.is_none()) {
            return Err(Error::InvalidRange);
        }
        if colors
            .iter()
            .flatten()
            .filter(|a| a.texture.is_none())
            .count()
            > 1
        {
            return Err(Error::InvalidRange);
        }
        for a in colors.iter().flatten() {
            if let Some(t) = a.texture {
                self.same_owner(&t.owner)?;
            } else if a
                .surface_format
                .and_then(super::super::format_map::internal_format)
                .is_none()
            {
                return Err(Error::Unsupported("WebGL2 surface attachment format"));
            }
            if let Some(r) = &a.resolve {
                if let Some(t) = r.texture {
                    self.same_owner(&t.owner)?;
                }
            }
        }
        if let Some(a) = &depth {
            self.same_owner(&a.texture.owner)?;
        }
        if let Some(count) = view_count {
            if count == 0 || count > i32::MAX as u32 {
                return Err(Error::InvalidRange);
            }
            let mut common_base = None;
            let mut check = |texture: &BrowserTexture, mip: u32, layer: u32| -> Result<()> {
                if texture.desc.dimension != GlTextureDimension::D2Array
                    || texture.desc.sample_count != 1
                    || texture.raw.is_none()
                {
                    return Err(Error::Unsupported(
                        "WebGL2 multiview requires single-sampled array textures",
                    ));
                }
                let bound = texture.desc.mip_extent(mip).ok_or(Error::InvalidRange)?;
                if bound.width != extent[0]
                    || bound.height != extent[1]
                    || layer
                        .checked_add(count)
                        .is_none_or(|end| end > bound.depth_or_layers)
                {
                    return Err(Error::InvalidRange);
                }
                if common_base.is_some_and(|base| base != layer) {
                    return Err(Error::Unsupported(
                        "WebGL2 multiview attachment base layers must match",
                    ));
                }
                common_base = Some(layer);
                Ok(())
            };
            for a in colors.iter().flatten() {
                if a.resolve.is_some() {
                    return Err(Error::Unsupported("WebGL2 multiview resolve"));
                }
                let t = a
                    .texture
                    .ok_or(Error::Unsupported("WebGL2 multiview surface attachment"))?;
                check(t, a.mip_level, a.layer)?;
            }
            if let Some(a) = &depth {
                check(a.texture, a.mip_level, a.layer)?;
            }
        }
        let (framebuffer, surface_color, surface_slot) = {
            let mut state = self.owner.state.borrow_mut();
            state.ensure_encoder(self.epoch)?;
            state.ensure_active()?;
            if state.raster_extent.is_some() || self.has_owned_framebuffer_pass() {
                return Err(Error::InvalidState);
            }
            let raw = &state.context.raw;
            let multiview = if let Some(count) = view_count {
                let Some(extension) = super::multiview::MultiviewExtension::acquire(raw) else {
                    return Err(Self::allocation_error(
                        &mut state,
                        Error::Unsupported("WebGL2 OVR_multiview2"),
                    ));
                };
                let max = raw
                    .get_parameter(0x9631)
                    .ok()
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0);
                if f64::from(count) > max {
                    return Err(Error::InvalidRange);
                }
                Some(extension)
            } else {
                None
            };
            let Some(framebuffer) = raw.create_framebuffer() else {
                return Err(Self::allocation_error(&mut state, Error::OutOfMemory));
            };
            raw.bind_framebuffer(Gl::FRAMEBUFFER, Some(&framebuffer));
            let mut surface_color = None;
            let mut surface_slot = None;
            let mut attach_failed = false;
            let buffers: Vec<u32> = colors
                .iter()
                .enumerate()
                .map(|(i, a)| {
                    if let Some(a) = a {
                        if let Some(texture) = a.texture {
                            if let Some(extension) = &multiview {
                                attach_failed |= extension
                                    .attach(
                                        Gl::COLOR_ATTACHMENT0 + i as u32,
                                        texture.raw.as_ref().expect("validated multiview texture"),
                                        a.mip_level,
                                        a.layer,
                                        view_count.expect("multiview count"),
                                    )
                                    .is_err();
                            } else {
                                attach(
                                    raw,
                                    Gl::COLOR_ATTACHMENT0 + i as u32,
                                    texture,
                                    a.mip_level,
                                    a.layer,
                                );
                            }
                        } else if let Some(color) = raw.create_texture() {
                            raw.bind_texture(Gl::TEXTURE_2D, Some(&color));
                            raw.tex_storage_2d(
                                Gl::TEXTURE_2D,
                                1,
                                super::super::format_map::internal_format(
                                    a.surface_format.expect("validated surface format"),
                                )
                                .expect("validated surface format mapping"),
                                extent[0] as i32,
                                extent[1] as i32,
                            );
                            raw.framebuffer_texture_2d(
                                Gl::FRAMEBUFFER,
                                Gl::COLOR_ATTACHMENT0 + i as u32,
                                Gl::TEXTURE_2D,
                                Some(&color),
                                0,
                            );
                            surface_color = Some(color);
                            surface_slot = Some(i as u32);
                        }
                        Gl::COLOR_ATTACHMENT0 + i as u32
                    } else {
                        Gl::NONE
                    }
                })
                .collect();
            raw.draw_buffers(&array(&buffers));
            if let Some(a) = &depth {
                let point = match a.aspect {
                    crate::backend::gl::api::GlTextureAspect::DepthOnly => Gl::DEPTH_ATTACHMENT,
                    crate::backend::gl::api::GlTextureAspect::StencilOnly => Gl::STENCIL_ATTACHMENT,
                    _ => depth_attachment(a.texture.desc.format),
                };
                if let Some(extension) = &multiview {
                    attach_failed |= extension
                        .attach(
                            point,
                            a.texture.raw.as_ref().expect("validated multiview texture"),
                            a.mip_level,
                            a.layer,
                            view_count.expect("multiview count"),
                        )
                        .is_err();
                } else {
                    attach(raw, point, a.texture, a.mip_level, a.layer);
                }
            }
            let status = raw.check_framebuffer_status(Gl::FRAMEBUFFER);
            let error = raw.get_error();
            if attach_failed
                || (colors.iter().flatten().any(|a| a.texture.is_none()) && surface_color.is_none())
                || status != Gl::FRAMEBUFFER_COMPLETE
                || error != Gl::NO_ERROR
            {
                raw.bind_framebuffer(Gl::FRAMEBUFFER, None);
                if let Some(color) = &surface_color {
                    raw.delete_texture(Some(color));
                }
                raw.delete_framebuffer(Some(&framebuffer));
                return Err(Self::allocation_error(
                    &mut state,
                    Error::Backend("WebGL2 offscreen framebuffer is incomplete".into()),
                ));
            }
            (framebuffer, surface_color, surface_slot)
        };
        let surface_srgb = colors
            .iter()
            .flatten()
            .any(|a| a.texture.is_none() && a.surface_format == Some(GlFormat::Rgba8Srgb));
        let mut conversion = Ok(());
        let result = self.with_context("begin_offscreen_attachments", |raw| {
            if let Some(slot) = surface_slot {
                raw.disable(Gl::SCISSOR_TEST);
                raw.bind_framebuffer(Gl::READ_FRAMEBUFFER, None);
                raw.bind_framebuffer(Gl::DRAW_FRAMEBUFFER, Some(&framebuffer));
                let select: Vec<u32> = (0..colors.len())
                    .map(|i| {
                        if i as u32 == slot {
                            Gl::COLOR_ATTACHMENT0 + slot
                        } else {
                            Gl::NONE
                        }
                    })
                    .collect();
                raw.draw_buffers(&array(&select));
                if surface_srgb {
                    conversion = super::surface_color::preload(raw, &framebuffer, slot, extent);
                } else {
                    raw.blit_framebuffer(
                        0,
                        extent[1] as i32,
                        extent[0] as i32,
                        0,
                        0,
                        0,
                        extent[0] as i32,
                        extent[1] as i32,
                        Gl::COLOR_BUFFER_BIT,
                        Gl::NEAREST,
                    );
                }
                let buffers: Vec<u32> = colors
                    .iter()
                    .enumerate()
                    .map(|(i, a)| {
                        if a.is_some() {
                            Gl::COLOR_ATTACHMENT0 + i as u32
                        } else {
                            Gl::NONE
                        }
                    })
                    .collect();
                raw.draw_buffers(&array(&buffers));
                raw.bind_framebuffer(Gl::FRAMEBUFFER, Some(&framebuffer));
            }
            raw.viewport(0, 0, extent[0] as i32, extent[1] as i32);
            raw.enable(Gl::SCISSOR_TEST);
            raw.scissor(
                render_area.x as i32,
                render_area.y as i32,
                render_area.width as i32,
                render_area.height as i32,
            );
            raw.color_mask(true, true, true, true);
            raw.depth_mask(true);
            raw.stencil_mask(u32::MAX);
            for (i, a) in colors.iter().enumerate() {
                if let Some(a) = a {
                    match a.load {
                        LoadOp::Clear(ClearColor::Float(v)) => {
                            raw.clear_bufferfv_with_f32_array(Gl::COLOR, i as i32, &v)
                        }
                        LoadOp::Clear(ClearColor::Uint(v)) => {
                            raw.clear_bufferuiv_with_u32_array(Gl::COLOR, i as i32, &v)
                        }
                        LoadOp::Clear(ClearColor::Sint(v)) => {
                            raw.clear_bufferiv_with_i32_array(Gl::COLOR, i as i32, &v)
                        }
                        _ => {}
                    }
                }
            }
            if let Some(a) = &depth {
                if let Some((LoadOp::Clear(v), _)) = a.depth {
                    raw.clear_bufferfv_with_f32_array(Gl::DEPTH, 0, &[v]);
                }
                if let Some((LoadOp::Clear(v), _)) = a.stencil {
                    raw.clear_bufferiv_with_i32_array(Gl::STENCIL, 0, &[v as i32]);
                }
            }
            raw.disable(Gl::SCISSOR_TEST);
        });
        if let Err(e) = result.and(conversion) {
            let state = self.owner.state.borrow();
            state.context.raw.bind_framebuffer(Gl::FRAMEBUFFER, None);
            if let Some(color) = &surface_color {
                state.context.raw.delete_texture(Some(color));
            }
            state.context.raw.delete_framebuffer(Some(&framebuffer));
            return Err(e);
        }
        let mut discard = Vec::new();
        for (i, a) in colors.iter().enumerate() {
            if a.as_ref()
                .is_some_and(|a| matches!(a.store, StoreOp::Discard))
            {
                discard.push(Gl::COLOR_ATTACHMENT0 + i as u32);
            }
        }
        if let Some(a) = &depth {
            if a.depth.is_some_and(|(_, s)| matches!(s, StoreOp::Discard)) {
                discard.push(Gl::DEPTH_ATTACHMENT);
            }
            if a.stencil
                .is_some_and(|(_, s)| matches!(s, StoreOp::Discard))
            {
                discard.push(Gl::STENCIL_ATTACHMENT);
            }
        }
        if render_area.x != 0
            || render_area.y != 0
            || render_area.width != extent[0]
            || render_area.height != extent[1]
        {
            discard.clear();
        }
        self.owner.state.borrow_mut().raster_extent = Some(extent);
        let resolves = colors
            .iter()
            .enumerate()
            .filter_map(|(slot, a)| {
                a.as_ref().and_then(|a| {
                    a.resolve.as_ref().map(|r| {
                        (
                            slot as u32,
                            r.texture.cloned(),
                            r.mip_level,
                            r.layer,
                            a.texture.expect("resolve source is a texture").desc.format,
                        )
                    })
                })
            })
            .collect();
        self.offscreen_pass = Some(BrowserOffscreenPass {
            framebuffer,
            resolves,
            render_area,
            surface_color,
            surface_slot,
            surface_srgb,
            colors: colors
                .iter()
                .flatten()
                .filter_map(|a| a.texture.cloned())
                .collect(),
            depth: depth.map(|a| a.texture.clone()),
            extent,
            discard,
        });
        Ok(())
    }
}
