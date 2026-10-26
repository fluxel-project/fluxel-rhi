//! Private stencil sampling mirrors, refreshed before each draw.
use super::owner::{BrowserEncoder, BrowserTexture};
use crate::backend::api::GlTextureDimension;
use fluxel_rhi_core::api::{Error, Result};
use std::rc::Rc;
use wasm_bindgen::JsCast;
use web_sys::{WebGl2RenderingContext as Gl, WebGlFramebuffer, WebGlTexture};

impl BrowserTexture {
    pub(crate) fn create_stencil_mirror(
        &self,
        base_mip: u32,
        mip_count: u32,
        dimension: GlTextureDimension,
        layers: u32,
    ) -> Result<Self> {
        let mut desc = self.desc;
        if !matches!(
            desc.dimension,
            GlTextureDimension::D2 | GlTextureDimension::D2Array
        ) || !matches!(
            dimension,
            GlTextureDimension::D2 | GlTextureDimension::D2Array
        ) || layers == 0
            || desc.sample_count != 1
            || mip_count == 0
            || base_mip
                .checked_add(mip_count)
                .is_none_or(|v| v > desc.mip_level_count)
        {
            return Err(Error::Unsupported("WebGL2 stencil mirror texture shape"));
        }
        self.texture()?;
        let mut state = self.owner.state.borrow_mut();
        state.ensure_active()?;
        let gl = &state.context.raw;
        let target = if dimension == GlTextureDimension::D2 {
            Gl::TEXTURE_2D
        } else {
            Gl::TEXTURE_2D_ARRAY
        };
        let binding = if dimension == GlTextureDimension::D2 {
            Gl::TEXTURE_BINDING_2D
        } else {
            Gl::TEXTURE_BINDING_2D_ARRAY
        };
        let old = gl
            .get_parameter(binding)
            .map_err(|_| Error::InvalidState)?
            .dyn_into::<WebGlTexture>()
            .ok();
        let raw = gl.create_texture().ok_or(Error::OutOfMemory)?;
        desc.extent.width = (desc.extent.width >> base_mip).max(1);
        desc.extent.height = (desc.extent.height >> base_mip).max(1);
        desc.mip_level_count = mip_count;
        desc.dimension = dimension;
        desc.extent.depth_or_layers = layers;
        let format = super::format_map::internal_format(desc.format)
            .ok_or(Error::Unsupported("WebGL2 stencil mirror format"))?;
        gl.bind_texture(target, Some(&raw));
        if dimension == GlTextureDimension::D2 {
            gl.tex_storage_2d(
                target,
                mip_count as i32,
                format,
                desc.extent.width as i32,
                desc.extent.height as i32,
            );
        } else {
            gl.tex_storage_3d(
                target,
                mip_count as i32,
                format,
                desc.extent.width as i32,
                desc.extent.height as i32,
                layers as i32,
            );
        }
        gl.tex_parameteri(target, Gl::TEXTURE_MAX_LEVEL, mip_count as i32 - 1);
        gl.tex_parameteri(
            target,
            Gl::TEXTURE_MIN_FILTER,
            Gl::NEAREST_MIPMAP_NEAREST as i32,
        );
        gl.tex_parameteri(target, Gl::TEXTURE_MAG_FILTER, Gl::NEAREST as i32);
        gl.bind_texture(target, old.as_ref());
        if gl.get_error() != Gl::NO_ERROR {
            gl.delete_texture(Some(&raw));
            return Err(Error::Backend("WebGL2 stencil mirror allocation".into()));
        }
        drop(state);
        Ok(Self {
            owner: self.owner.clone(),
            raw: Some(raw),
            renderbuffer: None,
            desc,
            lifetime: Rc::new(()),
        })
    }
}

impl BrowserEncoder {
    pub(crate) fn refresh_stencil_mirror(
        &mut self,
        source: &BrowserTexture,
        mirror: &BrowserTexture,
        base_mip: u32,
        base_layer: u32,
    ) -> Result<()> {
        self.same_owner(&source.owner)?;
        self.same_owner(&mirror.owner)?;
        let src = source.texture()?.clone();
        let dst = mirror.texture()?.clone();
        let desc = mirror.desc;
        let mut result = Ok(());
        self.with_context("refresh stencil mirror", |gl| {
            let read = gl
                .get_parameter(Gl::READ_FRAMEBUFFER_BINDING)
                .ok()
                .and_then(|v| v.dyn_into::<WebGlFramebuffer>().ok());
            let draw = gl
                .get_parameter(Gl::DRAW_FRAMEBUFFER_BINDING)
                .ok()
                .and_then(|v| v.dyn_into::<WebGlFramebuffer>().ok());
            let scissor = gl.is_enabled(Gl::SCISSOR_TEST);
            let Some(a) = gl.create_framebuffer() else {
                result = Err(Error::OutOfMemory);
                return;
            };
            let Some(b) = gl.create_framebuffer() else {
                gl.delete_framebuffer(Some(&a));
                result = Err(Error::OutOfMemory);
                return;
            };
            gl.disable(Gl::SCISSOR_TEST);
            for mip in 0..desc.mip_level_count {
                let width = (desc.extent.width >> mip).max(1) as i32;
                let height = (desc.extent.height >> mip).max(1) as i32;
                for layer in 0..desc.extent.depth_or_layers {
                    gl.bind_framebuffer(Gl::READ_FRAMEBUFFER, Some(&a));
                    if source.desc.dimension == GlTextureDimension::D2 {
                        gl.framebuffer_texture_2d(
                            Gl::READ_FRAMEBUFFER,
                            Gl::DEPTH_STENCIL_ATTACHMENT,
                            Gl::TEXTURE_2D,
                            Some(&src),
                            (base_mip + mip) as i32,
                        );
                    } else {
                        gl.framebuffer_texture_layer(
                            Gl::READ_FRAMEBUFFER,
                            Gl::DEPTH_STENCIL_ATTACHMENT,
                            Some(&src),
                            (base_mip + mip) as i32,
                            (base_layer + layer) as i32,
                        );
                    }
                    gl.read_buffer(Gl::NONE);
                    if gl.check_framebuffer_status(Gl::READ_FRAMEBUFFER) != Gl::FRAMEBUFFER_COMPLETE
                    {
                        result = Err(Error::Unsupported("WebGL2 stencil source framebuffer"));
                        break;
                    }
                    gl.bind_framebuffer(Gl::DRAW_FRAMEBUFFER, Some(&b));
                    if desc.dimension == GlTextureDimension::D2 {
                        gl.framebuffer_texture_2d(
                            Gl::DRAW_FRAMEBUFFER,
                            Gl::DEPTH_STENCIL_ATTACHMENT,
                            Gl::TEXTURE_2D,
                            Some(&dst),
                            mip as i32,
                        );
                    } else {
                        gl.framebuffer_texture_layer(
                            Gl::DRAW_FRAMEBUFFER,
                            Gl::DEPTH_STENCIL_ATTACHMENT,
                            Some(&dst),
                            mip as i32,
                            layer as i32,
                        );
                    }
                    gl.draw_buffers(&js_sys::Array::of1(&Gl::NONE.into()));
                    if gl.check_framebuffer_status(Gl::DRAW_FRAMEBUFFER) != Gl::FRAMEBUFFER_COMPLETE
                    {
                        result = Err(Error::Unsupported("WebGL2 stencil mirror framebuffer"));
                        break;
                    }
                    gl.blit_framebuffer(
                        0,
                        0,
                        width,
                        height,
                        0,
                        0,
                        width,
                        height,
                        Gl::STENCIL_BUFFER_BIT,
                        Gl::NEAREST,
                    );
                }
                if result.is_err() {
                    break;
                }
            }
            gl.bind_framebuffer(Gl::READ_FRAMEBUFFER, read.as_ref());
            gl.bind_framebuffer(Gl::DRAW_FRAMEBUFFER, draw.as_ref());
            if scissor {
                gl.enable(Gl::SCISSOR_TEST);
            }
            gl.delete_framebuffer(Some(&a));
            gl.delete_framebuffer(Some(&b));
        })?;
        result
    }
}
