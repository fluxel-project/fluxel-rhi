//! WebGL2 cube views mirrored from array layers for sampling.
//!
//! A mirror is private native storage: it is refreshed from the source directly
//! before sampling, including writes issued after bind. No CPU command replay is
//! involved; the owner retains both native objects through the immediate call.
use super::owner::{BrowserEncoder, BrowserTexture};
use crate::backend::gl::api::{
    GlFamilyApi as _, GlFormat, GlFormatResourceKind, GlTextureDimension,
};
use fluxel_rhi_core::api::{Error, Result};
use std::rc::Rc;
use wasm_bindgen::JsCast;
use web_sys::{WebGl2RenderingContext as Gl, WebGlFramebuffer, WebGlTexture};

impl BrowserTexture {
    pub(crate) fn create_cube_mirror(
        &self,
        base_mip: u32,
        mip_count: u32,
        base_layer: u32,
    ) -> Result<BrowserTexture> {
        let desc = self.desc;
        if desc.dimension != GlTextureDimension::D2Array
            || !cube_mirror_format(desc.format)
            || desc.sample_count != 1
            || mip_count == 0
            || base_mip
                .checked_add(mip_count)
                .is_none_or(|v| v > desc.mip_level_count)
            || base_layer
                .checked_add(6)
                .is_none_or(|v| v > desc.extent.depth_or_layers)
            || (desc.extent.width >> base_mip).max(1) != (desc.extent.height >> base_mip).max(1)
        {
            return Err(Error::InvalidRange);
        }
        self.texture()?;
        let mut state = self.owner.state.borrow_mut();
        state.ensure_active()?;
        let discovery = state.context.discovery();
        let side = (desc.extent.width >> base_mip).max(1);
        if side > discovery.limits().max_cube_map_texture_size {
            return Err(Error::InvalidRange);
        }
        if !discovery
            .formats()
            .get_for(GlFormatResourceKind::Texture, desc.format, 1)
            .is_some_and(|facts| facts.renderable)
        {
            return Err(Error::Unsupported(
                "WebGL2 cube mirror source format is not renderable",
            ));
        }
        let gl = &state.context.raw;
        let old = gl
            .get_parameter(Gl::TEXTURE_BINDING_CUBE_MAP)
            .map_err(|_| Error::InvalidState)?
            .dyn_into::<WebGlTexture>()
            .ok();
        let raw = gl.create_texture().ok_or(Error::OutOfMemory)?;
        gl.bind_texture(Gl::TEXTURE_CUBE_MAP, Some(&raw));
        gl.tex_parameteri(
            Gl::TEXTURE_CUBE_MAP,
            Gl::TEXTURE_MAX_LEVEL,
            (mip_count - 1) as i32,
        );
        gl.tex_storage_2d(
            Gl::TEXTURE_CUBE_MAP,
            mip_count as i32,
            super::format_map::internal_format(desc.format)
                .ok_or(Error::Unsupported("WebGL2 cube mirror format"))?,
            (desc.extent.width >> base_mip).max(1) as i32,
            (desc.extent.height >> base_mip).max(1) as i32,
        );
        gl.bind_texture(Gl::TEXTURE_CUBE_MAP, old.as_ref());
        if gl.is_context_lost() {
            state.mark_lost();
            return Err(Error::DeviceLost);
        }
        if gl.get_error() != Gl::NO_ERROR {
            gl.delete_texture(Some(&raw));
            return Err(Error::Backend(
                "WebGL2 cube mirror allocation failed".into(),
            ));
        }
        let mut mirror_desc = desc;
        mirror_desc.dimension = GlTextureDimension::Cube;
        mirror_desc.extent.width = (desc.extent.width >> base_mip).max(1);
        mirror_desc.extent.height = (desc.extent.height >> base_mip).max(1);
        mirror_desc.extent.depth_or_layers = 6;
        mirror_desc.mip_level_count = mip_count;
        drop(state);
        Ok(BrowserTexture {
            owner: self.owner.clone(),
            raw: Some(raw),
            renderbuffer: None,
            desc: mirror_desc,
            lifetime: Rc::new(()),
        })
    }
}

impl BrowserEncoder {
    pub(crate) fn refresh_cube_mirror(
        &mut self,
        source: &BrowserTexture,
        mirror: &BrowserTexture,
        base_mip: u32,
        mip_count: u32,
        base_layer: u32,
    ) -> Result<()> {
        self.same_owner(&source.owner)?;
        self.same_owner(&mirror.owner)?;
        if source.desc.dimension != GlTextureDimension::D2Array
            || mirror.desc.dimension != GlTextureDimension::Cube
            || !cube_mirror_format(source.desc.format)
            || mirror.desc.format != source.desc.format
            || mip_count != mirror.desc.mip_level_count
            || mip_count == 0
            || base_mip
                .checked_add(mip_count)
                .is_none_or(|v| v > source.desc.mip_level_count)
            || base_layer
                .checked_add(6)
                .is_none_or(|v| v > source.desc.extent.depth_or_layers)
            || mirror.desc.extent.width != (source.desc.extent.width >> base_mip).max(1)
            || mirror.desc.extent.height != (source.desc.extent.height >> base_mip).max(1)
        {
            return Err(Error::InvalidRange);
        }
        let source_raw = source.texture()?.clone();
        let mirror_raw = mirror.texture()?.clone();
        let mut inner = Ok(());
        self.with_context("refresh cube mirror", |gl| {
            let old_fb = match gl.get_parameter(Gl::READ_FRAMEBUFFER_BINDING) {
                Ok(v) => v.dyn_into::<WebGlFramebuffer>().ok(),
                Err(_) => {
                    inner = Err(Error::InvalidState);
                    return;
                }
            };
            let old_tex = match gl.get_parameter(Gl::TEXTURE_BINDING_CUBE_MAP) {
                Ok(v) => v.dyn_into::<WebGlTexture>().ok(),
                Err(_) => {
                    inner = Err(Error::InvalidState);
                    return;
                }
            };
            let Some(fb) = gl.create_framebuffer() else {
                inner = Err(Error::OutOfMemory);
                return;
            };
            gl.bind_framebuffer(Gl::READ_FRAMEBUFFER, Some(&fb));
            gl.bind_texture(Gl::TEXTURE_CUBE_MAP, Some(&mirror_raw));
            'levels: for mip in 0..mip_count {
                let width = (mirror.desc.extent.width >> mip).max(1) as i32;
                let height = (mirror.desc.extent.height >> mip).max(1) as i32;
                for face in 0..6 {
                    gl.framebuffer_texture_layer(
                        Gl::READ_FRAMEBUFFER,
                        Gl::COLOR_ATTACHMENT0,
                        Some(&source_raw),
                        (base_mip + mip) as i32,
                        (base_layer + face) as i32,
                    );
                    if gl.check_framebuffer_status(Gl::READ_FRAMEBUFFER) != Gl::FRAMEBUFFER_COMPLETE
                    {
                        inner = Err(Error::Unsupported("WebGL2 cube mirror source framebuffer"));
                        break 'levels;
                    }
                    gl.copy_tex_sub_image_2d(
                        Gl::TEXTURE_CUBE_MAP_POSITIVE_X + face,
                        mip as i32,
                        0,
                        0,
                        0,
                        0,
                        width,
                        height,
                    );
                    if gl.get_error() != Gl::NO_ERROR {
                        inner = Err(Error::Backend("WebGL2 cube mirror copy failed".into()));
                        break 'levels;
                    }
                }
            }
            gl.bind_texture(Gl::TEXTURE_CUBE_MAP, old_tex.as_ref());
            gl.bind_framebuffer(Gl::READ_FRAMEBUFFER, old_fb.as_ref());
            gl.delete_framebuffer(Some(&fb));
        })?;
        inner
    }
}

fn cube_mirror_format(format: GlFormat) -> bool {
    matches!(
        format,
        GlFormat::Rgba8Unorm | GlFormat::Rgba8Uint | GlFormat::Rgba8Sint
    )
}
