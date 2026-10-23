//! Per-context WebGL OVR multiview attachment completeness probe.
use wasm_bindgen::JsCast;
use web_sys::{WebGl2RenderingContext as Gl, WebGlFramebuffer, WebGlTexture};

pub(super) fn probe(raw: &Gl) -> bool {
    let Some(extension) = super::encoder::multiview::MultiviewExtension::acquire(raw) else {
        return false;
    };
    let old_read = raw
        .get_parameter(Gl::READ_FRAMEBUFFER_BINDING)
        .ok()
        .and_then(|v| v.dyn_into::<WebGlFramebuffer>().ok());
    let old_draw = raw
        .get_parameter(Gl::DRAW_FRAMEBUFFER_BINDING)
        .ok()
        .and_then(|v| v.dyn_into::<WebGlFramebuffer>().ok());
    let old_texture = raw
        .get_parameter(Gl::TEXTURE_BINDING_2D_ARRAY)
        .ok()
        .and_then(|v| v.dyn_into::<WebGlTexture>().ok());
    let Some(texture) = raw.create_texture() else {
        return false;
    };
    let Some(framebuffer) = raw.create_framebuffer() else {
        raw.delete_texture(Some(&texture));
        return false;
    };
    raw.bind_texture(Gl::TEXTURE_2D_ARRAY, Some(&texture));
    raw.tex_storage_3d(Gl::TEXTURE_2D_ARRAY, 1, Gl::RGBA8, 1, 1, 2);
    raw.bind_framebuffer(Gl::FRAMEBUFFER, Some(&framebuffer));
    let attached = extension
        .attach(Gl::COLOR_ATTACHMENT0, &texture, 0, 0, 2)
        .is_ok();
    let complete = raw.check_framebuffer_status(Gl::FRAMEBUFFER) == Gl::FRAMEBUFFER_COMPLETE;
    let error = raw.get_error();
    raw.bind_framebuffer(Gl::READ_FRAMEBUFFER, old_read.as_ref());
    raw.bind_framebuffer(Gl::DRAW_FRAMEBUFFER, old_draw.as_ref());
    raw.bind_texture(Gl::TEXTURE_2D_ARRAY, old_texture.as_ref());
    raw.delete_framebuffer(Some(&framebuffer));
    raw.delete_texture(Some(&texture));
    attached && complete && error == Gl::NO_ERROR && raw.get_error() == Gl::NO_ERROR
}
