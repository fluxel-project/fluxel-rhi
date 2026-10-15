//! Context-bound WebGL object wrappers and their final-release deletion.
//!
//! The last shared lifetime token triggers deletion on the browser owner;
//! context loss makes deletion unnecessary because WebGL invalidates names.

use super::*;
use crate::backend::gl::api::GlTextureDesc;

/// A WebGL buffer tied to exactly one browser owner.
#[derive(Clone)]
pub(crate) struct BrowserBuffer {
    pub(in crate::backend::gl::browser) owner: BrowserOwner,
    pub(in crate::backend::gl::browser) raw: WebGlBuffer,
    pub(in crate::backend::gl::browser) size: u64,
    pub(in crate::backend::gl::browser) lifetime: Rc<()>,
}

/// RGBA8 texture storage tied to exactly one browser owner.
#[derive(Clone)]
pub(crate) struct BrowserTexture {
    pub(in crate::backend::gl::browser) owner: BrowserOwner,
    pub(in crate::backend::gl::browser) raw: WebGlTexture,
    pub(in crate::backend::gl::browser) desc: GlTextureDesc,
    pub(in crate::backend::gl::browser) lifetime: Rc<()>,
}

/// Linked ESSL 3.00 raster program owned by one browser context.
#[derive(Clone)]
pub(crate) struct BrowserRasterProgram {
    pub(in crate::backend::gl::browser) owner: BrowserOwner,
    pub(in crate::backend::gl::browser) raw: WebGlProgram,
    pub(in crate::backend::gl::browser) lifetime: Rc<()>,
}

/// Vertex array state owned by one browser context.
#[derive(Clone)]
pub(crate) struct BrowserVertexArray {
    pub(in crate::backend::gl::browser) owner: BrowserOwner,
    pub(in crate::backend::gl::browser) raw: WebGlVertexArrayObject,
    pub(in crate::backend::gl::browser) lifetime: Rc<()>,
}

/// Sampler state tied to one browser context.
#[derive(Clone)]
pub(crate) struct BrowserSampler {
    pub(in crate::backend::gl::browser) owner: BrowserOwner,
    pub(in crate::backend::gl::browser) raw: WebGlSampler,
    pub(in crate::backend::gl::browser) lifetime: Rc<()>,
}

impl Drop for BrowserBuffer {
    fn drop(&mut self) {
        if Rc::strong_count(&self.lifetime) == 1 {
            self.owner.delete_buffer(&self.raw);
        }
    }
}

impl Drop for BrowserTexture {
    fn drop(&mut self) {
        if Rc::strong_count(&self.lifetime) == 1 {
            self.owner.delete_texture(&self.raw);
        }
    }
}

impl Drop for BrowserRasterProgram {
    fn drop(&mut self) {
        if Rc::strong_count(&self.lifetime) == 1 {
            self.owner.delete_program(&self.raw);
        }
    }
}

impl Drop for BrowserVertexArray {
    fn drop(&mut self) {
        if Rc::strong_count(&self.lifetime) == 1 {
            self.owner.delete_vertex_array(&self.raw);
        }
    }
}

impl Drop for BrowserSampler {
    fn drop(&mut self) {
        if Rc::strong_count(&self.lifetime) == 1 {
            self.owner.delete_sampler(&self.raw);
        }
    }
}
